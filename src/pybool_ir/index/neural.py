import json
import string
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Type

import lucene
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from lupyne import engine
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from pybool_ir.index.document import Document
from pybool_ir.index.pubmed import PubmedIndexer, PubmedArticle
from pybool_ir.query.ast import ASTNode, AtomNode, OperatorNode

Q = engine.queries.Query
D = engine.documents.Document

assert lucene.getVMEnv() or lucene.initVM()


class MemoryMappedArray:
    def __init__(self, path: Path, dim: int, dtype: Type = np.float16) -> None:
        self.path = path
        self.dim = dim
        self.dtype = dtype
        if not path.exists():
            self.array = np.empty((0, dim), dtype=dtype)
        else:
            file_size = path.stat().st_size
            if dim:
                num_elements = file_size // (dim * np.dtype(dtype).itemsize)
            else:
                num_elements = file_size // np.dtype(dtype).itemsize
            if dim:
                shape = (num_elements, dim)
            else:
                shape = (num_elements,)
            self.array = np.memmap(path, dtype=dtype, shape=shape)

    def add(self, array: np.ndarray) -> "MemoryMappedArray":
        length = self.array.shape[0] + array.shape[0]
        if self.dim:
            shape = (length, self.dim)
        else:
            shape = (length,)
        newarray = np.memmap(
            self.path,
            dtype=self.dtype,
            mode="r+" if self.path.exists() else "w+",
            shape=shape,
        )
        newarray[self.array.shape[0] :] = array
        self.array = newarray
        return self

    def flush(self) -> "MemoryMappedArray":
        if isinstance(self.array, np.memmap):
            self.array.flush()
        return self


def NeuralClassFactory(transformer_model_class: Type[PreTrainedModel]):
    def __init__(
        self,
        config: PretrainedConfig,
        dim: int,
        *args,
        **kwargs,
    ) -> None:
        transformer_model_class.__init__(self, config, *args, **kwargs)
        self.linear = torch.nn.Linear(config.hidden_size, dim, bias=False)

    neural_class = type(
        "NeuralModel", (transformer_model_class,), {"__init__": __init__}
    )
    return neural_class


class NeuralIndex:
    def __init__(
        self,
        index: engine.indexers.Indexer,
        index_path: Path | str,
        model_name_or_path: Path | str,
        prune_tokens: List[str] | None = None,
        dtype: Type = np.float16,
    ) -> None:
        self.index = index
        self.index_path = Path(index_path)
        self.dtype = dtype
        self.model_name_or_path = Path(model_name_or_path)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = self.load_model()
        # prune punctuation by default
        self.prune_tokens: torch.Tensor = self.tokenizer(
            ([] if prune_tokens is None else prune_tokens) + [string.punctuation],
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]

        if not self.index_path.exists():
            print(f"Index {self.index_path} does not exist, creating it now.")
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.embeddings = MemoryMappedArray(
            self.index_path / "embeddings.npy",
            dim=self.model.linear.out_features,
            dtype=dtype,
        )
        self.doc_lengths = MemoryMappedArray(
            self.index_path / "doc_lengths.npy", 0, np.int32
        )

    def search(self, node: ASTNode, n_hits: int | None = 10) -> Iterable[D]:
        scores = self._search(node)
        if n_hits is None:
            n_hits = len(scores)
        doc_ids = np.argsort(scores)[::-1][:n_hits].tolist()
        yield from (self.index[doc_id] for doc_id in doc_ids)

    def _search(self, node: ASTNode) -> np.ndarray:
        if isinstance(node, AtomNode):
            return self._search_atom(node)
        if isinstance(node, AtomNode):
            raise ValueError("Use NeuralPubmedQueryParser instead")
        assert isinstance(node, OperatorNode)
        child_scores = []
        for child in node.children:
            child_scores.append(self._search(child))
        if node.operator.lower() == "and":
            return np.minimum.reduce(child_scores)
        elif node.operator.lower() == "or":
            return np.maximum.reduce(child_scores)
        elif node.operator.lower() == "not":
            # TODO does this makes sense?!
            # we do not want to retrieve the opposite documents but exclude the ones
            # that are retrieved by the child query
            # therefore perhaps clamp to 0?!
            assert (
                len(child_scores) == 2
            ), "NOT operator should always have only two children"
            return np.maximum.reduce((1 - child_scores[0], child_scores[1]))
        else:
            raise ValueError(f"Unknown operator {node.operator}")

    def _search_atom(self, node: AtomNode) -> np.ndarray:
        if node.field.field != "tiab":
            hits = self.index.search(node.lucene_query, scores=False)
            scores = np.zeros(self.num_docs)
            scores[list(hits.ids)] = 1
            return scores
        tokenized = self.tokenizer(". " + node.query, return_tensors="pt")
        # 1 is the default special query token_id for neural
        tokenized.input_ids[:, 1] = 1
        tokenized = tokenized.to(self._device)
        with torch.no_grad():
            query_embeddings = self.model(**tokenized).last_hidden_state
            query_embeddings = self.model.linear(query_embeddings)
        # remove batch dim and special tokens
        query_embeddings = query_embeddings[0, 2:-1]
        query_embeddings = torch.nn.functional.normalize(
            query_embeddings, p=2, dim=1
        ).half()
        query_embeddings = query_embeddings.cpu().numpy()
        scores = np.dot(self.embeddings.array, query_embeddings.T)
        scores = (scores + 1) * 0.5
        split_idcs = np.cumsum(self.doc_lengths.array)[:-1].tolist()
        split_scores = np.split(scores, split_idcs)
        scores = np.array([s.max(0).mean(0) for s in split_scores])
        return scores

    @property
    def num_docs(self) -> int:
        return self.doc_lengths.array.shape[0]

    @property
    def num_tokens(self) -> int:
        return self.doc_lengths.array.sum()

    def load_model(self) -> PreTrainedModel:
        if self.model_name_or_path.is_dir():
            neural_config = json.loads(
                (self.model_name_or_path / "artifact.metadata").read_text()
            )
        else:
            neural_config = json.loads(
                Path(
                    hf_hub_download(
                        repo_id=str(self.model_name_or_path),
                        filename="artifact.metadata",
                    )
                ).read_text()
            )
            state_dict_path = hf_hub_download(
                repo_id=str(self.model_name_or_path),
                filename="pytorch_model.bin",
            )
            state_dict = torch.load(
                state_dict_path,
                map_location=self._device,
            )
            if "linear.weight" in state_dict:
                state_dict["bert.linear.weight"] = state_dict.pop("linear.weight")
                torch.save(state_dict, state_dict_path)
        config = AutoConfig.from_pretrained(self.model_name_or_path)
        model_class = AutoModel._model_mapping[type(config)]
        NeuralModel = NeuralClassFactory(model_class)
        return NeuralModel.from_pretrained(
            self.model_name_or_path, dim=neural_config["dim"]
        ).to(self._device)

    def add(self, docs: List[Document], index_fields: List[str] | None = None) -> None:
        if index_fields is None:
            index_fields = ["title", "abstract"]
        texts = []
        doc_ids = []
        for doc in docs:
            texts.append(". " + " ".join(getattr(doc, field) for field in index_fields))
            doc_ids.append(doc.id)
        tokenized = self.tokenizer(
            text=texts, padding=True, truncation=True, return_tensors="pt"
        )
        # 2 is the default special document token_id for neural
        tokenized.input_ids[:, 1] = 2
        mask = tokenized["attention_mask"].bool()
        mask = mask & ~(tokenized["input_ids"][..., None] == self.prune_tokens).any(-1)
        # remove special tokens
        mask[:, :2] = False
        mask = mask & ~(tokenized["input_ids"] == self.tokenizer.sep_token_id)
        doc_lengths = mask.sum(dim=1).cpu().numpy()
        tokenized = tokenized.to(self._device)
        with torch.no_grad():
            embeddings = self.model(**tokenized).last_hidden_state
            embeddings = self.model.linear(embeddings)
        embeddings = embeddings[mask]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).half()
        embeddings = embeddings.cpu().numpy()
        self.doc_lengths.add(doc_lengths)
        self.embeddings.add(embeddings)

    def commit(self) -> None:
        self.doc_lengths.flush()
        self.embeddings.flush()


class NeuralIndexer(PubmedIndexer):
    def __init__(
        self,
        index_path: Path | str,
        model_name_or_path: Path | str,
        store_fields: bool = False,
        store_termvectors: bool = False,
        optional_fields: List[str] | None = None,
        neural_index_fields: List[str] | None = None,
        prune_tokens: List[str] | None = None,
        dtype: Type = np.float16,
        batch_size: int = 64,
        index_only_neural: bool = False,
    ):
        super().__init__(
            index_path=index_path,
            store_fields=store_fields,
            store_termvectors=store_termvectors,
            optional_fields=optional_fields,
        )
        self.model_name_or_path = model_name_or_path
        self.neural_index_fields = neural_index_fields
        self.prune_tokens = prune_tokens
        self.dtype = dtype
        self.batch_size = batch_size
        self.index_only_neural = index_only_neural
        self.neural_index: NeuralIndex

    def add_documents_neural(
        self,
        docs: List[Document],
    ) -> None:
        self.neural_index.add(docs, index_fields=self.neural_index_fields)

    def _bulk_index(
        self,
        docs: Iterable[Document],
        total=None,
        optional_fields: Dict[str, Callable[[Document], Any]] | None = None,
    ) -> None:
        _docs = []
        for i, doc in tqdm(
            enumerate(docs), desc="indexing progress", position=1, total=total
        ):
            doc = self.process_document(doc)
            if i in self.index:
                continue
            _docs.append(doc)
            if not self.index_only_neural:
                self.add_document(doc, optional_fields)
            if (i + 1) % self.batch_size == 0:
                self.add_documents_neural(_docs)
                _docs = []
            if (i + 1) % 1_000 == 0:
                self.neural_index.commit()
                if not self.index_only_neural:
                    self.index.commit()
        if _docs:
            self.add_documents_neural(_docs)
            self.neural_index.commit()
        if not self.index_only_neural:
            self.index.commit()

    def retrieve(self, query: str):
        return self.neural_index.search(query, scores=False)

    def search(self, query: ASTNode, n_hits: int | None = 10) -> Iterable[Document]:
        hits = self.neural_index.search(query, n_hits=n_hits)
        for hit in hits:
            if self.store_fields:
                article: PubmedArticle = PubmedArticle.from_dict(
                    hit.dict(
                        "mesh_heading_list",
                        "mesh_qualifier_list",
                        "mesh_major_heading_list",
                        "supplementary_concept_list",
                        "keyword_list",
                        "publication_type",
                    )
                )
                yield article
            else:
                yield PubmedArticle.from_dict(hit.dict())

    def __enter__(self):
        super().__enter__()
        self.neural_index = NeuralIndex(
            self.index,
            self.index_path,
            self.model_name_or_path,
            prune_tokens=self.prune_tokens,
            dtype=self.dtype,
        )
        if self.neural_index.num_docs != self.index.maxDoc():
            raise RuntimeError("neural index and lucene index do not match")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.index.close()
