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
from pybool_ir.index.pubmed import PubmedArticle, PubmedIndexer
from pybool_ir.query.ast import ASTNode, AtomNode, OperatorNode

Q = engine.queries.Query
D = engine.documents.Document

assert lucene.getVMEnv() or lucene.initVM()


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
        for i, doc in tqdm(enumerate(docs), desc="indexing progress", position=1, total=total):
            doc = self.process_document(doc)
            if i in self.index:
                continue
            _docs.append(doc)
            if not self.index_only_neural:
                self.add_document(doc, optional_fields)
            # TODO index using lightning-ir
            if (i + 1) % self.batch_size == 0:
                self.add_documents_neural(_docs)
                _docs = []
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
