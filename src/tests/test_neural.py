from pathlib import Path

import numpy as np
from pytest import TempPathFactory

from pybool_ir.index.neural import NeuralIndexer, MemoryMappedArray
from pybool_ir.query.neural.parser import NeuralPubmedQueryParser


def test_embeddings(tmp_path_factory: TempPathFactory):
    embeddings_1 = MemoryMappedArray(
        tmp_path_factory.mktemp("embeddings") / "embeddings.npy", 128
    )
    embeddings_1.add(np.random.rand(100, 128))
    embeddings_1.add(np.random.rand(100, 128))
    embeddings_1.flush()
    embeddings_2 = MemoryMappedArray(embeddings_1.embeddings_path, 128)
    assert embeddings_1.array is not None
    assert embeddings_2.array is not None
    assert embeddings_1.array.shape == (200, 128)
    assert embeddings_2.array.shape == (200, 128)
    assert np.allclose(embeddings_1.array, embeddings_2.array)


def test_indexer(tmp_path_factory: TempPathFactory):
    with NeuralIndexer(
        tmp_path_factory.mktemp("index"), "colbert-ir/colbertv2.0", batch_size=4
    ) as ix:
        ix.bulk_index(Path(__file__).parent / "data" / "pubmed")
        index = ix.neural_index
        assert index.num_docs == 20
        assert index.num_tokens == 7262


def test_search():
    raw_query = (
        "('Acne Vulgaris'[Mesh] OR Acne[tiab] OR Blackheads[tiab] OR Whiteheads[tiab] "
        "OR Pimples[tiab]) AND ('Phototherapy'[Mesh] OR 'Blue light'[tiab] OR "
        "Phototherapy[tiab] OR Phototherapies[tiab] OR 'Photoradiation therapy'[tiab] "
        "OR 'Photoradiation Therapies'[tiab] OR 'Light Therapy'[tiab] OR "
        "'Light Therapies'[tiab]) AND (Randomized controlled trial[pt] OR controlled "
        "clinical trial[pt] OR randomized[tiab] OR randomised[tiab] OR placebo[tiab] "
        "OR 'drug therapy'[sh] OR randomly[tiab] OR trial[tiab] OR groups[tiab]) NOT "
        "(Animals[Mesh] not (Animals[Mesh] and Humans[Mesh]))"
    )
    query_parser = NeuralPubmedQueryParser()
    query = query_parser.parse_ast_lucene(raw_query)
    with NeuralIndexer(
        Path(__file__).parent / "data" / "index", "colbert-ir/colbertv2.0"
    ) as ix:
        hits = list(ix.search(query, None))
    assert len(hits) == 20
