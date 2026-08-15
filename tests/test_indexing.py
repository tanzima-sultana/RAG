import os
import pickle

import numpy as np
import pytest

from constants import LOCAL
from src.indexing import Indexing


def _write_embeddings(path, embedding_map):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(embedding_map, f)
    return path


def _write_chunks(path, chunk_ids_and_texts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    chunks_map = {
        cid: {"doc_id": "d1", "chunk_id": cid, "chunk_text": text}
        for cid, text in chunk_ids_and_texts
    }
    with open(path, "wb") as f:
        pickle.dump(chunks_map, f)
    return path


@pytest.fixture()
def indexer():
    return Indexing(mode=LOCAL, dataset_size=10, device="cpu")


def test_generate_faiss_index_writes_flatip_ivf_hnsw_and_ids(indexer, isolated_cwd):
    embedding_map = {f"d1_{i}": np.random.RandomState(i).rand(8).astype("float32") for i in range(50)}
    embedding_path = _write_embeddings("embeddings/10_fixed/size_8_overlap_2.pkl", embedding_map)

    flatip, ivf, hnsw, ids_path = indexer.generate_faiss_index(embedding_path, ivf_nlist=4, hnsw_m=8)

    assert flatip == "index/flatip/10_fixed/size_8_overlap_2"
    assert ivf == "index/ivf/10_fixed/size_8_overlap_2"
    assert hnsw == "index/hnsw/10_fixed/size_8_overlap_2"
    assert ids_path == "index/faiss_ids/10_fixed/size_8_overlap_2"

    for path in (flatip, ivf, hnsw, ids_path):
        assert os.path.exists(path)

    with open(ids_path, "rb") as f:
        chunk_ids = pickle.load(f)
    assert chunk_ids == list(embedding_map.keys())

    loaded = indexer.load_faiss(flatip)
    assert loaded.ntotal == 50


def test_generate_faiss_index_reuses_cached_files(indexer, isolated_cwd):
    embedding_map = {f"d1_{i}": np.random.RandomState(i).rand(8).astype("float32") for i in range(50)}
    embedding_path = _write_embeddings("embeddings/10_fixed/size_8_overlap_2.pkl", embedding_map)
    first = indexer.generate_faiss_index(embedding_path, ivf_nlist=4, hnsw_m=8)

    os.remove(embedding_path)  # would blow up a recompute attempt
    second = indexer.generate_faiss_index(embedding_path, ivf_nlist=4, hnsw_m=8)

    assert first == second


def test_generate_faiss_index_raises_when_embedding_file_missing(indexer, isolated_cwd):
    with pytest.raises(FileNotFoundError):
        indexer.generate_faiss_index("embeddings/missing/size_8_overlap_2.pkl", ivf_nlist=4, hnsw_m=8)


def test_generate_bm25_index_writes_index_and_ids(indexer, isolated_cwd):
    chunk_path = _write_chunks(
        "chunks/10_fixed/size_8_overlap_2.pkl",
        [
            ("d1_0", "the quick brown fox"),
            ("d1_1", "jumps over the lazy dog"),
            ("d1_2", "a completely different sentence about cars"),
        ],
    )

    bm25_path, ids_path = indexer.generate_bm25_index(chunk_path)

    assert bm25_path == "index/bm25/10_fixed/size_8_overlap_2"
    assert ids_path == "index/bm25_ids/10_fixed/size_8_overlap_2"
    assert os.path.exists(bm25_path)
    assert os.path.exists(ids_path)

    with open(bm25_path, "rb") as f:
        bm25_index = pickle.load(f)
    with open(ids_path, "rb") as f:
        chunk_ids = pickle.load(f)

    assert chunk_ids == ["d1_0", "d1_1", "d1_2"]
    scores = bm25_index.get_scores(indexer.tokenize_chunk_text("quick fox"))
    assert len(scores) == 3
    assert scores[0] > scores[1]  # "quick fox" only overlaps with the first chunk
    assert scores[0] > scores[2]


def test_generate_bm25_index_reuses_cached_files(indexer, isolated_cwd):
    chunk_path = _write_chunks("chunks/10_fixed/size_8_overlap_2.pkl", [("d1_0", "the quick brown fox")])
    first = indexer.generate_bm25_index(chunk_path)

    os.remove(chunk_path)
    second = indexer.generate_bm25_index(chunk_path)

    assert first == second


def test_generate_bm25_index_returns_none_none_on_failure(indexer, isolated_cwd):
    bad_chunk_path = "chunks/10_fixed/does_not_exist.pkl"
    result = indexer.generate_bm25_index(bad_chunk_path)
    assert result == (None, None)
