import pickle

import numpy as np
import pytest

from src.local.embedding import Embedding
from tests.fakes import FakeSentenceTransformer


@pytest.fixture()
def embedder(monkeypatch):
    monkeypatch.setattr("src.local.embedding.SentenceTransformer", FakeSentenceTransformer)
    return Embedding(dataset_size=10, device="cpu", model_name="fake-model")


def _write_chunks(path, chunk_ids_and_texts):
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    chunks_map = {
        cid: {"doc_id": "d1", "chunk_id": cid, "chunk_text": text, "chunk_size": len(text.split())}
        for cid, text in chunk_ids_and_texts
    }
    with open(path, "wb") as f:
        pickle.dump(chunks_map, f)
    return path


def test_generate_embeddings_writes_one_vector_per_chunk(embedder, isolated_cwd):
    chunk_path = _write_chunks(
        "chunks/10_fixed/size_8_overlap_2.pkl",
        [("d1_0", "one two three"), ("d1_1", "four five six")],
    )

    out_path = embedder.generate_embeddings(chunk_path)

    assert out_path == "embeddings/10_fixed/size_8_overlap_2.pkl"
    with open(out_path, "rb") as f:
        embedding_map = pickle.load(f)

    assert set(embedding_map.keys()) == {"d1_0", "d1_1"}
    for vec in embedding_map.values():
        assert isinstance(vec, np.ndarray)


def test_generate_embeddings_reuses_cached_file(embedder, isolated_cwd, monkeypatch):
    chunk_path = _write_chunks("chunks/10_fixed/size_8_overlap_2.pkl", [("d1_0", "one two three")])
    embedder.generate_embeddings(chunk_path)

    def boom(*args, **kwargs):
        raise AssertionError("should not re-encode when the embedding file already exists")

    monkeypatch.setattr(embedder.model, "encode", boom)

    out_path = embedder.generate_embeddings(chunk_path)
    assert out_path == "embeddings/10_fixed/size_8_overlap_2.pkl"


def test_generate_embeddings_returns_none_when_chunk_file_missing(embedder, isolated_cwd):
    out_path = embedder.generate_embeddings("chunks/does_not_exist/size_8_overlap_2.pkl")
    assert out_path is None
    assert not (isolated_cwd / "embeddings/does_not_exist/size_8_overlap_2.pkl").exists()
