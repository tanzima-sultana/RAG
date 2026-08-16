import os
import pickle

import numpy as np
import pytest

from src.vector_db import VectorDB
from tests.fakes import FakeQdrantClient


@pytest.fixture()
def vdb(monkeypatch):
    monkeypatch.setattr("src.vector_db.QdrantClient", FakeQdrantClient)
    return VectorDB(db_name="vectordb")


def _write_embeddings(path, n):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    embedding_map = {f"d1_{i}": np.random.RandomState(i).rand(8).astype("float32") for i in range(n)}
    with open(path, "wb") as f:
        pickle.dump(embedding_map, f)
    return path


def test_create_vector_db_creates_collection_and_inserts_points(vdb, isolated_cwd):
    embedding_path = _write_embeddings("embeddings/10_fixed/size_8_overlap_2.pkl", 5)

    name = vdb.create_vector_db(embedding_path)

    assert name == "vectordb_10_fixed_size_8_overlap_2"
    assert vdb.client.collection_exists(name)
    assert len(vdb.client.collections[name]["points"]) == 5


def test_create_vector_db_skips_creation_when_collection_already_exists(vdb, isolated_cwd):
    embedding_path = _write_embeddings("embeddings/10_fixed/size_8_overlap_2.pkl", 5)
    name = vdb.create_vector_db(embedding_path)

    os.remove(embedding_path)  # would blow up a recompute attempt
    second = vdb.create_vector_db(embedding_path)

    assert second == name


def test_create_vector_db_force_recreate_replaces_stale_vectors(vdb, isolated_cwd):
    embedding_path = _write_embeddings("embeddings/10_fixed/size_8_overlap_2.pkl", 8)
    name = vdb.create_vector_db(embedding_path)
    assert len(vdb.client.collections[name]["points"]) == 8

    # regenerate the embeddings file with fewer, different vectors at the same path,
    # simulating a chunking/embedding change that doesn't rename anything. If the old
    # collection isn't actually deleted first, the leftover stale points (ids 3-7)
    # would still be sitting there after the fresh upsert.
    os.remove(embedding_path)
    _write_embeddings(embedding_path, 3)

    second = vdb.create_vector_db(embedding_path, force_recreate=True)

    assert second == name
    assert len(vdb.client.collections[name]["points"]) == 3


def test_create_vector_db_without_force_recreate_still_skips_when_it_exists(vdb, isolated_cwd):
    embedding_path = _write_embeddings("embeddings/10_fixed/size_8_overlap_2.pkl", 5)
    name = vdb.create_vector_db(embedding_path)

    os.remove(embedding_path)  # would blow up a recompute attempt
    second = vdb.create_vector_db(embedding_path, force_recreate=False)

    assert second == name
    assert len(vdb.client.collections[name]["points"]) == 5


def test_create_vector_db_returns_none_when_embedding_file_missing(vdb, isolated_cwd):
    result = vdb.create_vector_db("embeddings/does_not_exist/size_8_overlap_2.pkl")
    assert result is None


def test_insert_update_embeddings_batches_upserts(vdb):
    name = "some_collection"
    vdb.client.create_collection(name, vectors_config=None)
    chunk_ids = [f"c{i}" for i in range(5)]
    embeddings = [np.zeros(4) for _ in range(5)]

    vdb.insert_update_embeddings(name, chunk_ids, embeddings, batch_size=2)

    assert len(vdb.client.collections[name]["points"]) == 5
