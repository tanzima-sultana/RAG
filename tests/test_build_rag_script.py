"""Orchestration tests for scripts/build_rag.py: each of the six build phases
(dataset, chunking, embedding, indexing, vector DB, eval-set) is replaced with a
lightweight fake so we can verify build_rag.py wires them together correctly and
assembles the final manifest — without paying for real model loads or Claude calls.
"""
import json
import runpy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = str(REPO_ROOT / "scripts" / "build_rag.py")


class FakeDataset:
    def __init__(self, mode, dataset_size):
        self.mode = mode
        self.dataset_size = dataset_size

    def load_parquet_dataset_local(self):
        return ["doc1", "doc2"]


class FakeChunking:
    def __init__(self, dataset, dataset_size, device, model_name):
        self.dataset = dataset
        self.dataset_size = dataset_size

    def compute_chunks(self, max_chunk_size, fix_chunk_overlap, semantic_threshold):
        return (
            f"chunks/{self.dataset_size}_fixed.pkl",
            f"chunks/{self.dataset_size}_sentence.pkl",
            f"chunks/{self.dataset_size}_semantic.pkl",
        )


class FakeEmbedding:
    def __init__(self, dataset_size, device, model_name):
        pass

    def generate_embeddings(self, chunk_path):
        return chunk_path.replace("chunks/", "embeddings/")


class FakeIndexing:
    def __init__(self, mode, dataset_size, device):
        pass

    def generate_faiss_index(self, embedding_path, ivf_nlist, hnsw_m):
        stem = embedding_path.replace("embeddings/", "")
        return f"index/flatip/{stem}", f"index/ivf/{stem}", f"index/hnsw/{stem}", f"index/faiss_ids/{stem}"

    def generate_bm25_index(self, chunk_path):
        stem = chunk_path.replace("chunks/", "")
        return f"index/bm25/{stem}", f"index/bm25_ids/{stem}"


class FakeVectorDB:
    def __init__(self, db_name):
        self.db_name = db_name

    def create_vector_db(self, embedding_path, force_recreate=False):
        return f"{self.db_name}_{embedding_path}"


class FakeEvalQA:
    def __init__(self, mock_run, mode, dataset_size, num_queries):
        pass

    def build_eval_set(self, chunking_type, chunks_path, min_chunk_size):
        return f"eval_qa/{chunking_type}"


@pytest.fixture()
def patched_collaborators(monkeypatch):
    monkeypatch.setattr("src.dataset.Dataset", FakeDataset)
    monkeypatch.setattr("src.local.chunking.Chunking", FakeChunking)
    monkeypatch.setattr("src.local.embedding.Embedding", FakeEmbedding)
    monkeypatch.setattr("src.indexing.Indexing", FakeIndexing)
    monkeypatch.setattr("src.vector_db.VectorDB", FakeVectorDB)
    monkeypatch.setattr("src.eval_qa.EvalQA", FakeEvalQA)


def run_build_rag(argv_tail):
    argv = ["build_rag.py"] + argv_tail
    old_argv = sys.argv
    sys.argv = argv
    try:
        runpy.run_path(BUILD_SCRIPT, run_name="__main__")
    finally:
        sys.argv = old_argv


BASE_ARGS = [
    "--mode", "local",
    "--device", "cpu",
    "--model_name", "fake-model",
    "--dataset_size", "10",
    "--mock_run", "1",
    "--num_queries", "2",
]


def test_build_rag_runs_all_phases_and_writes_manifest(patched_collaborators, isolated_cwd):
    run_build_rag(BASE_ARGS)

    manifest_path = isolated_cwd / "manifests" / "10_manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert set(manifest.keys()) == {"fixed", "sentence", "semantic"}

    fixed = manifest["fixed"]
    assert fixed["chunk_path"] == "chunks/10_fixed.pkl"
    assert fixed["flatip"] == "index/flatip/10_fixed.pkl"
    assert fixed["ivf"] == "index/ivf/10_fixed.pkl"
    assert fixed["hnsw"] == "index/hnsw/10_fixed.pkl"
    assert fixed["chunk_ids"] == "index/faiss_ids/10_fixed.pkl"
    assert fixed["bm25"] == "index/bm25/10_fixed.pkl"
    assert fixed["bm25_ids"] == "index/bm25_ids/10_fixed.pkl"
    assert fixed["vectordb"] == "vectordb_embeddings/10_fixed.pkl"
    assert fixed["eval_path"] == "eval_qa/fixed"

    assert manifest["semantic"]["eval_path"] == "eval_qa/semantic"


def test_build_rag_exits_when_dataset_load_fails(patched_collaborators, isolated_cwd, monkeypatch):
    monkeypatch.setattr(FakeDataset, "load_parquet_dataset_local", lambda self: None)

    with pytest.raises(SystemExit) as exc_info:
        run_build_rag(BASE_ARGS)

    assert exc_info.value.code == 1


def test_build_rag_exits_when_chunking_fails(patched_collaborators, isolated_cwd, monkeypatch):
    monkeypatch.setattr(FakeChunking, "compute_chunks", lambda self, *a, **k: (None, None, None))

    with pytest.raises(SystemExit) as exc_info:
        run_build_rag(BASE_ARGS)

    assert exc_info.value.code == 1


def test_build_rag_exits_when_vector_db_creation_fails(patched_collaborators, isolated_cwd, monkeypatch):
    monkeypatch.setattr(FakeVectorDB, "create_vector_db", lambda self, embedding_path, force_recreate=False: None)

    with pytest.raises(SystemExit) as exc_info:
        run_build_rag(BASE_ARGS)

    assert exc_info.value.code == 1


def test_build_rag_exits_when_eval_set_building_fails(patched_collaborators, isolated_cwd, monkeypatch):
    monkeypatch.setattr(FakeEvalQA, "build_eval_set", lambda self, *a, **k: None)

    with pytest.raises(SystemExit) as exc_info:
        run_build_rag(BASE_ARGS)

    assert exc_info.value.code == 1
