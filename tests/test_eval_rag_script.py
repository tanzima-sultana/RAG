"""Orchestration tests for scripts/eval_rag.py: manifest + artifact loading, the
retrieval-mode dispatch (dense/bm25/hybrid/vectordb), and the evaluation call.
Retrieval and Evaluation are replaced with fakes so no real model inference or
Claude calls happen; the manifest points at small real FAISS/BM25 artifacts so the
artifact-loading phase itself is exercised for real.
"""
import json
import os
import pickle
import runpy
import sys
from pathlib import Path

import faiss
import numpy as np
import pytest
from rank_bm25 import BM25Okapi

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = str(REPO_ROOT / "scripts" / "eval_rag.py")


def _build_manifest(root: Path, dataset_size=10, chunking_type="fixed", index_type="flatip"):
    chunk_path = f"chunks/{dataset_size}_{chunking_type}.pkl"
    chunk_ids_path = f"index/faiss_ids/{dataset_size}_{chunking_type}"
    bm25_path = f"index/bm25/{dataset_size}_{chunking_type}"
    bm25_ids_path = f"index/bm25_ids/{dataset_size}_{chunking_type}"
    index_path = f"index/{index_type}/{dataset_size}_{chunking_type}"
    eval_path = f"eval_qa/{dataset_size}_2_{chunking_type}_mock"

    for path in (chunk_path, chunk_ids_path, bm25_path, bm25_ids_path, index_path, eval_path):
        os.makedirs((root / path).parent, exist_ok=True)

    chunks_map = {
        "d1_0": {"doc_id": "d1", "chunk_id": "d1_0", "chunk_text": "cats are pets"},
        "d2_0": {"doc_id": "d2", "chunk_id": "d2_0", "chunk_text": "rockets reach orbit"},
    }
    with open(root / chunk_path, "wb") as f:
        pickle.dump(chunks_map, f)

    chunk_ids = list(chunks_map.keys())
    vectors = np.random.RandomState(0).rand(len(chunk_ids), 4).astype("float32")
    index = faiss.IndexFlatIP(4)
    index.add(vectors)
    faiss.write_index(index, str(root / index_path))
    with open(root / chunk_ids_path, "wb") as f:
        pickle.dump(chunk_ids, f)

    bm25_index = BM25Okapi([c["chunk_text"].split() for c in chunks_map.values()])
    with open(root / bm25_path, "wb") as f:
        pickle.dump(bm25_index, f)
    with open(root / bm25_ids_path, "wb") as f:
        pickle.dump(chunk_ids, f)

    eval_set = [{"chunk_id": "d1_0", "question": "about cats", "answer": "cats are pets"}]
    with open(root / eval_path, "w") as f:
        json.dump(eval_set, f)

    manifest = {
        chunking_type: {
            "chunk_path": chunk_path,
            "flatip": index_path if index_type == "flatip" else f"index/flatip/{dataset_size}_{chunking_type}",
            "ivf": index_path if index_type == "ivf" else f"index/ivf/{dataset_size}_{chunking_type}",
            "hnsw": index_path if index_type == "hnsw" else f"index/hnsw/{dataset_size}_{chunking_type}",
            "chunk_ids": chunk_ids_path,
            "bm25": bm25_path,
            "bm25_ids": bm25_ids_path,
            "vectordb": f"vectordb_{dataset_size}_{chunking_type}",
            "eval_path": eval_path,
        }
    }
    manifest_path = root / "manifests" / f"{dataset_size}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


class FakeSTForScript:
    def __init__(self, model_name=None, device=None):
        pass

    def encode(self, sentences, normalize_embeddings=False):
        items = list(sentences)
        return np.zeros((len(items), 4), dtype="float32")


class FakeCrossEncoderForScript:
    def __init__(self, *a, **kw):
        pass


RETRIEVAL_CALLS = []


class FakeRetrieval:
    def __init__(self, mock_run, mode, chunking_type, chunks_map, eval_set, k, re_ranking, rerank_k, model, cross_encoder):
        self.eval_set = eval_set
        self.k = k

    def _output(self, retrieval_type):
        return [
            {
                "retrieval_type": retrieval_type,
                "chunk_type": "fixed",
                "chunk_id": item["chunk_id"],
                "retrieved_chunk_ids": [item["chunk_id"]],
                "retrieved_chunk_texts": [],
                "qus": item["question"],
                "context": "",
                "generated_ans": "MOCK_ANSWER",
                "ground_truth_ans": item["answer"],
                "k": self.k,
                "cost": 0,
                "latency": 0,
            }
            for item in self.eval_set
        ]

    def retrieval_dense(self, faiss_index, faiss_ids):
        RETRIEVAL_CALLS.append("dense")
        return self._output("dense")

    def retrieval_bm25(self, bm25_index, bm25_ids):
        RETRIEVAL_CALLS.append("bm25")
        return self._output("bm25")

    def retrieval_hybrid(self, faiss_index, faiss_ids, bm25_index, bm25_ids):
        RETRIEVAL_CALLS.append("hybrid")
        return self._output("hybrid")

    def retrieval_qdrant(self, collection_name):
        RETRIEVAL_CALLS.append("vectordb")
        return self._output("vector_db")


EVALUATE_CALLS = []


class FakeEvaluation:
    def __init__(self, mode, dataset_size, model_name, use_llm_judge):
        self.use_llm_judge = use_llm_judge

    def evaluate(self, k, retrieved_output):
        EVALUATE_CALLS.append({"k": k, "use_llm_judge": self.use_llm_judge, "n": len(retrieved_output)})
        return {"recall": 1.0, "num_questions": len(retrieved_output)}


@pytest.fixture(autouse=True)
def _reset_call_logs():
    RETRIEVAL_CALLS.clear()
    EVALUATE_CALLS.clear()
    yield


@pytest.fixture()
def patched_collaborators(monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSTForScript)
    monkeypatch.setattr("sentence_transformers.CrossEncoder", FakeCrossEncoderForScript)
    monkeypatch.setattr("src.retrieval.Retrieval", FakeRetrieval)
    monkeypatch.setattr("src.evaluation.Evaluation", FakeEvaluation)


def run_eval_rag(argv_tail):
    argv = ["eval_rag.py"] + argv_tail
    old_argv = sys.argv
    sys.argv = argv
    try:
        runpy.run_path(EVAL_SCRIPT, run_name="__main__")
    finally:
        sys.argv = old_argv


def base_args(retrieval_type, mock_run="1"):
    return [
        "--mock_run", mock_run,
        "--mode", "local",
        "--device", "cpu",
        "--model_name", "fake-model",
        "--dataset_size", "10",
        "--chunking_type", "fixed",
        "--index_type", "flatip",
        "--retrieval_type", retrieval_type,
        "--num_queries", "2",
        "--k", "1",
        "--re_ranking", "0",
        "--rerank_k", "4",
    ]


@pytest.mark.parametrize(
    "cli_retrieval_type,expected_method",
    [("dense", "dense"), ("bm25", "bm25"), ("hybrid", "hybrid"), ("vectordb", "vectordb")],
)
def test_eval_rag_dispatches_to_the_right_retrieval_method(
    patched_collaborators, isolated_cwd, cli_retrieval_type, expected_method
):
    _build_manifest(isolated_cwd)

    run_eval_rag(base_args(cli_retrieval_type))

    assert RETRIEVAL_CALLS == [expected_method]
    assert len(EVALUATE_CALLS) == 1
    assert EVALUATE_CALLS[0]["n"] == 1


def test_eval_rag_passes_use_llm_judge_false_when_mock_run(patched_collaborators, isolated_cwd):
    _build_manifest(isolated_cwd)

    run_eval_rag(base_args("dense", mock_run="1"))

    assert EVALUATE_CALLS[0]["use_llm_judge"] is False


def test_eval_rag_passes_use_llm_judge_true_when_not_mock_run(patched_collaborators, isolated_cwd):
    _build_manifest(isolated_cwd)

    run_eval_rag(base_args("dense", mock_run="0"))

    assert EVALUATE_CALLS[0]["use_llm_judge"] is True


def test_eval_rag_exits_when_chunks_map_is_empty(patched_collaborators, isolated_cwd):
    manifest_path = _build_manifest(isolated_cwd)
    manifest = json.loads(manifest_path.read_text())
    chunk_path = manifest["fixed"]["chunk_path"]
    with open(isolated_cwd / chunk_path, "wb") as f:
        pickle.dump({}, f)

    with pytest.raises(SystemExit) as exc_info:
        run_eval_rag(base_args("dense"))

    assert exc_info.value.code == 1


def test_eval_rag_exits_when_eval_set_is_missing(patched_collaborators, isolated_cwd):
    manifest_path = _build_manifest(isolated_cwd)
    manifest = json.loads(manifest_path.read_text())
    os.remove(isolated_cwd / manifest["fixed"]["eval_path"])

    with pytest.raises(FileNotFoundError):
        run_eval_rag(base_args("dense"))
