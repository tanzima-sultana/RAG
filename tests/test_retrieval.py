import numpy as np
import pytest
from rank_bm25 import BM25Okapi

import faiss

from constants import LOCAL, DENSE, BM25 as BM25_TYPE, HYBRID, VECTOR_DB
from src.retrieval import Retrieval
from tests.fakes import FakeCrossEncoder, FakeQdrantClient, FakeSentenceTransformer, embed_text


CHUNKS_MAP = {
    "d1_0": {"doc_id": "d1", "chunk_id": "d1_0", "chunk_text": "cats are small pets"},
    "d1_1": {"doc_id": "d1", "chunk_id": "d1_1", "chunk_text": "dogs are loyal pets"},
    "d2_0": {"doc_id": "d2", "chunk_id": "d2_0", "chunk_text": "rockets launch into orbit"},
    "d2_1": {"doc_id": "d2", "chunk_id": "d2_1", "chunk_text": "the ocean is deep and blue"},
}
CHUNK_IDS = list(CHUNKS_MAP.keys())

EVAL_SET = [
    {"chunk_id": "d1_0", "question": "cats pets", "answer": "cats are pets"},
    {"chunk_id": "d2_0", "question": "rockets orbit", "answer": "rockets go to orbit"},
]


@pytest.fixture()
def faiss_index():
    model = FakeSentenceTransformer()
    vectors = model.encode([CHUNKS_MAP[cid]["chunk_text"] for cid in CHUNK_IDS], normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.array(vectors, dtype="float32"))
    return index


@pytest.fixture()
def bm25_index():
    tokenized = [CHUNKS_MAP[cid]["chunk_text"].lower().split() for cid in CHUNK_IDS]
    return BM25Okapi(tokenized)


def make_retrieval(k=1, reranking=0, rerank_k=4, dry_run=True):
    return Retrieval(
        dry_run, LOCAL, "fixed", CHUNKS_MAP, EVAL_SET, k, reranking, rerank_k,
        FakeSentenceTransformer(), FakeCrossEncoder(),
    )


def test_reciprocal_rank_fusion_merges_and_ranks_by_combined_score():
    ret = make_retrieval()

    merged = ret.reciprocal_rank_fusion(["a", "b", "c"], ["b", "a", "d"], k_const=60)

    assert merged[0] in ("a", "b")  # both appear in both lists near the top
    assert set(merged) == {"a", "b", "c", "d"}
    assert len(merged) == 4


def test_parse_answers_batch_handles_multiple_ids_and_multiline_answers():
    ret = make_retrieval()
    text = "ID: c1\nAnswer: first\nanswer continues\n\nID: c2\nAnswer: second answer\n"

    parsed = ret.parse_answers_batch(text)

    assert parsed == {"c1": "first answer continues", "c2": "second answer"}


def test_get_answers_batch_dry_run_returns_mock_answers_with_zero_cost():
    ret = make_retrieval(dry_run=True)

    result = ret.get_answers_batch([{"chunk_id": "d1_0"}, {"chunk_id": "d2_0"}])

    assert "ID: d1_0" in result["response"]
    assert "MOCK_ANSWER" in result["response"]
    assert result["cost"] == 0
    assert result["latency"] == 0


def test_retrieval_dense_returns_k_chunks_per_question(faiss_index):
    ret = make_retrieval(k=1, reranking=0)

    output = ret.retrieval_dense(faiss_index, CHUNK_IDS)

    assert len(output) == len(EVAL_SET)
    assert all(o["retrieval_type"] == DENSE for o in output)
    assert all(len(o["retrieved_chunk_ids"]) == 1 for o in output)
    assert all(o["generated_ans"] == "MOCK_ANSWER" for o in output)
    # the top dense hit for "cats pets" should be the cats chunk
    cats_output = next(o for o in output if o["chunk_id"] == "d1_0")
    assert cats_output["retrieved_chunk_ids"][0] == "d1_0"


def test_retrieval_dense_with_reranking_narrows_rerank_k_down_to_k(faiss_index):
    ret = make_retrieval(k=1, reranking=1, rerank_k=4)

    output = ret.retrieval_dense(faiss_index, CHUNK_IDS)

    assert all(len(o["retrieved_chunk_ids"]) == 1 for o in output)


def test_retrieval_bm25_ranks_lexical_overlap_first(bm25_index):
    ret = make_retrieval(k=1, reranking=0)

    output = ret.retrieval_bm25(bm25_index, CHUNK_IDS)

    rocket_output = next(o for o in output if o["chunk_id"] == "d2_0")
    assert rocket_output["retrieved_chunk_ids"][0] == "d2_0"
    assert all(o["retrieval_type"] == BM25_TYPE for o in output)


def test_retrieval_hybrid_combines_dense_and_bm25(faiss_index, bm25_index):
    ret = make_retrieval(k=2, reranking=0, rerank_k=4)

    output = ret.retrieval_hybrid(faiss_index, CHUNK_IDS, bm25_index, CHUNK_IDS)

    assert len(output) == len(EVAL_SET)
    assert all(o["retrieval_type"] == HYBRID for o in output)
    assert all(len(o["retrieved_chunk_ids"]) == 2 for o in output)


def test_retrieval_hybrid_returns_empty_list_when_an_index_is_missing():
    ret = make_retrieval()

    assert ret.retrieval_hybrid(None, CHUNK_IDS, object(), CHUNK_IDS) == []


def test_retrieval_qdrant_uses_payload_chunk_ids(monkeypatch):
    ret = make_retrieval(k=1, reranking=0)

    fake_client = FakeQdrantClient()
    fake_client.set_hits("my_collection", ["d1_0", "d1_1"])
    monkeypatch.setattr("src.retrieval.QdrantClient", lambda *a, **k: fake_client)

    output = ret.retrieval_qdrant("my_collection")

    assert len(output) == len(EVAL_SET)
    assert all(o["retrieval_type"] == VECTOR_DB for o in output)
    assert all(o["retrieved_chunk_ids"] == ["d1_0"] for o in output)
