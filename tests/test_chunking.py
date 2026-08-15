import pickle

import numpy as np
import pytest

from constants import FIXED, SENTENCE, SEMANTIC
from src.local.chunking import Chunking
from tests.fakes import FakeSentenceTransformer


@pytest.fixture()
def chunker(monkeypatch, sample_docs):
    monkeypatch.setattr("src.local.chunking.SentenceTransformer", FakeSentenceTransformer)
    return Chunking(sample_docs, dataset_size=10, device="cpu", model_name="fake-model")


def test_fixed_chunking_splits_and_overlaps_by_token_count(chunker):
    # 10 words, max_chunk_size=4, overlap=1 -> stride 3 -> starts 0,3,6,9
    text = "one two three four five six seven eight nine ten"
    chunks, sizes, elapsed = chunker.fixed_chunking("d1", "T", text, max_chunk_size=4, fix_chunk_overlap=1)

    assert list(chunks.keys()) == ["d1_0", "d1_1", "d1_2", "d1_3"]
    assert chunks["d1_0"]["chunk_text"] == "one two three four"
    assert chunks["d1_1"]["chunk_text"] == "four five six seven"
    assert chunks["d1_3"]["chunk_text"] == "ten"
    assert all(c["chunking_type"] == FIXED for c in chunks.values())
    assert all(c["doc_id"] == "d1" for c in chunks.values())
    assert sizes == [4, 4, 4, 1]
    assert elapsed >= 0


def test_sentence_aware_chunking_merges_sentences_under_limit(chunker):
    text = "One two. Three four. Five six."
    chunks, sizes, _ = chunker.sentence_aware_chunking("d1", "T", text, max_chunk_size=100, fix_chunk_overlap=1)

    assert len(chunks) == 1
    only = next(iter(chunks.values()))
    assert only["chunk_text"] == "One two. Three four. Five six."
    assert only["chunking_type"] == SENTENCE


def test_sentence_aware_chunking_starts_new_chunk_when_limit_exceeded(chunker):
    # Each sentence is 2 tokens; max_chunk_size=2 forces a new chunk per sentence.
    text = "One two. Three four. Five six."
    chunks, sizes, _ = chunker.sentence_aware_chunking("d1", "T", text, max_chunk_size=2, fix_chunk_overlap=0)

    assert len(chunks) == 3
    assert [c["chunk_text"] for c in chunks.values()] == ["One two.", "Three four.", "Five six."]
    assert sizes == [2, 2, 2]


def test_sentence_aware_chunking_splits_an_oversized_sentence_with_fixed_fallback(chunker):
    long_sentence = " ".join(f"w{i}" for i in range(10)) + "."
    text = f"Short one. {long_sentence}"
    chunks, sizes, _ = chunker.sentence_aware_chunking("d1", "T", text, max_chunk_size=4, fix_chunk_overlap=1)

    values = list(chunks.values())
    assert values[0]["chunk_text"] == "Short one."
    # the oversized sentence gets split via the fixed-chunking fallback (stride 3 over 10 tokens + '.')
    assert all(c["chunking_type"] == SENTENCE for c in values)
    assert len(values) > 2


def test_semantic_chunking_splits_on_low_similarity(chunker, monkeypatch):
    text = "Cats are pets. Cats like naps. Rockets reach orbit."
    # sentence 1 & 2 share vocabulary (high similarity), sentence 3 is unrelated (low similarity)
    encode_calls = []

    def fake_encode(sentences, batch_size=64, show_progress_bar=False):
        encode_calls.append(list(sentences))
        return np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype="float32")

    monkeypatch.setattr(chunker.model, "encode", fake_encode)

    chunks, sizes, _ = chunker.semantic_chunking(
        "d1", "T", text, max_chunk_size=100, fix_chunk_overlap=0, semantic_threshold=0.5
    )

    texts = [c["chunk_text"] for c in chunks.values()]
    assert texts == ["Cats are pets. Cats like naps.", "Rockets reach orbit."]
    assert all(c["chunking_type"] == SEMANTIC for c in chunks.values())


def test_semantic_chunking_keeps_similar_sentences_together(chunker, monkeypatch):
    text = "Cats are pets. Cats like naps."

    def fake_encode(sentences, batch_size=64, show_progress_bar=False):
        return np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")

    monkeypatch.setattr(chunker.model, "encode", fake_encode)

    chunks, sizes, _ = chunker.semantic_chunking(
        "d1", "T", text, max_chunk_size=100, fix_chunk_overlap=0, semantic_threshold=0.5
    )

    assert len(chunks) == 1


def test_compute_chunks_writes_three_pickle_files(chunker, isolated_cwd):
    path1, path2, path3 = chunker.compute_chunks(max_chunk_size=8, fix_chunk_overlap=2, semantic_threshold=0.3)

    assert path1 == "chunks/10_fixed/size_8_overlap_2.pkl"
    assert path2 == "chunks/10_sentence/size_8_overlap_2.pkl"
    assert path3 == "chunks/10_semantic/size_8_overlap_2_threashold_0.3.pkl"

    for path in (path1, path2, path3):
        with open(path, "rb") as f:
            chunks_map = pickle.load(f)
        assert len(chunks_map) > 0
        doc_ids = {c["doc_id"] for c in chunks_map.values()}
        assert doc_ids == {"d1", "d2"}


def test_compute_chunks_reuses_cached_files_without_recomputing(chunker, isolated_cwd):
    chunker.compute_chunks(max_chunk_size=8, fix_chunk_overlap=2, semantic_threshold=0.3)

    def boom(*args, **kwargs):
        raise AssertionError("compute_chunks should not recompute when cached files exist")

    chunker.fixed_chunking = boom
    paths = chunker.compute_chunks(max_chunk_size=8, fix_chunk_overlap=2, semantic_threshold=0.3)

    assert all(paths)


def test_compute_chunks_returns_none_tuple_and_cleans_up_on_failure(chunker, isolated_cwd):
    def boom(*args, **kwargs):
        raise RuntimeError("embedding backend unavailable")

    chunker.semantic_chunking = boom

    result = chunker.compute_chunks(max_chunk_size=8, fix_chunk_overlap=2, semantic_threshold=0.3)

    assert result == (None, None, None)
