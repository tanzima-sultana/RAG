import os
import pickle

import numpy as np
import pytest
from nltk.tokenize import sent_tokenize

from constants import FIXED, SENTENCE, SEMANTIC
from src.local.chunking import Chunking
from tests.fakes import FakeSentenceTransformer


@pytest.fixture()
def chunker(monkeypatch, sample_docs):
    monkeypatch.setattr("src.local.chunking.SentenceTransformer", FakeSentenceTransformer)
    return Chunking(sample_docs, dataset_size=10, device="cpu", model_name="fake-model")


def sentences_and_tokens(chunker, text):
    sentences = sent_tokenize(text)
    sentence_tokens = [chunker.tokenizer.encode(s, add_special_tokens=False) for s in sentences]
    return sentences, sentence_tokens


# -------------------- Bug fixes -------------------- #

@pytest.mark.parametrize("max_chunk_size,fix_chunk_overlap", [(4, 4), (4, 5)])
def test_fixed_chunking_rejects_overlap_that_would_stall_the_sliding_window(chunker, max_chunk_size, fix_chunk_overlap):
    with pytest.raises(ValueError):
        chunker.fixed_chunking("d1", "T", "one two three four five", max_chunk_size, fix_chunk_overlap)


def test_compute_chunks_rejects_bad_overlap_before_doing_any_work(chunker, isolated_cwd):
    with pytest.raises(ValueError):
        chunker.compute_chunks(max_chunk_size=4, fix_chunk_overlap=4, semantic_threshold=0.3)

    assert not os.path.exists("chunks/10_fixed/size_4_overlap_4.pkl")


def test_compute_chunks_returns_none_and_cleans_up_partial_files_on_save_failure(chunker, isolated_cwd):
    original_save = chunker.save
    save_calls = []

    def flaky_save(chunks, out_path):
        save_calls.append(out_path)
        if len(save_calls) == 3:
            raise RuntimeError("disk full")
        original_save(chunks, out_path)

    chunker.save = flaky_save

    result = chunker.compute_chunks(max_chunk_size=8, fix_chunk_overlap=2, semantic_threshold=0.3)

    assert result == (None, None, None)
    # the first two saves succeeded and wrote real files before the third one failed;
    # cleanup must remove them (os.remove) rather than crash trying to rmtree a file.
    assert not os.path.exists("chunks/10_fixed/size_8_overlap_2.pkl")
    assert not os.path.exists("chunks/10_sentence/size_8_overlap_2.pkl")


# -------------------- Fixed chunking -------------------- #

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


# -------------------- Sentence-aware chunking -------------------- #

def test_sentence_aware_chunking_merges_sentences_under_limit(chunker):
    sentences, tokens = sentences_and_tokens(chunker, "One two. Three four. Five six.")
    chunks, sizes, _ = chunker.sentence_aware_chunking("d1", "T", sentences, tokens, max_chunk_size=100, fix_chunk_overlap=1)

    assert len(chunks) == 1
    only = next(iter(chunks.values()))
    assert only["chunk_text"] == "One two. Three four. Five six."
    assert only["chunking_type"] == SENTENCE


def test_sentence_aware_chunking_starts_new_chunk_when_limit_exceeded(chunker):
    # Each sentence is 2 tokens; max_chunk_size=2 forces a new chunk per sentence.
    sentences, tokens = sentences_and_tokens(chunker, "One two. Three four. Five six.")
    chunks, sizes, _ = chunker.sentence_aware_chunking("d1", "T", sentences, tokens, max_chunk_size=2, fix_chunk_overlap=0)

    assert len(chunks) == 3
    assert [c["chunk_text"] for c in chunks.values()] == ["One two.", "Three four.", "Five six."]
    assert sizes == [2, 2, 2]


def test_sentence_aware_chunking_splits_an_oversized_sentence_with_fixed_fallback(chunker):
    long_sentence = " ".join(f"w{i}" for i in range(10)) + "."
    text = f"Short one. {long_sentence}"
    sentences, tokens = sentences_and_tokens(chunker, text)
    chunks, sizes, _ = chunker.sentence_aware_chunking("d1", "T", sentences, tokens, max_chunk_size=4, fix_chunk_overlap=1)

    values = list(chunks.values())
    assert values[0]["chunk_text"] == "Short one."
    assert all(c["chunking_type"] == SENTENCE for c in values)
    assert len(values) > 2


def test_sentence_aware_chunking_reports_exact_joined_token_count_not_sum_of_sentences(chunker, monkeypatch):
    sentence1, sentence2 = "Sentence one.", "Sentence two."
    joined = f"{sentence1} {sentence2}"
    token_map = {
        sentence1: ["a", "b"],
        sentence2: ["c", "d"],
        joined: ["x", "y", "z"],  # re-tokenizing the joined text yields 3, not 2+2=4
    }
    monkeypatch.setattr(chunker.tokenizer, "encode", lambda text, add_special_tokens=False: token_map[text])

    sentences = [sentence1, sentence2]
    sentence_tokens = [token_map[sentence1], token_map[sentence2]]
    chunks, sizes, _ = chunker.sentence_aware_chunking(
        "d1", "T", sentences, sentence_tokens, max_chunk_size=100, fix_chunk_overlap=1
    )

    only = next(iter(chunks.values()))
    assert only["chunk_text"] == joined
    assert only["chunk_size"] == 3
    assert sizes == [3]


# -------------------- Semantic chunking -------------------- #

def test_semantic_chunking_splits_on_low_similarity(chunker):
    text = "Cats are pets. Cats like naps. Rockets reach orbit."
    sentences, tokens = sentences_and_tokens(chunker, text)
    # sentence 1 & 2 share vocabulary (high similarity), sentence 3 is unrelated (low similarity)
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype="float32")

    chunks, sizes, _ = chunker.semantic_chunking(
        "d1", "T", sentences, tokens, embeddings, max_chunk_size=100, fix_chunk_overlap=0, semantic_threshold=0.5
    )

    texts = [c["chunk_text"] for c in chunks.values()]
    assert texts == ["Cats are pets. Cats like naps.", "Rockets reach orbit."]
    assert all(c["chunking_type"] == SEMANTIC for c in chunks.values())


def test_semantic_chunking_keeps_similar_sentences_together(chunker):
    sentences, tokens = sentences_and_tokens(chunker, "Cats are pets. Cats like naps.")
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")

    chunks, sizes, _ = chunker.semantic_chunking(
        "d1", "T", sentences, tokens, embeddings, max_chunk_size=100, fix_chunk_overlap=0, semantic_threshold=0.5
    )

    assert len(chunks) == 1


def test_semantic_chunking_zero_vector_does_not_crash_and_does_not_force_a_split(chunker):
    sentences, tokens = sentences_and_tokens(chunker, "Alpha bravo. Charlie delta.")
    embeddings = np.array([[0.0, 0.0], [1.0, 0.0]], dtype="float32")

    chunks, sizes, _ = chunker.semantic_chunking(
        "d1", "T", sentences, tokens, embeddings, max_chunk_size=100, fix_chunk_overlap=0, semantic_threshold=0.5
    )

    assert len(chunks) == 1


# -------------------- compute_chunks orchestration -------------------- #

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


def test_compute_chunks_batch_encodes_sentences_once_for_the_whole_dataset(chunker, isolated_cwd):
    original_encode = chunker.model.encode
    encode_call_sizes = []

    def spy_encode(sentences, **kwargs):
        encode_call_sizes.append(len(list(sentences)))
        return original_encode(sentences, **kwargs)

    chunker.model.encode = spy_encode

    chunker.compute_chunks(max_chunk_size=8, fix_chunk_overlap=2, semantic_threshold=0.3)

    assert len(encode_call_sizes) == 1  # one batched call across both documents, not one per document
    assert encode_call_sizes[0] > 0


def test_compute_chunks_skips_a_failing_document_and_keeps_the_rest(chunker, isolated_cwd):
    original_semantic_chunking = chunker.semantic_chunking

    def flaky_semantic_chunking(doc_id, *args, **kwargs):
        if doc_id == "d1":
            raise RuntimeError("simulated failure for d1")
        return original_semantic_chunking(doc_id, *args, **kwargs)

    chunker.semantic_chunking = flaky_semantic_chunking

    path1, path2, path3 = chunker.compute_chunks(max_chunk_size=8, fix_chunk_overlap=2, semantic_threshold=0.3)

    assert (path1, path2, path3) != (None, None, None)
    with open(path1, "rb") as f:
        fixed_chunks = pickle.load(f)
    # d1 fails (in the semantic step) so the whole document is skipped for all three
    # strategies; d2 should still make it through.
    assert {c["doc_id"] for c in fixed_chunks.values()} == {"d2"}
