import json
import os
import pickle

import pytest

from constants import FIXED, LOCAL
from src.eval_qa import EvalQA


@pytest.fixture()
def eval_qa():
    return EvalQA(mock_run=1, mode=LOCAL, dataset_size=10, num_queries=2)


def _write_chunks(path, chunk_ids_and_sizes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    chunks_map = {
        cid: {"doc_id": "d1", "chunk_id": cid, "chunk_text": f"text for {cid}", "chunk_size": size}
        for cid, size in chunk_ids_and_sizes
    }
    with open(path, "wb") as f:
        pickle.dump(chunks_map, f)
    return path


def test_get_sample_chunks_filters_by_min_size_and_is_seeded(eval_qa):
    chunks = [
        {"chunk_id": "a", "chunk_size": 50},
        {"chunk_id": "b", "chunk_size": 150},
        {"chunk_id": "c", "chunk_size": 200},
    ]

    sample1 = eval_qa.get_sample_chunks(chunks, min_chunk_size=100, no_of_chunks=2)
    sample2 = eval_qa.get_sample_chunks(chunks, min_chunk_size=100, no_of_chunks=2)

    assert len(sample1) == 2
    assert all(c["chunk_size"] >= 100 for c in sample1)
    assert sample1 == sample2  # SEED makes selection deterministic


def test_get_sample_chunks_raises_when_not_enough_chunks_meet_threshold(eval_qa):
    chunks = [{"chunk_id": "a", "chunk_size": 50}]

    with pytest.raises(ValueError):
        eval_qa.get_sample_chunks(chunks, min_chunk_size=100, no_of_chunks=2)


def test_parse_qa_batch_response_handles_multiple_blocks_and_multiline_answers():
    text = (
        "CHUNK_ID: d1_0\n"
        "Question: What is X?\n"
        "Answer: X is a thing\n"
        "that spans two lines.\n"
        "\n"
        "CHUNK_ID: d1_1\n"
        "Question: What is Y?\n"
        "Answer: Y is simple.\n"
    )

    parsed = EvalQA(1, LOCAL, 10, 2).parse_qa_batch_response(text)

    assert parsed["d1_0"] == {"question": "What is X?", "answer": "X is a thing that spans two lines."}
    assert parsed["d1_1"] == {"question": "What is Y?", "answer": "Y is simple."}


def test_parse_qa_batch_response_drops_incomplete_trailing_block():
    text = "CHUNK_ID: d1_0\nQuestion: What is X?\n"  # no Answer line

    parsed = EvalQA(1, LOCAL, 10, 2).parse_qa_batch_response(text)

    assert parsed == {}


def test_generate_qa_batch_mock_run_produces_one_block_per_chunk(eval_qa):
    chunks = [{"chunk_id": "d1_0"}, {"chunk_id": "d1_1"}]

    response = eval_qa.generate_qa_batch(chunks)

    assert "CHUNK_ID: d1_0" in response
    assert "CHUNK_ID: d1_1" in response
    assert "Mock question for d1_0?" in response


def test_build_eval_set_end_to_end_mock_run(eval_qa, isolated_cwd):
    chunk_path = _write_chunks(
        "chunks/10_fixed/size_8_overlap_2.pkl",
        [(f"d1_{i}", 150) for i in range(5)],
    )

    eval_path = eval_qa.build_eval_set(FIXED, chunk_path, min_chunk_size=100)

    assert eval_path == "eval_qa/10_2_fixed_mock"
    with open(eval_path) as f:
        eval_set = json.load(f)

    assert len(eval_set) == 2
    for item in eval_set:
        assert set(item.keys()) == {"chunk_id", "question", "answer"}


def test_build_eval_set_reuses_cached_file_when_size_matches(eval_qa, isolated_cwd):
    chunk_path = _write_chunks(
        "chunks/10_fixed/size_8_overlap_2.pkl",
        [(f"d1_{i}", 150) for i in range(5)],
    )
    eval_qa.build_eval_set(FIXED, chunk_path, min_chunk_size=100)

    def boom(*args, **kwargs):
        raise AssertionError("should not regenerate an eval set that is already cached")

    eval_qa.generate_qa_batch = boom
    eval_path = eval_qa.build_eval_set(FIXED, chunk_path, min_chunk_size=100)

    assert eval_path == "eval_qa/10_2_fixed_mock"


def test_get_eval_set_returns_none_when_query_count_does_not_match(eval_qa, isolated_cwd):
    path = "eval_qa/10_2_fixed_mock"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([{"chunk_id": "d1_0", "question": "q", "answer": "a"}], f)  # only 1, but num_queries=2

    assert eval_qa.get_eval_set(path) is None
