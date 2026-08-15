import json

import pytest

from constants import LOCAL
from src.evaluation import Evaluation
from tests.fakes import FakeSentenceTransformer


@pytest.fixture()
def evaluation(monkeypatch):
    monkeypatch.setattr("src.evaluation.SentenceTransformer", FakeSentenceTransformer)
    return Evaluation(mode=LOCAL, dataset_size=10, model_name="fake-model", use_llm_judge=False)


def make_retrieved_output(chunk_id, retrieved_chunk_ids, generated_ans, ground_truth_ans, k=2, cost=0.0, latency=0.0):
    return {
        "retrieval_type": "dense",
        "chunk_type": "fixed",
        "chunk_id": chunk_id,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_chunk_texts": [],
        "qus": "some question",
        "context": "some context",
        "generated_ans": generated_ans,
        "ground_truth_ans": ground_truth_ans,
        "k": k,
        "cost": cost,
        "latency": latency,
    }


def test_doc_id_from_chunk_id_strips_trailing_index():
    ev = Evaluation.__new__(Evaluation)
    assert ev._doc_id_from_chunk_id("doc42_7") == "doc42"


@pytest.mark.parametrize(
    "gt,retrieved,expected",
    [
        ("d1_0", ["d1_5", "d2_0"], 1),
        ("d1_0", ["d2_0", "d3_0"], 0),
    ],
)
def test_compute_recall_single_chunk_matches_by_doc_id(evaluation, gt, retrieved, expected):
    assert evaluation.compute_recall_single_chunk(gt, retrieved) == expected


def test_compute_precision_at_k_counts_hits_by_doc_id(evaluation):
    precision = evaluation.compute_precision_at_k_single_chunk("d1_0", ["d1_3", "d1_9", "d2_0"], k=3)
    assert precision == pytest.approx(2 / 3)


def test_compute_mrr_returns_inverse_rank_of_first_hit(evaluation):
    assert evaluation.compute_mrr("d1_0", ["d2_0", "d1_5", "d3_0"]) == pytest.approx(1 / 2)
    assert evaluation.compute_mrr("d1_0", ["d2_0", "d3_0"]) == 0


def test_compute_semantic_similarity_is_higher_for_similar_answers(evaluation):
    high = evaluation.compute_semantic_similarity("cats are pets", "cats are pets too")
    low = evaluation.compute_semantic_similarity("cats are pets", "rockets reach orbit")
    assert -1.0 <= low <= 1.0
    assert high > low


def test_parse_handles_plain_json():
    ev = Evaluation.__new__(Evaluation)
    text = '{"0": {"faithfulness": 0.9, "relevancy": 0.8, "correctness": 0.7}}'

    parsed = ev.parse(text, batch=[{}])

    assert parsed == {0: (0.9, 0.8, 0.7)}


def test_parse_strips_markdown_code_fences():
    ev = Evaluation.__new__(Evaluation)
    text = '```json\n{"0": {"faithfulness": 1.0, "relevancy": 1.0, "correctness": 1.0}}\n```'

    parsed = ev.parse(text, batch=[{}])

    assert parsed == {0: (1.0, 1.0, 1.0)}


def test_parse_returns_empty_dict_on_unparseable_text():
    ev = Evaluation.__new__(Evaluation)
    assert ev.parse("not json at all", batch=[{}]) == {}


def test_regex_fallback_batch_score_extracts_values_from_malformed_json():
    ev = Evaluation.__new__(Evaluation)
    text = '{"3": {"faithfulness": 0.5, "relevancy": 0.6, "correctness": 0.4},'  # truncated / invalid JSON

    assert ev._regex_fallback_batch_score(text, 3) == (0.5, 0.6, 0.4)
    assert ev._regex_fallback_batch_score(text, 99) == (0.0, 0.0, 0.0)


def test_evaluate_without_llm_judge_computes_averages_and_saves(evaluation, isolated_cwd):
    retrieved_output = [
        make_retrieved_output("d1_0", ["d1_0", "d2_0"], "ans1", "ans1"),
        make_retrieved_output("d2_0", ["d2_0", "d1_1"], "ans2", "ans2 ref"),
    ]

    summary = evaluation.evaluate(k=2, retrieved_output=retrieved_output)

    assert summary["num_questions"] == 2
    assert summary["recall"] == pytest.approx(1.0)  # both ground-truth docs found in top-k
    assert summary["precision"] == pytest.approx((0.5 + 0.5) / 2)
    assert summary["avg_faithfulness"] == 0
    assert summary["avg_relevancy"] == 0

    with open("evals/10/summary.json") as f:
        saved_summary = json.load(f)
    assert saved_summary == summary

    with open("evals/10/details.json") as f:
        details = json.load(f)
    assert len(details["eval_metrices"]) == 2


def test_evaluate_with_llm_judge_aggregates_batched_scores(monkeypatch, isolated_cwd):
    monkeypatch.setattr("src.evaluation.SentenceTransformer", FakeSentenceTransformer)
    ev = Evaluation(mode=LOCAL, dataset_size=10, model_name="fake-model", use_llm_judge=True)

    judgement_json = json.dumps(
        {
            "0": {"faithfulness": 0.9, "relevancy": 0.8, "correctness": 0.7},
            "1": {"faithfulness": 0.5, "relevancy": 0.4, "correctness": 0.3},
        }
    )
    monkeypatch.setattr(
        ev.anthropic,
        "anthropic_msg_api",
        lambda prompt: {"response": judgement_json, "cost": 0.02, "latency": 0.1},
    )

    retrieved_output = [
        make_retrieved_output("d1_0", ["d1_0"], "ans1", "ans1"),
        make_retrieved_output("d2_0", ["d2_0"], "ans2", "ans2"),
    ]

    summary = ev.evaluate(k=1, retrieved_output=retrieved_output)

    assert summary["avg_faithfulness"] == pytest.approx((0.9 + 0.5) / 2)
    assert summary["avg_relevancy"] == pytest.approx((0.8 + 0.4) / 2)
    assert summary["avg_ans_lmm_correctness"] == pytest.approx((0.7 + 0.3) / 2)
    assert summary["total_cost"] == pytest.approx(0.02)
