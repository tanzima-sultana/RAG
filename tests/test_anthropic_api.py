import pytest

from constants import INPUT_COST_PER_MTOK, OUTPUT_COST_PER_MTOK
from src.anthropic_api import AnthropicAPI
from tests.fakes import FakeAnthropicClient


@pytest.fixture()
def api(monkeypatch):
    fake_client = FakeAnthropicClient(text="  The answer is 42.  ", input_tokens=100, output_tokens=20)
    monkeypatch.setattr("src.anthropic_api.anthropic.Anthropic", lambda api_key: fake_client)
    return AnthropicAPI(anthropic_model="claude-test-model", max_tokens=100), fake_client


def test_anthropic_msg_api_strips_response_and_computes_cost(api):
    an, fake_client = api

    result = an.anthropic_msg_api("What is the answer?")

    assert result["response"] == "The answer is 42."
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 20
    expected_cost = (100 / 1_000_000) * INPUT_COST_PER_MTOK + (20 / 1_000_000) * OUTPUT_COST_PER_MTOK
    assert result["cost"] == pytest.approx(expected_cost)
    assert result["latency"] >= 0


def test_anthropic_msg_api_calls_client_with_model_and_prompt(api):
    an, fake_client = api

    an.anthropic_msg_api("hello there")

    assert len(fake_client.messages.calls) == 1
    call = fake_client.messages.calls[0]
    assert call["model"] == "claude-test-model"
    assert call["max_tokens"] == 100
    assert call["messages"] == [{"role": "user", "content": "hello there"}]


def test_api_response_shape():
    an = AnthropicAPI.__new__(AnthropicAPI)  # avoid constructing a real client
    result = an.API_RESPONSE("resp", 1, 2, 0.01, 0.5)

    assert result == {
        "response": "resp",
        "input_tokens": 1,
        "output_tokens": 2,
        "cost": 0.01,
        "latency": 0.5,
    }
