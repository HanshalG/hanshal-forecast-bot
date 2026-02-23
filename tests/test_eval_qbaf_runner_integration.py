import asyncio
import json
from datetime import datetime, timezone

import pytest

from src.eval.runner import _run_single_strategy_question
from src.eval.types import EvalQuestion, EvalStrategyConfig, QbafAgentProfile


def _question() -> EvalQuestion:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return EvalQuestion(
        post_id=1,
        question_id=11,
        title="Will X happen?",
        type="binary",
        description="desc",
        resolution_criteria="criteria",
        fine_print="fine",
        label=1,
        resolution_raw="yes",
        status="resolved",
        open_time=now,
        actual_resolve_time=None,
        metadata={},
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_qbaf_runner_dispatch_and_prediction_record(monkeypatch):
    async def fake_call_llm(prompt, model, temperature, reasoning_effort="medium", component="call_llm"):
        if component == "qbaf_similarity_judge":
            return json.dumps({"score": 1.0})
        if component == "qbaf_root_base_estimate":
            return json.dumps({"root_base_score": 0.55})
        if component == "qbaf_generate":
            return json.dumps(
                {
                    "root": {"text": "Will X happen?", "base_score": 0.5},
                    "arguments": [
                        {
                            "id": "a1",
                            "parent_id": "root",
                            "depth": 1,
                            "relation": "support",
                            "text": "supporting evidence",
                            "base_score": 0.8,
                        }
                    ],
                }
            )
        raise AssertionError(f"Unexpected component: {component}")

    async def fake_asknews(question_or_details, *, as_of_time=None):
        return "asknews context"

    async def fake_exa_context(question_details):
        return "exa context"

    async def forecast_should_not_be_called(*args, **kwargs):
        raise AssertionError("forecast pipeline should not be called for qbaf strategy")

    monkeypatch.setattr("src.eval.qbaf_strategy.call_llm", fake_call_llm)
    monkeypatch.setattr("src.eval.qbaf_strategy.call_asknews_async", fake_asknews)
    monkeypatch.setattr("src.eval.qbaf_strategy._fetch_exa_context", fake_exa_context)
    monkeypatch.setattr("src.forecast.get_binary_prediction", forecast_should_not_be_called)

    strategy = EvalStrategyConfig(
        id="qbaf_test",
        enabled=True,
        num_runs=1,
        strategy_kind="qbaf_multi_agent",
        qbaf_similarity_backend="llm_pairwise",
        qbaf_similarity_threshold=0.5,
        qbaf_base_aggregation="avg",
        qbaf_root_base_mode="fixed_0_5",
        qbaf_depth=2,
        qbaf_max_nodes_per_depth=4,
        qbaf_generation_model="gpt-5-nano",
        qbaf_pairwise_model="gpt-5-nano",
        qbaf_agent_profiles=[
            QbafAgentProfile(id="base", retrieval_mode="argllm_base"),
            QbafAgentProfile(id="ask", retrieval_mode="rag_asknews"),
            QbafAgentProfile(id="exa", retrieval_mode="rag_exa"),
        ],
    )

    row = asyncio.run(
        _run_single_strategy_question(
            strategy=strategy,
            question=_question(),
            num_runs=1,
        )
    )

    assert row.strategy_id == "qbaf_test"
    assert row.error is None
    assert row.probability_yes > 0.5
    assert row.brier >= 0.0
    assert row.log_loss >= 0.0
