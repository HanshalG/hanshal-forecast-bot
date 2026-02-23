import math

import pytest

from src.eval.scoring import compute_brier, compute_log_loss, summarize_strategies
from src.eval.types import PredictionRecord


def _row(strategy_id: str, p: float, y: int, brier: float | None = None, log_loss: float | None = None):
    if brier is None:
        brier = compute_brier(p, y)
    if log_loss is None:
        log_loss = compute_log_loss(p, y)
    return PredictionRecord(
        strategy_id=strategy_id,
        post_id=1,
        question_id=1,
        title="Q",
        as_of_time="2024-01-01T00:00:00Z",
        probability_yes=p,
        label=y,
        resolution_raw="yes" if y == 1 else "no",
        brier=brier,
        log_loss=log_loss,
        runtime_seconds=1.0,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost_usd=0.001,
        num_runs=1,
        outside_view_enabled=True,
        inside_view_enabled=True,
        prediction_market_enabled=False,
        final_forecast_use_agent=True,
        error=None,
    )


def test_brier_and_log_loss_basic_values():
    assert compute_brier(0.8, 1) == pytest.approx(0.04)
    assert round(compute_log_loss(0.8, 1), 6) == round(-math.log(0.8), 6)


def test_leaderboard_order_prefers_lower_brier_then_logloss():
    rows = [
        _row("A", 0.9, 1),
        _row("A", 0.9, 1),
        _row("B", 0.6, 1),
        _row("B", 0.6, 1),
    ]

    summaries = summarize_strategies(rows)

    assert summaries[0].strategy_id == "A"
    assert summaries[1].strategy_id == "B"
    assert summaries[0].mean_brier < summaries[1].mean_brier
