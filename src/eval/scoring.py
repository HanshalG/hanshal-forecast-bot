from __future__ import annotations

import math
from collections import defaultdict

from .types import CalibrationBin, PredictionRecord, StrategySummary


def compute_brier(probability_yes: float, label: int) -> float:
    p = float(probability_yes)
    y = int(label)
    return (p - y) ** 2


def compute_log_loss(probability_yes: float, label: int, *, epsilon: float = 1e-6) -> float:
    p = min(max(float(probability_yes), epsilon), 1.0 - epsilon)
    y = int(label)
    if y == 1:
        return -math.log(p)
    return -math.log(1.0 - p)


def compute_calibration_bins(
    records: list[PredictionRecord],
    *,
    n_bins: int = 10,
) -> dict[str, list[CalibrationBin]]:
    by_strategy: dict[str, list[PredictionRecord]] = defaultdict(list)
    for row in records:
        if row.error is None:
            by_strategy[row.strategy_id].append(row)

    result: dict[str, list[CalibrationBin]] = {}

    for strategy_id, rows in by_strategy.items():
        bins: list[list[PredictionRecord]] = [[] for _ in range(n_bins)]
        for row in rows:
            idx = min(n_bins - 1, int(row.probability_yes * n_bins))
            bins[idx].append(row)

        calibrated: list[CalibrationBin] = []
        for idx, bucket in enumerate(bins):
            start = idx / n_bins
            end = (idx + 1) / n_bins
            if bucket:
                avg_pred = sum(x.probability_yes for x in bucket) / len(bucket)
                avg_outcome = sum(x.label for x in bucket) / len(bucket)
            else:
                avg_pred = 0.0
                avg_outcome = 0.0
            calibrated.append(
                CalibrationBin(
                    strategy_id=strategy_id,
                    bin_index=idx,
                    bin_start=start,
                    bin_end=end,
                    count=len(bucket),
                    avg_pred=avg_pred,
                    avg_outcome=avg_outcome,
                    abs_gap=abs(avg_pred - avg_outcome) if bucket else 0.0,
                )
            )

        result[strategy_id] = calibrated

    return result


def summarize_strategies(records: list[PredictionRecord]) -> list[StrategySummary]:
    by_strategy: dict[str, list[PredictionRecord]] = defaultdict(list)
    for row in records:
        by_strategy[row.strategy_id].append(row)

    calibration = compute_calibration_bins(records)
    summaries: list[StrategySummary] = []

    for strategy_id, rows in by_strategy.items():
        successful = [r for r in rows if r.error is None and math.isfinite(r.brier) and math.isfinite(r.log_loss)]
        n = len(rows)
        n_success = len(successful)
        n_failed = n - n_success

        if successful:
            mean_brier = sum(r.brier for r in successful) / n_success
            mean_log_loss = sum(r.log_loss for r in successful) / n_success
            mean_cost = sum(r.cost_usd for r in successful) / n_success
            mean_runtime = sum(r.runtime_seconds for r in successful) / n_success
        else:
            mean_brier = math.inf
            mean_log_loss = math.inf
            mean_cost = math.inf
            mean_runtime = math.inf

        bins = calibration.get(strategy_id, [])
        if successful and bins:
            total = sum(b.count for b in bins)
            ece = 0.0
            if total > 0:
                for b in bins:
                    ece += (b.count / total) * b.abs_gap
            else:
                ece = math.inf
        else:
            ece = math.inf

        summaries.append(
            StrategySummary(
                strategy_id=strategy_id,
                n=n,
                n_success=n_success,
                n_failed=n_failed,
                mean_brier=mean_brier,
                mean_log_loss=mean_log_loss,
                ece=ece,
                mean_cost_usd=mean_cost,
                mean_runtime_seconds=mean_runtime,
            )
        )

    return sorted(
        summaries,
        key=lambda s: (
            s.mean_brier,
            s.mean_log_loss,
            s.ece,
            s.mean_cost_usd,
            s.strategy_id,
        ),
    )
