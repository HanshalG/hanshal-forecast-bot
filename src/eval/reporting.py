from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .types import CalibrationBin, PredictionRecord, StrategySummary


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False))
            handle.write("\n")


def _write_strategy_summary_csv(path: Path, summaries: list[StrategySummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy_id",
                "n",
                "n_success",
                "n_failed",
                "mean_brier",
                "mean_log_loss",
                "ece",
                "mean_cost_usd",
                "mean_runtime_seconds",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def _write_calibration_csv(path: Path, bins: list[CalibrationBin]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy_id",
                "bin_index",
                "bin_start",
                "bin_end",
                "count",
                "avg_pred",
                "avg_outcome",
                "abs_gap",
            ],
        )
        writer.writeheader()
        for b in bins:
            writer.writerow(asdict(b))


def _render_leaderboard_md(summaries: list[StrategySummary]) -> str:
    lines: list[str] = []
    lines.append("# Eval Leaderboard")
    lines.append("")
    lines.append("Ranking: Brier asc, then log loss asc, then ECE asc, then mean cost asc.")
    lines.append("")
    lines.append("| Rank | Strategy | N | Success | Failed | Mean Brier | Mean Log Loss | ECE | Mean Cost (USD) | Mean Latency (s) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for idx, summary in enumerate(summaries, start=1):
        lines.append(
            "| {rank} | {strategy} | {n} | {n_success} | {n_failed} | {brier:.6f} | {log_loss:.6f} | {ece:.6f} | {cost:.6f} | {latency:.3f} |".format(
                rank=idx,
                strategy=summary.strategy_id,
                n=summary.n,
                n_success=summary.n_success,
                n_failed=summary.n_failed,
                brier=summary.mean_brier,
                log_loss=summary.mean_log_loss,
                ece=summary.ece,
                cost=summary.mean_cost_usd,
                latency=summary.mean_runtime_seconds,
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_eval_reports(
    *,
    output_dir: str,
    predictions: list[PredictionRecord],
    strategy_summaries: list[StrategySummary],
    calibration_by_strategy: dict[str, list[CalibrationBin]],
    run_config_snapshot: dict[str, Any],
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_jsonl_path = out_dir / "predictions.jsonl"
    strategy_summary_csv_path = out_dir / "strategy_summary.csv"
    leaderboard_md_path = out_dir / "leaderboard.md"
    run_config_snapshot_path = out_dir / "run_config_snapshot.json"

    prediction_rows = [_to_jsonable(asdict(p)) for p in predictions]
    _write_jsonl(predictions_jsonl_path, prediction_rows)
    _write_strategy_summary_csv(strategy_summary_csv_path, strategy_summaries)
    leaderboard_md_path.write_text(_render_leaderboard_md(strategy_summaries), encoding="utf-8")
    _write_json(run_config_snapshot_path, run_config_snapshot)

    calibration_paths: dict[str, str] = {}
    for strategy_id, bins in calibration_by_strategy.items():
        path = out_dir / f"calibration_{strategy_id}.csv"
        _write_calibration_csv(path, bins)
        calibration_paths[strategy_id] = str(path)

    return {
        "predictions_jsonl": str(predictions_jsonl_path),
        "strategy_summary_csv": str(strategy_summary_csv_path),
        "leaderboard_md": str(leaderboard_md_path),
        "run_config_snapshot_json": str(run_config_snapshot_path),
        "calibration_csv_by_strategy": calibration_paths,
    }
