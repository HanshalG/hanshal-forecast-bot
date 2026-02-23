from __future__ import annotations

import asyncio
import math
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.forecast_logger import set_supabase_logging_enabled
from src.prediction_market_check import format_semipublic_market_data, get_prediction_market_data
from src.token_cost import (
    clear_usage_scope,
    get_total_usage,
    reset_usage,
    reset_usage_scope,
    set_usage_scope,
)
from src.utils import ASKNEWS_STATS

from .eval_question_file_loader import load_eval_question_file
from .scoring import compute_brier, compute_calibration_bins, compute_log_loss, summarize_strategies
from .strategy_config import load_strategy_files
from .timebox import to_iso_z, with_question_as_of_time
from .types import EvalQuestion, EvalStrategyConfig, PredictionRecord


def _snapshot_counter(counter: dict[str, int]) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.items()}


def _diff_counter(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    keys = set(before.keys()) | set(after.keys())
    return {k: int(after.get(k, 0) - before.get(k, 0)) for k in sorted(keys)}


def _timestamp_run_id() -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"eval-{now}-{uuid.uuid4().hex[:8]}"


@contextmanager
def _temporary_eval_overrides(strategy: EvalStrategyConfig, *, as_of_time_iso: str) -> Iterator[None]:
    import src.agent_infrastructure as agent_infrastructure
    import src.exa_utils as exa_utils
    import src.final_forecast as final_forecast
    import src.forecast as forecast_module
    import src.inside_view as inside_view
    import src.outside_view as outside_view
    import src.utils as utils_module

    env_updates: dict[str, str] = {}
    env_updates.update(strategy.env_overrides)
    env_updates.update(strategy.model_overrides)
    env_updates["FINAL_FORECAST_USE_AGENT"] = "true" if strategy.final_forecast_use_agent else "false"
    env_updates["EVAL_AS_OF_TIME"] = as_of_time_iso

    old_env: dict[str, str | None] = {}
    for key, value in env_updates.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = str(value)

    patches: list[tuple[Any, str, Any]] = []

    def patch(module: Any, name: str, value: Any) -> None:
        if hasattr(module, name):
            patches.append((module, name, getattr(module, name)))
            setattr(module, name, value)

    model_outside = strategy.model_overrides.get("OUTSIDE_VIEW_MODEL")
    model_inside = strategy.model_overrides.get("INSIDE_VIEW_MODEL")
    model_final = strategy.model_overrides.get("FINAL_FORECAST_MODEL")
    model_summary = strategy.model_overrides.get("SUMMARY_MODEL")
    reasoning_effort = env_updates.get("REASONING_EFFORT")
    final_forecast_reasoning_effort = env_updates.get("FINAL_FORECAST_REASONING_EFFORT")
    tool_summary_reasoning_effort = env_updates.get("TOOL_SUMMARY_REASONING_EFFORT")
    clear_summary_llm_cache = (
        hasattr(agent_infrastructure, "_get_tool_summary_llm")
        and hasattr(agent_infrastructure._get_tool_summary_llm, "cache_clear")
    )
    if clear_summary_llm_cache:
        agent_infrastructure._get_tool_summary_llm.cache_clear()

    if model_outside:
        patch(outside_view, "OUTSIDE_VIEW_MODEL", model_outside)
        patch(forecast_module, "OUTSIDE_VIEW_MODEL", model_outside)

    if model_inside:
        patch(inside_view, "LLM_MODEL", model_inside)
        patch(agent_infrastructure, "LLM_MODEL", model_inside)
        patch(utils_module, "INSIDE_VIEW_MODEL", model_inside)
        patch(exa_utils, "INSIDE_VIEW_MODEL", model_inside)
        patch(forecast_module, "INSIDE_VIEW_MODEL", model_inside)

    if model_final:
        patch(final_forecast, "FINAL_FORECAST_MODEL", model_final)
        patch(forecast_module, "FINAL_FORECAST_MODEL", model_final)

    if model_summary:
        patch(forecast_module, "SUMMARY_MODEL", model_summary)
        patch(utils_module, "SUMMARY_MODEL", model_summary)
        patch(exa_utils, "SUMMARY_MODEL", model_summary)
        patch(agent_infrastructure, "TOOL_SUMMARY_MODEL", model_summary)

    if reasoning_effort:
        patch(agent_infrastructure, "REASONING_EFFORT", reasoning_effort)

    if tool_summary_reasoning_effort:
        patch(agent_infrastructure, "TOOL_SUMMARY_REASONING_EFFORT", tool_summary_reasoning_effort)
    elif reasoning_effort:
        patch(agent_infrastructure, "TOOL_SUMMARY_REASONING_EFFORT", reasoning_effort)

    if final_forecast_reasoning_effort:
        patch(final_forecast, "FINAL_FORECAST_REASONING_EFFORT", final_forecast_reasoning_effort)

    patch(final_forecast, "FINAL_FORECAST_USE_AGENT", strategy.final_forecast_use_agent)

    try:
        yield
    finally:
        if clear_summary_llm_cache:
            agent_infrastructure._get_tool_summary_llm.cache_clear()

        for module, name, old_value in reversed(patches):
            setattr(module, name, old_value)

        for key, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


async def _run_single_strategy_question(
    *,
    strategy: EvalStrategyConfig,
    question: EvalQuestion,
    num_runs: int,
) -> PredictionRecord:
    import src.forecast as forecast_module
    from src.agent_infrastructure import TOOL_CACHE_HIT_COUNTS, TOOL_CACHE_MISS_COUNTS, TOOL_CALL_COUNTS

    set_supabase_logging_enabled(False)

    question_details = {
        "id": question.question_id,
        "post_id": question.post_id,
        "title": question.title,
        "type": question.type,
        "description": question.description,
        "resolution_criteria": question.resolution_criteria,
        "fine_print": question.fine_print,
    }
    question_details = with_question_as_of_time(question_details, question.open_time)

    prediction_market_data = "None"
    if strategy.prediction_market_enabled:
        try:
            markets = await get_prediction_market_data(question.title)
            prediction_market_data = format_semipublic_market_data(markets)
        except Exception as exc:
            prediction_market_data = f"Prediction market lookup failed: {exc}"

    token_scope = f"eval-{strategy.id}-{question.post_id}-{uuid.uuid4().hex}"
    scope_token = set_usage_scope(token_scope)

    tool_counts_before = _snapshot_counter(TOOL_CALL_COUNTS)
    tool_cache_hit_before = _snapshot_counter(TOOL_CACHE_HIT_COUNTS)
    tool_cache_miss_before = _snapshot_counter(TOOL_CACHE_MISS_COUNTS)
    asknews_before_total = int(ASKNEWS_STATS.get("total", 0))
    asknews_before_removed = int(ASKNEWS_STATS.get("removed", 0))

    started = time.perf_counter()
    probability_yes = 0.5
    all_probabilities: list[float] = []
    forecast_stddev: float | None = None
    error: str | None = None

    try:
        reset_usage(scope=token_scope)
        as_of_iso = to_iso_z(question.open_time)

        with _temporary_eval_overrides(strategy, as_of_time_iso=as_of_iso):
            probability_yes, _comment, diagnostics = await forecast_module.get_binary_prediction(
                question_details,
                num_runs=num_runs,
                prediction_market_data=prediction_market_data,
                outside_view_enabled=strategy.outside_view_enabled,
                inside_view_enabled=strategy.inside_view_enabled,
                final_forecast_use_agent=strategy.final_forecast_use_agent,
            )

        maybe_probs = diagnostics.get("all_probabilities") if isinstance(diagnostics, dict) else None
        if isinstance(maybe_probs, list):
            all_probabilities = [float(x) for x in maybe_probs if isinstance(x, (float, int))]
        maybe_std = diagnostics.get("forecast_stddev") if isinstance(diagnostics, dict) else None
        if isinstance(maybe_std, (float, int)):
            forecast_stddev = float(maybe_std)

    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"

    elapsed = time.perf_counter() - started

    total_usage = get_total_usage(scope=token_scope)
    prompt_tokens = int(total_usage.get("prompt", 0))
    completion_tokens = int(total_usage.get("completion", 0))
    total_tokens = int(total_usage.get("total", 0))
    cost_usd = float(total_usage.get("cost", 0.0))

    tool_call_counts = _diff_counter(_snapshot_counter(TOOL_CALL_COUNTS), tool_counts_before)
    tool_cache_hit_counts = _diff_counter(_snapshot_counter(TOOL_CACHE_HIT_COUNTS), tool_cache_hit_before)
    tool_cache_miss_counts = _diff_counter(_snapshot_counter(TOOL_CACHE_MISS_COUNTS), tool_cache_miss_before)
    asknews_total_fetched = max(0, int(ASKNEWS_STATS.get("total", 0)) - asknews_before_total)
    asknews_removed_by_filter = max(0, int(ASKNEWS_STATS.get("removed", 0)) - asknews_before_removed)

    clear_usage_scope(scope=token_scope)
    reset_usage_scope(scope_token)

    if error is None:
        brier = compute_brier(probability_yes, question.label)
        log_loss = compute_log_loss(probability_yes, question.label)
    else:
        brier = math.inf
        log_loss = math.inf

    return PredictionRecord(
        strategy_id=strategy.id,
        post_id=question.post_id,
        question_id=question.question_id,
        title=question.title,
        as_of_time=to_iso_z(question.open_time),
        probability_yes=float(probability_yes),
        label=question.label,
        resolution_raw=question.resolution_raw,
        brier=float(brier),
        log_loss=float(log_loss),
        runtime_seconds=float(elapsed),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        num_runs=int(num_runs),
        outside_view_enabled=strategy.outside_view_enabled,
        inside_view_enabled=strategy.inside_view_enabled,
        prediction_market_enabled=strategy.prediction_market_enabled,
        final_forecast_use_agent=strategy.final_forecast_use_agent,
        tool_call_counts=tool_call_counts,
        tool_cache_hit_counts=tool_cache_hit_counts,
        tool_cache_miss_counts=tool_cache_miss_counts,
        asknews_total_fetched=asknews_total_fetched,
        asknews_removed_by_filter=asknews_removed_by_filter,
        all_probabilities=all_probabilities,
        forecast_stddev=forecast_stddev,
        error=error,
    )


async def run_eval(
    *,
    eval_question_file: str,
    strategy_files: list[str],
    output_dir: str,
    question_concurrency: int = 1,
) -> dict[str, Any]:
    strategies = load_strategy_files(strategy_files, force_nano_models=True)
    enabled_strategies = [s for s in strategies if s.enabled]
    if not enabled_strategies:
        raise ValueError("No enabled strategies found. At least one strategy must have enabled: true")
    if question_concurrency < 1:
        raise ValueError("question_concurrency must be >= 1")

    eval_questions_document = load_eval_question_file(eval_question_file)
    questions = eval_questions_document.questions

    run_id = _timestamp_run_id()
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[PredictionRecord] = []
    for strategy in enabled_strategies:
        semaphore = asyncio.Semaphore(question_concurrency)

        async def _run_with_limit(question: EvalQuestion) -> PredictionRecord:
            async with semaphore:
                return await _run_single_strategy_question(
                    strategy=strategy,
                    question=question,
                    num_runs=strategy.num_runs,
                )

        strategy_rows = await asyncio.gather(
            *[_run_with_limit(question) for question in questions]
        )
        predictions.extend(strategy_rows)

    summaries = summarize_strategies(predictions)
    calibration = compute_calibration_bins(predictions)

    run_config_snapshot = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_question_file": eval_question_file,
        "eval_question_schema_version": eval_questions_document.schema_version,
        "eval_question_count": len(questions),
        "strategy_files": strategy_files,
        "question_concurrency": question_concurrency,
        "output_dir": str(out_dir),
        "strategies": [asdict(s) for s in enabled_strategies],
        "questions": [
            {
                "post_id": q.post_id,
                "question_id": q.question_id,
                "title": q.title,
                "label": q.label,
                "resolution_raw": q.resolution_raw,
                "open_time": to_iso_z(q.open_time),
                "actual_resolve_time": to_iso_z(q.actual_resolve_time) if q.actual_resolve_time else None,
            }
            for q in questions
        ],
    }

    from .reporting import write_eval_reports

    report_paths = write_eval_reports(
        output_dir=str(out_dir),
        predictions=predictions,
        strategy_summaries=summaries,
        calibration_by_strategy=calibration,
        run_config_snapshot=run_config_snapshot,
    )

    return {
        "run_id": run_id,
        "output_dir": str(out_dir),
        "predictions": predictions,
        "summaries": summaries,
        "report_paths": report_paths,
    }
