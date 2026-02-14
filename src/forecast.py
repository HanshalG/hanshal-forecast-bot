import asyncio
import traceback
import os
import uuid
import json
import re
from typing import Tuple, List, Dict, Any

import numpy as np

from src.outside_view import (
    generate_outside_view,
    OUTSIDE_VIEW_MODEL,
)
from src.inside_view import (
    generate_inside_view,
    generate_inside_view_multiple_choice,
    generate_inside_view_numeric,
    LLM_MODEL as INSIDE_VIEW_MODEL,
)
from src.final_forecast import generate_final_forecast, FINAL_FORECAST_MODEL
from src.utils import (
    generate_continuous_cdf,
    enforce_cdf_monotonicity,
    read_prompt,
    call_llm,
    call_asknews_async,
)
from src.metaculus_utils import (
    forecast_is_already_made,
    post_question_comment,
    post_question_prediction,
    create_forecast_payload,
    get_post_details,
)
from src.forecast_logger import log_forecast_event
from src.agent_infrastructure import TOOL_CALL_COUNTS, TOOL_CACHE_HIT_COUNTS, TOOL_CACHE_MISS_COUNTS
from src.prediction_market_check import get_prediction_market_data, format_semipublic_market_data

# Module-level identifiers for logging context
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "openai/gpt-5-nano")
RUN_ID = os.getenv("RUN_ID") or f"run-{uuid.uuid4()}"
# This will be set by callers before forecasting session begins as needed
TOURNAMENT_ID = None


def _snapshot_counter(counter: dict[str, int]) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.items()}


def _diff_counter(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    keys = set(before.keys()) | set(after.keys())
    return {k: int(after.get(k, 0) - before.get(k, 0)) for k in sorted(keys)}


def _empty_prediction_diagnostics() -> dict[str, Any]:
    return {
        "outside_view_text": None,
        "inside_view_text": None,
        "final_forecast_analysis": None,
        "all_probabilities": None,
        "forecast_stddev": None,
    }


def _try_parse_json_dict_from_text(text: str) -> dict | None:
    if not isinstance(text, str):
        return None

    candidates = [text.strip()]
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        candidates.append(match.group(0).strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _normalize_final_forecast_data(final_forecast_data: dict) -> dict:
    if not isinstance(final_forecast_data, dict):
        return {}

    if any(k in final_forecast_data for k in ("probability", "percentiles", "probabilities")):
        return final_forecast_data

    for key in ("rationale", "raw_output", "result", "raw_result"):
        parsed = _try_parse_json_dict_from_text(final_forecast_data.get(key, ""))
        if parsed is not None:
            return parsed

    return final_forecast_data


async def get_binary_prediction(
    question_details: dict,
    num_runs: int,
    prediction_market_data: str = "None",
) -> Tuple[float, str, dict[str, Any]]:
    shared_news_context_task: asyncio.Task[str] | None = None
    shared_news_context_lock = asyncio.Lock()

    async def _get_shared_news_context() -> str | None:
        nonlocal shared_news_context_task
        if num_runs <= 1:
            return None

        if shared_news_context_task is None:
            async with shared_news_context_lock:
                if shared_news_context_task is None:
                    print("Fetching AskNews once for all binary runs...")
                    shared_news_context_task = asyncio.create_task(call_asknews_async(question_details))

        try:
            return await shared_news_context_task
        except Exception:
            async with shared_news_context_lock:
                if shared_news_context_task is not None and shared_news_context_task.done():
                    shared_news_context_task = None
            raise

    async def get_probability_and_comment(n) -> Tuple[float, str, str, str]:
        # Per-run retry logic (2 retries -> up to 3 attempts total)
        attempts = 0
        last_err: Exception | None = None
        while attempts < 3:
            attempts += 1
            try:
                # Generate outside view per run
                print(f"Generating outside view {n+1} (attempt {attempts})")

                outside_view_text = await generate_outside_view(question_details)
                
                print("\n" + "#" * 80 + "\n")
                print(f"Outside view {n+1} (attempt {attempts}): \"...{outside_view_text}\"")
                print("\n" + "#" * 80 + "\n")

                print(f"Generating inside view {n+1} (attempt {attempts})")
                shared_news_context = await _get_shared_news_context()
                inside_view_text = await generate_inside_view(
                    question_details,
                    news_context=shared_news_context,
                )

                print("\n" + "#" * 80 + "\n")
                print(f"Inside view {n+1} (attempt {attempts}): \"...{inside_view_text}\"")
                print("\n" + "#" * 80 + "\n")

                print("\n" + "#" * 80 + "\n")
                print("Prediction Market Data:", prediction_market_data)
                print("\n" + "#" * 80 + "\n")

                # Run Final Forecast Agent per run
                print(f"Running Final Forecast Agent for run {n+1}...")
                final_forecast_data = await generate_final_forecast(
                        question_details,
                        outside_view_text=outside_view_text,
                        inside_view_text=inside_view_text,
                        prediction_market_data=prediction_market_data,
                )
                final_forecast_data = _normalize_final_forecast_data(final_forecast_data)

                print("\n" + "#" * 80 + "\n")
                print(f"Final Forecast Agent output for run {n+1}: {final_forecast_data}")
                print("\n" + "#" * 80 + "\n")
                
                # Robust parsing of final forecast data - raise errors instead of using fallbacks
                # Extract probability
                p_val = final_forecast_data.get("probability")
                if p_val is None:
                    raise ValueError(f"No probability found in forecast data: {final_forecast_data}")
                
                # Handle various probability formats
                if isinstance(p_val, str):
                    # Handle "45%", "0.45", "45", etc.
                    clean_p = p_val.replace("%", "").strip()
                    try:
                        p_val = float(clean_p)
                    except ValueError:
                        raise ValueError(f"Could not parse probability string '{p_val}' to float")
                    # Convert percentage to decimal if > 1
                    if p_val > 1.0:
                        p_val /= 100.0
                
                probability = float(p_val)
                
                # Clamp to valid range
                probability = max(0.001, min(0.999, probability))
                
                # Extract rationale
                final_forecast_text = final_forecast_data.get("rationale", "")
                if not final_forecast_text:
                    raise ValueError(f"No rationale found in forecast data: {final_forecast_data}")

                return probability, final_forecast_text, outside_view_text, inside_view_text

            except Exception as e:
                print(f"Error in run {n+1} (attempt {attempts}): {e}")
                last_err = e
                await asyncio.sleep(0.5 * attempts)
        
        raise last_err if last_err is not None else RuntimeError("Unknown error in run")

    # Run tasks
    results = await asyncio.gather(
        *[get_probability_and_comment(n) for n in range(num_runs)],
        return_exceptions=True,
    )
    
    successes: list[tuple[float, str, str, str]] = [r for r in results if not isinstance(r, Exception)]
    
    if len(successes) == 0:
        first_err = next((r for r in results if isinstance(r, Exception)), RuntimeError("All runs failed"))
        raise first_err

    probs = np.array([p for p, _, _, _ in successes], dtype=float)
    rationales = [r for _, r, _, _ in successes]
    outside_views = [o for _, _, o, _ in successes]
    inside_views = [i for _, _, _, i in successes]
    run_probabilities = [float(p) for p in probs.tolist()]

    print("Probabilities from runs:")
    for p in probs:
        print(f"{p:.3f}")

    # Calculate and Clip Median
    median_probability = np.clip(np.median(probs), 0.001, 0.999)

    # Find the median rationale
    closest_index = np.argmin(np.abs(probs - median_probability))
    best_rationale = rationales[closest_index]

    print(f"Median probability: {median_probability:.3f}")
    probability_stddev = float(np.std(probs))
    print(f"Run probability stddev: {probability_stddev:.4f}")
    print(f"Best Rationale: {best_rationale[:50]}...") # Print a snippet

    # Generate Final Comment Summary using gpt-5-nano
    print("Generating Final Forecast Comment...")
    final_comment = ""
    try:
        summary_prompt_template = read_prompt("analysis_summary_bullets.txt")
        summary_prompt = summary_prompt_template.format(
            title=question_details.get("title", ""),
            question_type=question_details.get("type", "binary"),
            resolution_criteria=question_details.get("resolution_criteria", ""),
            fine_print=question_details.get("fine_print", ""),
            analysis=best_rationale,
        )        
        final_comment = await call_llm(summary_prompt, SUMMARY_MODEL, 0.3, "medium")
        final_comment = final_comment.strip()
    except Exception as e:
        print(f"Summary generation failed: {e}")
        final_comment = best_rationale

    if not final_comment:
         final_comment = best_rationale

    diagnostics = {
        "outside_view_text": outside_views[int(closest_index)] if outside_views else None,
        "inside_view_text": inside_views[int(closest_index)] if inside_views else None,
        "final_forecast_analysis": best_rationale,
        "all_probabilities": run_probabilities,
        "forecast_stddev": probability_stddev,
    }
    return median_probability, final_comment, diagnostics


async def get_numeric_prediction(
    question_details: dict,
    num_runs: int,
    prediction_market_data: str = "",
) -> Tuple[List[float], str, dict[str, Any]]:
    shared_news_context_task: asyncio.Task[str] | None = None
    shared_news_context_lock = asyncio.Lock()

    async def _get_shared_news_context() -> str | None:
        nonlocal shared_news_context_task
        if num_runs <= 1:
            return None

        if shared_news_context_task is None:
            async with shared_news_context_lock:
                if shared_news_context_task is None:
                    print("Fetching AskNews once for all numeric runs...")
                    shared_news_context_task = asyncio.create_task(call_asknews_async(question_details))

        try:
            return await shared_news_context_task
        except Exception:
            async with shared_news_context_lock:
                if shared_news_context_task is not None and shared_news_context_task.done():
                    shared_news_context_task = None
            raise

    question_type = question_details["type"]
    scaling = question_details["scaling"]
    open_upper_bound = question_details["open_upper_bound"]
    open_lower_bound = question_details["open_lower_bound"]
    unit_of_measure = (
        question_details["unit"] if question_details["unit"] else "Not stated (please infer this)"
    )
    upper_bound = scaling["range_max"]
    lower_bound = scaling["range_min"]
    zero_point = scaling["zero_point"]
    if question_type == "discrete":
        outcome_count = question_details["scaling"]["inbound_outcome_count"]
        cdf_size = outcome_count + 1
    else:
        cdf_size = 201

    if open_upper_bound:
        upper_bound_message = ""
    else:
        upper_bound_message = f"The outcome can not be higher than {upper_bound}."
    if open_lower_bound:
        lower_bound_message = ""
    else:
        lower_bound_message = f"The outcome can not be lower than {lower_bound}."

    async def get_numeric_cdf_and_comment(n) -> Tuple[List[float], str, str, str]:
        attempts = 0
        last_err: Exception | None = None
        while attempts < 3:
            attempts += 1
            try:
                # Generate outside view per run
                print(f"Generating outside view {n+1} (attempt {attempts})")
                outside_view_text = await generate_outside_view(question_details)
                try:
                    print(f"Outside view {n+1} (attempt {attempts}): \"...{outside_view_text[-200:]}\"")
                except Exception:
                    pass
                    
                print(f"Generating inside view {n+1} (attempt {attempts})")
                shared_news_context = await _get_shared_news_context()
                inside_view_text = await generate_inside_view_numeric(
                    question_details,
                    news_context=shared_news_context,
                    units=unit_of_measure,
                    lower_bound_message=lower_bound_message,
                    upper_bound_message=upper_bound_message,
                    hint="",
                )
                try:
                    print(f"Inside view {n+1} (attempt {attempts}): \"...{inside_view_text[-200:]}\"")
                except Exception:
                    pass

                # Run Final Forecast Agent per run
                print(f"Running Final Forecast Agent for run {n+1}...")
                final_forecast_data = await generate_final_forecast(
                    question_details,
                    outside_view_text=outside_view_text,
                    inside_view_text=inside_view_text,
                    prediction_market_data=prediction_market_data,
                )
                final_forecast_data = _normalize_final_forecast_data(final_forecast_data)
                print(f"Final Forecast Agent output for run {n+1}: {final_forecast_data}")
                
                # Parse percentiles from agent output
                percentiles_dict = final_forecast_data.get("percentiles")
                if not percentiles_dict:
                    raise ValueError(f"No percentiles found in forecast data: {final_forecast_data}")
                
                # Clean percentiles dict
                clean_percentiles = {}
                for k, v in percentiles_dict.items():
                    k_str = str(k).lower().replace("p", "").replace("%", "").strip()
                    if k_str.isdigit():
                        clean_percentiles[int(k_str)] = float(v)
                
                if not clean_percentiles:
                    raise ValueError(f"Could not parse percentiles from: {percentiles_dict}")
                
                # Extract rationale
                rationale = final_forecast_data.get("rationale", "")
                if not rationale:
                    raise ValueError(f"No rationale found in forecast data: {final_forecast_data}")

                # Generate CDF from percentiles
                cdf = generate_continuous_cdf(
                    clean_percentiles,
                    question_type,
                    open_upper_bound,
                    open_lower_bound,
                    upper_bound,
                    lower_bound,
                    zero_point,
                    cdf_size,
                )

                return cdf, rationale, outside_view_text, inside_view_text
                
            except Exception as e:
                print(f"Error in run {n+1} (attempt {attempts}): {e}")
                last_err = e
                await asyncio.sleep(0.5 * attempts)
                
        raise last_err if last_err is not None else RuntimeError("Unknown error in numeric run")

    cdf_results = await asyncio.gather(
        *[get_numeric_cdf_and_comment(n) for n in range(num_runs)], return_exceptions=True
    )
    
    successes: list[tuple[List[float], str, str, str]] = [r for r in cdf_results if not isinstance(r, Exception)]
    
    if len(successes) == 0:
        first_err = next((r for r in cdf_results if isinstance(r, Exception)), RuntimeError("All numeric runs failed"))
        raise first_err

    cdfs: List[List[float]] = [pair[0] for pair in successes]
    rationales = [pair[1] for pair in successes]
    outside_views = [pair[2] for pair in successes]
    inside_views = [pair[3] for pair in successes]
    
    all_cdfs = np.array(cdfs)
    median_cdf: List[float] = np.median(all_cdfs, axis=0).tolist()

    # Ensure median CDF satisfies API monotonicity constraints
    median_cdf = enforce_cdf_monotonicity(
        median_cdf,
        open_upper_bound=open_upper_bound,
        open_lower_bound=open_lower_bound,
        min_delta=5e-05,
    )

    # Choose the rationale whose CDF is closest (L2) to the median CDF
    diffs = [float(np.linalg.norm(np.array(c) - np.array(median_cdf))) for c in cdfs]
    closest_index = int(np.argmin(diffs))
    best_rationale = rationales[closest_index]

    print(f"Median CDF calculated (length: {len(median_cdf)})")
    cdf_pointwise_std = np.std(all_cdfs, axis=0)
    cdf_stddev_mean = float(np.mean(cdf_pointwise_std))
    print(f"Mean pointwise CDF stddev across runs: {cdf_stddev_mean:.6f}")
    print(f"Best Rationale: {best_rationale[:50]}...")

    # Generate Final Comment Summary using gpt-5-nano
    print("Generating Final Forecast Comment...")
    final_comment = ""
    try:
        summary_prompt_template = read_prompt("analysis_summary_bullets.txt")
        summary_prompt = summary_prompt_template.format(
            title=question_details.get("title", ""),
            question_type=question_details.get("type", "numeric"),
            resolution_criteria=question_details.get("resolution_criteria", ""),
            fine_print=question_details.get("fine_print", ""),
            analysis=best_rationale,
        )
        final_comment = await call_llm(summary_prompt, SUMMARY_MODEL, 0.3, "medium")
        final_comment = final_comment.strip()
    except Exception as e:
        print(f"Summary generation failed: {e}")
        final_comment = best_rationale

    if not final_comment:
        final_comment = best_rationale

    diagnostics = {
        "outside_view_text": outside_views[closest_index] if outside_views else None,
        "inside_view_text": inside_views[closest_index] if inside_views else None,
        "final_forecast_analysis": best_rationale,
        "all_probabilities": cdfs,
        "forecast_stddev": cdf_stddev_mean,
    }
    return median_cdf, final_comment, diagnostics


async def get_multiple_choice_prediction(
    question_details: dict,
    num_runs: int,
    prediction_market_data: str = "",
) -> Tuple[Dict[str, float], str, dict[str, Any]]:
    shared_news_context_task: asyncio.Task[str] | None = None
    shared_news_context_lock = asyncio.Lock()

    async def _get_shared_news_context() -> str | None:
        nonlocal shared_news_context_task
        if num_runs <= 1:
            return None

        if shared_news_context_task is None:
            async with shared_news_context_lock:
                if shared_news_context_task is None:
                    print("Fetching AskNews once for all multiple-choice runs...")
                    shared_news_context_task = asyncio.create_task(call_asknews_async(question_details))

        try:
            return await shared_news_context_task
        except Exception:
            async with shared_news_context_lock:
                if shared_news_context_task is not None and shared_news_context_task.done():
                    shared_news_context_task = None
            raise

    options = question_details["options"]

    async def ask_inside_view_mc(n) -> Tuple[Dict[str, float], str, str, str]:
        attempts = 0
        last_err: Exception | None = None
        while attempts < 3:
            attempts += 1
            try:
                # Generate outside view per run
                print(f"Generating outside view {n+1} (attempt {attempts})")
                outside_view_text = await generate_outside_view(question_details)
                try:
                    print(f"Outside view {n+1} (attempt {attempts}): \"...{outside_view_text[-200:]}\"")
                except Exception:
                    pass
                    
                print(f"Generating inside view {n+1} (attempt {attempts})")
                shared_news_context = await _get_shared_news_context()
                inside_view_text = await generate_inside_view_multiple_choice(
                    question_details,
                    news_context=shared_news_context,
                )
                try:
                    print(f"Inside view {n+1} (attempt {attempts}): \"...{inside_view_text[-200:]}\"")
                except Exception:
                    pass

                # Run Final Forecast Agent per run
                print(f"Running Final Forecast Agent for run {n+1}...")
                final_forecast_data = await generate_final_forecast(
                    question_details,
                    outside_view_text=outside_view_text,
                    inside_view_text=inside_view_text,
                    prediction_market_data=prediction_market_data,
                )
                final_forecast_data = _normalize_final_forecast_data(final_forecast_data)
                print(f"Final Forecast Agent output for run {n+1}: {final_forecast_data}")
                
                # Parse probabilities from agent output
                probs_dict = final_forecast_data.get("probabilities")
                if not probs_dict:
                    raise ValueError(f"No probabilities found in forecast data: {final_forecast_data}")
                
                # Match keys in 'options' list and normalize
                raw_probs = [float(probs_dict.get(opt, 0.0)) for opt in options]
                s = sum(raw_probs)
                if s <= 0:
                    raise ValueError(f"Probabilities sum to zero or less: {probs_dict}")
                    
                norm_probs = [p/s for p in raw_probs]
                probability_yes_per_category = {opt: norm_probs[i] for i, opt in enumerate(options)}
                
                # Extract rationale
                rationale = final_forecast_data.get("rationale", "")
                if not rationale:
                    raise ValueError(f"No rationale found in forecast data: {final_forecast_data}")

                return probability_yes_per_category, rationale, outside_view_text, inside_view_text
                
            except Exception as e:
                print(f"Error in run {n+1} (attempt {attempts}): {e}")
                last_err = e
                await asyncio.sleep(0.5 * attempts)
                
        raise last_err if last_err is not None else RuntimeError("Unknown error in MC run")

    mc_results = await asyncio.gather(
        *[ask_inside_view_mc(n) for n in range(num_runs)], return_exceptions=True
    )
    
    successes: list[tuple[Dict[str, float], str, str, str]] = [r for r in mc_results if not isinstance(r, Exception)]
    
    if len(successes) == 0:
        first_err = next((r for r in mc_results if isinstance(r, Exception)), RuntimeError("All MC runs failed"))
        raise first_err

    probability_yes_per_category_dicts: List[Dict[str, float]] = [pair[0] for pair in successes]
    rationales = [pair[1] for pair in successes]
    outside_views = [pair[2] for pair in successes]
    inside_views = [pair[3] for pair in successes]

    # --- Trimmed Linear Opinion Pool ---
    # Build matrix of shape (num_runs, num_options) in the provided option order
    run_matrix = np.array(
        [[float(run_probs.get(opt, 0.0)) for opt in options] for run_probs in probability_yes_per_category_dicts],
        dtype=float,
    )

    # Compute baseline mean and per-run L2 distances
    mean_vector = run_matrix.mean(axis=0)
    distances = np.linalg.norm(run_matrix - mean_vector, axis=1)

    # Determine how many runs to trim (top tail by distance). Use a modest 20% trim.
    trim_fraction = 0.2
    n_runs = run_matrix.shape[0]
    trim_k = int(np.floor(trim_fraction * n_runs))
    # Ensure we keep at least 1 run
    keep_count = max(1, n_runs - trim_k)

    # Keep the runs with smallest distances
    keep_indices = np.argsort(distances)[:keep_count]
    kept_matrix = run_matrix[keep_indices]

    # Average the kept runs and renormalize to ensure the probabilities sum to 1.0
    pooled_vector = kept_matrix.mean(axis=0)
    pooled_sum = float(pooled_vector.sum())
    if pooled_sum > 0.0:
        pooled_vector = pooled_vector / pooled_sum
    else:
        # Edge-case fallback: uniform distribution
        pooled_vector = np.ones_like(pooled_vector) / float(len(pooled_vector) or 1)

    final_probs_dict: Dict[str, float] = {
        opt: float(p) for opt, p in zip(options, pooled_vector)
    }

    print(f"Trimmed mean probabilities (kept {keep_count}/{n_runs} runs)")
    print(f"Final probabilities: {final_probs_dict}")
    option_std_values = np.std(run_matrix, axis=0)
    mc_stddev_mean = float(np.mean(option_std_values))
    print(f"Mean option-probability stddev across runs: {mc_stddev_mean:.6f}")

    # Select rationale closest to pooled vector
    dists_to_pooled = [float(np.linalg.norm(run_matrix[i] - pooled_vector)) for i in range(n_runs)]
    closest_index = int(np.argmin(dists_to_pooled))
    best_rationale = rationales[closest_index]

    print(f"Best Rationale: {best_rationale[:50]}...")

    # Generate Final Comment Summary using gpt-5-nano
    print("Generating Final Forecast Comment...")
    final_comment = ""
    try:
        summary_prompt_template = read_prompt("analysis_summary_bullets.txt")
        summary_prompt = summary_prompt_template.format(
            title=question_details.get("title", ""),
            question_type=question_details.get("type", "multiple_choice"),
            resolution_criteria=question_details.get("resolution_criteria", ""),
            fine_print=question_details.get("fine_print", ""),
            analysis=best_rationale,
        )
        final_comment = await call_llm(summary_prompt, SUMMARY_MODEL, 0.3, "medium")
        final_comment = final_comment.strip()
    except Exception as e:
        print(f"Summary generation failed: {e}")
        final_comment = best_rationale

    if not final_comment:
        final_comment = best_rationale
        
    diagnostics = {
        "outside_view_text": outside_views[closest_index] if outside_views else None,
        "inside_view_text": inside_views[closest_index] if inside_views else None,
        "final_forecast_analysis": best_rationale,
        "all_probabilities": probability_yes_per_category_dicts,
        "forecast_stddev": mc_stddev_mean,
    }
    return final_probs_dict, final_comment, diagnostics


################### FORECASTING ###################

async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
    get_prediction_market: bool = False,
) -> str:
    # Reset token usage at start of each forecast
    from src.token_cost import (
        clear_usage_scope,
        get_total_usage,
        get_usage_breakdown,
        print_total_usage,
        reset_usage,
        reset_usage_scope,
        set_usage_scope,
    )
    from src.utils import ASKNEWS_STATS

    token_scope = f"question-{question_id}-{post_id}-{uuid.uuid4().hex}"
    scope_token = set_usage_scope(token_scope)

    try:
        reset_usage()
        tool_counts_before = _snapshot_counter(TOOL_CALL_COUNTS)
        tool_cache_hit_before = _snapshot_counter(TOOL_CACHE_HIT_COUNTS)
        tool_cache_miss_before = _snapshot_counter(TOOL_CACHE_MISS_COUNTS)
        asknews_before = {
            "total": int(ASKNEWS_STATS.get("total", 0)),
            "removed": int(ASKNEWS_STATS.get("removed", 0)),
        }
        
        post_details = get_post_details(post_id)
        question_details = post_details["question"]
        title = question_details["title"]
        question_type = question_details["type"]

        print(f"Forecasting question: {title} ({question_type})")

        summary_of_forecast = ""
        summary_of_forecast += f"-----------------------------------------------\nQuestion: {title}\n"
        summary_of_forecast += f"URL: https://www.metaculus.com/questions/{post_id}/\n"

        if question_type == "multiple_choice":
            options = question_details["options"]
            print(f"Options: {options}")
            summary_of_forecast += f"Options: {options}\n"

        if (
            forecast_is_already_made(post_details)
            and skip_previously_forecasted_questions == True
        ):
            print(f"Skipped: Forecast already made")
            summary_of_forecast += f"Skipped: Forecast already made\n"
            return summary_of_forecast

        prediction_diagnostics = _empty_prediction_diagnostics()

        # Fetch prediction market data if requested
        prediction_market_data_str = ""
        if get_prediction_market:
            print(f"Fetching prediction market data for: {title}")
            try:
                markets = await get_prediction_market_data(title)
                prediction_market_data_str = format_semipublic_market_data(markets)
                print(f"Prediction Market Data:\n{prediction_market_data_str[:200]}...")
                summary_of_forecast += f"\nPrediction Markets checked:\n{prediction_market_data_str}\n" 
            except Exception as e:
                print(f"Error fetching prediction market data: {e}")
                summary_of_forecast += f"\nPrediction Markets check failed: {e}\n"

        if question_type == "binary":
            forecast, comment, prediction_diagnostics = await get_binary_prediction(
                question_details, num_runs_per_question, prediction_market_data_str
            )
        elif question_type == "numeric":
            forecast, comment, prediction_diagnostics = await get_numeric_prediction(
                question_details, num_runs_per_question, prediction_market_data_str
            )
        elif question_type == "discrete":
            forecast, comment, prediction_diagnostics = await get_numeric_prediction(
                question_details, num_runs_per_question, prediction_market_data_str
            )
        elif question_type == "multiple_choice":
            forecast, comment, prediction_diagnostics = await get_multiple_choice_prediction(
                question_details, num_runs_per_question, prediction_market_data_str
            )
        else:
            raise ValueError(f"Unknown question type: {question_type}")

        print(f"-----------------------------------------------\nPost {post_id} Question {question_id}:\n")
        print(f"Forecast for post {post_id} (question {question_id}):\n{forecast}")
        print(f"Comment for post {post_id} (question {question_id}):\n{comment}")

        total_usage = get_total_usage(scope=token_scope)
        usage_breakdown = get_usage_breakdown(scope=token_scope)
        tool_counts_after = _snapshot_counter(TOOL_CALL_COUNTS)
        tool_cache_hit_after = _snapshot_counter(TOOL_CACHE_HIT_COUNTS)
        tool_cache_miss_after = _snapshot_counter(TOOL_CACHE_MISS_COUNTS)
        asknews_after = {
            "total": int(ASKNEWS_STATS.get("total", 0)),
            "removed": int(ASKNEWS_STATS.get("removed", 0)),
        }
        tool_call_counts = _diff_counter(tool_counts_after, tool_counts_before)
        tool_cache_hit_counts = _diff_counter(tool_cache_hit_after, tool_cache_hit_before)
        tool_cache_miss_counts = _diff_counter(tool_cache_miss_after, tool_cache_miss_before)
        asknews_total_fetched = max(0, asknews_after["total"] - asknews_before["total"])
        asknews_removed_by_filter = max(0, asknews_after["removed"] - asknews_before["removed"])

        # Log forecast event (pre-submission)
        try:
            event = {
                "run_id": RUN_ID,
                "tournament_id": TOURNAMENT_ID,
                "question_id": question_id,
                "post_id": post_id,
                "question_title": title,
                "question_type": question_type,
                "model": SUMMARY_MODEL,
                "outside_view_model": OUTSIDE_VIEW_MODEL,
                "inside_view_model": INSIDE_VIEW_MODEL,
                "final_forecast_model": FINAL_FORECAST_MODEL,
                "summary_model": SUMMARY_MODEL,
                "num_runs": num_runs_per_question,
                "forecast": forecast,
                "comment": comment,
                "submit_attempted": bool(submit_prediction),
                "submitted": False,
                "outside_view_text": prediction_diagnostics.get("outside_view_text"),
                "inside_view_text": prediction_diagnostics.get("inside_view_text"),
                "final_forecast_analysis": prediction_diagnostics.get("final_forecast_analysis"),
                "all_probabilities": prediction_diagnostics.get("all_probabilities"),
                "forecast_stddev": prediction_diagnostics.get("forecast_stddev"),
                "prompt_tokens": int(total_usage.get("prompt", 0)),
                "completion_tokens": int(total_usage.get("completion", 0)),
                "cost_usd": float(total_usage.get("cost", 0.0)),
                "token_usage_by_component": usage_breakdown,
                "tool_call_counts": tool_call_counts,
                "tool_cache_hit_counts": tool_cache_hit_counts,
                "tool_cache_miss_counts": tool_cache_miss_counts,
                "asknews_total_fetched": asknews_total_fetched,
                "asknews_removed_by_filter": asknews_removed_by_filter,
            }
            log_forecast_event(event)
        except Exception as e:
            print(f"Logging error (pre-submission): {e}")

        if question_type == "numeric" or question_type == "discrete":
            summary_of_forecast += f"Forecast: {str(forecast)[:200]}...\n"
        else:
            summary_of_forecast += f"Forecast: {forecast}\n"

        summary_of_forecast += f"Comment:\n```\n{comment[:200]}...\n```\n\n"

        if submit_prediction == True:
            forecast_payload = create_forecast_payload(forecast, question_type)
            post_question_prediction(question_id, forecast_payload)
            post_question_comment(post_id, comment)
            summary_of_forecast += "Posted: Forecast was posted to Metaculus.\n"

            # Log successful submission
            try:
                submitted_event = {
                    "run_id": RUN_ID,
                    "tournament_id": TOURNAMENT_ID,
                    "question_id": question_id,
                    "post_id": post_id,
                    "question_title": title,
                    "question_type": question_type,
                    "model": SUMMARY_MODEL,
                    "outside_view_model": OUTSIDE_VIEW_MODEL,
                    "inside_view_model": INSIDE_VIEW_MODEL,
                    "final_forecast_model": FINAL_FORECAST_MODEL,
                    "summary_model": SUMMARY_MODEL,
                    "num_runs": num_runs_per_question,
                    "forecast": forecast,
                    "comment": comment,
                    "submit_attempted": True,
                    "submitted": True,
                    "outside_view_text": prediction_diagnostics.get("outside_view_text"),
                    "inside_view_text": prediction_diagnostics.get("inside_view_text"),
                    "final_forecast_analysis": prediction_diagnostics.get("final_forecast_analysis"),
                    "all_probabilities": prediction_diagnostics.get("all_probabilities"),
                    "forecast_stddev": prediction_diagnostics.get("forecast_stddev"),
                    "prompt_tokens": int(total_usage.get("prompt", 0)),
                    "completion_tokens": int(total_usage.get("completion", 0)),
                    "cost_usd": float(total_usage.get("cost", 0.0)),
                    "token_usage_by_component": usage_breakdown,
                    "tool_call_counts": tool_call_counts,
                    "tool_cache_hit_counts": tool_cache_hit_counts,
                    "tool_cache_miss_counts": tool_cache_miss_counts,
                    "asknews_total_fetched": asknews_total_fetched,
                    "asknews_removed_by_filter": asknews_removed_by_filter,
                }
                log_forecast_event(submitted_event)
            except Exception as e:
                print(f"Logging error (post-submission): {e}")

        # Print token usage summary for this forecast
        print_total_usage()

        return summary_of_forecast
    finally:
        clear_usage_scope(scope=token_scope)
        reset_usage_scope(scope_token)


async def forecast_questions(
    open_question_id_post_id: list[tuple[int, int]],
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
    get_prediction_market: bool = False,
) -> None:
    forecast_tasks = [
        forecast_individual_question(
            question_id,
            post_id,
            submit_prediction,
            num_runs_per_question,
            skip_previously_forecasted_questions,
            get_prediction_market,
        )
        for question_id, post_id in open_question_id_post_id
    ]
    forecast_summaries = await asyncio.gather(*forecast_tasks, return_exceptions=True)
    
    print("\n", "#" * 100, "\nForecast Summaries\n", "#" * 100)

    errors = []
    for question_id_post_id, forecast_summary in zip(
        open_question_id_post_id, forecast_summaries
    ):
        question_id, post_id = question_id_post_id
        if isinstance(forecast_summary, Exception):
            # Print concise error header
            print(
                f"-----------------------------------------------\nPost {post_id} Question {question_id}:\nError: {forecast_summary.__class__.__name__} {forecast_summary}\nURL: https://www.metaculus.com/questions/{post_id}/\n"
            )
            # Also print full traceback for debugging
            try:
                trace_str = "".join(
                    traceback.format_exception(
                        type(forecast_summary), forecast_summary, forecast_summary.__traceback__
                    )
                )
                print(trace_str)
            except Exception:
                trace_str = None
            errors.append(forecast_summary)
            # Log error event
            try:
                error_event = {
                    "run_id": RUN_ID,
                    "tournament_id": TOURNAMENT_ID,
                    "question_id": question_id,
                    "post_id": post_id,
                    "question_title": None,
                    "question_type": None,
                    "model": SUMMARY_MODEL,
                    "outside_view_model": OUTSIDE_VIEW_MODEL,
                    "inside_view_model": INSIDE_VIEW_MODEL,
                    "final_forecast_model": FINAL_FORECAST_MODEL,
                    "summary_model": SUMMARY_MODEL,
                    "num_runs": num_runs_per_question,
                    "forecast": None,
                    "comment": None,
                    "submit_attempted": bool(submit_prediction),
                    "submitted": False,
                    "error": f"{forecast_summary.__class__.__name__}: {forecast_summary}",
                    **({"error_trace": trace_str} if trace_str else {}),
                }
                log_forecast_event(error_event)
            except Exception as e:
                print(f"Logging error (error-event): {e}")
        else:
            print(forecast_summary)

    if errors:
        print("-----------------------------------------------\nErrors:\n")
        error_message = f"Errors were encountered: {errors}"
        print(error_message)
        print(error_message)
        raise RuntimeError(error_message)

    print("\n" + "=" * 50)
    print("Tool Usage Statistics:", flush=True)
    for tool_name, count in TOOL_CALL_COUNTS.items():
        cache_hits = TOOL_CACHE_HIT_COUNTS.get(tool_name, 0)
        cache_misses = TOOL_CACHE_MISS_COUNTS.get(tool_name, 0)
        print(
            f"{tool_name}: {count} (cache_hits={cache_hits}, cache_misses={cache_misses})",
            flush=True,
        )
    
    from src.utils import ASKNEWS_STATS
    print("\nAskNews Statistics:", flush=True)
    print(f"Total Fetched: {ASKNEWS_STATS['total']}", flush=True)
    print(f"Removed by Filter: {ASKNEWS_STATS['removed']}", flush=True)
    print("=" * 50 + "\n", flush=True)
