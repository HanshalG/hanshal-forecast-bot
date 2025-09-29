import asyncio
import traceback
import datetime
import os
import uuid
from typing import Tuple, List, Dict

import numpy as np

from src.outside_view import (
    generate_outside_view,
    prepare_outside_view_context,
)
from src.inside_view import (
    prepare_inside_view_context,
    generate_inside_view,
    generate_inside_view_multiple_choice,
    generate_inside_view_numeric,
)
from src.utils import (
    extract_probability_from_response_as_percentage_not_decimal,
    extract_percentiles_from_response,
    generate_continuous_cdf,
    enforce_cdf_monotonicity,
    extract_option_probabilities_from_response,
    read_prompt,
    call_llm,
)
from src.metaculus_utils import (
    forecast_is_already_made,
    post_question_comment,
    post_question_prediction,
    create_forecast_payload,
    get_open_question_ids_from_tournament,
    get_post_details,
)
from src.forecast_logger import log_forecast_event

# Module-level identifiers for logging context
LLM_MODEL = "openai/gpt-5-mini"
RUN_ID = os.getenv("RUN_ID") or f"run-{uuid.uuid4()}"
# This will be set by callers before forecasting session begins as needed
TOURNAMENT_ID = None


def generate_multiple_choice_forecast(options, option_probabilities) -> dict:
    """
    Returns: dict corresponding to the probabilities of each option.
    """

    if len(options) != len(option_probabilities):
        raise ValueError(
            f"Number of options ({len(options)}) does not match number of probabilities ({len(option_probabilities)})"
        )

    total_sum = sum(option_probabilities)
    decimal_list = [x / total_sum for x in option_probabilities]

    def normalize_list(float_list):
        clamped_list = [max(min(x, 0.99), 0.01) for x in float_list]
        total_sum = sum(clamped_list)
        normalized_list = [x / total_sum for x in clamped_list]
        adjustment = 1.0 - sum(normalized_list)
        normalized_list[-1] += adjustment
        return normalized_list

    normalized_option_probabilities = normalize_list(decimal_list)

    probability_yes_per_category = {}
    for i in range(len(options)):
        probability_yes_per_category[options[i]] = normalized_option_probabilities[i]

    return probability_yes_per_category


async def get_binary_prediction(
    question_details: dict,
    num_runs: int,
) -> Tuple[float, str]:
    print(f"Preparing outside view context")
    historical_context = await prepare_outside_view_context(question_details)

    # Prepare inside view context once
    print(f"Preparing inside view context")
    pre_news_ctx, pre_exa_ctx = await prepare_inside_view_context(question_details)

    async def get_inside_probability_and_comment(n) -> Tuple[float, str, str]:
        # Per-run retry logic (2 retries -> up to 3 attempts total)
        attempts = 0
        last_err: Exception | None = None
        while attempts < 3:
            attempts += 1
            try:
                # Generate outside view per run using the precomputed historical context
                print(f"Generating outside view {n+1} (attempt {attempts})")
                outside_view_text = await generate_outside_view(question_details, historical_context)
                print(f"Outside view {n+1} (attempt {attempts}): \"...{outside_view_text[-200:]}\"")

                print(f"Generating inside view {n+1} (attempt {attempts})")
                inside_view_text = await generate_inside_view(
                    question_details,
                    outside_view_text,
                    precomputed_news_context=pre_news_ctx,
                    precomputed_exa_context=pre_exa_ctx,
                )
                print(f"Inside view {n+1} (attempt {attempts}): \"...{inside_view_text[-200:]}\"")

                probability = extract_probability_from_response_as_percentage_not_decimal(
                    inside_view_text
                )
                comment = (
                    f"## Outside View\n{outside_view_text}\n\n"
                    f"## Inside View Analysis\n{inside_view_text}\n\n"
                    f"Extracted Probability: {probability}%\n\n"
                )
                return probability, comment, inside_view_text
            except Exception as e:
                print(f"Error in inside-view run {n+1} (attempt {attempts}): {e}")
                last_err = e
                # small backoff before retrying
                await asyncio.sleep(0.5 * attempts)
        # After retries, re-raise last error
        raise last_err if last_err is not None else RuntimeError("Unknown error in inside-view run")

    # Run inside-view tasks; tolerate per-run failures
    results = await asyncio.gather(
        *[get_inside_probability_and_comment(n) for n in range(num_runs)],
        return_exceptions=True,
    )
    successes: list[tuple[float, str, str]] = [r for r in results if not isinstance(r, Exception)]  # type: ignore[list-item]
    if len(successes) == 0:
        # If all runs failed, raise the first error
        first_err = next((r for r in results if isinstance(r, Exception)), RuntimeError("All inside-view runs failed"))
        raise first_err  # type: ignore[misc]

    comments = [pair[1] for pair in successes]
    final_comment_sections = [f"### Run {i+1}\n{comment}" for i, comment in enumerate(comments)]
    probabilities = [pair[0] for pair in successes]
    inside_views = [pair[2] for pair in successes]

    #sort probabilities by descending order
    probabilities.sort(reverse=True)

    print("Probabilities: ")
    for p in probabilities:
        print(p)

    median_probability = float(np.median(probabilities)) / 100

    print("Median probability: ", median_probability)

    # Select the inside-view analysis whose probability is closest to the median
    try:
        probs_array = np.array(probabilities, dtype=float)
        target_pct = median_probability * 100.0
        closest_index = int(np.argmin(np.abs(probs_array - target_pct)))
        median_inside_view_analysis = inside_views[closest_index]
    except Exception:
        # Fallback: pick the middle element
        median_inside_view_analysis = inside_views[len(inside_views) // 2]

    # Summarize the median analysis using gpt-5-nano with a reusable prompt
    summary_prompt_template = read_prompt("analysis_summary_bullets.txt")
    title = question_details.get("title", "")
    question_type = question_details.get("type", "binary")
    resolution_criteria = question_details.get("resolution_criteria", "")
    fine_print = question_details.get("fine_print", "")

    summary_prompt = summary_prompt_template.format(
        title=title,
        question_type=question_type,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        analysis=median_inside_view_analysis,
    )

    summary_comment = None
    try:
        summary_comment = await call_llm(summary_prompt, "openai/gpt-5-nano", 0.3, "medium")
    except Exception as e:
        print(f"Summary generation failed: {e}. Falling back to full run comments.")

    if isinstance(summary_comment, str):
        final_comment = summary_comment.strip()
    else:
        # Fallback to the original detailed comments if summarization fails
        final_comment = "\n\n".join(final_comment_sections)

    return median_probability, final_comment


async def get_numeric_prediction(
    question_details: dict, num_runs: int
) -> Tuple[List[float], str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
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

    historical_context = await prepare_outside_view_context(question_details)
    # Prepare inside view context once
    pre_news_ctx, pre_exa_ctx = await prepare_inside_view_context(question_details)

    async def get_numeric_cdf_and_comment() -> Tuple[List[float], str]:
        attempts = 0
        last_err: Exception | None = None
        while attempts < 3:
            attempts += 1
            try:
                # Generate outside view per run using the precomputed historical context
                print(f"Generating outside view (attempt {attempts})")
                outside_view_text = await generate_outside_view(question_details, historical_context)
                try:
                    print(f"Outside view (attempt {attempts}): \"...{outside_view_text[-200:]}\"")
                except Exception:
                    pass
                print(f"Generating inside view (attempt {attempts})")
                rationale = await generate_inside_view_numeric(
                    question_details,
                    outside_view_text,
                    units=unit_of_measure,
                    lower_bound_message=lower_bound_message,
                    upper_bound_message=upper_bound_message,
                    hint="",
                    precomputed_news_context=pre_news_ctx,
                    precomputed_exa_context=pre_exa_ctx,
                )
                try:
                    print(f"Inside view (attempt {attempts}): \"...{rationale[-200:]}\"")
                except Exception:
                    pass
                percentile_values = extract_percentiles_from_response(rationale)

                comment = (
                    f"## Outside View\n{outside_view_text}\n\n"
                    f"## Inside View Analysis (Numeric)\n{rationale}\n\n"
                    f"Extracted Percentile_values: {percentile_values}\n\n"
                )

                cdf = generate_continuous_cdf(
                    percentile_values,
                    question_type,
                    open_upper_bound,
                    open_lower_bound,
                    upper_bound,
                    lower_bound,
                    zero_point,
                    cdf_size,
                )

                return cdf, comment
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.5 * attempts)
        raise last_err if last_err is not None else RuntimeError("Unknown error in numeric inside-view run")

    cdf_results = await asyncio.gather(
        *[get_numeric_cdf_and_comment() for _ in range(num_runs)], return_exceptions=True
    )
    cdf_and_comment_pairs = [r for r in cdf_results if not isinstance(r, Exception)]  # type: ignore[list-item]
    if len(cdf_and_comment_pairs) == 0:
        first_err = next((r for r in cdf_results if isinstance(r, Exception)), RuntimeError("All numeric runs failed"))
        raise first_err  # type: ignore[misc]
    comments = [pair[1] for pair in cdf_and_comment_pairs]
    final_comment_sections = [f"### Run {i+1}\n{comment}" for i, comment in enumerate(comments)]
    cdfs: List[List[float]] = [pair[0] for pair in cdf_and_comment_pairs]
    all_cdfs = np.array(cdfs)
    median_cdf: List[float] = np.median(all_cdfs, axis=0).tolist()

    # Ensure median CDF also satisfies API monotonicity constraints
    median_cdf = enforce_cdf_monotonicity(
        median_cdf,
        open_upper_bound=open_upper_bound,
        open_lower_bound=open_lower_bound,
        min_delta=5e-05,
    )

    # Choose the rationale whose CDF is closest (L2) to the median CDF
    try:
        diffs = [float(np.linalg.norm(np.array(c) - np.array(median_cdf))) for c in cdfs]
        closest_index = int(np.argmin(diffs))
        median_rationale = comments[closest_index]
    except Exception:
        median_rationale = comments[len(comments) // 2]

    # Summarize with the shared prompt
    summary_prompt_template = read_prompt("analysis_summary_bullets.txt")
    title = question_details.get("title", "")
    question_type = question_details.get("type", "numeric")
    resolution_criteria = question_details.get("resolution_criteria", "")
    fine_print = question_details.get("fine_print", "")
    summary_prompt = summary_prompt_template.format(
        title=title,
        question_type=question_type,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        analysis=median_rationale,
    )

    summary_comment = None
    try:
        summary_comment = await call_llm(summary_prompt, "openai/gpt-5-nano", 0.3, "medium")
    except Exception as e:
        print(f"Summary generation failed (numeric): {e}. Falling back to detailed comments.")

    if isinstance(summary_comment, str) and summary_comment.strip():
        final_comment = summary_comment.strip()
    else:
        final_comment = f"# Median CDF length: {len(median_cdf)}\n\n" + "\n\n".join(final_comment_sections)

    return median_cdf, final_comment


async def get_multiple_choice_prediction(
    question_details: dict,
    num_runs: int,
) -> Tuple[Dict[str, float], str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    options = question_details["options"]

    historical_context = await prepare_outside_view_context(question_details)
    # Prepare inside view context once
    pre_news_ctx, pre_exa_ctx = await prepare_inside_view_context(question_details)

    async def ask_inside_view_mc() -> Tuple[Dict[str, float], str]:
        attempts = 0
        last_err: Exception | None = None
        while attempts < 3:
            attempts += 1
            try:
                # Generate outside view per run using the precomputed historical context
                print(f"Generating outside view (attempt {attempts})")
                outside_view_text = await generate_outside_view(question_details, historical_context)
                try:
                    print(f"Outside view (attempt {attempts}): \"...{outside_view_text[-200:]}\"")
                except Exception:
                    pass
                print(f"Generating inside view (attempt {attempts})")
                rationale = await generate_inside_view_multiple_choice(
                    question_details,
                    outside_view_text,
                    precomputed_news_context=pre_news_ctx,
                    precomputed_exa_context=pre_exa_ctx,
                )
                try:
                    print(f"Inside view (attempt {attempts}): \"...{rationale[-200:]}\"")
                except Exception:
                    pass

                option_probabilities = extract_option_probabilities_from_response(
                    rationale, options
                )

                comment = (
                    f"## Outside View\n{outside_view_text}\n\n"
                    f"## Inside View Analysis (Multiple Choice)\n{rationale}\n\n"
                    f"EXTRACTED_PROBABILITIES: {option_probabilities}\n\n"
                )

                probability_yes_per_category = generate_multiple_choice_forecast(
                    options, option_probabilities
                )
                return probability_yes_per_category, comment
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.5 * attempts)
        raise last_err if last_err is not None else RuntimeError("Unknown error in MC inside-view run")

    mc_results = await asyncio.gather(
        *[ask_inside_view_mc() for _ in range(num_runs)], return_exceptions=True
    )
    probability_yes_per_category_and_comment_pairs = [r for r in mc_results if not isinstance(r, Exception)]  # type: ignore[list-item]
    if len(probability_yes_per_category_and_comment_pairs) == 0:
        first_err = next((r for r in mc_results if isinstance(r, Exception)), RuntimeError("All MC runs failed"))
        raise first_err  # type: ignore[misc]
    comments = [pair[1] for pair in probability_yes_per_category_and_comment_pairs]
    final_comment_sections = [f"### Run {i+1}\n{comment}" for i, comment in enumerate(comments)]
    probability_yes_per_category_dicts: List[Dict[str, float]] = [
        pair[0] for pair in probability_yes_per_category_and_comment_pairs
    ]

    # --- Trimmed Linear Opinion Pool ---
    # Build matrix of shape (num_runs, num_options) in the provided option order
    try:
        run_matrix = np.array(
            [[float(run_probs.get(opt, 0.0)) for opt in options] for run_probs in probability_yes_per_category_dicts],
            dtype=float,
        )
    except Exception:
        # Fallback: if something unexpected happens, default to simple mean over available dicts
        run_matrix = np.array(
            [[float(run_probs[opt]) for opt in options] for run_probs in probability_yes_per_category_dicts],
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

    trimmed_probability_yes_per_category: Dict[str, float] = {
        opt: float(p) for opt, p in zip(options, pooled_vector)
    }

    trimmed_note = f"(kept {keep_count}/{n_runs} runs; trimmed {n_runs - keep_count})"

    # Select rationale closest to pooled vector
    try:
        dists_to_pooled = [float(np.linalg.norm(run_matrix[i] - pooled_vector)) for i in range(n_runs)]
        closest_index = int(np.argmin(dists_to_pooled))
        pooled_rationale = comments[closest_index]
    except Exception:
        pooled_rationale = comments[len(comments) // 2]

    # Summarize with shared prompt
    summary_prompt_template = read_prompt("analysis_summary_bullets.txt")
    title = question_details.get("title", "")
    question_type = question_details.get("type", "multiple_choice")
    resolution_criteria = question_details.get("resolution_criteria", "")
    fine_print = question_details.get("fine_print", "")
    summary_prompt = summary_prompt_template.format(
        title=title,
        question_type=question_type,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        analysis=pooled_rationale,
    )

    summary_comment = None
    try:
        summary_comment = await call_llm(summary_prompt, "openai/gpt-5-nano", 0.3, "medium")
    except Exception as e:
        print(f"Summary generation failed (multiple choice): {e}. Falling back to detailed comments.")

    if isinstance(summary_comment, str) and summary_comment.strip():
        final_comment = summary_comment.strip()
    else:
        final_comment = (
            f"Trimmed-mean Probability Yes Per Category {trimmed_note}: `{trimmed_probability_yes_per_category}`\n\n"
            + "\n\n".join(final_comment_sections)
        )

    return trimmed_probability_yes_per_category, final_comment


################### FORECASTING ###################

async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> str:
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

    if question_type == "binary":
        forecast, comment = await get_binary_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "numeric":
        forecast, comment = await get_numeric_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "discrete":
        forecast, comment = await get_numeric_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "multiple_choice":
        forecast, comment = await get_multiple_choice_prediction(
            question_details, num_runs_per_question
        )
    else:
        raise ValueError(f"Unknown question type: {question_type}")

    print(f"-----------------------------------------------\nPost {post_id} Question {question_id}:\n")
    print(f"Forecast for post {post_id} (question {question_id}):\n{forecast}")
    print(f"Comment for post {post_id} (question {question_id}):\n{comment}")

    # Log forecast event (pre-submission)
    try:
        event = {
            "run_id": RUN_ID,
            "tournament_id": TOURNAMENT_ID,
            "question_id": question_id,
            "post_id": post_id,
            "question_title": title,
            "question_type": question_type,
            "model": LLM_MODEL,
            "num_runs": num_runs_per_question,
            "forecast": forecast,
            "comment": comment,
            "submit_attempted": bool(submit_prediction),
            "submitted": False,
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
                "model": LLM_MODEL,
                "num_runs": num_runs_per_question,
                "forecast": forecast,
                "comment": comment,
                "submit_attempted": True,
                "submitted": True,
            }
            log_forecast_event(submitted_event)
        except Exception as e:
            print(f"Logging error (post-submission): {e}")

    return summary_of_forecast


async def forecast_questions(
    open_question_id_post_id: list[tuple[int, int]],
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> None:
    forecast_tasks = [
        forecast_individual_question(
            question_id,
            post_id,
            submit_prediction,
            num_runs_per_question,
            skip_previously_forecasted_questions,
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
                    "model": LLM_MODEL,
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
        raise RuntimeError(error_message)