import asyncio
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
    extract_option_probabilities_from_response,
)
from src.metaculus_utils import (
    forecast_is_already_made,
    post_question_comment,
    post_question_prediction,
    create_forecast_payload,
    get_open_question_ids_from_tournament,
    get_post_details,
    get_metaculus_community_prediction,
    get_metaculus_community_prediction_and_count,
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
    community_prediction: float | None = None,
    post_id: int | None = None,
) -> Tuple[float, str]:
    print(f"Preparing outside view context")
    historical_context = await prepare_outside_view_context(question_details)

    print(f"Generating outside view")
    outside_view_text = await generate_outside_view(question_details, historical_context)
    print(f"Outside view text: {outside_view_text}")


    print(f"Preparing inside view context")
    pre_news_ctx, pre_exa_ctx = await prepare_inside_view_context(question_details)

    async def get_inside_probability_and_comment() -> Tuple[float, str]:
        try:
            crowd_pct, n_forecasters = (
                get_metaculus_community_prediction_and_count(post_id)
                if post_id is not None
                else (community_prediction, None)
            )
        except Exception:
            crowd_pct, n_forecasters = (None, None)

        inside_view_text = await generate_inside_view(
            question_details,
            outside_view_text,
            community_prediction_pct=crowd_pct,
            n_forecasters=(n_forecasters or 0),
            precomputed_news_context=pre_news_ctx,
            precomputed_exa_context=pre_exa_ctx,
        )

        print(f"Inside view text: {inside_view_text}")
        probability = extract_probability_from_response_as_percentage_not_decimal(
            inside_view_text
        )
        comment = (
            f"Extracted Probability: {probability}%\n\n"
            f"Inside View Analysis:\n{inside_view_text}\n\n"
        )
        return probability, comment

    probability_and_comment_pairs = await asyncio.gather(
        *[get_inside_probability_and_comment() for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_and_comment_pairs]
    final_comment_sections = [
        f"## Inside View {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probabilities = [pair[0] for pair in probability_and_comment_pairs]
    median_probability = float(np.median(probabilities)) / 100

    final_comment_preamble = (
        "# Outside View\n" + outside_view_text + "\n\n"
        f"# Median Probability (Inside View): {median_probability}\n\n"
    )
    final_comment = final_comment_preamble + "\n\n".join(final_comment_sections)
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
    outside_view_text = await generate_outside_view(question_details, historical_context)
    pre_news_ctx, pre_exa_ctx = await prepare_inside_view_context(question_details)

    async def get_numeric_cdf_and_comment() -> Tuple[List[float], str]:
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
        percentile_values = extract_percentiles_from_response(rationale)

        comment = (
            f"Extracted Percentile_values: {percentile_values}\n\nInside View Analysis (Numeric): "
            f"{rationale}\n\n"
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

    cdf_and_comment_pairs = await asyncio.gather(
        *[get_numeric_cdf_and_comment() for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in cdf_and_comment_pairs]
    final_comment_sections = [
        f"## Inside View {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    cdfs: List[List[float]] = [pair[0] for pair in cdf_and_comment_pairs]
    all_cdfs = np.array(cdfs)
    median_cdf: List[float] = np.median(all_cdfs, axis=0).tolist()

    final_comment_preamble = (
        "# Outside View\n" + outside_view_text + "\n\n"
        f"# Median CDF length: {len(median_cdf)}\n\n"
    )
    final_comment = final_comment_preamble + "\n\n".join(final_comment_sections)
    return median_cdf, final_comment


async def get_multiple_choice_prediction(
    question_details: dict,
    num_runs: int,
) -> Tuple[Dict[str, float], str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    options = question_details["options"]

    historical_context = await prepare_outside_view_context(question_details)
    outside_view_text = await generate_outside_view(question_details, historical_context)
    pre_news_ctx, pre_exa_ctx = await prepare_inside_view_context(question_details)

    async def ask_inside_view_mc() -> Tuple[Dict[str, float], str]:
        rationale = await generate_inside_view_multiple_choice(
            question_details,
            outside_view_text,
            precomputed_news_context=pre_news_ctx,
            precomputed_exa_context=pre_exa_ctx,
        )

        option_probabilities = extract_option_probabilities_from_response(
            rationale, options
        )

        comment = (
            f"EXTRACTED_PROBABILITIES: {option_probabilities}\n\nInside View Analysis (Multiple Choice): "
            f"{rationale}\n\n"
        )

        probability_yes_per_category = generate_multiple_choice_forecast(
            options, option_probabilities
        )
        return probability_yes_per_category, comment

    probability_yes_per_category_and_comment_pairs = await asyncio.gather(
        *[ask_inside_view_mc() for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_yes_per_category_and_comment_pairs]
    final_comment_sections = [
        f"## Inside View {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probability_yes_per_category_dicts: List[Dict[str, float]] = [
        pair[0] for pair in probability_yes_per_category_and_comment_pairs
    ]
    average_probability_yes_per_category: Dict[str, float] = {}
    for option in options:
        probabilities_for_current_option: List[float] = [
            d[option] for d in probability_yes_per_category_dicts
        ]
        average_probability_yes_per_category[option] = sum(
            probabilities_for_current_option
        ) / len(probabilities_for_current_option)

    final_comment_preamble = (
        "# Outside View\n" + outside_view_text + "\n\n"
    )
    final_comment = (
        final_comment_preamble
        + f"Average Probability Yes Per Category: `{average_probability_yes_per_category}`\n\n"
        + "\n\n".join(final_comment_sections)
    )
    return average_probability_yes_per_category, final_comment


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
        try:
            community_prediction = get_metaculus_community_prediction(post_id)
        except Exception:
            community_prediction = None

        forecast, comment = await get_binary_prediction(
            question_details, num_runs_per_question, community_prediction, post_id
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
            "community_prediction": community_prediction if question_type == "binary" else None,
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
                "community_prediction": community_prediction if question_type == "binary" else None,
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
            print(
                f"-----------------------------------------------\nPost {post_id} Question {question_id}:\nError: {forecast_summary.__class__.__name__} {forecast_summary}\nURL: https://www.metaculus.com/questions/{post_id}/\n"
            )
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
                    "community_prediction": None,
                    "forecast": None,
                    "comment": None,
                    "submit_attempted": bool(submit_prediction),
                    "submitted": False,
                    "error": f"{forecast_summary.__class__.__name__}: {forecast_summary}",
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