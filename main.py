import asyncio
import datetime
import os
from pickle import FALSE
import re
from pathlib import Path

import time

import dotenv

dotenv.load_dotenv()

import numpy as np
from asknews_sdk import AskNewsSDK
from openai import AsyncOpenAI
import uuid
from src.forecast_logger import log_forecast_event
from src.metaculus_utils import (
    forecast_is_already_made,
    post_question_comment,
    post_question_prediction,
    create_forecast_payload,
    list_posts_from_tournament,
    get_open_question_ids_from_tournament,
    get_post_details,
    get_metaculus_community_prediction,
    forecast_is_already_made
)
from src.utils import (
    read_prompt,
    call_llm,
    call_asknews,
    extract_probability_from_response_as_percentage_not_decimal,
    extract_percentiles_from_response,
    generate_continuous_cdf,
    extract_option_probabilities_from_response,
    PROMPTS_DIR,
    run_research,
)


######################### CONSTANTS #########################
# Constants
SUBMIT_PREDICTION = True  # set to True to publish your predictions to Metaculus
USE_EXAMPLE_QUESTIONS = False  # set to True to forecast example questions rather than the tournament questions
NUM_RUNS_PER_QUESTION = 5  # The median forecast is taken between NUM_RUNS_PER_QUESTION runs
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = True

LLM_MODEL = "openai/gpt-5-mini"

# A unique identifier for this process run to correlate events
RUN_ID = os.getenv("RUN_ID") or f"run-{uuid.uuid4()}"

# Environment variables
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# The tournament IDs below can be used for testing your bot.
Q4_2024_AI_BENCHMARKING_ID = 32506
Q1_2025_AI_BENCHMARKING_ID = 32627
FALL_2025_AI_BENCHMARKING_ID = "fall-aib-2025"
CURRENT_MINIBENCH_ID = "minibench"

Q4_2024_QUARTERLY_CUP_ID = 3672
Q1_2025_QUARTERLY_CUP_ID = 32630
CURRENT_METACULUS_CUP_ID = "metaculus-cup"

AXC_2025_TOURNAMENT_ID = 32564
AI_2027_TOURNAMENT_ID = "ai-2027"

TOURNAMENT_ID = FALL_2025_AI_BENCHMARKING_ID

# The example questions can be used for testing your bot. (note that question and post id are not always the same)
EXAMPLE_QUESTIONS = [  # (question_id, post_id)
    (39724, 39724),
    #(578, 578),  # Human Extinction - Binary - https://www.metaculus.com/questions/578/human-extinction-by-2100/
    #(14333, 14333),  # Age of Oldest Human - Numeric - https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/
    #(22427, 22427),  # Number of New Leading AI Labs - Multiple Choice - https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/
    #(38195, 38880), # Number of US Labor Strikes Due to AI in 2029 - Discrete - https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/
]


######################### HELPER FUNCTIONS #########################

# @title Helper functions
AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api"




CONCURRENT_REQUESTS_LIMIT = 5
llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)

# Removed external research providers (Perplexity/Exa); only AskNews is supported now.


############### BINARY ###############
# @title Binary prompt & functions

# This section includes functionality for binary questions.

# Load prompt templates from files
BINARY_PROMPT_TEMPLATE = read_prompt("binary_question_prompt.txt")




async def get_binary_gpt_prediction(
    question_details: dict, num_runs: int, community_prediction: float | None = None
) -> tuple[float, str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        # Prefer IANA tz name if available
        timezone_str = datetime.datetime.now().astimezone().tzinfo.key  # type: ignore[attr-defined]
    except Exception:
        timezone_str = str(datetime.datetime.now().astimezone().tzinfo)
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]

    summary_report = run_research(title)

    content = BINARY_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        timezone=timezone_str,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        community_prediction=(
            f"{community_prediction:.2f}" if community_prediction is not None else "N/A"
        ),
    )

    async def get_rationale_and_probability(content: str) -> tuple[float, str]:
        rationale = await call_llm(content, LLM_MODEL, 0.3)

        probability = extract_probability_from_response_as_percentage_not_decimal(
            rationale
        )
        comment = (
            f"Extracted Probability: {probability}%\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
        )
        return probability, comment

    probability_and_comment_pairs = await asyncio.gather(
        *[get_rationale_and_probability(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probabilities = [pair[0] for pair in probability_and_comment_pairs]
    median_probability = float(np.median(probabilities)) / 100

    final_comment = f"Median Probability: {median_probability}\n\n" + "\n\n".join(
        final_comment_sections
    )
    return median_probability, final_comment


####################### NUMERIC ###############
# @title Numeric prompt & functions

NUMERIC_PROMPT_TEMPLATE = read_prompt("numeric_question_prompt.txt")






async def get_numeric_gpt_prediction(
    question_details: dict, num_runs: int
) -> tuple[list[float], str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]
    scaling = question_details["scaling"]
    open_upper_bound = question_details["open_upper_bound"]
    open_lower_bound = question_details["open_lower_bound"]
    unit_of_measure = question_details["unit"] if question_details["unit"] else "Not stated (please infer this)"
    upper_bound = scaling["range_max"]
    lower_bound = scaling["range_min"]
    zero_point = scaling["zero_point"]
    if question_type == "discrete":
        outcome_count = question_details["scaling"]["inbound_outcome_count"]
        cdf_size = outcome_count + 1
    else:
        cdf_size = 201

    # Create messages about the bounds that are passed in the LLM prompt
    if open_upper_bound:
        upper_bound_message = ""
    else:
        upper_bound_message = f"The outcome can not be higher than {upper_bound}."
    if open_lower_bound:
        lower_bound_message = ""
    else:
        lower_bound_message = f"The outcome can not be lower than {lower_bound}."

    summary_report = run_research(title)

    content = NUMERIC_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        lower_bound_message=lower_bound_message,
        upper_bound_message=upper_bound_message,
        units=unit_of_measure,
    )

    async def ask_llm_to_get_cdf(content: str) -> tuple[list[float], str]:
        rationale = await call_llm(content, LLM_MODEL, 0.3)
        percentile_values = extract_percentiles_from_response(rationale)

        comment = (
            f"Extracted Percentile_values: {percentile_values}\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
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
        *[ask_llm_to_get_cdf(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in cdf_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    cdfs: list[list[float]] = [pair[0] for pair in cdf_and_comment_pairs]
    all_cdfs = np.array(cdfs)
    median_cdf: list[float] = np.median(all_cdfs, axis=0).tolist()

    final_comment = f"Median CDF: `{str(median_cdf)[:100]}...`\n\n" + "\n\n".join(
        final_comment_sections
    )
    return median_cdf, final_comment


########################## MULTIPLE CHOICE ###############
# @title Multiple Choice prompt & functions

MULTIPLE_CHOICE_PROMPT_TEMPLATE = read_prompt("multiple_choice_question_prompt.txt")

# Validate that required placeholders exist in templates
def validate_prompt_template(template_name: str, template: str, required_placeholders: list[str]):
    """Validate that a prompt template contains all required placeholders."""
    missing = []
    for placeholder in required_placeholders:
        if placeholder not in template:
            missing.append(placeholder)
    if missing:
        raise ValueError(f"Missing placeholders in {template_name}: {missing}")

# Validate all templates
validate_prompt_template("binary_question_prompt.txt", BINARY_PROMPT_TEMPLATE,
                        ["{title}", "{background}", "{resolution_criteria}", "{fine_print}",
                         "{summary_report}", "{today}", "{timezone}", "{community_prediction}"])

validate_prompt_template("numeric_question_prompt.txt", NUMERIC_PROMPT_TEMPLATE,
                        ["{title}", "{background}", "{resolution_criteria}", "{fine_print}",
                         "{units}", "{summary_report}", "{today}", "{lower_bound_message}", "{upper_bound_message}"])

validate_prompt_template("multiple_choice_question_prompt.txt", MULTIPLE_CHOICE_PROMPT_TEMPLATE,
                        ["{title}", "{options}", "{background}", "{resolution_criteria}",
                         "{fine_print}", "{summary_report}", "{today}"])

print(f"Loaded prompt templates from {PROMPTS_DIR}")




def generate_multiple_choice_forecast(options, option_probabilities) -> dict:
    """
    Returns: dict corresponding to the probabilities of each option.
    """

    # confirm that there is a probability for each option
    if len(options) != len(option_probabilities):
        raise ValueError(
            f"Number of options ({len(options)}) does not match number of probabilities ({len(option_probabilities)})"
        )

    # Ensure we are using decimals
    total_sum = sum(option_probabilities)
    decimal_list = [x / total_sum for x in option_probabilities]

    def normalize_list(float_list):
        # Step 1: Clamp values
        clamped_list = [max(min(x, 0.99), 0.01) for x in float_list]

        # Step 2: Calculate the sum of all elements
        total_sum = sum(clamped_list)

        # Step 3: Normalize the list so that all elements add up to 1
        normalized_list = [x / total_sum for x in clamped_list]

        # Step 4: Adjust for any small floating-point errors
        adjustment = 1.0 - sum(normalized_list)
        normalized_list[-1] += adjustment

        return normalized_list

    normalized_option_probabilities = normalize_list(decimal_list)

    probability_yes_per_category = {}
    for i in range(len(options)):
        probability_yes_per_category[options[i]] = normalized_option_probabilities[i]

    return probability_yes_per_category


async def get_multiple_choice_gpt_prediction(
    question_details: dict,
    num_runs: int,
) -> tuple[dict[str, float], str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]
    options = question_details["options"]

    summary_report = run_research(title)

    content = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        options=options,
    )

    async def ask_llm_for_multiple_choice_probabilities(
        content: str,
    ) -> tuple[dict[str, float], str]:
        rationale = await call_llm(content, LLM_MODEL, 0.3)


        option_probabilities = extract_option_probabilities_from_response(
            rationale, options
        )

        comment = (
            f"EXTRACTED_PROBABILITIES: {option_probabilities}\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
        )

        probability_yes_per_category = generate_multiple_choice_forecast(
            options, option_probabilities
        )
        return probability_yes_per_category, comment

    probability_yes_per_category_and_comment_pairs = await asyncio.gather(
        *[ask_llm_for_multiple_choice_probabilities(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_yes_per_category_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probability_yes_per_category_dicts: list[dict[str, float]] = [
        pair[0] for pair in probability_yes_per_category_and_comment_pairs
    ]
    average_probability_yes_per_category: dict[str, float] = {}
    for option in options:
        probabilities_for_current_option: list[float] = [
            dict[option] for dict in probability_yes_per_category_dicts
        ]
        average_probability_yes_per_category[option] = sum(
            probabilities_for_current_option
        ) / len(probabilities_for_current_option)

    final_comment = (
        f"Average Probability Yes Per Category: `{average_probability_yes_per_category}`\n\n"
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

    summary_of_forecast = ""
    summary_of_forecast += f"-----------------------------------------------\nQuestion: {title}\n"
    summary_of_forecast += f"URL: https://www.metaculus.com/questions/{post_id}/\n"

    if question_type == "multiple_choice":
        options = question_details["options"]
        summary_of_forecast += f"options: {options}\n"

    if (
        forecast_is_already_made(post_details)
        and skip_previously_forecasted_questions == True
    ):
        summary_of_forecast += f"Skipped: Forecast already made\n"
        return summary_of_forecast

    if question_type == "binary":
        try:
            community_prediction = get_metaculus_community_prediction(post_id)
        except Exception:
            community_prediction = None
        
        print('-------')
        print('community ', community_prediction)

        forecast, comment = await get_binary_gpt_prediction(
            question_details, num_runs_per_question, community_prediction
        )
    elif question_type == "numeric":
        forecast, comment = await get_numeric_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "discrete":
        forecast, comment = await get_numeric_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "multiple_choice":
        forecast, comment = await get_multiple_choice_gpt_prediction(
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




######################## FINAL RUN #########################
if __name__ == "__main__":
    if USE_EXAMPLE_QUESTIONS:
        open_question_id_post_id = EXAMPLE_QUESTIONS
    else:
        open_question_id_post_id = get_open_question_ids_from_tournament(TOURNAMENT_ID)

    asyncio.run(
        forecast_questions(
            open_question_id_post_id,
            SUBMIT_PREDICTION,
            NUM_RUNS_PER_QUESTION,
            SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
        )
    )
