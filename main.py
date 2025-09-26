import asyncio
import argparse

import dotenv

dotenv.load_dotenv()
import src.forecast as forecast_module
from src.forecast import forecast_questions
from src.metaculus_utils import get_open_question_ids_from_tournament


######################### CONSTANTS #########################
# Constants
SUBMIT_PREDICTION = False
NUM_RUNS_PER_QUESTION = 1
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = False

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

NUMERIC_EXAMPLE_QUESTIONS = [
    (39606, 39606),
]
MC_EXAMPLE_QUESTIONS = [
    (39997, 39997),
]
# The example questions can be used for testing your bot. (note that question and post id are not always the same)
EXAMPLE_QUESTIONS = [  # (question_id, post_id)
    (30651, 30651),
    #(578, 578),  # Human Extinction - Binary - https://www.metaculus.com/questions/578/human-extinction-by-2100/
    #(14333, 14333),  # Age of Oldest Human - Numeric - https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/
    #(22427, 22427),  # Number of New Leading AI Labs - Multiple Choice - https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/
    #(38195, 38880), # Number of US Labor Strikes Due to AI in 2029 - Discrete - https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/
]


######################### HELPER #########################

################### FORECASTING ###################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the forecasting bot")
    parser.add_argument(
        "--mode",
        choices=["fall-aib-2025", "minibench", "example_questions"],
        default="fall-aib-2025",
    )
    args = parser.parse_args()

    if args.mode == "example_questions":
        open_question_id_post_id = EXAMPLE_QUESTIONS
        forecast_module.TOURNAMENT_ID = "example_questions"
    elif args.mode == "minibench":
        open_question_id_post_id = get_open_question_ids_from_tournament(CURRENT_MINIBENCH_ID)
        forecast_module.TOURNAMENT_ID = CURRENT_MINIBENCH_ID
    elif args.mode == "fall-aib-2025":
        open_question_id_post_id = get_open_question_ids_from_tournament(FALL_2025_AI_BENCHMARKING_ID)
        forecast_module.TOURNAMENT_ID = FALL_2025_AI_BENCHMARKING_ID
    
    asyncio.run(
        forecast_questions(
            open_question_id_post_id,
            SUBMIT_PREDICTION,
            NUM_RUNS_PER_QUESTION,
            SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
        )
    )
