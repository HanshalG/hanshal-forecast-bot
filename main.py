import asyncio
import argparse

import dotenv

dotenv.load_dotenv()
import src.forecast as forecast_module
from src.forecast import forecast_questions
from src.metaculus_utils import get_open_question_ids_from_tournament

NUMERIC_EXAMPLE_QUESTIONS = [
    (39606, 39606),
]
MC_EXAMPLE_QUESTIONS = [
    (39997, 39997),
]
# The example questions can be used for testing your bot. (note that question and post id are not always the same)
EXAMPLE_QUESTIONS = [  # (question_id, post_id)
    (30651, 30651),
]

################### FORECASTING ###################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the forecasting bot")
    parser.add_argument(
        "--mode",
        choices=["fall-aib-2025", "minibench", "example_questions"],
        default="fall-aib-2025",
    )
    parser.add_argument(
        "--submit",
        "--submit-prediction",
        dest="submit_prediction",
        action="store_true",
        default=False,
        help="Submit forecasts and comments to Metaculus (default: False)",
    )
    parser.add_argument(
        "--num-runs",
        "--num-runs-per-question",
        dest="num_runs_per_question",
        type=int,
        default=1,
        help="Number of inside-view runs per question (default: 1)",
    )
    parser.add_argument(
        "--skip-prev",
        "--skip-previously-forecasted-questions",
        dest="skip_previously_forecasted_questions",
        action="store_true",
        default=False,
        help="Skip questions where a forecast already exists (default: False)",
    )
    args = parser.parse_args()

    if args.mode == "example_questions":
        open_question_id_post_id = EXAMPLE_QUESTIONS
        forecast_module.TOURNAMENT_ID = "example_questions"
    elif args.mode == "minibench":
        open_question_id_post_id = get_open_question_ids_from_tournament("minibench")
        forecast_module.TOURNAMENT_ID = "minibench"
    elif args.mode == "fall-aib-2025":
        open_question_id_post_id = get_open_question_ids_from_tournament("fall-aib-2025")
        forecast_module.TOURNAMENT_ID = "fall-aib-2025",

    asyncio.run(
        forecast_questions(
            open_question_id_post_id,
            args.submit_prediction,
            args.num_runs_per_question,
            args.skip_previously_forecasted_questions,
        )
    )
