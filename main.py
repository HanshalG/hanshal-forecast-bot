import asyncio
import argparse

import dotenv

dotenv.load_dotenv()
import src.forecast as forecast_module
from src.forecast import forecast_questions
from src.forecast_logger import set_supabase_logging_enabled
from src.metaculus_utils import get_open_question_ids_from_tournament

# The example questions can be used for testing your bot. (note that question and post id are not always the same)
EXAMPLE_QUESTIONS = [  # (question_id, post_id)
    (40966, 40966),
]

################### FORECASTING ###################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the forecasting bot")
    parser.add_argument(
        "--mode",
        choices=["spring-aib-2026", "minibench", "example_questions"],
        default="example_questions",
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
        help="Number of runs per question (default: 1)",
    )
    parser.add_argument(
        "--skip-prev",
        "--skip-previously-forecasted-questions",
        dest="skip_previously_forecasted_questions",
        action="store_true",
        default=False,
        help="Skip questions where a forecast already exists (default: False)",
    )

    #TO REMOVE
    parser.add_argument(
        "--max-outside-searches",
        dest="max_outside_searches",
        type=int,
        default=10,
        help="Maximum number of Exa searches for historical/outside view research (default: 10)",
    )
    parser.add_argument(
        "--max-inside-searches",
        dest="max_inside_searches",
        type=int,
        default=10,
        help="Maximum number of Exa searches for current/inside view research (default: 10)",
    )
    parser.add_argument(
        "--get-prediction-market",
        action="store_true",
        help="Check prediction markets (Polymarket) for relevant data before forecasting",
    )
    parser.add_argument(
        "--log-to-supabase",
        action="store_true",
        default=False,
        help="Enable Supabase logging for this run (default: disabled unless this flag is provided)",
    )
    #TO REMOVE

    args = parser.parse_args()
    set_supabase_logging_enabled(args.log_to_supabase)

    if args.mode == "example_questions":
        open_question_id_post_id = EXAMPLE_QUESTIONS
        forecast_module.TOURNAMENT_ID = "example_questions"
    elif args.mode == "minibench":
        open_question_id_post_id = get_open_question_ids_from_tournament("minibench")
        forecast_module.TOURNAMENT_ID = "minibench"
    elif args.mode == "spring-aib-2026":
        open_question_id_post_id = get_open_question_ids_from_tournament("spring-aib-2026")
        forecast_module.TOURNAMENT_ID = "spring-aib-2026"

    asyncio.run(
        forecast_questions(
            open_question_id_post_id,
            args.submit_prediction,
            args.num_runs_per_question,
            args.skip_previously_forecasted_questions,
            args.max_outside_searches,
            args.max_inside_searches,
            get_prediction_market=args.get_prediction_market,
        )
    )
