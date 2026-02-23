import argparse
import asyncio

import dotenv

dotenv.load_dotenv()

from src.eval.runner import run_eval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resolved-question eval backtests")
    parser.add_argument(
        "--eval-question-file",
        required=True,
        help="Path to canonical eval question JSON file",
    )
    parser.add_argument(
        "--strategy-files",
        nargs="+",
        required=True,
        help="One or more strategy YAML files (required, no defaults)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/evals",
        help="Base output directory for eval artifacts",
    )
    parser.add_argument(
        "--question-concurrency",
        type=int,
        default=1,
        help="How many questions to run in parallel within each strategy",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()

    result = await run_eval(
        eval_question_file=args.eval_question_file,
        strategy_files=args.strategy_files,
        output_dir=args.output_dir,
        question_concurrency=args.question_concurrency,
    )

    print("Eval completed")
    print(f"Run ID: {result['run_id']}")
    print(f"Output directory: {result['output_dir']}")
    print("Artifacts:")
    for key, value in result["report_paths"].items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    asyncio.run(_main())
