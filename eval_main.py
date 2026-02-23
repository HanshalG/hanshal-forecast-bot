import argparse
import asyncio

import dotenv

dotenv.load_dotenv()

from src.eval.runner import run_eval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resolved-question eval backtests")
    parser.add_argument(
        "--post-ids",
        nargs="+",
        type=int,
        required=True,
        help="Resolved Metaculus post IDs to evaluate (e.g. --post-ids 39523 39575 39476)",
    )
    parser.add_argument(
        "--context-file",
        type=str,
        required=True,
        help="Path to strict manual context JSON file",
    )
    parser.add_argument(
        "--strategy-files",
        nargs="+",
        required=True,
        help="One or more strategy YAML files (required, no defaults)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs per question inside each strategy forecast",
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
        post_ids=args.post_ids,
        context_file=args.context_file,
        strategy_files=args.strategy_files,
        num_runs=args.num_runs,
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
