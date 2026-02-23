from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/project"
RESULTS_VOLUME_NAME = "hanshal-forecast-eval-results"
RESULTS_MOUNT_PATH = "/cloud-results"
DEFAULT_OUTPUT_DIR = f"{RESULTS_MOUNT_PATH}/evals"
DEFAULT_EVAL_QUESTION_FILE = "eval_inputs/eval_questions_39523_39575_39476.json"
DEFAULT_STRATEGY_FILES = ["eval_inputs/strategies/nano_baseline.yaml"]

_ignore_patterns = [
    ".git",
    ".git/**",
    ".venv",
    ".venv/**",
    "__pycache__",
    "**/__pycache__/**",
    ".pytest_cache",
    ".pytest_cache/**",
    "logs",
    "logs/**",
    ".DS_Store",
    "*.pyc",
    ".env",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .poetry_install_from_file(
        poetry_pyproject_toml=str(PROJECT_ROOT / "pyproject.toml"),
        poetry_lockfile=str(PROJECT_ROOT / "poetry.lock"),
    )
    .add_local_dir(
        local_path=str(PROJECT_ROOT),
        remote_path=REMOTE_ROOT,
        ignore=_ignore_patterns,
    )
)

dotenv_path = PROJECT_ROOT / ".env"
env_secret = (
    modal.Secret.from_dotenv(str(dotenv_path))
    if dotenv_path.exists()
    else modal.Secret.from_dict({})
)

app = modal.App("hanshal-forecast-eval")
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


@app.function(
    image=image,
    secrets=[env_secret],
    volumes={RESULTS_MOUNT_PATH: results_volume},
    timeout=60 * 60,
    cpu=2,
)
def run_eval_cloud(
    eval_question_file: str = DEFAULT_EVAL_QUESTION_FILE,
    strategy_files: list[str] = DEFAULT_STRATEGY_FILES,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    question_concurrency: int = 1,
) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    os.environ["PYTHONPATH"] = REMOTE_ROOT
    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)

    from src.eval.runner import run_eval

    result = __import__("asyncio").run(
        run_eval(
            eval_question_file=eval_question_file,
            strategy_files=strategy_files,
            output_dir=output_dir,
            question_concurrency=question_concurrency,
        )
    )

    summaries = _serialize(result.get("summaries", []))
    report_paths = _serialize(result.get("report_paths", {}))
    run_id = str(result.get("run_id"))
    output_dir_str = str(result.get("output_dir"))
    metadata_path = Path(output_dir_str) / "modal_result.json"
    metadata_payload = {
        "run_id": run_id,
        "output_dir": output_dir_str,
        "prediction_count": len(result.get("predictions", [])),
        "summary_count": len(summaries),
        "summaries": summaries,
        "report_paths": report_paths,
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    results_volume.commit()

    return {
        "run_id": run_id,
        "output_dir": output_dir_str,
        "prediction_count": len(result.get("predictions", [])),
        "summary_count": len(summaries),
        "summaries": summaries,
        "report_paths": report_paths,
        "metadata_path": str(metadata_path),
        "results_volume": RESULTS_VOLUME_NAME,
    }


@app.local_entrypoint()
def main(
    eval_question_file: str = DEFAULT_EVAL_QUESTION_FILE,
    strategy_files: str = ",".join(DEFAULT_STRATEGY_FILES),
    output_dir: str = DEFAULT_OUTPUT_DIR,
    question_concurrency: int = 1,
) -> None:
    strategy_file_list = [part.strip() for part in strategy_files.split(",") if part.strip()]
    result = run_eval_cloud.remote(
        eval_question_file=eval_question_file,
        strategy_files=strategy_file_list,
        output_dir=output_dir,
        question_concurrency=question_concurrency,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
