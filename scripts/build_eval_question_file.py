#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
dotenv.load_dotenv(REPO_ROOT / ".env")

from src.eval.eval_question_file_loader import EVAL_QUESTION_FILE_SCHEMA_VERSION
from src.eval.manual_context_loader import load_manual_context_file
from src.eval.metaculus_resolution_loader import load_resolution_records
from src.eval.timebox import to_iso_z


class BuildEvalQuestionFileError(RuntimeError):
    """Raised when eval question file generation fails."""


def build_eval_question_file(
    *,
    post_ids: list[int],
    context_file: str,
    output_file: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not post_ids:
        raise BuildEvalQuestionFileError("post_ids cannot be empty")

    out_path = Path(output_file)
    if out_path.exists() and not overwrite:
        raise BuildEvalQuestionFileError(
            f"Output file already exists: {out_path}. Use --overwrite to replace it."
        )

    manual_context_by_post_id = load_manual_context_file(
        context_file,
        expected_post_ids=post_ids,
    )
    resolution_by_post_id = load_resolution_records(post_ids)

    questions: list[dict[str, Any]] = []
    for post_id in post_ids:
        context = manual_context_by_post_id.get(post_id)
        if context is None:
            raise BuildEvalQuestionFileError(f"Missing manual context for post_id={post_id}")

        resolution = resolution_by_post_id.get(post_id)
        if resolution is None:
            raise BuildEvalQuestionFileError(f"Missing resolution payload for post_id={post_id}")

        questions.append(
            {
                "post_id": context.post_id,
                "question_id": resolution.question_id,
                "title": context.title,
                "type": context.type,
                "description": context.description,
                "resolution_criteria": context.resolution_criteria,
                "fine_print": context.fine_print,
                "label": resolution.label,
                "resolution_raw": resolution.resolution_raw,
                "status": resolution.status,
                "open_time": to_iso_z(resolution.open_time),
                "actual_resolve_time": (
                    to_iso_z(resolution.actual_resolve_time)
                    if resolution.actual_resolve_time is not None
                    else None
                ),
                "metadata": context.metadata,
            }
        )

    payload = {
        "schema_version": EVAL_QUESTION_FILE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "builder": "scripts/build_eval_question_file.py",
            "post_ids": post_ids,
            "context_file": context_file,
        },
        "questions": questions,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical eval_question_file from post IDs + context")
    parser.add_argument(
        "--post-ids",
        nargs="+",
        type=int,
        required=True,
        help="Resolved Metaculus post IDs to include",
    )
    parser.add_argument(
        "--context-file",
        type=str,
        required=True,
        help="Path to strict manual context JSON file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output path for canonical eval question file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Allow replacing an existing output file",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_eval_question_file(
        post_ids=args.post_ids,
        context_file=args.context_file,
        output_file=args.output_file,
        overwrite=args.overwrite,
    )
    print(f"Wrote eval question file: {args.output_file}")
    print(f"Schema version: {payload['schema_version']}")
    print(f"Question count: {len(payload.get('questions', []))}")


if __name__ == "__main__":
    main()
