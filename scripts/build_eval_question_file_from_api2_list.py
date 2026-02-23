#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import dotenv
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
dotenv.load_dotenv(REPO_ROOT / ".env")

from src.eval.eval_question_file_loader import EVAL_QUESTION_FILE_SCHEMA_VERSION
from src.eval.timebox import parse_datetime, to_iso_z

API2_BASE_URL = "https://www.metaculus.com/api2/questions/"
DEFAULT_FINE_PRINT_PLACEHOLDER = (
    "No additional fine print was provided by Metaculus for this question."
)


class BuildEvalQuestionFileFromApi2Error(RuntimeError):
    """Raised when eval question file generation from /api2/questions/ fails."""


def _normalize_resolution_label(raw_resolution: Any, *, post_id: int) -> tuple[str, int]:
    value = str(raw_resolution or "").strip().lower()
    if value == "yes":
        return value, 1
    if value == "no":
        return value, 0
    raise BuildEvalQuestionFileFromApi2Error(
        f"Post {post_id}: unsupported resolution '{raw_resolution}'. Expected yes/no."
    )


def _parse_required_iso_datetime(value: Any, *, field: str, post_id: int) -> str:
    dt = parse_datetime(value)
    if dt is None:
        raise BuildEvalQuestionFileFromApi2Error(
            f"Post {post_id}: missing or invalid datetime field '{field}'."
        )
    return to_iso_z(dt)


def _parse_optional_iso_datetime(value: Any, *, field: str, post_id: int) -> str | None:
    if value in (None, ""):
        return None
    dt = parse_datetime(value)
    if dt is None:
        raise BuildEvalQuestionFileFromApi2Error(
            f"Post {post_id}: invalid datetime field '{field}'."
        )
    return to_iso_z(dt)


def _non_empty_str(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _extract_context_field(row: dict[str, Any], field: str) -> str:
    question = row.get("question")
    if isinstance(question, dict):
        value = _non_empty_str(question.get(field))
        if value:
            return value
    return _non_empty_str(row.get(field))


def _extract_next_offset(next_url: str | None, *, current_offset: int, page_limit: int) -> int | None:
    if not next_url:
        return None
    parsed = urlparse(next_url)
    offset_values = parse_qs(parsed.query).get("offset")
    if offset_values:
        try:
            return int(offset_values[0])
        except Exception:
            return current_offset + page_limit
    return current_offset + page_limit


def _build_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "hanshal-forecast-bot/eval-api2-list",
    }
    token = _non_empty_str(os.getenv("METACULUS_TOKEN"))
    if token:
        headers["Authorization"] = f"Token {token}"
    return headers


def _fetch_page(
    *,
    offset: int,
    page_limit: int,
    timeout_s: float,
    retries: int,
    headers: dict[str, str],
) -> dict[str, Any]:
    url = f"{API2_BASE_URL}?limit={int(page_limit)}&offset={int(offset)}"

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout_s)
            if response.status_code == 429 and attempt < retries:
                time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise BuildEvalQuestionFileFromApi2Error(
                    f"/api2/questions/ response at offset={offset} must be an object."
                )
            return payload
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
                continue
            break

    raise BuildEvalQuestionFileFromApi2Error(
        f"Failed to fetch /api2/questions/?limit={page_limit}&offset={offset}: {last_err}"
    )


def _load_rows_for_post_ids(
    *,
    post_ids: list[int],
    page_limit: int,
    timeout_s: float,
    retries: int,
    max_pages: int | None,
) -> dict[int, dict[str, Any]]:
    targets = set(post_ids)
    found: dict[int, dict[str, Any]] = {}
    headers = _build_headers()

    offset = 0
    pages_loaded = 0
    while True:
        if max_pages is not None and pages_loaded >= max_pages:
            break

        payload = _fetch_page(
            offset=offset,
            page_limit=page_limit,
            timeout_s=timeout_s,
            retries=retries,
            headers=headers,
        )
        pages_loaded += 1

        rows = payload.get("results")
        if not isinstance(rows, list):
            raise BuildEvalQuestionFileFromApi2Error(
                f"/api2/questions/ page offset={offset} missing 'results' list."
            )

        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                row_post_id = int(row.get("id"))
            except Exception:
                continue
            if row_post_id in targets and row_post_id not in found:
                found[row_post_id] = row

        if len(found) == len(targets):
            break

        next_url = payload.get("next")
        next_offset = _extract_next_offset(next_url, current_offset=offset, page_limit=page_limit)
        if next_offset is None or next_offset == offset:
            break
        offset = next_offset

    missing = sorted(targets.difference(found.keys()))
    if missing:
        raise BuildEvalQuestionFileFromApi2Error(
            f"Could not find requested post IDs via /api2/questions/ pagination: {missing} "
            f"(scanned pages={pages_loaded}, page_limit={page_limit})."
        )

    return found


def _build_question_row(*, post_id: int, row: dict[str, Any]) -> dict[str, Any]:
    question = row.get("question")
    if not isinstance(question, dict):
        raise BuildEvalQuestionFileFromApi2Error(f"Post {post_id}: missing 'question' object in list row.")

    status = _non_empty_str(question.get("status")).lower()
    if status != "resolved":
        raise BuildEvalQuestionFileFromApi2Error(
            f"Post {post_id}: expected status='resolved', got '{question.get('status')}'."
        )

    qtype = _non_empty_str(question.get("type")).lower()
    if qtype != "binary":
        raise BuildEvalQuestionFileFromApi2Error(
            f"Post {post_id}: only binary questions are supported in eval v1; got '{question.get('type')}'."
        )

    try:
        question_id = int(question.get("id"))
    except Exception as exc:
        raise BuildEvalQuestionFileFromApi2Error(
            f"Post {post_id}: invalid or missing question.id in list row."
        ) from exc

    title = _non_empty_str(row.get("title")) or _non_empty_str(question.get("title"))
    if not title:
        raise BuildEvalQuestionFileFromApi2Error(f"Post {post_id}: missing title.")

    description = _extract_context_field(row, "description")
    resolution_criteria = _extract_context_field(row, "resolution_criteria")
    fine_print = _extract_context_field(row, "fine_print")
    missing_context_fields = []
    if not description:
        missing_context_fields.append("description")
    if not resolution_criteria:
        missing_context_fields.append("resolution_criteria")
    if missing_context_fields:
        raise BuildEvalQuestionFileFromApi2Error(
            f"Post {post_id}: missing required context fields from /api2/questions/ list row: "
            f"{', '.join(missing_context_fields)}."
        )
    fine_print_was_missing = False
    if not fine_print:
        fine_print = DEFAULT_FINE_PRINT_PLACEHOLDER
        fine_print_was_missing = True

    resolution_raw, label = _normalize_resolution_label(question.get("resolution"), post_id=post_id)
    open_time = _parse_required_iso_datetime(
        question.get("open_time") or row.get("open_time"),
        field="open_time",
        post_id=post_id,
    )
    actual_resolve_time = _parse_optional_iso_datetime(
        question.get("actual_resolve_time") or row.get("actual_resolve_time"),
        field="actual_resolve_time",
        post_id=post_id,
    )

    metadata: dict[str, Any] = {
        "source": "api2_questions_pagination",
        "row_url": f"https://www.metaculus.com/questions/{post_id}/",
    }
    if fine_print_was_missing:
        metadata["context_warnings"] = ["fine_print_missing_replaced_with_placeholder"]

    return {
        "post_id": post_id,
        "question_id": question_id,
        "title": title,
        "type": qtype,
        "description": description,
        "resolution_criteria": resolution_criteria,
        "fine_print": fine_print,
        "label": label,
        "resolution_raw": resolution_raw,
        "status": status,
        "open_time": open_time,
        "actual_resolve_time": actual_resolve_time,
        "metadata": metadata,
    }


def build_eval_question_file_from_api2_list(
    *,
    post_ids: list[int],
    output_file: str,
    overwrite: bool = False,
    page_limit: int = 100,
    timeout_s: float = 20.0,
    retries: int = 5,
    max_pages: int | None = None,
) -> dict[str, Any]:
    if not post_ids:
        raise BuildEvalQuestionFileFromApi2Error("post_ids cannot be empty.")
    if len(set(post_ids)) != len(post_ids):
        raise BuildEvalQuestionFileFromApi2Error("post_ids contains duplicates.")
    if page_limit <= 0:
        raise BuildEvalQuestionFileFromApi2Error("page_limit must be > 0.")
    if retries <= 0:
        raise BuildEvalQuestionFileFromApi2Error("retries must be > 0.")
    if timeout_s <= 0:
        raise BuildEvalQuestionFileFromApi2Error("timeout_s must be > 0.")
    if max_pages is not None and max_pages <= 0:
        raise BuildEvalQuestionFileFromApi2Error("max_pages must be > 0 when provided.")

    out_path = Path(output_file)
    if out_path.exists() and not overwrite:
        raise BuildEvalQuestionFileFromApi2Error(
            f"Output file already exists: {out_path}. Use --overwrite to replace it."
        )

    rows_by_post_id = _load_rows_for_post_ids(
        post_ids=post_ids,
        page_limit=page_limit,
        timeout_s=timeout_s,
        retries=retries,
        max_pages=max_pages,
    )
    questions = [
        _build_question_row(post_id=post_id, row=rows_by_post_id[post_id])
        for post_id in post_ids
    ]

    payload = {
        "schema_version": EVAL_QUESTION_FILE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "builder": "scripts/build_eval_question_file_from_api2_list.py",
            "post_ids": post_ids,
            "api_endpoint": API2_BASE_URL,
            "context_strategy": "api2_questions_pagination",
            "page_limit": page_limit,
        },
        "questions": questions,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical eval_question_file by crawling /api2/questions/ pages for context."
    )
    parser.add_argument(
        "--post-ids",
        nargs="+",
        type=int,
        required=True,
        help="Resolved Metaculus post IDs to include",
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
    parser.add_argument(
        "--page-limit",
        type=int,
        default=100,
        help="Pagination page size for /api2/questions/ crawling (default: 100)",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=20.0,
        help="HTTP timeout per request in seconds (default: 20)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Request retries per page on failure/429 (default: 5)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional hard limit on number of pages to scan",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_eval_question_file_from_api2_list(
        post_ids=args.post_ids,
        output_file=args.output_file,
        overwrite=args.overwrite,
        page_limit=args.page_limit,
        timeout_s=args.timeout_s,
        retries=args.retries,
        max_pages=args.max_pages,
    )
    print(f"Wrote eval question file: {args.output_file}")
    print(f"Schema version: {payload['schema_version']}")
    print(f"Question count: {len(payload.get('questions', []))}")


if __name__ == "__main__":
    main()
