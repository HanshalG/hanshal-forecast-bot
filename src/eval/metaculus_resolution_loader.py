from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import requests
from dotenv import load_dotenv

from .types import EvalQuestion, ResolutionRecord, ResolvedContextQuestion

load_dotenv()

API2_BASE_URL = "https://www.metaculus.com/api2/questions"


class ResolutionLoaderError(RuntimeError):
    """Raised when loading resolved outcomes from Metaculus fails."""


def _parse_datetime(value: Any, *, field: str, post_id: int, required: bool) -> datetime | None:
    if value in (None, ""):
        if required:
            raise ResolutionLoaderError(f"Post {post_id}: missing required datetime field '{field}'.")
        return None

    text = str(value).strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except Exception as exc:
        raise ResolutionLoaderError(
            f"Post {post_id}: invalid ISO datetime for '{field}': {value}"
        ) from exc


def _normalize_resolution_label(raw_resolution: Any, *, post_id: int) -> tuple[str, int]:
    value = str(raw_resolution or "").strip().lower()
    if value == "yes":
        return value, 1
    if value == "no":
        return value, 0
    raise ResolutionLoaderError(
        f"Post {post_id}: unsupported resolution '{raw_resolution}'. Expected yes/no."
    )


def _fetch_question_from_api2(post_id: int, *, timeout_s: float = 20.0, retries: int = 3) -> dict[str, Any]:
    token = (os.getenv("METACULUS_TOKEN") or "").strip()
    if not token:
        raise ResolutionLoaderError("METACULUS_TOKEN is required to query /api2/questions/<post_id>/.")

    headers = {
        "Authorization": f"Token {token}",
        "Accept": "application/json",
        "User-Agent": "hanshal-forecast-bot/eval",
    }
    url = f"{API2_BASE_URL}/{int(post_id)}/"

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout_s)
            if response.status_code == 429 and attempt < retries:
                time.sleep(0.5 * (2 ** (attempt - 1)))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ResolutionLoaderError(f"Post {post_id}: API2 response must be an object.")
            return payload
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(0.5 * (2 ** (attempt - 1)))
                continue
            break

    raise ResolutionLoaderError(f"Failed to fetch /api2/questions/{post_id}/: {last_err}")


def load_resolution_records(post_ids: list[int]) -> dict[int, ResolutionRecord]:
    results: dict[int, ResolutionRecord] = {}
    for post_id in post_ids:
        payload = _fetch_question_from_api2(post_id)
        question = payload.get("question")
        if not isinstance(question, dict):
            raise ResolutionLoaderError(f"Post {post_id}: missing 'question' object in API2 response.")

        status = str(question.get("status") or "").strip().lower()
        if status != "resolved":
            raise ResolutionLoaderError(
                f"Post {post_id}: expected status='resolved', got '{question.get('status')}'."
            )

        raw_resolution, label = _normalize_resolution_label(question.get("resolution"), post_id=post_id)
        open_time = _parse_datetime(question.get("open_time"), field="open_time", post_id=post_id, required=True)
        actual_resolve_time = _parse_datetime(
            question.get("actual_resolve_time"),
            field="actual_resolve_time",
            post_id=post_id,
            required=False,
        )

        try:
            question_id = int(question.get("id"))
        except Exception as exc:
            raise ResolutionLoaderError(f"Post {post_id}: invalid or missing question.id") from exc

        results[post_id] = ResolutionRecord(
            post_id=post_id,
            question_id=question_id,
            status=status,
            resolution_raw=raw_resolution,
            label=label,
            open_time=open_time,
            actual_resolve_time=actual_resolve_time,
        )

    return results


def join_context_and_resolutions(
    *,
    post_ids: list[int],
    manual_context_by_post_id: dict[int, ResolvedContextQuestion],
) -> list[EvalQuestion]:
    resolution_by_post_id = load_resolution_records(post_ids)

    joined: list[EvalQuestion] = []
    for post_id in post_ids:
        context = manual_context_by_post_id.get(post_id)
        if context is None:
            raise ResolutionLoaderError(
                f"Missing manual context for post_id={post_id}; eval requires all context fields."
            )

        missing_context_fields = []
        if not context.description.strip():
            missing_context_fields.append("description")
        if not context.resolution_criteria.strip():
            missing_context_fields.append("resolution_criteria")
        if not context.fine_print.strip():
            missing_context_fields.append("fine_print")
        if missing_context_fields:
            raise ResolutionLoaderError(
                f"Post {post_id}: missing required manual context fields: {', '.join(missing_context_fields)}"
            )

        resolution = resolution_by_post_id.get(post_id)
        if resolution is None:
            raise ResolutionLoaderError(f"Missing API2 resolution for post_id={post_id}")

        joined.append(
            EvalQuestion(
                post_id=post_id,
                question_id=resolution.question_id,
                title=context.title,
                type=context.type,
                description=context.description,
                resolution_criteria=context.resolution_criteria,
                fine_print=context.fine_print,
                label=resolution.label,
                resolution_raw=resolution.resolution_raw,
                status=resolution.status,
                open_time=resolution.open_time,
                actual_resolve_time=resolution.actual_resolve_time,
                metadata=context.metadata,
            )
        )

    return joined
