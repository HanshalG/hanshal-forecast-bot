from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .timebox import parse_datetime
from .types import EvalQuestion

EVAL_QUESTION_FILE_SCHEMA_VERSION = "v1"


class EvalQuestionFileError(ValueError):
    """Raised when an eval_question_file is malformed."""


@dataclass(frozen=True)
class EvalQuestionFileDocument:
    schema_version: str
    generated_at: str | None
    source: dict[str, Any]
    questions: list[EvalQuestion]


def _require_non_empty_str(raw: dict[str, Any], field: str, *, row_label: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EvalQuestionFileError(f"{row_label}: field '{field}' must be a non-empty string.")
    return value.strip()


def _parse_required_datetime(raw: dict[str, Any], field: str, *, row_label: str) -> datetime:
    value = raw.get(field)
    dt = parse_datetime(value)
    if dt is None:
        raise EvalQuestionFileError(f"{row_label}: field '{field}' must be an ISO datetime string.")
    return dt


def _parse_optional_datetime(raw: dict[str, Any], field: str, *, row_label: str) -> datetime | None:
    value = raw.get(field)
    if value in (None, ""):
        return None
    dt = parse_datetime(value)
    if dt is None:
        raise EvalQuestionFileError(f"{row_label}: field '{field}' must be an ISO datetime string when present.")
    return dt


def _parse_question(raw: dict[str, Any], *, row_label: str) -> EvalQuestion:
    required_fields = (
        "post_id",
        "question_id",
        "title",
        "type",
        "description",
        "resolution_criteria",
        "fine_print",
        "label",
        "resolution_raw",
        "status",
        "open_time",
    )
    missing = [f for f in required_fields if f not in raw]
    if missing:
        raise EvalQuestionFileError(f"{row_label}: missing required fields: {', '.join(missing)}")

    try:
        post_id = int(raw["post_id"])
    except Exception as exc:
        raise EvalQuestionFileError(f"{row_label}: field 'post_id' must be an integer.") from exc

    try:
        question_id = int(raw["question_id"])
    except Exception as exc:
        raise EvalQuestionFileError(f"{row_label}: field 'question_id' must be an integer.") from exc

    qtype = _require_non_empty_str(raw, "type", row_label=row_label).lower()
    if qtype != "binary":
        raise EvalQuestionFileError(f"{row_label}: only binary questions are supported in eval v1.")

    try:
        label = int(raw["label"])
    except Exception as exc:
        raise EvalQuestionFileError(f"{row_label}: field 'label' must be 0 or 1.") from exc
    if label not in (0, 1):
        raise EvalQuestionFileError(f"{row_label}: field 'label' must be 0 or 1.")

    status = _require_non_empty_str(raw, "status", row_label=row_label).lower()
    if status != "resolved":
        raise EvalQuestionFileError(f"{row_label}: field 'status' must be 'resolved'.")

    metadata_raw = raw.get("metadata", {})
    if metadata_raw is None:
        metadata_raw = {}
    if not isinstance(metadata_raw, dict):
        raise EvalQuestionFileError(f"{row_label}: field 'metadata' must be an object when present.")

    return EvalQuestion(
        post_id=post_id,
        question_id=question_id,
        title=_require_non_empty_str(raw, "title", row_label=row_label),
        type=qtype,
        description=_require_non_empty_str(raw, "description", row_label=row_label),
        resolution_criteria=_require_non_empty_str(raw, "resolution_criteria", row_label=row_label),
        fine_print=_require_non_empty_str(raw, "fine_print", row_label=row_label),
        label=label,
        resolution_raw=_require_non_empty_str(raw, "resolution_raw", row_label=row_label),
        status=status,
        open_time=_parse_required_datetime(raw, "open_time", row_label=row_label),
        actual_resolve_time=_parse_optional_datetime(raw, "actual_resolve_time", row_label=row_label),
        metadata={str(k): v for k, v in metadata_raw.items()},
    )


def load_eval_question_file(eval_question_file: str) -> EvalQuestionFileDocument:
    path = Path(eval_question_file)
    if not path.exists():
        raise FileNotFoundError(f"Eval question file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvalQuestionFileError(f"Could not parse JSON from '{path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise EvalQuestionFileError("eval_question_file must be a JSON object.")

    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version != EVAL_QUESTION_FILE_SCHEMA_VERSION:
        raise EvalQuestionFileError(
            f"Unsupported schema_version '{schema_version}'. Expected '{EVAL_QUESTION_FILE_SCHEMA_VERSION}'."
        )

    questions_raw = payload.get("questions")
    if not isinstance(questions_raw, list) or not questions_raw:
        raise EvalQuestionFileError("Field 'questions' must be a non-empty list.")

    questions: list[EvalQuestion] = []
    seen_post_ids: set[int] = set()
    for idx, item in enumerate(questions_raw, start=1):
        if not isinstance(item, dict):
            raise EvalQuestionFileError(f"questions[{idx}]: each row must be an object.")
        row_label = f"{path}.questions[{idx}]"
        parsed = _parse_question(item, row_label=row_label)
        if parsed.post_id in seen_post_ids:
            raise EvalQuestionFileError(f"{row_label}: duplicate post_id {parsed.post_id}.")
        seen_post_ids.add(parsed.post_id)
        questions.append(parsed)

    source_raw = payload.get("source", {})
    if source_raw is None:
        source_raw = {}
    if not isinstance(source_raw, dict):
        raise EvalQuestionFileError("Field 'source' must be an object when present.")

    generated_at = payload.get("generated_at")
    if generated_at is not None and not isinstance(generated_at, str):
        raise EvalQuestionFileError("Field 'generated_at' must be an ISO datetime string when present.")

    return EvalQuestionFileDocument(
        schema_version=schema_version,
        generated_at=generated_at,
        source={str(k): v for k, v in source_raw.items()},
        questions=questions,
    )
