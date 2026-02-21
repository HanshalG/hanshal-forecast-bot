import json
from pathlib import Path
from typing import Any


SYNTHETIC_ID_START = 900_000_000


class ManualQuestionError(ValueError):
    """Raised when a manual question payload is invalid."""


def _normalize_question_type(raw_type: Any) -> str:
    value = str(raw_type or "").strip().lower()
    aliases = {
        "binary": "binary",
        "yes_no": "binary",
        "yes/no": "binary",
        "multiple_choice": "multiple_choice",
        "multiple choice": "multiple_choice",
        "multiple-choice": "multiple_choice",
        "mc": "multiple_choice",
        "numeric": "numeric",
        "discrete": "discrete",
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise ManualQuestionError(
            f"Unsupported question type '{raw_type}'. Supported: binary, multiple_choice, numeric, discrete."
        )
    return normalized


def _to_int_or_fallback(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _to_float(value: Any, *, field: str, index: int) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ManualQuestionError(f"Question #{index}: '{field}' must be numeric.") from exc


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _load_json_or_jsonl(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        items = []
        for lineno, line in enumerate(raw.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ManualQuestionError(
                    f"Invalid JSONL at line {lineno} in '{path}'."
                ) from exc
            if not isinstance(parsed, dict):
                raise ManualQuestionError(
                    f"Invalid JSONL at line {lineno} in '{path}': each line must be a JSON object."
                )
            items.append(parsed)
        if not items:
            raise ManualQuestionError(f"No valid questions found in '{path}'.")
        return items


def _normalize_question(raw_question: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(raw_question, dict):
        raise ManualQuestionError(f"Question #{index}: question entry must be a JSON object.")

    fallback_id = SYNTHETIC_ID_START + index
    question_id = _to_int_or_fallback(raw_question.get("id"), fallback_id)
    post_id = _to_int_or_fallback(raw_question.get("post_id"), question_id)

    title = str(raw_question.get("title") or raw_question.get("question") or "").strip()
    if not title:
        raise ManualQuestionError(f"Question #{index}: 'title' is required.")

    question_type = _normalize_question_type(raw_question.get("type"))

    normalized: dict[str, Any] = {
        "id": question_id,
        "post_id": post_id,
        "title": title,
        "type": question_type,
        "description": str(raw_question.get("description") or "").strip(),
        "resolution_criteria": str(raw_question.get("resolution_criteria") or "").strip(),
        "fine_print": str(raw_question.get("fine_print") or "").strip(),
        "unit": str(raw_question.get("unit") or "").strip(),
    }

    url = raw_question.get("url")
    if isinstance(url, str) and url.strip():
        normalized["url"] = url.strip()

    if question_type == "multiple_choice":
        options = raw_question.get("options")
        if not isinstance(options, list) or len(options) < 2:
            raise ManualQuestionError(
                f"Question #{index}: multiple_choice questions require an 'options' list with at least 2 items."
            )
        normalized["options"] = [str(option) for option in options]
        return normalized

    if question_type in {"numeric", "discrete"}:
        scaling = raw_question.get("scaling")
        if not isinstance(scaling, dict):
            raise ManualQuestionError(
                f"Question #{index}: {question_type} questions require a 'scaling' object."
            )

        range_min = _to_float(scaling.get("range_min"), field="scaling.range_min", index=index)
        range_max = _to_float(scaling.get("range_max"), field="scaling.range_max", index=index)
        if range_max <= range_min:
            raise ManualQuestionError(
                f"Question #{index}: scaling.range_max must be greater than scaling.range_min."
            )

        zero_point_raw = scaling.get("zero_point")
        zero_point = (
            None
            if zero_point_raw is None or zero_point_raw == ""
            else _to_float(zero_point_raw, field="scaling.zero_point", index=index)
        )

        normalized_scaling: dict[str, Any] = {
            "range_min": range_min,
            "range_max": range_max,
            "zero_point": zero_point,
        }

        if question_type == "discrete":
            inbound = scaling.get("inbound_outcome_count")
            try:
                inbound_int = int(inbound)
            except Exception as exc:
                raise ManualQuestionError(
                    f"Question #{index}: discrete questions require scaling.inbound_outcome_count (integer)."
                ) from exc
            if inbound_int < 1:
                raise ManualQuestionError(
                    f"Question #{index}: scaling.inbound_outcome_count must be >= 1."
                )
            normalized_scaling["inbound_outcome_count"] = inbound_int

        normalized["scaling"] = normalized_scaling
        normalized["open_upper_bound"] = _to_bool(raw_question.get("open_upper_bound"), default=False)
        normalized["open_lower_bound"] = _to_bool(raw_question.get("open_lower_bound"), default=False)
        return normalized

    return normalized


def load_manual_questions(path_str: str) -> list[dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Manual questions file not found: {path}")

    loaded = _load_json_or_jsonl(path)
    if isinstance(loaded, dict):
        questions = loaded.get("questions")
        if not isinstance(questions, list):
            raise ManualQuestionError(
                "Manual questions file must be either a JSON array or an object containing a 'questions' array."
            )
    elif isinstance(loaded, list):
        questions = loaded
    else:
        raise ManualQuestionError(
            "Manual questions file must be either a JSON array or an object containing a 'questions' array."
        )

    if not questions:
        raise ManualQuestionError("Manual questions file contains no questions.")

    normalized = []
    for index, raw in enumerate(questions, start=1):
        normalized.append(_normalize_question(raw, index))
    return normalized

