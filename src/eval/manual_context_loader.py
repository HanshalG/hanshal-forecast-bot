from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import ResolvedContextQuestion


class ManualContextError(ValueError):
    """Raised when the resolved context JSON file is malformed."""


_REQUIRED_FIELDS = (
    "post_id",
    "title",
    "type",
    "description",
    "resolution_criteria",
    "fine_print",
)


def _require_non_empty_str(raw: dict[str, Any], field: str, *, row_label: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManualContextError(f"{row_label}: field '{field}' must be a non-empty string.")
    return value.strip()


def _load_raw_entries(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ManualContextError(f"Could not parse JSON from '{path}': {exc}") from exc

    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        maybe_entries = payload.get("questions")
        if not isinstance(maybe_entries, list):
            raise ManualContextError("Context JSON object must contain a 'questions' list.")
        entries = maybe_entries
    else:
        raise ManualContextError("Context JSON must be either a list or {\"questions\": [...]}.")

    if not entries:
        raise ManualContextError("Context file contains no question entries.")

    out: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ManualContextError(f"Entry #{idx} must be an object.")
        out.append(entry)
    return out


def load_manual_context_file(context_file: str, *, expected_post_ids: list[int] | None = None) -> dict[int, ResolvedContextQuestion]:
    path = Path(context_file)
    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")

    entries = _load_raw_entries(path)

    result: dict[int, ResolvedContextQuestion] = {}
    for idx, row in enumerate(entries, start=1):
        row_label = f"{path}[{idx}]"
        missing = [field for field in _REQUIRED_FIELDS if field not in row]
        if missing:
            raise ManualContextError(f"{row_label}: missing required fields: {', '.join(missing)}")

        try:
            post_id = int(row["post_id"])
        except Exception as exc:
            raise ManualContextError(f"{row_label}: field 'post_id' must be an integer.") from exc

        if post_id in result:
            raise ManualContextError(f"{row_label}: duplicate post_id {post_id}.")

        qtype = _require_non_empty_str(row, "type", row_label=row_label).lower()
        if qtype != "binary":
            raise ManualContextError(f"{row_label}: only binary questions are supported in eval v1.")

        metadata_raw = row.get("metadata", {})
        if metadata_raw is None:
            metadata_raw = {}
        if not isinstance(metadata_raw, dict):
            raise ManualContextError(f"{row_label}: field 'metadata' must be an object when present.")

        result[post_id] = ResolvedContextQuestion(
            post_id=post_id,
            title=_require_non_empty_str(row, "title", row_label=row_label),
            type=qtype,
            description=_require_non_empty_str(row, "description", row_label=row_label),
            resolution_criteria=_require_non_empty_str(row, "resolution_criteria", row_label=row_label),
            fine_print=_require_non_empty_str(row, "fine_print", row_label=row_label),
            metadata={str(k): v for k, v in metadata_raw.items()},
        )

    if expected_post_ids is not None:
        expected_set = {int(pid) for pid in expected_post_ids}
        missing = sorted(pid for pid in expected_set if pid not in result)
        if missing:
            raise ManualContextError(
                "Context file is missing required post IDs: " + ", ".join(str(x) for x in missing)
            )
        result = {pid: result[pid] for pid in expected_post_ids}

    return result
