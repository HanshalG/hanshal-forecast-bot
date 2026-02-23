from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any


def parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, datetime.min.time())
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except Exception:
            return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_iso_z(dt: datetime) -> str:
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.isoformat().replace("+00:00", "Z")


def today_string_for_prompt(as_of_time: datetime | str | None) -> str:
    dt = parse_datetime(as_of_time)
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d")


def resolve_as_of_time(question_details: dict[str, Any] | None, explicit_as_of_time: datetime | str | None = None) -> datetime | None:
    if explicit_as_of_time is not None:
        return parse_datetime(explicit_as_of_time)
    if not isinstance(question_details, dict):
        return None
    return parse_datetime(question_details.get("as_of_time"))


def with_question_as_of_time(question_details: dict[str, Any], as_of_time: datetime | str) -> dict[str, Any]:
    enriched = dict(question_details)
    dt = parse_datetime(as_of_time)
    if dt is None:
        raise ValueError(f"Invalid as_of_time: {as_of_time}")
    enriched["as_of_time"] = to_iso_z(dt)
    return enriched


def _extract_published_time(item: dict[str, Any]) -> datetime | None:
    candidates = (
        item.get("pub_date"),
        item.get("publishedDate"),
        item.get("published_date"),
        item.get("published"),
        item.get("date"),
    )
    for candidate in candidates:
        dt = parse_datetime(candidate)
        if dt is not None:
            return dt
    return None


def filter_items_before_as_of(
    items: list[dict[str, Any]],
    *,
    as_of_time: datetime | str | None,
    keep_unparseable: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    cutoff = parse_datetime(as_of_time)
    if cutoff is None:
        return list(items), 0

    kept: list[dict[str, Any]] = []
    removed = 0
    for item in items:
        published_dt = _extract_published_time(item)
        if published_dt is None:
            if keep_unparseable:
                kept.append(item)
            else:
                removed += 1
            continue
        if published_dt <= cutoff:
            kept.append(item)
        else:
            removed += 1

    return kept, removed
