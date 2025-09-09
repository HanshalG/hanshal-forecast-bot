import datetime
import json
import os
import threading
from typing import Any, Dict, Optional


LOG_ENABLE = os.getenv("LOG_ENABLE", "true").lower() == "true"
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_INCLUDE_COMMENT = os.getenv("LOG_INCLUDE_COMMENT", "true").lower() == "true"
LOG_GLOBAL_STREAM = os.getenv("LOG_GLOBAL_STREAM", "true").lower() == "true"
LOG_PLAINTEXT = os.getenv("LOG_PLAINTEXT", "true").lower() == "true"


_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _init_dirs() -> str:
    if not LOG_ENABLE:
        return ""
    base_dir = os.path.join(LOG_DIR, "forecasts")
    by_question_dir = os.path.join(base_dir, "by_question")
    os.makedirs(by_question_dir, exist_ok=True)
    return base_dir


BASE_DIR = _init_dirs()


def _json_dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _format_percent(value: Optional[float]) -> str:
    try:
        if value is None:
            return "N/A"
        return f"{float(value):.2f}%"
    except Exception:
        return "N/A"

def _format_plaintext(event: Dict[str, Any]) -> str:
    ts = event.get("timestamp_iso", _now_iso())
    qid = event.get("question_id")
    pid = event.get("post_id")
    title = event.get("question_title") or ""
    qtype = event.get("question_type") or ""
    model = event.get("model") or ""
    submitted = bool(event.get("submitted", False))
    submit_attempted = bool(event.get("submit_attempted", False))
    community = event.get("community_prediction")
    forecast = event.get("forecast")
    error = event.get("error")

    # Render forecast compactly by type
    forecast_str = ""
    try:
        if error:
            forecast_str = f"ERROR: {error}"
        elif qtype == "binary" and isinstance(forecast, (float, int)):
            # binary stored as decimal (0..1)
            forecast_str = f"Yes: {float(forecast) * 100:.2f}%"
        elif qtype == "multiple_choice" and isinstance(forecast, dict):
            parts = []
            for k, v in forecast.items():
                try:
                    parts.append(f"{k}={float(v) * 100:.1f}%")
                except Exception:
                    parts.append(f"{k}={v}")
            forecast_str = ", ".join(parts)
        elif qtype in ("numeric", "discrete") and isinstance(forecast, list):
            # Write full CDF without truncation
            forecast_str = f"CDF: {forecast}"
        else:
            forecast_str = str(forecast)
    except Exception:
        forecast_str = str(forecast)

    lines = []
    lines.append("=" * 80)
    lines.append(f"Time: {ts}")
    lines.append(f"Question: {title}")
    lines.append(f"IDs: question_id={qid} post_id={pid}")
    lines.append(f"Type: {qtype}  Model: {model}")
    lines.append(f"Community: {_format_percent(community)}  Submitted: {submitted}  Attempted: {submit_attempted}")
    lines.append(f"Forecast: {forecast_str}")

    comment = event.get("comment_preview") or event.get("comment")
    if comment is not None:
        lines.append("")
        lines.append("Comment:")
        lines.append(str(comment))

    lines.append("")
    return "\n".join(lines)


def log_forecast_event(event: Dict[str, Any]) -> None:
    """
    Append a single forecast event to per-question JSONL and global JSONL.

    The `event` dict is augmented with `timestamp_iso` if missing.
    If LOG_INCLUDE_COMMENT is false, the `comment` field is removed and
    a `comment_preview` (first 200 chars) is added instead.
    """
    if not LOG_ENABLE:
        return

    if not BASE_DIR:
        return

    event = dict(event)  # shallow copy so we can modify safely
    event.setdefault("timestamp_iso", _now_iso())

    # Handle comment inclusion policy
    # If LOG_INCLUDE_COMMENT is False, still avoid truncation by removing the field entirely.
    if not LOG_INCLUDE_COMMENT and "comment" in event and event["comment"] is not None:
        event.pop("comment", None)

    question_id: Optional[int] = event.get("question_id")  # type: ignore[assignment]

    # Write per-question stream
    if question_id is not None:
        by_q_path = os.path.join(BASE_DIR, "by_question", f"{question_id}.jsonl")
        with _lock:
            with open(by_q_path, "a", encoding="utf-8") as f:
                f.write(_json_dumps(event) + "\n")
        if LOG_PLAINTEXT:
            by_q_txt = os.path.join(BASE_DIR, "by_question", f"{question_id}.txt")
            text = _format_plaintext(event)
            with _lock:
                with open(by_q_txt, "a", encoding="utf-8") as tf:
                    tf.write(text)

    # Write global stream
    if LOG_GLOBAL_STREAM:
        all_path = os.path.join(BASE_DIR, "all_forecasts.jsonl")
        with _lock:
            with open(all_path, "a", encoding="utf-8") as f:
                f.write(_json_dumps(event) + "\n")
        if LOG_PLAINTEXT:
            all_txt = os.path.join(BASE_DIR, "all_forecasts.txt")
            text = _format_plaintext(event)
            with _lock:
                with open(all_txt, "a", encoding="utf-8") as tf:
                    tf.write(text)


