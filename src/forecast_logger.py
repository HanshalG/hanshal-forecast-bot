import datetime
import json
import os
import sys
import threading
import traceback
from typing import Any, Optional

try:
    import requests
except Exception:
    requests = None

try:
    import numpy as np
except Exception:
    np = None


_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
_lock = threading.Lock()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


LOG_ENABLE = _env_bool("LOG_ENABLE", True)
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_INCLUDE_COMMENT = _env_bool("LOG_INCLUDE_COMMENT", True)
LOG_GLOBAL_STREAM = _env_bool("LOG_GLOBAL_STREAM", True)
LOG_PLAINTEXT = _env_bool("LOG_PLAINTEXT", True)
LOG_CONSOLE = _env_bool("LOG_CONSOLE", True)
LOG_CONSOLE_JSON = _env_bool("LOG_CONSOLE_JSON", False)
LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").strip().upper()
LOG_LEVEL = _LEVELS.get(LOG_LEVEL_NAME, 20)
LOG_MESSAGE_PREVIEW_CHARS = _env_int("LOG_MESSAGE_PREVIEW_CHARS", 300)
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_KEY = (os.getenv("SUPABASE_KEY") or "").strip()
SUPABASE_TABLE = (os.getenv("SUPABASE_FORECAST_TABLE") or "forecast_events").strip()
SUPABASE_TIMEOUT_S = _env_int("SUPABASE_TIMEOUT_S", 5)
SUPABASE_LOG_ENABLE = _env_bool("SUPABASE_LOG_ENABLE", True)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return "<unrepresentable>"


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if np is not None:
        try:
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:
            pass

    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        return value.isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, BaseException):
        return {"error_type": value.__class__.__name__, "error_message": str(value)}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    return _safe_repr(value)


def _json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(_to_jsonable(data), ensure_ascii=False, separators=(",", ":"))


def _init_dirs() -> dict[str, str]:
    if not LOG_ENABLE:
        return {}

    base_dir = os.path.join(LOG_DIR, "forecasts")
    by_question_dir = os.path.join(base_dir, "by_question")
    by_run_dir = os.path.join(base_dir, "by_run")
    runtime_dir = os.path.join(LOG_DIR, "runtime")

    os.makedirs(by_question_dir, exist_ok=True)
    os.makedirs(by_run_dir, exist_ok=True)
    os.makedirs(runtime_dir, exist_ok=True)

    return {
        "base": base_dir,
        "by_question": by_question_dir,
        "by_run": by_run_dir,
        "runtime": runtime_dir,
    }


PATHS = _init_dirs()


def _append_line(path: str, line: str) -> None:
    with _lock:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")


def _append_text(path: str, text: str) -> None:
    with _lock:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)


def _format_plaintext_forecast(event: dict[str, Any]) -> str:
    ts = event.get("timestamp_iso", _now_iso())
    qid = event.get("question_id")
    pid = event.get("post_id")
    title = event.get("question_title") or ""
    qtype = event.get("question_type") or ""
    model = event.get("model") or ""
    run_id = event.get("run_id") or ""
    tournament_id = event.get("tournament_id") or ""
    submitted = bool(event.get("submitted", False))
    submit_attempted = bool(event.get("submit_attempted", False))
    forecast = event.get("forecast")
    error = event.get("error")

    forecast_str = ""
    try:
        if error:
            forecast_str = f"ERROR: {error}"
        elif qtype == "binary" and isinstance(forecast, (float, int)):
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
            forecast_str = f"CDF: {forecast}"
        else:
            forecast_str = str(forecast)
    except Exception:
        forecast_str = str(forecast)

    lines = []
    lines.append("=" * 80)
    lines.append(f"Time: {ts}")
    lines.append(f"Run: {run_id}  Tournament: {tournament_id}")
    lines.append(f"Question: {title}")
    lines.append(f"IDs: question_id={qid} post_id={pid}")
    lines.append(f"Type: {qtype}  Model: {model}")
    lines.append(f"Submitted: {submitted}  Attempted: {submit_attempted}")
    lines.append(f"Forecast: {forecast_str}")

    comment = event.get("comment_preview") or event.get("comment")
    if comment is not None:
        lines.append("")
        lines.append("Comment:")
        lines.append(str(comment))

    lines.append("")
    return "\n".join(lines)


def _format_console_line(record: dict[str, Any]) -> str:
    level = str(record.get("level", "INFO")).upper()
    ts = str(record.get("timestamp_iso", _now_iso()))
    message = str(record.get("message", "")).replace("\n", " ").strip()
    if len(message) > LOG_MESSAGE_PREVIEW_CHARS:
        message = f"{message[:LOG_MESSAGE_PREVIEW_CHARS]}..."

    run_id = record.get("run_id")
    question_id = record.get("question_id")
    tags = []
    if run_id:
        tags.append(f"run={run_id}")
    if question_id is not None:
        tags.append(f"q={question_id}")
    tag_block = f" [{' '.join(tags)}]" if tags else ""
    return f"{ts} {level}{tag_block} {message}"


def _safe_stderr_write(line: str) -> None:
    try:
        sys.stderr.write(line + "\n")
    except Exception:
        pass


def _should_log_level(level: str) -> bool:
    return _LEVELS.get(level.upper(), 20) >= LOG_LEVEL


def _emit_console(record: dict[str, Any]) -> None:
    if not LOG_CONSOLE:
        return
    if not _should_log_level(str(record.get("level", "INFO"))):
        return
    if LOG_CONSOLE_JSON:
        _safe_stderr_write(_json_dumps(record))
    else:
        _safe_stderr_write(_format_console_line(record))


def _coerce_int_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _supabase_enabled() -> bool:
    return bool(
        SUPABASE_LOG_ENABLE
        and SUPABASE_URL
        and SUPABASE_KEY
        and SUPABASE_TABLE
        and requests is not None
    )


def set_supabase_logging_enabled(enabled: bool) -> None:
    """Override Supabase logging behavior at runtime."""
    global SUPABASE_LOG_ENABLE
    SUPABASE_LOG_ENABLE = bool(enabled)


def _supabase_endpoint() -> str:
    return f"{SUPABASE_URL.rstrip('/')}/rest/v1/{SUPABASE_TABLE}"


def _map_forecast_event_to_supabase_row(event: dict[str, Any]) -> dict[str, Any]:
    question_type = event.get("question_type")
    if question_type not in {"binary", "numeric", "discrete", "multiple_choice"}:
        question_type = None

    token_usage_by_component = event.get("token_usage_by_component")
    if not isinstance(token_usage_by_component, dict):
        token_usage_by_component = {}

    tool_call_counts = event.get("tool_call_counts")
    if not isinstance(tool_call_counts, dict):
        tool_call_counts = {}

    tool_cache_hit_counts = event.get("tool_cache_hit_counts")
    if not isinstance(tool_cache_hit_counts, dict):
        tool_cache_hit_counts = {}

    tool_cache_miss_counts = event.get("tool_cache_miss_counts")
    if not isinstance(tool_cache_miss_counts, dict):
        tool_cache_miss_counts = {}

    metadata = event.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "event_at": event.get("timestamp_iso") or _now_iso(),
        "run_id": event.get("run_id") or "unknown",
        "tournament_id": event.get("tournament_id"),
        "question_id": _coerce_int_id(event.get("question_id")),
        "post_id": _coerce_int_id(event.get("post_id")),
        "question_title": event.get("question_title"),
        "question_type": question_type,
        "model": event.get("model"),
        "num_runs": _coerce_int_id(event.get("num_runs")),
        "event_type": event.get("event_type") or "forecast",
        "level": event.get("level") or "INFO",
        "submit_attempted": bool(event.get("submit_attempted", False)),
        "submitted": bool(event.get("submitted", False)),
        "forecast_json": event.get("forecast"),
        "all_probabilities_json": event.get("all_probabilities"),
        "forecast_stddev": _coerce_float(event.get("forecast_stddev")),
        "comment": event.get("comment"),
        "error": event.get("error"),
        "error_trace": event.get("error_trace"),
        "outside_view_text": event.get("outside_view_text"),
        "inside_view_text": event.get("inside_view_text"),
        "final_forecast_analysis": event.get("final_forecast_analysis"),
        "prompt_tokens": _coerce_int_id(event.get("prompt_tokens")),
        "completion_tokens": _coerce_int_id(event.get("completion_tokens")),
        "cost_usd": _coerce_float(event.get("cost_usd")),
        "token_usage_by_component": token_usage_by_component,
        "tool_call_counts": tool_call_counts,
        "tool_cache_hit_counts": tool_cache_hit_counts,
        "tool_cache_miss_counts": tool_cache_miss_counts,
        "asknews_total_fetched": _coerce_int_id(event.get("asknews_total_fetched")),
        "asknews_removed_by_filter": _coerce_int_id(event.get("asknews_removed_by_filter")),
        "metadata": metadata,
        "raw_event": event,
    }


def _insert_supabase_row(row: dict[str, Any]) -> None:
    if not _supabase_enabled():
        return

    assert requests is not None
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    response = requests.post(
        _supabase_endpoint(),
        headers=headers,
        json=row,
        timeout=SUPABASE_TIMEOUT_S,
    )
    if response.status_code >= 300:
        raise RuntimeError(
            f"Supabase insert failed ({response.status_code}): {response.text[:400]}"
        )


def _prepare_record(
    *,
    event_type: str,
    level: str = "INFO",
    message: Optional[str] = None,
    **fields: Any,
) -> dict[str, Any]:
    record = {
        "timestamp_iso": _now_iso(),
        "event_type": event_type,
        "level": level.upper(),
    }
    if message is not None:
        record["message"] = message
    record.update(_to_jsonable(fields))
    return record


def log_runtime_event(level: str, message: str, **fields: Any) -> None:
    record = _prepare_record(event_type="runtime", level=level, message=message, **fields)
    _emit_console(record)

    if not LOG_ENABLE or not PATHS:
        return

    try:
        runtime_path = os.path.join(PATHS["runtime"], "runtime.jsonl")
        _append_line(runtime_path, _json_dumps(record))
        if LOG_GLOBAL_STREAM:
            all_events_path = os.path.join(PATHS["base"], "all_events.jsonl")
            _append_line(all_events_path, _json_dumps(record))
    except Exception as exc:
        _safe_stderr_write(f"Logger write failure (runtime): {exc}")


def log_debug(message: str, **fields: Any) -> None:
    log_runtime_event("DEBUG", message, **fields)


def log_info(message: str, **fields: Any) -> None:
    log_runtime_event("INFO", message, **fields)


def log_warning(message: str, **fields: Any) -> None:
    log_runtime_event("WARNING", message, **fields)


def log_error(message: str, **fields: Any) -> None:
    log_runtime_event("ERROR", message, **fields)


def log_exception(message: str, exc: BaseException, **fields: Any) -> None:
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_runtime_event(
        "ERROR",
        message,
        error_type=exc.__class__.__name__,
        error_message=str(exc),
        traceback=trace,
        **fields,
    )


def log_forecast_event(event: dict[str, Any]) -> None:
    """
    Append a single forecast event to per-question JSONL and global JSONL.

    This remains backward-compatible with existing call sites while adding:
    - safe serialization for numpy/datetime/exception values
    - optional run-level logs under logs/forecasts/by_run
    - optional runtime-style console mirroring
    """
    event = dict(event)
    event.setdefault("timestamp_iso", _now_iso())
    event.setdefault("event_type", "forecast")
    event.setdefault("level", "INFO")

    if not LOG_INCLUDE_COMMENT and event.get("comment") is not None:
        event.pop("comment", None)

    safe_event = _to_jsonable(event)

    preview_title = safe_event.get("question_title") or "forecast event"
    _emit_console(
        _prepare_record(
            event_type="forecast",
            level=str(safe_event.get("level", "INFO")),
            message=f"Forecast logged: {preview_title}",
            run_id=safe_event.get("run_id"),
            question_id=safe_event.get("question_id"),
            post_id=safe_event.get("post_id"),
            submitted=safe_event.get("submitted"),
        )
    )

    if LOG_ENABLE and PATHS:
        try:
            line = _json_dumps(safe_event)

            question_id = _coerce_int_id(safe_event.get("question_id"))
            if question_id is not None:
                by_q_path = os.path.join(PATHS["by_question"], f"{question_id}.jsonl")
                _append_line(by_q_path, line)
                if LOG_PLAINTEXT:
                    by_q_txt = os.path.join(PATHS["by_question"], f"{question_id}.txt")
                    _append_text(by_q_txt, _format_plaintext_forecast(safe_event))

            run_id = safe_event.get("run_id")
            if run_id:
                by_run_path = os.path.join(PATHS["by_run"], f"{run_id}.jsonl")
                _append_line(by_run_path, line)

            if LOG_GLOBAL_STREAM:
                all_path = os.path.join(PATHS["base"], "all_forecasts.jsonl")
                _append_line(all_path, line)
                if LOG_PLAINTEXT:
                    all_txt = os.path.join(PATHS["base"], "all_forecasts.txt")
                    _append_text(all_txt, _format_plaintext_forecast(safe_event))

                all_events_path = os.path.join(PATHS["base"], "all_events.jsonl")
                _append_line(all_events_path, line)
        except Exception as exc:
            _safe_stderr_write(f"Logger write failure (forecast): {exc}")

    if _supabase_enabled():
        try:
            row = _map_forecast_event_to_supabase_row(safe_event)
            _insert_supabase_row(row)
        except Exception as exc:
            _safe_stderr_write(f"Supabase logging failure (forecast): {exc}")
