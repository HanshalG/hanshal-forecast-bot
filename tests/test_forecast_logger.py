import importlib.util
import json
from pathlib import Path

import numpy as np


def _reload_logger(monkeypatch, tmp_path, **extra_env):
    env = {
        "LOG_ENABLE": "true",
        "LOG_DIR": str(tmp_path / "logs"),
        "LOG_INCLUDE_COMMENT": "true",
        "LOG_GLOBAL_STREAM": "true",
        "LOG_PLAINTEXT": "false",
        "LOG_CONSOLE": "false",
    }
    env.update({k: str(v) for k, v in extra_env.items()})
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    module_path = Path(__file__).resolve().parents[1] / "src" / "forecast_logger.py"
    spec = importlib.util.spec_from_file_location("forecast_logger_test_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    forecast_logger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forecast_logger)

    return forecast_logger


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_runtime_and_forecast_logs_are_written(monkeypatch, tmp_path):
    logger = _reload_logger(monkeypatch, tmp_path)

    logger.log_info("runtime-check", run_id="run-1", question_id=np.int64(100))
    logger.log_forecast_event(
        {
            "run_id": "run-1",
            "question_id": np.int64(100),
            "post_id": 200,
            "question_title": "Sample",
            "question_type": "binary",
            "forecast": np.float64(0.41),
            "comment": "ok",
            "submitted": False,
            "submit_attempted": False,
        }
    )

    runtime_path = tmp_path / "logs" / "runtime" / "runtime.jsonl"
    by_question_path = tmp_path / "logs" / "forecasts" / "by_question" / "100.jsonl"
    by_run_path = tmp_path / "logs" / "forecasts" / "by_run" / "run-1.jsonl"
    all_events_path = tmp_path / "logs" / "forecasts" / "all_events.jsonl"

    runtime_events = _read_jsonl(runtime_path)
    forecast_events = _read_jsonl(by_question_path)
    by_run_events = _read_jsonl(by_run_path)
    all_events = _read_jsonl(all_events_path)

    assert runtime_events[0]["event_type"] == "runtime"
    assert runtime_events[0]["message"] == "runtime-check"
    assert runtime_events[0]["question_id"] == 100

    assert forecast_events[0]["event_type"] == "forecast"
    assert forecast_events[0]["forecast"] == 0.41
    assert forecast_events[0]["comment"] == "ok"

    assert by_run_events[0]["run_id"] == "run-1"
    assert len(all_events) == 2


def test_comment_can_be_excluded(monkeypatch, tmp_path):
    logger = _reload_logger(monkeypatch, tmp_path, LOG_INCLUDE_COMMENT="false")

    logger.log_forecast_event(
        {
            "run_id": "run-2",
            "question_id": 300,
            "post_id": 301,
            "question_title": "Sample 2",
            "question_type": "binary",
            "forecast": 0.5,
            "comment": "sensitive text",
            "submitted": False,
            "submit_attempted": False,
        }
    )

    by_question_path = tmp_path / "logs" / "forecasts" / "by_question" / "300.jsonl"
    forecast_events = _read_jsonl(by_question_path)
    assert "comment" not in forecast_events[0]


def test_supabase_insert_is_called_with_enriched_payload(monkeypatch, tmp_path):
    logger = _reload_logger(
        monkeypatch,
        tmp_path,
        SUPABASE_LOG_ENABLE="true",
        SUPABASE_URL="https://example-project.supabase.co",
        SUPABASE_KEY="test-key",
        SUPABASE_FORECAST_TABLE="forecast_events",
        SUPABASE_TIMEOUT_S="9",
    )

    captured = {}

    class _Resp:
        status_code = 201
        text = ""

    def _fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(logger.requests, "post", _fake_post)

    logger.log_forecast_event(
        {
            "run_id": "run-supa",
            "question_id": 999,
            "post_id": 888,
            "question_title": "Supa test",
            "question_type": "binary",
            "forecast": 0.73,
            "comment": "comment body",
            "outside_view_model": "openai/gpt-5-mini",
            "inside_view_model": "openai/gpt-5-mini",
            "final_forecast_model": "openai/gpt-5-mini",
            "summary_model": "openai/gpt-5-nano",
            "submit_attempted": True,
            "submitted": True,
            "outside_view_text": "outside",
            "inside_view_text": "inside",
            "final_forecast_analysis": "analysis",
            "all_probabilities": [0.7, 0.8, 0.69],
            "forecast_stddev": 0.05,
            "prompt_tokens": 101,
            "completion_tokens": 202,
            "cost_usd": 0.012345,
            "token_usage_by_component": {"outside_view": {"total": 50}},
            "tool_call_counts": {"exa_search": 3},
            "tool_cache_hit_counts": {"exa_search": 1},
            "tool_cache_miss_counts": {"exa_search": 2},
            "asknews_total_fetched": 20,
            "asknews_removed_by_filter": 5,
        }
    )

    assert captured["url"] == "https://example-project.supabase.co/rest/v1/forecast_events"
    assert captured["headers"]["apikey"] == "test-key"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["timeout"] == 9

    payload = captured["json"]
    assert payload["run_id"] == "run-supa"
    assert payload["question_id"] == 999
    assert payload["outside_view_model"] == "openai/gpt-5-mini"
    assert payload["inside_view_model"] == "openai/gpt-5-mini"
    assert payload["final_forecast_model"] == "openai/gpt-5-mini"
    assert payload["summary_model"] == "openai/gpt-5-nano"
    assert payload["forecast_json"] == 0.73
    assert payload["all_probabilities_json"] == [0.7, 0.8, 0.69]
    assert payload["forecast_stddev"] == 0.05
    assert payload["outside_view_text"] == "outside"
    assert payload["prompt_tokens"] == 101
    assert payload["completion_tokens"] == 202
    assert payload["cost_usd"] == 0.012345
    assert payload["tool_call_counts"]["exa_search"] == 3
    assert payload["raw_event"]["question_title"] == "Supa test"


def test_supabase_dedup_skips_pre_submit_event(monkeypatch, tmp_path):
    logger = _reload_logger(
        monkeypatch,
        tmp_path,
        SUPABASE_LOG_ENABLE="true",
        SUPABASE_URL="https://example-project.supabase.co",
        SUPABASE_KEY="test-key",
    )

    calls = {"count": 0}

    class _Resp:
        status_code = 201
        text = ""

    def _fake_post(url, headers, json, timeout):
        calls["count"] += 1
        return _Resp()

    monkeypatch.setattr(logger.requests, "post", _fake_post)

    logger.log_forecast_event(
        {
            "run_id": "run-skip",
            "question_id": 1234,
            "question_type": "binary",
            "forecast": 0.61,
            "submit_attempted": True,
            "submitted": False,
        }
    )
    assert calls["count"] == 0

    logger.log_forecast_event(
        {
            "run_id": "run-skip",
            "question_id": 1234,
            "question_type": "binary",
            "forecast": 0.61,
            "submit_attempted": True,
            "submitted": True,
        }
    )
    assert calls["count"] == 1


def test_supabase_runtime_toggle_controls_writes(monkeypatch, tmp_path):
    logger = _reload_logger(
        monkeypatch,
        tmp_path,
        SUPABASE_LOG_ENABLE="true",
        SUPABASE_URL="https://example-project.supabase.co",
        SUPABASE_KEY="test-key",
    )

    calls = {"count": 0}

    class _Resp:
        status_code = 201
        text = ""

    def _fake_post(url, headers, json, timeout):
        calls["count"] += 1
        return _Resp()

    monkeypatch.setattr(logger.requests, "post", _fake_post)

    logger.set_supabase_logging_enabled(False)
    logger.log_forecast_event({"run_id": "toggle-1", "question_id": 1, "question_type": "binary", "forecast": 0.5})
    assert calls["count"] == 0

    logger.set_supabase_logging_enabled(True)
    logger.log_forecast_event({"run_id": "toggle-2", "question_id": 2, "question_type": "binary", "forecast": 0.5})
    assert calls["count"] == 1
