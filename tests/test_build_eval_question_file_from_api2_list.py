import importlib.util
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from src.eval.eval_question_file_loader import load_eval_question_file


def _load_builder_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_eval_question_file_from_api2_list.py"
    )
    spec = importlib.util.spec_from_file_location("build_eval_question_file_from_api2_list_test_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, *, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


def test_build_eval_question_file_from_api2_list_success(monkeypatch, tmp_path):
    module = _load_builder_module()

    page_0 = {
        "next": "https://www.metaculus.com/api2/questions/?limit=100&offset=100",
        "results": [
            {
                "id": 99999,
                "title": "Irrelevant question",
                "description": "Ignore me",
                "resolution_criteria": "Ignore me",
                "fine_print": "Ignore me",
                "open_time": "2025-01-01T00:00:00Z",
                "actual_resolve_time": "2025-02-01T00:00:00Z",
                "question": {
                    "id": 90000,
                    "type": "binary",
                    "status": "resolved",
                    "resolution": "yes",
                    "open_time": "2025-01-01T00:00:00Z",
                    "actual_resolve_time": "2025-02-01T00:00:00Z",
                },
            }
        ],
    }
    page_100 = {
        "next": None,
        "results": [
            {
                "id": 39523,
                "title": "Q1",
                "description": "D1",
                "resolution_criteria": "R1",
                "fine_print": "F1",
                "open_time": "2025-10-21T17:26:35Z",
                "actual_resolve_time": "2025-12-28T22:00:00Z",
                "question": {
                    "id": 38900,
                    "type": "binary",
                    "status": "resolved",
                    "resolution": "yes",
                    "open_time": "2025-10-21T17:26:35Z",
                    "actual_resolve_time": "2025-12-28T22:00:00Z",
                    "description": "D1",
                    "resolution_criteria": "R1",
                    "fine_print": "F1",
                },
            },
            {
                "id": 39575,
                "title": "Q2",
                "description": "D2",
                "resolution_criteria": "R2",
                "fine_print": "F2",
                "open_time": "2025-10-17T16:08:26Z",
                "actual_resolve_time": "2026-01-27T17:47:00Z",
                "question": {
                    "id": 38952,
                    "type": "binary",
                    "status": "resolved",
                    "resolution": "no",
                    "open_time": "2025-10-17T16:08:26Z",
                    "actual_resolve_time": "2026-01-27T17:47:00Z",
                    "description": "D2",
                    "resolution_criteria": "R2",
                    "fine_print": "F2",
                },
            },
        ],
    }

    def _fake_get(url, headers, timeout):
        parsed = urlparse(url)
        offset = int(parse_qs(parsed.query)["offset"][0])
        if offset == 0:
            return _FakeResponse(payload=page_0)
        if offset == 100:
            return _FakeResponse(payload=page_100)
        raise AssertionError(f"Unexpected offset requested: {offset}")

    monkeypatch.setattr(module.requests, "get", _fake_get)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    out = tmp_path / "eval_questions.json"
    payload = module.build_eval_question_file_from_api2_list(
        post_ids=[39523, 39575],
        output_file=str(out),
        overwrite=False,
        page_limit=100,
    )

    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "v1"
    assert loaded["source"]["context_strategy"] == "api2_questions_pagination"
    assert [row["post_id"] for row in loaded["questions"]] == [39523, 39575]
    assert loaded["questions"][0]["label"] == 1
    assert loaded["questions"][1]["label"] == 0
    assert payload["schema_version"] == "v1"

    doc = load_eval_question_file(str(out))
    assert len(doc.questions) == 2
    assert doc.questions[0].post_id == 39523
    assert doc.questions[1].post_id == 39575


def test_build_eval_question_file_from_api2_list_missing_post_id_fails(monkeypatch, tmp_path):
    module = _load_builder_module()

    page_0 = {"next": None, "results": []}

    monkeypatch.setattr(module.requests, "get", lambda url, headers, timeout: _FakeResponse(payload=page_0))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    out = tmp_path / "eval_questions.json"
    with pytest.raises(module.BuildEvalQuestionFileFromApi2Error, match="Could not find requested post IDs"):
        module.build_eval_question_file_from_api2_list(
            post_ids=[39523],
            output_file=str(out),
            overwrite=False,
        )


def test_build_eval_question_file_from_api2_list_missing_fine_print_uses_placeholder(monkeypatch, tmp_path):
    module = _load_builder_module()

    page_0 = {
        "next": None,
        "results": [
            {
                "id": 40210,
                "title": "Q40210",
                "description": "D40210",
                "resolution_criteria": "R40210",
                "fine_print": "",
                "open_time": "2025-10-01T00:00:00Z",
                "actual_resolve_time": "2025-11-01T00:00:00Z",
                "question": {
                    "id": 39762,
                    "type": "binary",
                    "status": "resolved",
                    "resolution": "no",
                    "open_time": "2025-10-01T00:00:00Z",
                    "actual_resolve_time": "2025-11-01T00:00:00Z",
                    "description": "D40210",
                    "resolution_criteria": "R40210",
                    "fine_print": "",
                },
            }
        ],
    }

    monkeypatch.setattr(module.requests, "get", lambda url, headers, timeout: _FakeResponse(payload=page_0))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    out = tmp_path / "eval_questions.json"
    module.build_eval_question_file_from_api2_list(
        post_ids=[40210],
        output_file=str(out),
        overwrite=False,
    )

    loaded = json.loads(out.read_text(encoding="utf-8"))
    row = loaded["questions"][0]
    assert row["fine_print"] == module.DEFAULT_FINE_PRINT_PLACEHOLDER
    assert row["metadata"]["context_warnings"] == ["fine_print_missing_replaced_with_placeholder"]
