import json

import pytest

from src.eval.eval_question_file_loader import (
    EVAL_QUESTION_FILE_SCHEMA_VERSION,
    EvalQuestionFileError,
    load_eval_question_file,
)


def _valid_payload() -> dict:
    return {
        "schema_version": EVAL_QUESTION_FILE_SCHEMA_VERSION,
        "generated_at": "2026-02-23T00:00:00+00:00",
        "source": {"builder": "tests", "post_ids": [39523]},
        "questions": [
            {
                "post_id": 39523,
                "question_id": 123,
                "title": "Q",
                "type": "binary",
                "description": "d",
                "resolution_criteria": "r",
                "fine_print": "f",
                "label": 1,
                "resolution_raw": "yes",
                "status": "resolved",
                "open_time": "2025-10-21T17:26:35Z",
                "actual_resolve_time": "2025-12-31T00:00:00Z",
                "metadata": {"source": "manual"},
            }
        ],
    }


def test_load_eval_question_file_valid(tmp_path):
    path = tmp_path / "eval_questions.json"
    path.write_text(json.dumps(_valid_payload()), encoding="utf-8")

    doc = load_eval_question_file(str(path))

    assert doc.schema_version == EVAL_QUESTION_FILE_SCHEMA_VERSION
    assert len(doc.questions) == 1
    assert doc.questions[0].post_id == 39523
    assert doc.questions[0].label == 1


def test_load_eval_question_file_bad_schema_version_fails(tmp_path):
    path = tmp_path / "eval_questions.json"
    payload = _valid_payload()
    payload["schema_version"] = "v999"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvalQuestionFileError):
        load_eval_question_file(str(path))


def test_load_eval_question_file_missing_required_field_fails(tmp_path):
    path = tmp_path / "eval_questions.json"
    payload = _valid_payload()
    del payload["questions"][0]["open_time"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvalQuestionFileError):
        load_eval_question_file(str(path))


def test_load_eval_question_file_bad_label_fails(tmp_path):
    path = tmp_path / "eval_questions.json"
    payload = _valid_payload()
    payload["questions"][0]["label"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvalQuestionFileError):
        load_eval_question_file(str(path))
