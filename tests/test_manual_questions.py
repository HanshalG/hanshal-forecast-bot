import json
import importlib.util
from pathlib import Path

import pytest


def _load_manual_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "manual_questions.py"
    spec = importlib.util.spec_from_file_location("manual_questions_test_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_manual_questions_from_json_object(tmp_path):
    module = _load_manual_module()
    payload = {
        "questions": [
            {
                "id": 42,
                "type": "binary",
                "title": "Will X happen?",
            }
        ]
    }
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    questions = module.load_manual_questions(str(path))

    assert len(questions) == 1
    assert questions[0]["id"] == 42
    assert questions[0]["post_id"] == 42
    assert questions[0]["type"] == "binary"
    assert questions[0]["title"] == "Will X happen?"


def test_load_manual_questions_supports_jsonl(tmp_path):
    module = _load_manual_module()
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": 1, "type": "binary", "title": "Q1"}),
                json.dumps(
                    {
                        "id": 2,
                        "type": "multiple_choice",
                        "title": "Q2",
                        "options": ["A", "B"],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    questions = module.load_manual_questions(str(path))

    assert len(questions) == 2
    assert questions[0]["id"] == 1
    assert questions[1]["type"] == "multiple_choice"
    assert questions[1]["options"] == ["A", "B"]


def test_multiple_choice_requires_options(tmp_path):
    module = _load_manual_module()
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": 7,
                    "type": "multiple_choice",
                    "title": "Missing options",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(module.ManualQuestionError):
        module.load_manual_questions(str(path))
