import json

import pytest

from src.eval.manual_context_loader import ManualContextError, load_manual_context_file


def test_load_manual_context_file_strict_binary_schema(tmp_path):
    path = tmp_path / "context.json"
    payload = {
        "questions": [
            {
                "post_id": 1,
                "title": "Q1",
                "type": "binary",
                "description": "Background",
                "resolution_criteria": "Criteria",
                "fine_print": "Fine print",
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = load_manual_context_file(str(path), expected_post_ids=[1])

    assert list(result.keys()) == [1]
    assert result[1].type == "binary"
    assert result[1].title == "Q1"


def test_load_manual_context_file_missing_required_field_fails(tmp_path):
    path = tmp_path / "context.json"
    payload = {
        "questions": [
            {
                "post_id": 1,
                "title": "Q1",
                "type": "binary",
                "description": "Background",
                "resolution_criteria": "Criteria"
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManualContextError):
        load_manual_context_file(str(path), expected_post_ids=[1])
