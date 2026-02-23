import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.eval.types import ResolutionRecord, ResolvedContextQuestion


def _load_builder_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_eval_question_file.py"
    )
    spec = importlib.util.spec_from_file_location("build_eval_question_file_test_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_eval_question_file_outputs_canonical_schema(monkeypatch, tmp_path):
    module = _load_builder_module()

    def _fake_load_context(context_file, expected_post_ids):
        assert expected_post_ids == [39523]
        return {
            39523: ResolvedContextQuestion(
                post_id=39523,
                title="Q",
                type="binary",
                description="d",
                resolution_criteria="r",
                fine_print="f",
                metadata={"source": "manual"},
            )
        }

    def _fake_load_resolutions(post_ids):
        assert post_ids == [39523]
        return {
            39523: ResolutionRecord(
                post_id=39523,
                question_id=123,
                status="resolved",
                resolution_raw="yes",
                label=1,
                open_time=datetime(2025, 10, 21, 17, 26, 35, tzinfo=timezone.utc),
                actual_resolve_time=datetime(2025, 12, 31, 0, 0, 0, tzinfo=timezone.utc),
            )
        }

    monkeypatch.setattr(module, "load_manual_context_file", _fake_load_context)
    monkeypatch.setattr(module, "load_resolution_records", _fake_load_resolutions)

    out = tmp_path / "eval_questions.json"
    payload = module.build_eval_question_file(
        post_ids=[39523],
        context_file="ctx.json",
        output_file=str(out),
        overwrite=False,
    )

    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "v1"
    assert len(loaded["questions"]) == 1
    row = loaded["questions"][0]
    assert row["post_id"] == 39523
    assert row["question_id"] == 123
    assert row["label"] == 1
    assert row["type"] == "binary"
    assert payload["schema_version"] == "v1"


def test_build_eval_question_file_fails_on_missing_resolution(monkeypatch, tmp_path):
    module = _load_builder_module()

    def _fake_load_context(context_file, expected_post_ids):
        return {
            39523: ResolvedContextQuestion(
                post_id=39523,
                title="Q",
                type="binary",
                description="d",
                resolution_criteria="r",
                fine_print="f",
                metadata={},
            )
        }

    monkeypatch.setattr(module, "load_manual_context_file", _fake_load_context)
    monkeypatch.setattr(module, "load_resolution_records", lambda post_ids: {})

    out = tmp_path / "eval_questions.json"
    with pytest.raises(module.BuildEvalQuestionFileError):
        module.build_eval_question_file(
            post_ids=[39523],
            context_file="ctx.json",
            output_file=str(out),
            overwrite=False,
        )
