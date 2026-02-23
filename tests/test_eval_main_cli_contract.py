import sys

import pytest

import eval_main


def test_eval_main_requires_eval_question_file(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_main.py",
            "--strategy-files",
            "s.yaml",
        ],
    )

    with pytest.raises(SystemExit):
        eval_main._parse_args()


def test_eval_main_rejects_removed_post_ids_and_context_file_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_main.py",
            "--post-ids",
            "39523",
            "--context-file",
            "ctx.json",
            "--strategy-files",
            "s.yaml",
            "--eval-question-file",
            "eq.json",
        ],
    )

    with pytest.raises(SystemExit):
        eval_main._parse_args()


def test_eval_main_rejects_removed_num_runs_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_main.py",
            "--eval-question-file",
            "eq.json",
            "--strategy-files",
            "s.yaml",
            "--num-runs",
            "2",
        ],
    )

    with pytest.raises(SystemExit):
        eval_main._parse_args()


def test_eval_main_parses_new_runtime_contract(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_main.py",
            "--eval-question-file",
            "eq.json",
            "--strategy-files",
            "s.yaml",
            "--question-concurrency",
            "3",
            "--output-dir",
            "logs/evals",
        ],
    )

    args = eval_main._parse_args()

    assert args.eval_question_file == "eq.json"
    assert args.strategy_files == ["s.yaml"]
    assert args.question_concurrency == 3
