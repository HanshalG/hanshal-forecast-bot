import pytest

from src.eval.strategy_config import StrategyConfigError, load_strategy_files


def test_load_strategy_file_applies_nano_defaults_but_respects_explicit_model_fields(tmp_path):
    path = tmp_path / "strategy.yaml"
    path.write_text(
        """
id: baseline
enabled: true
num_runs: 3
outside_view_enabled: true
inside_view_enabled: false
prediction_market_enabled: true
final_forecast_use_agent: false
env_overrides:
  LOG_ENABLE: "false"
outside_view_model: gpt-5-mini
inside_view_model: gpt-5.2
reasoning_effort: low
final_forecast_reasoning_effort: medium
tool_summary_reasoning_effort: high
""",
        encoding="utf-8",
    )

    strategies = load_strategy_files([str(path)], force_nano_models=True)

    assert len(strategies) == 1
    strategy = strategies[0]
    assert strategy.id == "baseline"
    assert strategy.enabled is True
    assert strategy.num_runs == 3
    assert strategy.inside_view_enabled is False
    assert strategy.model_overrides["OUTSIDE_VIEW_MODEL"] == "gpt-5-mini"
    assert strategy.model_overrides["INSIDE_VIEW_MODEL"] == "gpt-5.2"
    assert strategy.model_overrides["FINAL_FORECAST_MODEL"] == "gpt-5-nano"
    assert strategy.model_overrides["SUMMARY_MODEL"] == "gpt-5-nano"
    assert strategy.env_overrides["REASONING_EFFORT"] == "low"
    assert strategy.env_overrides["FINAL_FORECAST_REASONING_EFFORT"] == "medium"
    assert strategy.env_overrides["TOOL_SUMMARY_REASONING_EFFORT"] == "high"


def test_load_strategy_file_requires_id_and_enabled(tmp_path):
    path = tmp_path / "strategy.yaml"
    path.write_text("outside_view_enabled: true\n", encoding="utf-8")

    with pytest.raises(StrategyConfigError):
        load_strategy_files([str(path)])


def test_load_strategy_file_requires_positive_num_runs(tmp_path):
    path = tmp_path / "strategy.yaml"
    path.write_text(
        """
id: bad
enabled: true
num_runs: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(StrategyConfigError):
        load_strategy_files([str(path)])


def test_load_strategy_file_can_disable_nano_defaults(tmp_path):
    path = tmp_path / "strategy.yaml"
    path.write_text(
        """
id: baseline
enabled: true
num_runs: 1
outside_view_model: gpt-5-mini
""",
        encoding="utf-8",
    )

    strategies = load_strategy_files([str(path)], force_nano_models=False)
    assert strategies[0].model_overrides == {"OUTSIDE_VIEW_MODEL": "gpt-5-mini"}
