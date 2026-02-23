import pytest

from src.eval.strategy_config import StrategyConfigError, load_strategy_files


def test_load_strategy_file_enforces_nano_models(tmp_path):
    path = tmp_path / "strategy.yaml"
    path.write_text(
        """
id: baseline
enabled: true
outside_view_enabled: true
inside_view_enabled: false
prediction_market_enabled: true
final_forecast_use_agent: false
env_overrides:
  LOG_ENABLE: "false"
model_overrides:
  OUTSIDE_VIEW_MODEL: gpt-5-mini
""",
        encoding="utf-8",
    )

    strategies = load_strategy_files([str(path)], force_nano_models=True)

    assert len(strategies) == 1
    strategy = strategies[0]
    assert strategy.id == "baseline"
    assert strategy.enabled is True
    assert strategy.inside_view_enabled is False
    assert strategy.model_overrides["OUTSIDE_VIEW_MODEL"] == "gpt-5-nano"
    assert strategy.model_overrides["INSIDE_VIEW_MODEL"] == "gpt-5-nano"
    assert strategy.model_overrides["FINAL_FORECAST_MODEL"] == "gpt-5-nano"
    assert strategy.model_overrides["SUMMARY_MODEL"] == "gpt-5-nano"


def test_load_strategy_file_requires_id_and_enabled(tmp_path):
    path = tmp_path / "strategy.yaml"
    path.write_text("outside_view_enabled: true\n", encoding="utf-8")

    with pytest.raises(StrategyConfigError):
        load_strategy_files([str(path)])
