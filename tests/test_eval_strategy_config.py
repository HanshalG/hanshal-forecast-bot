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


def test_load_qbaf_strategy_valid(tmp_path):
    path = tmp_path / "strategy_qbaf.yaml"
    path.write_text(
        """
id: qbaf_core
enabled: true
num_runs: 2
strategy_kind: qbaf_multi_agent
qbaf_depth: 3
qbaf_similarity_threshold: 0.7
qbaf_similarity_backend: tfidf_cosine
qbaf_base_aggregation: max
qbaf_root_base_mode: estimated
qbaf_generation_model: gpt-5-mini
qbaf_pairwise_model: gpt-5-nano
qbaf_max_nodes_per_depth: 5
qbaf_agent_profiles:
  - id: base
    retrieval_mode: argllm_base
  - id: ask
    retrieval_mode: rag_asknews
  - id: exa
    retrieval_mode: rag_exa
""",
        encoding="utf-8",
    )

    strategy = load_strategy_files([str(path)], force_nano_models=False)[0]

    assert strategy.strategy_kind == "qbaf_multi_agent"
    assert strategy.qbaf_depth == 3
    assert strategy.qbaf_similarity_threshold == pytest.approx(0.7)
    assert strategy.qbaf_similarity_backend == "tfidf_cosine"
    assert strategy.qbaf_base_aggregation == "max"
    assert strategy.qbaf_root_base_mode == "estimated"
    assert strategy.qbaf_generation_model == "gpt-5-mini"
    assert strategy.qbaf_pairwise_model == "gpt-5-nano"
    assert strategy.qbaf_max_nodes_per_depth == 5
    assert [p.id for p in strategy.qbaf_agent_profiles] == ["base", "ask", "exa"]


@pytest.mark.parametrize(
    "body",
    [
        """
id: bad_qbaf_kind
enabled: true
num_runs: 1
strategy_kind: qbaf_multi_agent
qbaf_similarity_backend: unsupported
""",
        """
id: bad_qbaf_threshold
enabled: true
num_runs: 1
strategy_kind: qbaf_multi_agent
qbaf_similarity_threshold: 1.5
""",
        """
id: bad_qbaf_depth
enabled: true
num_runs: 1
strategy_kind: qbaf_multi_agent
qbaf_depth: 0
""",
    ],
)
def test_load_qbaf_strategy_rejects_invalid_fields(tmp_path, body):
    path = tmp_path / "bad.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(StrategyConfigError):
        load_strategy_files([str(path)], force_nano_models=False)


def test_strategy_config_backward_compatible_defaults_strategy_kind(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(
        """
id: legacy
enabled: true
num_runs: 1
outside_view_enabled: false
inside_view_enabled: true
""",
        encoding="utf-8",
    )

    strategy = load_strategy_files([str(path)], force_nano_models=False)[0]
    assert strategy.strategy_kind == "forecast_pipeline"
    assert strategy.qbaf_depth == 2
    assert strategy.qbaf_similarity_backend == "llm_pairwise"
    assert len(strategy.qbaf_agent_profiles) == 3
