from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(frozen=True)
class ResolvedContextQuestion:
    post_id: int
    title: str
    type: str
    description: str
    resolution_criteria: str
    fine_print: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolutionRecord:
    post_id: int
    question_id: int
    status: str
    resolution_raw: str
    label: int
    open_time: datetime
    actual_resolve_time: datetime | None


@dataclass(frozen=True)
class EvalQuestion:
    post_id: int
    question_id: int
    title: str
    type: str
    description: str
    resolution_criteria: str
    fine_print: str
    label: int
    resolution_raw: str
    status: str
    open_time: datetime
    actual_resolve_time: datetime | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QbafAgentProfile:
    id: str
    retrieval_mode: Literal["argllm_base", "rag_asknews", "rag_exa"]
    description: str = ""


def _default_qbaf_agent_profiles() -> list[QbafAgentProfile]:
    return [
        QbafAgentProfile(
            id="argllm_base",
            retrieval_mode="argllm_base",
            description="Base argument-only agent without retrieval context.",
        ),
        QbafAgentProfile(
            id="rag_asknews",
            retrieval_mode="rag_asknews",
            description="RAG agent using AskNews time-boxed context.",
        ),
        QbafAgentProfile(
            id="rag_exa",
            retrieval_mode="rag_exa",
            description="RAG agent using Exa historical summaries.",
        ),
    ]


@dataclass(frozen=True)
class EvalStrategyConfig:
    id: str
    enabled: bool
    num_runs: int = 1
    strategy_kind: Literal["forecast_pipeline", "qbaf_multi_agent"] = "forecast_pipeline"
    outside_view_enabled: bool = True
    inside_view_enabled: bool = True
    prediction_market_enabled: bool = False
    final_forecast_use_agent: bool = True
    qbaf_depth: int = 2
    qbaf_similarity_threshold: float = 0.5
    qbaf_similarity_backend: Literal["llm_pairwise", "embedding_cosine", "tfidf_cosine"] = "llm_pairwise"
    qbaf_base_aggregation: Literal["avg", "max"] = "avg"
    qbaf_root_base_mode: Literal["fixed_0_5", "estimated"] = "fixed_0_5"
    qbaf_agent_profiles: list[QbafAgentProfile] = field(default_factory=_default_qbaf_agent_profiles)
    qbaf_pairwise_model: str | None = None
    qbaf_generation_model: str | None = None
    qbaf_max_nodes_per_depth: int = 6
    env_overrides: dict[str, str] = field(default_factory=dict)
    model_overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionRecord:
    strategy_id: str
    post_id: int
    question_id: int
    title: str
    as_of_time: str
    probability_yes: float
    label: int
    resolution_raw: str
    brier: float
    log_loss: float
    runtime_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    num_runs: int
    outside_view_enabled: bool
    inside_view_enabled: bool
    prediction_market_enabled: bool
    final_forecast_use_agent: bool
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    tool_cache_hit_counts: dict[str, int] = field(default_factory=dict)
    tool_cache_miss_counts: dict[str, int] = field(default_factory=dict)
    asknews_total_fetched: int = 0
    asknews_removed_by_filter: int = 0
    all_probabilities: list[float] = field(default_factory=list)
    forecast_stddev: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class CalibrationBin:
    strategy_id: str
    bin_index: int
    bin_start: float
    bin_end: float
    count: int
    avg_pred: float
    avg_outcome: float
    abs_gap: float


@dataclass(frozen=True)
class StrategySummary:
    strategy_id: str
    n: int
    n_success: int
    n_failed: int
    mean_brier: float
    mean_log_loss: float
    ece: float
    mean_cost_usd: float
    mean_runtime_seconds: float
