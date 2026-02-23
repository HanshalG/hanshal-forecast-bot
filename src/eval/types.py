from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
class EvalStrategyConfig:
    id: str
    enabled: bool
    outside_view_enabled: bool = True
    inside_view_enabled: bool = True
    prediction_market_enabled: bool = False
    final_forecast_use_agent: bool = True
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
