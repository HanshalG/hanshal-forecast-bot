from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import EvalStrategyConfig

REQUIRED_NANO_MODEL_OVERRIDES = {
    "OUTSIDE_VIEW_MODEL": "gpt-5-nano",
    "INSIDE_VIEW_MODEL": "gpt-5-nano",
    "FINAL_FORECAST_MODEL": "gpt-5-nano",
    "SUMMARY_MODEL": "gpt-5-nano",
}

TOP_LEVEL_MODEL_FIELDS = {
    "outside_view_model": "OUTSIDE_VIEW_MODEL",
    "inside_view_model": "INSIDE_VIEW_MODEL",
    "final_forecast_model": "FINAL_FORECAST_MODEL",
    "summary_model": "SUMMARY_MODEL",
}

TOP_LEVEL_ENV_FIELDS = {
    "reasoning_effort": "REASONING_EFFORT",
    "final_forecast_reasoning_effort": "FINAL_FORECAST_REASONING_EFFORT",
    "tool_summary_reasoning_effort": "TOOL_SUMMARY_REASONING_EFFORT",
}


class StrategyConfigError(ValueError):
    """Raised when strategy config YAML is invalid."""


def _to_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise StrategyConfigError(f"Field '{field}' must be a boolean.")


def _to_str_map(value: Any, *, field: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise StrategyConfigError(f"Field '{field}' must be a mapping.")
    out: dict[str, str] = {}
    for k, v in value.items():
        if v is None:
            continue
        out[str(k)] = str(v)
    return out


def _to_positive_int(value: Any, *, field: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise StrategyConfigError(f"Field '{field}' must be a positive integer.") from exc
    if parsed < 1:
        raise StrategyConfigError(f"Field '{field}' must be >= 1.")
    return parsed


def _to_optional_str(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip()
    if not parsed:
        raise StrategyConfigError(f"Field '{field}' cannot be empty when provided.")
    return parsed


def _parse_one_strategy(raw: dict[str, Any], *, source: str, force_nano_models: bool) -> EvalStrategyConfig:
    if not isinstance(raw, dict):
        raise StrategyConfigError(f"{source}: each strategy entry must be a mapping.")

    if "id" not in raw:
        raise StrategyConfigError(f"{source}: missing required field 'id'.")
    if "enabled" not in raw:
        raise StrategyConfigError(f"{source}: missing required field 'enabled'.")
    if "num_runs" not in raw:
        raise StrategyConfigError(f"{source}: missing required field 'num_runs'.")

    strategy_id = str(raw["id"]).strip()
    if not strategy_id:
        raise StrategyConfigError(f"{source}: field 'id' cannot be empty.")

    enabled = _to_bool(raw["enabled"], field=f"{source}.enabled")

    model_overrides = _to_str_map(raw.get("model_overrides"), field=f"{source}.model_overrides")
    for field_name, env_name in TOP_LEVEL_MODEL_FIELDS.items():
        value = _to_optional_str(raw.get(field_name), field=f"{source}.{field_name}")
        if value is not None:
            model_overrides[env_name] = value

    env_overrides = _to_str_map(raw.get("env_overrides"), field=f"{source}.env_overrides")
    for field_name, env_name in TOP_LEVEL_ENV_FIELDS.items():
        value = _to_optional_str(raw.get(field_name), field=f"{source}.{field_name}")
        if value is not None:
            env_overrides[env_name] = value

    if force_nano_models:
        for key, value in REQUIRED_NANO_MODEL_OVERRIDES.items():
            model_overrides.setdefault(key, value)

    return EvalStrategyConfig(
        id=strategy_id,
        enabled=enabled,
        num_runs=_to_positive_int(raw.get("num_runs"), field=f"{source}.num_runs"),
        outside_view_enabled=_to_bool(raw.get("outside_view_enabled", True), field=f"{source}.outside_view_enabled"),
        inside_view_enabled=_to_bool(raw.get("inside_view_enabled", True), field=f"{source}.inside_view_enabled"),
        prediction_market_enabled=_to_bool(raw.get("prediction_market_enabled", False), field=f"{source}.prediction_market_enabled"),
        final_forecast_use_agent=_to_bool(raw.get("final_forecast_use_agent", True), field=f"{source}.final_forecast_use_agent"),
        env_overrides=env_overrides,
        model_overrides=model_overrides,
    )


def _load_yaml(path: Path) -> Any:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StrategyConfigError(f"Unable to parse YAML at '{path}': {exc}") from exc
    return payload


def load_strategy_files(strategy_files: list[str], *, force_nano_models: bool = True) -> list[EvalStrategyConfig]:
    if not strategy_files:
        raise StrategyConfigError("At least one --strategy-files path is required.")

    parsed: list[EvalStrategyConfig] = []
    seen_ids: set[str] = set()

    for raw_path in strategy_files:
        path = Path(raw_path)
        if not path.exists():
            raise StrategyConfigError(f"Strategy file not found: {path}")

        payload = _load_yaml(path)
        source = str(path)

        if isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict):
            maybe_entries = payload.get("strategies")
            if isinstance(maybe_entries, list):
                entries = maybe_entries
            else:
                entries = [payload]
        else:
            raise StrategyConfigError(f"{source}: YAML root must be a map or list.")

        for idx, entry in enumerate(entries, start=1):
            strategy = _parse_one_strategy(
                entry,
                source=f"{source}[{idx}]",
                force_nano_models=force_nano_models,
            )
            if strategy.id in seen_ids:
                raise StrategyConfigError(f"Duplicate strategy id '{strategy.id}' across strategy files.")
            seen_ids.add(strategy.id)
            parsed.append(strategy)

    return parsed
