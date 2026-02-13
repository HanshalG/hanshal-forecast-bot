from langchain_community.callbacks.openai_info import OpenAICallbackHandler
import threading
import sys
import contextvars

# Define cost per 1M tokens (in USD)
# Format: "model_name": (input_cost_per_1m, output_cost_per_1m)
MODEL_COSTS = {
    "gpt-5-mini": (0.25, 2.00),
    "openai/gpt-5-mini": (0.25, 2.00),
    "gpt-4o": (2.50, 10.00),
    "openai/gpt-4o": (2.50, 10.00),
    "o1-mini": (1.10, 4.40),
    "openai/o1-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
    "openai/o3-mini": (1.10, 4.40),
    "gpt-5-nano": (0.05, 0.40),
    "openai/gpt-5-nano": (0.05, 0.40),
    "gpt-5.2": (1.75, 14.00),
    "openai/gpt-5.2": (1.75, 14.00),
}

DEFAULT_COMPONENTS = (
    "outside_view",
    "inside_view",
    "final_forecast",
    "call_llm",
)
DEFAULT_USAGE_SCOPE = "__default__"


def _empty_component_usage() -> dict:
    return {"prompt": 0, "completion": 0, "total": 0, "cost": 0.0, "calls": 0}


def _new_scope_usage() -> dict:
    return {component: _empty_component_usage() for component in DEFAULT_COMPONENTS}


# Backward-compatible default scope mapping.
GLOBAL_TOKEN_USAGE = _new_scope_usage()
_USAGE_BY_SCOPE = {DEFAULT_USAGE_SCOPE: GLOBAL_TOKEN_USAGE}
_usage_scope_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "token_usage_scope",
    default=DEFAULT_USAGE_SCOPE,
)

_token_usage_lock = threading.Lock()


def _resolve_scope(scope: str | None = None) -> str:
    if scope is not None and str(scope).strip():
        return str(scope).strip()
    return _usage_scope_var.get()


def _ensure_scope_usage(scope: str) -> dict:
    usage = _USAGE_BY_SCOPE.get(scope)
    if usage is None:
        usage = _new_scope_usage()
        _USAGE_BY_SCOPE[scope] = usage
    return usage


def set_usage_scope(scope: str):
    """Set token-usage scope for the current async/task context."""
    resolved_scope = _resolve_scope(scope)
    return _usage_scope_var.set(resolved_scope)


def reset_usage_scope(scope_token):
    """Restore prior token-usage scope from a token returned by set_usage_scope."""
    _usage_scope_var.reset(scope_token)

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost based on model name and token usage."""
    if model_name in MODEL_COSTS:
        input_cost, output_cost = MODEL_COSTS[model_name]
        cost = (prompt_tokens / 1_000_000 * input_cost) + (completion_tokens / 1_000_000 * output_cost)
        return cost
    return 0.0

def add_token_usage(
    component: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cost: float,
    *,
    scope: str | None = None,
):
    """Add token usage to the global tracker for a specific component.
    
    Args:
        component: Tracking bucket name (e.g. outside_view, inside_view, final_forecast, call_llm)
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens used
        total_tokens: Total tokens used
        cost: Estimated cost in USD
    """
    resolved_scope = _resolve_scope(scope)
    with _token_usage_lock:
        scope_usage = _ensure_scope_usage(resolved_scope)
        if component not in scope_usage:
            scope_usage[component] = _empty_component_usage()
        scope_usage[component]["prompt"] += prompt_tokens
        scope_usage[component]["completion"] += completion_tokens
        scope_usage[component]["total"] += total_tokens
        scope_usage[component]["cost"] += cost
        scope_usage[component]["calls"] += 1

def reset_usage(*, scope: str | None = None):
    """Reset all token usage counters for the provided (or current) scope."""
    resolved_scope = _resolve_scope(scope)
    with _token_usage_lock:
        scope_usage = _ensure_scope_usage(resolved_scope)
        scope_usage.clear()
        for component in DEFAULT_COMPONENTS:
            scope_usage[component] = _empty_component_usage()

def clear_usage_scope(*, scope: str | None = None):
    """Delete usage data for a non-default scope to avoid unbounded growth."""
    resolved_scope = _resolve_scope(scope)
    with _token_usage_lock:
        if resolved_scope == DEFAULT_USAGE_SCOPE:
            scope_usage = _ensure_scope_usage(resolved_scope)
            scope_usage.clear()
            for component in DEFAULT_COMPONENTS:
                scope_usage[component] = _empty_component_usage()
            return
        _USAGE_BY_SCOPE.pop(resolved_scope, None)

def get_total_usage(*, scope: str | None = None) -> dict:
    """Get aggregated token usage across all components.
    
    Returns:
        Dict with total prompt, completion, total tokens, cost, and calls
    """
    resolved_scope = _resolve_scope(scope)
    with _token_usage_lock:
        scope_usage = _ensure_scope_usage(resolved_scope)
        total = {
            "prompt": sum(v["prompt"] for v in scope_usage.values()),
            "completion": sum(v["completion"] for v in scope_usage.values()),
            "total": sum(v["total"] for v in scope_usage.values()),
            "cost": sum(v["cost"] for v in scope_usage.values()),
            "calls": sum(v["calls"] for v in scope_usage.values()),
        }
        return total

def print_total_usage(*, scope: str | None = None):
    """Print comprehensive token usage summary for the provided (or current) scope."""
    resolved_scope = _resolve_scope(scope)
    with _token_usage_lock:
        print("\n" + "=" * 80)
        if resolved_scope == DEFAULT_USAGE_SCOPE:
            print("TOKEN USAGE SUMMARY")
        else:
            print(f"TOKEN USAGE SUMMARY (scope={resolved_scope})")
        print("=" * 80)
        scope_usage = _ensure_scope_usage(resolved_scope)

        # Print per-component breakdown
        for component in sorted(scope_usage.keys()):
            usage = scope_usage[component]
            if usage["calls"] > 0:
                print(f"\n{component.upper().replace('_', ' ')}:")
                print(f"  Calls: {usage['calls']}")
                print(f"  Prompt Tokens: {usage['prompt']:,}")
                print(f"  Completion Tokens: {usage['completion']:,}")
                print(f"  Total Tokens: {usage['total']:,}")
                print(f"  Cost: ${usage['cost']:.6f}")

        # Compute total directly (don't call get_total_usage to avoid deadlock)
        total = {
            "prompt": sum(v["prompt"] for v in scope_usage.values()),
            "completion": sum(v["completion"] for v in scope_usage.values()),
            "total": sum(v["total"] for v in scope_usage.values()),
            "cost": sum(v["cost"] for v in scope_usage.values()),
            "calls": sum(v["calls"] for v in scope_usage.values()),
        }

        print("\n" + "-" * 80)
        print("TOTAL ACROSS ALL COMPONENTS:")
        print(f"  Total Calls: {total['calls']}")
        print(f"  Total Prompt Tokens: {total['prompt']:,}")
        print(f"  Total Completion Tokens: {total['completion']:,}")
        print(f"  Total Tokens: {total['total']:,}")
        print(f"  Total Cost: ${total['cost']:.6f}")
        print("=" * 80 + "\n")
        sys.stdout.flush()


def print_token_usage(cb: OpenAICallbackHandler, model_name: str = "gpt-5-mini", messages: list = None, component: str = "call_llm"):
    """Print token usage and calculated cost, and add to global tracker.
    
    Args:
        cb: OpenAI callback handler with token usage data
        model_name: Name of the model used
        messages: Optional message list/dict with final_output for metadata fallback
        component: Component name for tracking (outside_view, inside_view, final_forecast, call_llm)
    """
    total_tokens = cb.total_tokens
    prompt_tokens = cb.prompt_tokens
    completion_tokens = cb.completion_tokens
    total_cost = cb.total_cost
    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)

    # Fallback to message metadata if callback failed (common with OpenRouter/custom endpoints)
    if total_tokens == 0 and messages:
        # Extract messages list from dict if needed
        msgs_to_process = []
        if isinstance(messages, dict) and "messages" in messages:
            msgs_to_process = messages["messages"]
        elif isinstance(messages, list):
            msgs_to_process = messages
            
        # Sum usage from all messages in the history
        step_count = 0
        for m in msgs_to_process:
            if hasattr(m, "response_metadata") and "token_usage" in m.response_metadata:
                usage = m.response_metadata["token_usage"]
                pt = int(usage.get("prompt_tokens", 0) or 0)
                ct = int(usage.get("completion_tokens", 0) or 0)
                tt = usage.get("total_tokens", None)
                if tt is None:
                    tt = pt + ct
                prompt_tokens += pt
                completion_tokens += ct
                total_tokens += int(tt or 0)
                step_count += 1
        
        if step_count > 0:
            print(f"  (Extracted usage from {step_count} steps in message history)")

    # If LangChain didn't calculate cost (e.g. unknown model), calculate manually
    if total_cost == 0 and model_name:
        total_cost = calculate_cost(model_name, prompt_tokens, completion_tokens)
    
    # Add to global tracker
    add_token_usage(component, prompt_tokens, completion_tokens, total_tokens, total_cost)
        
    print(f"\n--- Token Usage ({component}) ---")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Prompt Tokens: {prompt_tokens:,}")
    print(f"Completion Tokens: {completion_tokens:,}")
    print(f"Total Cost (USD): ${total_cost:.6f}")
    sys.stdout.flush()
