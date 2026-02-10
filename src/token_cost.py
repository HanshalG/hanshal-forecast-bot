from langchain_community.callbacks.openai_info import OpenAICallbackHandler
import threading
import sys

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

# Global token usage tracker by component
GLOBAL_TOKEN_USAGE = {
    "outside_view": {"prompt": 0, "completion": 0, "total": 0, "cost": 0.0, "calls": 0},
    "inside_view": {"prompt": 0, "completion": 0, "total": 0, "cost": 0.0, "calls": 0},
    "final_forecast": {"prompt": 0, "completion": 0, "total": 0, "cost": 0.0, "calls": 0},
    "call_llm": {"prompt": 0, "completion": 0, "total": 0, "cost": 0.0, "calls": 0},
}

_token_usage_lock = threading.Lock()

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost based on model name and token usage."""
    if model_name in MODEL_COSTS:
        input_cost, output_cost = MODEL_COSTS[model_name]
        cost = (prompt_tokens / 1_000_000 * input_cost) + (completion_tokens / 1_000_000 * output_cost)
        return cost
    return 0.0

def add_token_usage(component: str, prompt_tokens: int, completion_tokens: int, total_tokens: int, cost: float):
    """Add token usage to the global tracker for a specific component.
    
    Args:
        component: One of 'outside_view', 'inside_view', 'final_forecast', 'call_llm'
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens used
        total_tokens: Total tokens used
        cost: Estimated cost in USD
    """
    with _token_usage_lock:
        if component in GLOBAL_TOKEN_USAGE:
            GLOBAL_TOKEN_USAGE[component]["prompt"] += prompt_tokens
            GLOBAL_TOKEN_USAGE[component]["completion"] += completion_tokens
            GLOBAL_TOKEN_USAGE[component]["total"] += total_tokens
            GLOBAL_TOKEN_USAGE[component]["cost"] += cost
            GLOBAL_TOKEN_USAGE[component]["calls"] += 1

def reset_usage():
    """Reset all token usage counters."""
    with _token_usage_lock:
        for component in GLOBAL_TOKEN_USAGE:
            GLOBAL_TOKEN_USAGE[component]["prompt"] = 0
            GLOBAL_TOKEN_USAGE[component]["completion"] = 0
            GLOBAL_TOKEN_USAGE[component]["total"] = 0
            GLOBAL_TOKEN_USAGE[component]["cost"] = 0.0
            GLOBAL_TOKEN_USAGE[component]["calls"] = 0

def get_total_usage() -> dict:
    """Get aggregated token usage across all components.
    
    Returns:
        Dict with total prompt, completion, total tokens, cost, and calls
    """
    with _token_usage_lock:
        total = {
            "prompt": sum(v["prompt"] for v in GLOBAL_TOKEN_USAGE.values()),
            "completion": sum(v["completion"] for v in GLOBAL_TOKEN_USAGE.values()),
            "total": sum(v["total"] for v in GLOBAL_TOKEN_USAGE.values()),
            "cost": sum(v["cost"] for v in GLOBAL_TOKEN_USAGE.values()),
            "calls": sum(v["calls"] for v in GLOBAL_TOKEN_USAGE.values()),
        }
        return total

def print_total_usage():
    """Print comprehensive token usage summary."""
    with _token_usage_lock:
        print("\n" + "=" * 80)
        print("TOKEN USAGE SUMMARY")
        print("=" * 80)
        
        # Print per-component breakdown
        for component, usage in GLOBAL_TOKEN_USAGE.items():
            if usage["calls"] > 0:
                print(f"\n{component.upper().replace('_', ' ')}:")
                print(f"  Calls: {usage['calls']}")
                print(f"  Prompt Tokens: {usage['prompt']:,}")
                print(f"  Completion Tokens: {usage['completion']:,}")
                print(f"  Total Tokens: {usage['total']:,}")
                print(f"  Cost: ${usage['cost']:.6f}")
        
        # Compute total directly (don't call get_total_usage to avoid deadlock)
        total = {
            "prompt": sum(v["prompt"] for v in GLOBAL_TOKEN_USAGE.values()),
            "completion": sum(v["completion"] for v in GLOBAL_TOKEN_USAGE.values()),
            "total": sum(v["total"] for v in GLOBAL_TOKEN_USAGE.values()),
            "cost": sum(v["cost"] for v in GLOBAL_TOKEN_USAGE.values()),
            "calls": sum(v["calls"] for v in GLOBAL_TOKEN_USAGE.values()),
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
                prompt_tokens += usage.get("prompt_tokens", 0)
                completion_tokens += usage.get("completion_tokens", 0)
                total_tokens += usage.get("total_tokens", 0)
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
