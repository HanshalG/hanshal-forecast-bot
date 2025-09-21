import datetime
import json

from src.utils import read_prompt, call_llm, run_research

# Constants
LLM_MODEL = "openai/gpt-5-mini"

def _format_prompt(question_details: dict, summary_report: str, today: str, timezone_str: str, time_horizon: str) -> str:
    template = read_prompt("base_rate_agent_prompt.txt")
    content = template.format(
        title=question_details["title"],
        background=question_details["description"],
        resolution_criteria=question_details["resolution_criteria"],
        fine_print=question_details["fine_print"],
        summary_report=summary_report,
        today=today,
        timezone=timezone_str,
        time_horizon=time_horizon,
    )
    return content

def _parse_agent_json(answer: str) -> dict:
    # Try to extract JSON from the response, even if there's extra text
    import re

    # Look for JSON-like content in the response
    json_match = re.search(r'\{.*\}', answer, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # If that fails, try the whole response
    try:
        return json.loads(answer)
    except json.JSONDecodeError:
        raise ValueError(f"Could not extract valid JSON from agent response: {answer[:200]}...")

def _validate_agent_json(parsed: dict) -> tuple[int, int, float]:
    if not isinstance(parsed, dict):
        raise ValueError("Parsed agent output is not a dict")

    counts = parsed.get("counts")
    if not counts or not isinstance(counts, dict):
        raise ValueError("Missing or invalid counts")

    k_yes = counts.get("k_yes")
    n_total = counts.get("n_total")
    if not isinstance(k_yes, int) or not isinstance(n_total, int) or n_total <= 0 or not (0 <= k_yes <= n_total):
        raise ValueError("Invalid k_yes or n_total")

    mapping_discount = parsed.get("mapping_discount")
    if not isinstance(mapping_discount, (int, float)) or not (0.3 <= mapping_discount <= 1.0):
        raise ValueError("Invalid mapping_discount")
    return k_yes, n_total, float(mapping_discount)

def _compute_prior(k_yes: int, n_total: int, mapping_discount: float) -> float:
    base_rate = k_yes / n_total
    prior = base_rate * mapping_discount
    # Clamp to [0.01, 0.99]
    return max(0.01, min(0.99, prior))

async def compute_base_rate_prior_with_research(question_details: dict, summary_report: str) -> float:
    # Derive time_horizon
    if "scheduled_close_time" in question_details:
        time_horizon = str(question_details["scheduled_close_time"])
    else:
        time_horizon = "until close date"

    # Get today and timezone
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        timezone_str = datetime.datetime.now().astimezone().tzinfo.key  # type: ignore[attr-defined]
    except Exception:
        timezone_str = str(datetime.datetime.now().astimezone().tzinfo)

    # Format prompt
    prompt = _format_prompt(question_details, summary_report, today, timezone_str, time_horizon)

    # Call LLM
    answer = await call_llm(prompt, LLM_MODEL, 0.2)

    # Parse
    try:
        parsed = _parse_agent_json(answer)
    except ValueError:
        # Retry once with a stricter instruction
        retry_prompt = prompt + "\n\nYou returned invalid JSON. Please respond with valid JSON only."
        answer = await call_llm(retry_prompt, LLM_MODEL, 0.2)
        parsed = _parse_agent_json(answer)

    # Validate
    k_yes, n_total, mapping_discount = _validate_agent_json(parsed)

    # Compute
    prior = _compute_prior(k_yes, n_total, mapping_discount)

    return prior
