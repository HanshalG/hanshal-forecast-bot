import datetime
import json
from typing import Any, Dict, List

import src.utils as utils
from string import Template

LLM_MODEL = "openai/gpt-5-mini"

def _format_prompt(question_details: dict, summary_report: str, today: str, timezone_str: str, time_horizon: str) -> str:
    raw = utils.read_prompt("evidence_agent_prompt.txt")
    # Convert only known placeholders to Template-style to avoid clashing with JSON braces
    keys = [
        "title",
        "background",
        "resolution_criteria",
        "fine_print",
        "summary_report",
        "today",
        "timezone",
        "time_horizon",
    ]
    for k in keys:
        raw = raw.replace("{" + k + "}", "${" + k + "}")
    t = Template(raw)
    content = t.safe_substitute(
        title=question_details.get("title", ""),
        background=question_details.get("description", ""),
        resolution_criteria=question_details.get("resolution_criteria", ""),
        fine_print=question_details.get("fine_print", ""),
        summary_report=summary_report,
        today=today,
        timezone=timezone_str,
        time_horizon=time_horizon,
    )
    return content


def _parse_agent_json(answer: str) -> dict:
    import re

    json_match = re.search(r"\{.*\}", answer, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return json.loads(answer)


def _validate_and_normalize_items(parsed: dict) -> List[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        raise ValueError("Parsed output is not a dict")

    items = parsed.get("evidence")
    if not isinstance(items, list):
        raise ValueError("Missing or invalid 'evidence' list")

    validated: List[Dict[str, Any]] = []

    for raw in items:
        if not isinstance(raw, dict):
            continue

        text = raw.get("text")
        direction = raw.get("direction")
        lr = raw.get("likelihood_ratio")
        priced_in = raw.get("priced_in")
        confidence = raw.get("confidence", 0.5)
        sources = raw.get("sources", [])

        if not isinstance(text, str) or len(text.strip()) == 0:
            continue
        if direction not in {"for", "against"}:
            continue
        try:
            lr_val = float(lr)
        except Exception:
            continue
        if lr_val <= 0:
            continue
        # clamp extreme LRs to reasonable bounds
        if lr_val < 0.05:
            lr_val = 0.05
        if lr_val > 20.0:
            lr_val = 20.0

        if not isinstance(priced_in, bool):
            continue
        try:
            conf_val = float(confidence)
        except Exception:
            conf_val = 0.5
        if conf_val < 0.0:
            conf_val = 0.0
        if conf_val > 1.0:
            conf_val = 1.0

        if not isinstance(sources, list):
            sources = []
        sources = [str(s) for s in sources]

        validated.append(
            {
                "text": text.strip(),
                "direction": direction,
                "likelihood_ratio": lr_val,
                "priced_in": priced_in,
                "confidence": conf_val,
                "sources": sources,
            }
        )

    if len(validated) == 0:
        raise ValueError("No valid evidence items produced by agent")

    return validated


async def generate_evidence(question_details: dict, summary_report: str | None = None) -> dict:
    # time horizon derivation
    time_horizon = str(question_details.get("scheduled_close_time", "until close date"))

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        timezone_str = datetime.datetime.now().astimezone().tzinfo.key  # type: ignore[attr-defined]
    except Exception:
        timezone_str = str(datetime.datetime.now().astimezone().tzinfo)

    if summary_report is None:
        summary_report = utils.run_research(question_details.get("title", ""))

    prompt = _format_prompt(question_details, summary_report, today, timezone_str, time_horizon)

    answer = await utils.call_llm(prompt, LLM_MODEL, 0.2)

    try:
        parsed = _parse_agent_json(answer)
    except Exception:
        retry_prompt = prompt + "\n\nYou returned invalid JSON. Please respond with valid JSON only that matches the required schema."
        answer = await utils.call_llm(retry_prompt, LLM_MODEL, 0.2)
        parsed = _parse_agent_json(answer)

    evidence_items = _validate_and_normalize_items(parsed)

    return {
        "evidence": evidence_items,
        "meta": {
            "today": today,
            "timezone": timezone_str,
            "model": LLM_MODEL,
        },
    }


