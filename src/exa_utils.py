import json
import os
from typing import Any, Dict, List, Tuple
import asyncio

from .eval.timebox import filter_items_before_as_of, today_string_for_prompt
from .utils import read_prompt, call_llm, exa

# Model Configuration from .env
INSIDE_VIEW_MODEL = os.getenv("INSIDE_VIEW_MODEL", "gpt-5-mini")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-5-nano")

HISTORICAL_QUERIES_TMPL = "historical_search_queries.txt"
 


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    try:
        if isinstance(d, dict):
            return d.get(key, default)
    except Exception:
        pass
    return default


def _coerce_list_of_str(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val if isinstance(x, (str, int, float))]
    if isinstance(val, (str, int, float)):
        return [str(val)]
    return []


def _parse_json_lines(s: str) -> List[Dict[str, Any]]:
    """Parse all JSON objects from a string (one per line).

    Tolerates surrounding text by extracting the first top-level {...} per line.
    Silently skips lines that fail to parse.
    """
    results: List[Dict[str, Any]] = []
    for raw_line in s.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Extract JSON object if the line contains extra commentary
        start = line.find("{")
        end = line.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue
        candidate = line[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                results.append(obj)
        except Exception:
            continue
    return results


def build_historical_queries_prompt(question_details: Dict[str, Any]) -> str:
    """Render the historical Exa search prompt using question details.

    Expected keys in question_details: title, description, resolution_criteria, fine_print.
    """
    template = read_prompt(HISTORICAL_QUERIES_TMPL)
    title = _safe_get(question_details, "title", "")
    background = _safe_get(question_details, "description", "")
    resolution_criteria = _safe_get(question_details, "resolution_criteria", "")
    fine_print = _safe_get(question_details, "fine_print", "")
    today = today_string_for_prompt(_safe_get(question_details, "as_of_time"))

    # Important: Do targeted replacements only to avoid breaking JSON braces in the template.
    rendered = template
    rendered = rendered.replace("{title}", str(title))
    rendered = rendered.replace("{background}", str(background))
    rendered = rendered.replace("{resolution_criteria}", str(resolution_criteria))
    rendered = rendered.replace("{fine_print}", str(fine_print))
    rendered = rendered.replace("{today}", str(today))
    return rendered


async def generate_historical_exa_queries(question_details: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Use the LLM to generate Exa /search JSON payloads for historical context.

    Returns a list of dicts with keys like: query, includeDomains, startPublishedDate, endPublishedDate, includeText, category (optional).
    """
    prompt = build_historical_queries_prompt(question_details)
    response = await call_llm(prompt, INSIDE_VIEW_MODEL, 0.3, "high")
    payloads = _parse_json_lines(response)
    return payloads


def _extract_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        if hasattr(obj, name):
            return getattr(obj, name)
    except Exception:
        pass
    try:
        if isinstance(obj, dict):
            return obj.get(name, default)
    except Exception:
        pass
    return default


def _format_exa_result_item(item: Any) -> Dict[str, Any]:
    title = _extract_attr(item, "title", "")
    url = _extract_attr(item, "url", "")
    # Summaries only (text or structured)
    summary_obj = _extract_attr(item, "summary", None)
    summary_text: str = ""
    summary_structured: Any = None
    if isinstance(summary_obj, dict):
        # Some SDKs return structured summary containing both text and json
        summary_text = (
            _extract_attr(summary_obj, "text")
            or _extract_attr(summary_obj, "summary")
            or ""
        )
        if not isinstance(summary_text, str):
            summary_text = ""
        maybe_json = _extract_attr(summary_obj, "json")
        if isinstance(maybe_json, (dict, list)):
            summary_structured = maybe_json
    elif isinstance(summary_obj, str):
        summary_text = summary_obj
    score = _extract_attr(item, "score", None)
    published = _extract_attr(item, "publishedDate", None)
    return {
        "title": title,
        "url": url,
        "summary": summary_text,
        "summary_structured": summary_structured,
        "score": score,
        "publishedDate": published,
    }


def _call_exa_search_and_contents(
    params: Dict[str, Any],
    *,
    num_results: int = 8,
    end_published_date_override: str | None = None,
) -> List[Dict[str, Any]]:
    """Call Exa search_and_contents requesting summaries only (use summaryQuery/schema if provided).

    The incoming params follow the LLM JSON schema. We map a safe subset to Exa arguments.
    We intentionally ignore ambiguous fields like category if unsure of Exa's accepted values.
    """
    query: str = str(_safe_get(params, "query", "")).strip()
    if not query:
        return []

    include_domains = _coerce_list_of_str(_safe_get(params, "includeDomains", []))
    include_text = _coerce_list_of_str(_safe_get(params, "includeText", []))
    start_published_date = _safe_get(params, "startPublishedDate")
    end_published_date = (
        end_published_date_override
        or _safe_get(params, "endPublishedDate")
        or os.getenv("EVAL_AS_OF_TIME")
    )

    # Build a conservative kwargs set. Exa SDK is permissive about snake_case names.
    kwargs: Dict[str, Any] = {
        "query": query,
        "num_results": num_results,
        "use_autoprompt": True,
    }
    if include_domains:
        kwargs["include_domains"] = include_domains
    if include_text:
        kwargs["include_text"] = include_text
    if isinstance(start_published_date, str) and start_published_date:
        kwargs["start_published_date"] = start_published_date
    if isinstance(end_published_date, str) and end_published_date:
        kwargs["end_published_date"] = end_published_date

    # Summaries-only retrieval: use summaryQuery when available
    summary_query = None
    sq_raw = _safe_get(params, "summaryQuery")
    if isinstance(sq_raw, str) and sq_raw.strip():
        summary_query = sq_raw.strip()

    # Prefer search_and_contents with requested summary options; fall back to two-step if needed
    try:
        # Summaries with steering query (or plain summary if none provided)
        try:
            summary_opts: Dict[str, Any] = {}
            if summary_query is not None:
                summary_opts["query"] = summary_query
            results = exa.search_and_contents(
                contents_options={"summary": summary_opts if summary_opts else True},
                **kwargs,
            )
        except Exception:
            # Fallback via direct kwarg signatures
            try:
                if summary_query is not None:
                    results = exa.search_and_contents(summary={"query": summary_query}, **kwargs)  # type: ignore[arg-type]
                else:
                    results = exa.search_and_contents(summary=True, **kwargs)  # type: ignore[arg-type]
            except Exception:
                results = exa.search_and_contents(summary=True, **kwargs)  # type: ignore[arg-type]
    except Exception:
        # Fallback: run search then fetch contents with requested summary options
        try:
            search_resp = exa.search(**kwargs)
            ids: List[str] = []
            for it in _extract_attr(search_resp, "results", []) or []:
                doc_id = _extract_attr(it, "id")
                if doc_id:
                    ids.append(str(doc_id))
            if not ids:
                return []
            # Try different get_contents signatures for summary retrieval
            try:
                summary_opts2: Dict[str, Any] = {}
                if summary_query is not None:
                    summary_opts2["query"] = summary_query
                results = exa.get_contents(ids, contents_options={"summary": summary_opts2 if summary_opts2 else True})
            except Exception:
                try:
                    if summary_query is not None:
                        results = exa.get_contents(ids, summary={"query": summary_query})  # type: ignore[arg-type]
                    else:
                        results = exa.get_contents(ids, summary=True)  # type: ignore[arg-type]
                except Exception:
                    results = exa.get_contents(ids, summary=True)  # type: ignore[arg-type]
        except Exception:
            return []

    # Normalize and format
    items = _extract_attr(results, "results", results) or []
    formatted: List[Dict[str, Any]] = []
    for it in items:
        formatted.append(_format_exa_result_item(it))
    filtered, removed = filter_items_before_as_of(
        formatted,
        as_of_time=end_published_date,
        keep_unparseable=False,
    )
    if removed > 0:
        print(f"Exa timebox filter removed {removed} item(s) newer than as_of={end_published_date}")
    return filtered


def run_historical_exa_research(question_details: Dict[str, Any], *, num_results: int = 8) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Generate Exa queries via LLM and return summaries for each query's results.

    Returns a list of (query_payload, results[]) tuples. Each result contains title, url, summary.
    """
    async def _inner() -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        payloads = await generate_historical_exa_queries(question_details)
        pairs: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
        end_published_date_override = None
        maybe_as_of = _safe_get(question_details, "as_of_time")
        if isinstance(maybe_as_of, str) and maybe_as_of.strip():
            end_published_date_override = maybe_as_of.strip()
        for p in payloads:
            results = _call_exa_search_and_contents(
                p,
                num_results=num_results,
                end_published_date_override=end_published_date_override,
            )
            try:
                filtered = await filter_relevant_exa_results(question_details, results)
            except Exception:
                filtered = results
            pairs.append((p, filtered))
        return pairs

    return asyncio.run(_inner())


async def filter_relevant_exa_results(
    question_details: Dict[str, Any],
    items: List[Dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
) -> List[Dict[str, Any]]:
    """Filter Exa results for relevance to the forecasting question using an LLM.

    Adds a 'relevance_reason' to kept items. Mirrors the AskNews filtering style.
    """
    # Use default if model not provided
    if model is None:
        model = SUMMARY_MODEL
        
    if not items:
        return []

    def _format_result_context(d: Dict[str, Any]) -> str:
        title = d.get("title") or ""
        url = d.get("url") or ""
        summary = d.get("summary") or ""
        lines: List[str] = []
        if title:
            lines.append(f"Title: {title}")
        if url:
            lines.append(f"URL: {url}")
        if summary:
            lines.append(f"Summary: {summary}")
        return "\n".join(lines)

    def _build_prompt(q: Dict[str, Any], result_block: str) -> str:
        title = q.get("title", "")
        background = q.get("description", "")
        resolution_criteria = q.get("resolution_criteria", "")
        fine_print = q.get("fine_print", "")
        return (
            "You help a probabilistic forecaster decide if a web result's SUMMARY alone contains useful, question-relevant information.\n"
            "Consider ONLY the summary text (not the full page). If the summary is generic, directory-like, navigational, or lacks concrete evidence relevant to resolution criteria or key drivers, mark it as not relevant.\n"
            "Return strict JSON only matching: {\n  \"relevant\": true|false,\n  \"reason\": string\n}. Keep it brief.\n\n"
            "Question:\n"
            f"- Title: {title}\n"
            f"- Background: {background}\n"
            f"- Resolution criteria: {resolution_criteria}\n"
            f"- Fine print: {fine_print}\n\n"
            "Result:\n"
            f"{result_block}\n\n"
            "Answer with only the JSON object."
        )

    async def _judge(d: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        ctx = _format_result_context(d)
        prompt = _build_prompt(question_details, ctx)
        try:
            answer = await call_llm(prompt, model, temperature, "low")
        except Exception:
            return False, d
        relevant = False
        reason = ""
        try:
            parsed = None
            try:
                parsed = json.loads(answer)
            except Exception:
                import re as _re
                m = _re.search(r"\{[\s\S]*\}", answer)
                if m:
                    parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                rv = parsed.get("relevant")
                if isinstance(rv, bool):
                    relevant = rv
                elif isinstance(rv, str):
                    relevant = rv.strip().lower() in {"yes", "true"}
                rs = parsed.get("reason")
                if isinstance(rs, str):
                    reason = rs.strip()
        except Exception:
            pass
        if relevant:
            kept = dict(d)
            kept["relevance_reason"] = reason or "Relevant to resolution or key drivers."
            return True, kept
        return False, d

    # Run judgments concurrently; call_llm has its own rate limiter
    results = await asyncio.gather(*[_judge(d) for d in items])
    kept_items: List[Dict[str, Any]] = [ad for ok, ad in results if ok]

    # Debug: show removed items
    removed_items: List[Dict[str, Any]] = [ad for ok, ad in results if not ok]  # type: ignore[assignment]
    if removed_items:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Exa: kept={len(kept_items)}, removed={len(removed_items)} (removed shown below)")
        for i, ad in enumerate(removed_items, start=1):
            t = ad.get("title") or "(no title)"
            u = ad.get("url") or ""
            print(f"Removed #{i}: {t} | {u}")
            s = (ad.get("summary") or "").strip()
            if s:
                snippet = s if len(s) <= 300 else s[:300] + "..."
                for line in snippet.splitlines()[:4]:
                    print(f"  {line}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return kept_items


if __name__ == "__main__":
    # Allow running this file directly via `python src/exa_utils.py`
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # Use local relative import after path fix
    from src.metaculus_utils import get_post_details  # type: ignore

    example_post_id = 39425
    print(f"Fetching post details for {example_post_id}...")
    post = get_post_details(example_post_id)
    question = post.get("question", {}) if isinstance(post, dict) else {}

    print("Generating Exa historical queries and fetching results...\n")
    try:
        results = run_historical_exa_research(question, num_results=6)

        for idx, (payload, items) in enumerate(results, start=1):
            print(f"Query #{idx}: {json.dumps(payload, ensure_ascii=False)}")
            print(f"  Results: {len(items)}")
            if not items:
                print()
                continue
            for j, it in enumerate(items, start=1):
                title = it.get("title") or "(no title)"
                url = it.get("url") or ""
                summary = (it.get("summary") or "").strip()
                summary_structured = it.get("summary_structured")
                print(f"  [{j}] {title}")
                if url:
                    print(f"      {url}")
                if it.get("relevance_reason"):
                    print(f"      Reason: {it['relevance_reason']}")
                if summary:
                    print("      Summary (text):")
                    s = summary if len(summary) <= 500 else summary[:500] + "..."
                    for line in s.splitlines()[:6]:
                        print(f"        {line}")
                if summary_structured is not None:
                    try:
                        js = json.dumps(summary_structured, ensure_ascii=False)
                    except Exception:
                        js = str(summary_structured)
                    print("      Summary (structured JSON):")
                    # print compact; wrap at ~500 chars
                    if len(js) > 500:
                        print(f"        {js[:500]}...")
                    else:
                        print(f"        {js}")
                print()
    except Exception as e:
        print("Run failed:", e)
        print("Environment variables required: OPENROUTER_API_KEY, EXA_API_KEY, METACULUS_TOKEN")

