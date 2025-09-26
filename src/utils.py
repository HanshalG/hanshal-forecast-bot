from dotenv import load_dotenv
from pathlib import Path
import asyncio
import os
import re
import time
import json
import numpy as np
from asknews_sdk import AskNewsSDK
from openai import AsyncOpenAI
from exa_py import Exa

load_dotenv()

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Constants
CONCURRENT_REQUESTS_LIMIT = 10
llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)
EXA_CONCURRENT_REQUESTS_LIMIT = 10

# Simple per-1k token pricing map for rough cost estimates (USD)
# Prices derived from public references; adjust if your provider differs.
# Reference units converted from per-1M to per-1k tokens.
LLM_PRICES_PER_1K: dict[str, dict[str, float]] = {
    # Keys: model name; Values: {"input": price_per_1k_input_tokens, "output": price_per_1k_output_tokens}
    "gpt-5": {"input": 0.00125, "output": 0.0100},     # $1.25 / $10 per 1M
    "gpt-5-mini": {"input": 0.00025, "output": 0.0020}, # $0.25 / $2 per 1M
    "gpt-5-nano": {"input": 0.00005, "output": 0.0004}, # $0.05 / $0.40 per 1M
}

# Running total estimated LLM cost (USD)
LLM_TOTAL_COST_USD: float = 0.0
llm_cost_lock = asyncio.Lock()

def _canonicalize_model_for_pricing(model_name: str) -> str:
    """Return a canonical key used in LLM_PRICES_PER_1K.

    Normalizes provider prefixes (e.g., "openai/gpt-5-mini" -> "gpt-5-mini") and
    collapses model variants to their family when possible (e.g., "gpt-5-xyz" -> "gpt-5").
    """
    try:
        name = str(model_name).strip().lower()
        # Remove provider prefix if present
        if "/" in name:
            name = name.split("/")[-1]
        # Map by family precedence
        if name.startswith("gpt-5-nano"):
            return "gpt-5-nano"
        if name.startswith("gpt-5-mini"):
            return "gpt-5-mini"
        if name.startswith("gpt-5"):
            return "gpt-5"
        return str(model_name)
    except Exception:
        return str(model_name)

# Environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
EXA_API_KEY = os.getenv("EXA_API_KEY")

exa = Exa(api_key=EXA_API_KEY)

def _asknews_fetch_articles_with_retries(
    ask: AskNewsSDK,
    *,
    query: str,
    n_articles: int,
    return_type: str,
    strategy: str,
    retries: int = 2,
    base_sleep_seconds: float = 0.5,
):
    """Fetch AskNews articles with up to `retries` retries and raise on final failure."""
    last_err: Exception | None = None
    attempts = retries + 1
    for attempt in range(attempts):
        try:
            resp = ask.news.search_news(
                query=query,
                n_articles=n_articles,
                return_type=return_type,
                strategy=strategy,
            )
            try:
                articles = resp.as_dicts  # type: ignore[attr-defined]
            except Exception:
                articles = None
            if not articles:
                articles = []
            return articles
        except Exception as e:
            last_err = e
            if attempt < attempts - 1:
                sleep_s = base_sleep_seconds * (2 ** attempt)
                print(
                    f"AskNews retry {attempt + 1}/{retries} for strategy='{strategy}' query='{query[:60]}...': {e}"
                )
                time.sleep(sleep_s)
            else:
                break
    raise RuntimeError(f"AskNews failed after {retries} retries for strategy='{strategy}': {last_err}")

def _sanitize_question_text(text: str) -> str:
    """Remove wrapping curly braces or quotes from a question string.

    Examples:
      "{What happened?}" -> "What happened?"
      '"What happened?"' -> "What happened?"
    """
    try:
        s = str(text).strip()
        # Strip one layer of surrounding braces if present
        if len(s) >= 2 and s[0] == "{" and s[-1] == "}":
            s = s[1:-1].strip()
        # Strip one layer of surrounding matching quotes if present
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1].strip()
        return s
    except Exception:
        return str(text).strip()

async def get_historical_research_questions(content: str) -> list[str]:
    response = await call_llm(content, "gpt-5-mini", 0.3, "medium")

    # Only consider the portion after "Search questions:" to avoid the Analysis section
    lower_response = response.lower()
    start_index = lower_response.find("search questions:")
    tail = response[start_index:] if start_index != -1 else response

    # Extract questions that begin with [Question]
    pattern = re.compile(r"^\s*\[Question\]\s*(.+?)\s*$", re.MULTILINE)
    raw_questions = [match.strip() for match in pattern.findall(tail) if match.strip()]
    questions = [_sanitize_question_text(q) for q in raw_questions]
    questions = [q for q in questions if q]
    return questions[:10]

async def get_current_research_questions(content: str) -> list[str]:
    """Mirror of get_historical_research_questions for current-focused questions.

    The prompt format is identical: an Analysis section, then a "Search questions:" section
    with lines that start with [Question]. We only parse the questions section.
    """
    response = await call_llm(content, "gpt-5-mini", 0.3, "medium")

    lower_response = response.lower()
    start_index = lower_response.find("search questions:")
    tail = response[start_index:] if start_index != -1 else response

    pattern = re.compile(r"^\s*\[Question\]\s*(.+?)\s*$", re.MULTILINE)
    raw_questions = [match.strip() for match in pattern.findall(tail) if match.strip()]
    questions = [_sanitize_question_text(q) for q in raw_questions]
    questions = [q for q in questions if q]
    return questions[:10]

def get_exa_answers(questions: list[str]) -> dict[str, str]:
    """Return a mapping of question -> formatted Exa answer (with sources)."""
    answers: dict[str, str] = {}
    for question in questions:
        try:
            response = exa.answer(question)

            answer_text = None
            if response is not None:
                answer_text = getattr(response, "answer", None)
                if answer_text is None and isinstance(response, dict):
                    answer_text = response.get("answer") or response.get("text")

            citation_lines: list[str] = []
            citations = getattr(response, "citations", None)
            if citations and isinstance(citations, list):
                for idx, c in enumerate(citations, start=1):
                    url = getattr(c, "url", None) if hasattr(c, "url") else (
                        c.get("url") if isinstance(c, dict) else None
                    )
                    title = getattr(c, "title", None) if hasattr(c, "title") else (
                        c.get("title") if isinstance(c, dict) else None
                    )
                    if url:
                        if title:
                            citation_lines.append(f"[{idx}] {title} - {url}")
                        else:
                            citation_lines.append(f"[{idx}] {url}")

            formatted_answer = ""
            if answer_text:
                if citation_lines:
                    formatted_answer = f"{answer_text}\n\nSources:\n" + "\n".join(citation_lines)
                else:
                    formatted_answer = answer_text
            else:
                if citation_lines:
                    formatted_answer = "Sources:\n" + "\n".join(citation_lines)
                else:
                    formatted_answer = "No answer found."

            answers[question] = formatted_answer
        except Exception as e:
            answers[question] = f"Error fetching Exa answer: {e}"

    return answers

def _format_exa_answer_from_response(response) -> str:
    answer_text = None
    if response is not None:
        answer_text = getattr(response, "answer", None)
        if answer_text is None and isinstance(response, dict):
            answer_text = response.get("answer") or response.get("text")

    citation_lines: list[str] = []
    citations = getattr(response, "citations", None)
    if citations and isinstance(citations, list):
        for idx, c in enumerate(citations, start=1):
            url = getattr(c, "url", None) if hasattr(c, "url") else (
                c.get("url") if isinstance(c, dict) else None
            )
            title = getattr(c, "title", None) if hasattr(c, "title") else (
                c.get("title") if isinstance(c, dict) else None
            )
            if url:
                if title:
                    citation_lines.append(f"[{idx}] {title} - {url}")
                else:
                    citation_lines.append(f"[{idx}] {url}")

    formatted_answer = ""
    if answer_text:
        if citation_lines:
            formatted_answer = f"{answer_text}\n\nSources:\n" + "\n".join(citation_lines)
        else:
            formatted_answer = answer_text
    else:
        if citation_lines:
            formatted_answer = "Sources:\n" + "\n".join(citation_lines)
        else:
            formatted_answer = "No answer found."
    return formatted_answer

def _get_exa_answer_sync(question: str) -> str:
    try:
        response = exa.answer(question)
        result = _format_exa_answer_from_response(response)
        print(f"Received Exa answer: {result[:50]}...")
        return result
    except Exception as e:
        return f"Error fetching Exa answer: {e}"

async def get_exa_answers_async(questions: list[str], *, max_concurrency: int = EXA_CONCURRENT_REQUESTS_LIMIT) -> dict[str, str]:
    """Return mapping of question -> formatted Exa answer using concurrent threads.

    Uses asyncio.to_thread to run the sync Exa client without blocking the event loop.
    """
    if not questions:
        return {}

    print(f"Getting Exa answers async for {len(questions)} questions")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _worker(q: str) -> tuple[str, str]:
        async with semaphore:
            ans = await asyncio.to_thread(_get_exa_answer_sync, q)
            return q, ans

    results = await asyncio.gather(*[_worker(q) for q in questions], return_exceptions=True)
    answers: dict[str, str] = {}
    for item in results:
        if isinstance(item, Exception):
            # Should not happen often since we capture per-call errors, but guard anyway
            continue
        q, ans = item
        answers[q] = ans
    return answers

async def get_exa_research_report(content: str) -> str:
    """Get research report by asking questions to Exa."""
    # Get research questions from LLM
    questions_response = await call_llm(content, "gpt-5-mini", 0.3, "medium")

    #print("questions_response", questions_response)
    # Extract questions from response
    questions = [q.strip() for q in questions_response.split('Question:') if q.strip()]
    questions = [_sanitize_question_text(q) for q in questions]
    questions = [q for q in questions if q]
    questions = questions[:10] # limit the number of questions to 10
    # for question in questions:
    #     print("question", question)
    #     print("--------------------------------")

    # Get answers for each question via helper
    answers_by_question = get_exa_answers(questions)

    # Build report preserving question order
    reports = []
    for q in questions:
        reports.append("\n\n" + q)
        reports.append(answers_by_question.get(q, "No answer found."))

    result = "\n\n".join(reports) if reports else "No research results found."
    print("Exa Research Report:\n\n", result)
    # Combine all reports
    return result

def read_prompt(filename: str) -> str:
    """Read a prompt template from the prompts directory."""
    path = PROMPTS_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {path}")

async def call_llm(prompt: str, model: str, temperature: float, reasoning_effort: str = "medium") -> str:
    """
    Makes a completion request to OpenRouter (OpenAI SDK compatible) with concurrent
    request limiting and retry/backoff for transient API/JSON decode errors.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    async with llm_rate_limiter:
        max_attempts = 3
        base_sleep = 0.5
        last_err: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            start_time = time.time()
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    stream=False,
                )
                elapsed_s = time.time() - start_time

                # Extract token usage if available
                prompt_tokens = None
                completion_tokens = None
                total_tokens = None
                try:
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
                        completion_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
                        total_tokens = getattr(usage, "total_tokens", None)
                except Exception:
                    pass

                # Estimate cost using simple per-1k pricing if configured
                est_cost_usd = None
                pricing = LLM_PRICES_PER_1K.get(_canonicalize_model_for_pricing(model))
                if pricing is not None and (prompt_tokens is not None or completion_tokens is not None):
                    try:
                        pt = float(prompt_tokens or 0)
                        ct = float(completion_tokens or 0)
                        in_cost = (pt / 1000.0) * float(pricing.get("input", 0.0))
                        out_cost = (ct / 1000.0) * float(pricing.get("output", 0.0))
                        est_cost_usd = in_cost + out_cost
                    except Exception:
                        est_cost_usd = None

                # Update running total
                total_after_str = "n/a"
                if est_cost_usd is not None:
                    try:
                        global LLM_TOTAL_COST_USD
                        async with llm_cost_lock:
                            LLM_TOTAL_COST_USD += est_cost_usd
                            total_after_str = f"${LLM_TOTAL_COST_USD:.4f}"
                    except Exception:
                        pass

                # Print simple logging line
                try:
                    pt_str = str(prompt_tokens) if prompt_tokens is not None else "?"
                    ct_str = str(completion_tokens) if completion_tokens is not None else "?"
                    tt_str = str(total_tokens) if total_tokens is not None else "?"
                    cost_str = (f"${est_cost_usd:.4f}" if est_cost_usd is not None else "n/a")
                    print(
                        f"LLM call | model={model} temp={temperature} "
                        f"tokens: prompt={pt_str} completion={ct_str} total={tt_str} "
                        f"time={elapsed_s:.2f}s est_cost={cost_str} total_cost={total_after_str}"
                    )
                except Exception:
                    # Best-effort logging; never fail the call due to logging
                    pass

                answer = response.choices[0].message.content
                if answer is None:
                    raise ValueError("No answer returned from LLM")
                return answer
            except Exception as e:
                last_err = e
                if attempt < max_attempts:
                    sleep_s = base_sleep * (2 ** (attempt - 1))
                    print(f"LLM call retry {attempt}/{max_attempts - 1} for model={model}: {e}")
                    try:
                        await asyncio.sleep(sleep_s)
                    except Exception:
                        pass
                    continue
                break

        # Final failure
        assert last_err is not None
        raise last_err

async def call_asknews_async(question_or_details: str | dict) -> str:
    """
    Use the AskNews `news` endpoint to get news context for your query.
    The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
    """
    if not ASKNEWS_CLIENT_ID or not ASKNEWS_SECRET:
        raise ValueError("ASKNEWS_CLIENT_ID and ASKNEWS_SECRET environment variables must be set")

    ask = AskNewsSDK(
        client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_SECRET, scopes=set(["news"])
    )

    # Derive the text query and optional full question_details for filtering
    question_details: dict | None = None
    if isinstance(question_or_details, dict):
        question_text = question_or_details.get("title") or question_or_details.get("question") or ""
        question_details = question_or_details
    else:
        question_text = str(question_or_details)

    # get the latest news related to the query (within the past 48 hours) with retries
    hot_articles = _asknews_fetch_articles_with_retries(
        ask,
        query=question_text,
        n_articles=6,
        return_type="both",
        strategy="latest news",
        retries=2,
    )

    time.sleep(10)

    #get context from the "historical" database that contains a news archive going back to 2023
    historical_articles = _asknews_fetch_articles_with_retries(
        ask,
        query=question_text,
        n_articles=10,
        return_type="both",
        strategy="news knowledge",
        retries=2,
    )

    # Coerce SDK objects to dicts
    def _coerce_article_to_dict_local(a) -> dict:
        if isinstance(a, dict):
            return a
        try:
            return dict(getattr(a, "__dict__", {}))
        except Exception:
            return {"raw": str(a)}

    hot_articles = [_coerce_article_to_dict_local(a) for a in (hot_articles or [])]
    historical_articles = [_coerce_article_to_dict_local(a) for a in (historical_articles or [])]

    all_articles = hot_articles + historical_articles

    if len(all_articles) == 0:
        return "Here are the relevant news articles:\n\nNo articles were found.\n\n"

    kept_articles = all_articles
    if question_details is not None:
        # Apply LLM-based relevance filter
        try:
            import copy as _copy
            original = [_copy.copy(a) for a in all_articles]
            kept_articles = await filter_relevant_asknews_articles(question_details, all_articles)
            # Debugging: show removed articles
            def _key(ad: dict) -> tuple:
                return (
                    ad.get("eng_title") or ad.get("title") or "",
                    ad.get("article_url") or ad.get("url") or "",
                )
            original_keys = {_key(a): a for a in original}
            kept_keys = {_key(a) for a in kept_articles}
            removed_items = [v for k, v in original_keys.items() if k not in kept_keys]
            total_raw = len(original)
            kept_raw = len(kept_articles)
            removed_raw = total_raw - kept_raw
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(
                f"AskNews: total fetched={total_raw}, kept={kept_raw}, removed={removed_raw} (unique removed shown below: {len(removed_items)})"
            )
            for i, art in enumerate(removed_items, start=1):
                title = art.get("eng_title") or art.get("title") or "(no title)"
                url = art.get("article_url") or art.get("url") or ""
                print(f"Removed #{i}: {title} | {url}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        except Exception as e:
            print(f"AskNews relevance filtering failed: {e}. Proceeding without filtering.")
            kept_articles = all_articles

    # Sort kept by pub_date desc if available
    try:
        kept_articles = sorted(
            kept_articles,
            key=lambda x: x.get("pub_date"),
            reverse=True,
        )
    except Exception:
        pass

    formatted_articles = "Here are the relevant news articles:\n\n"
    try:
        for article in kept_articles:
            pub_date = article.get("pub_date")
            try:
                pub_date_str = pub_date.strftime("%B %d, %Y %I:%M %p") if hasattr(pub_date, "strftime") else str(pub_date or "")
            except Exception:
                pub_date_str = str(pub_date or "")
            title = article.get("eng_title") or article.get("title") or "(no title)"
            summary = article.get("summary") or article.get("eng_summary") or ""
            language = article.get("language") or ""
            source_id = article.get("source_id") or article.get("source") or ""
            url = article.get("article_url") or article.get("url") or ""
            formatted_articles += (
                f"**{title}**\n{summary}\n"
                f"Original language: {language}\n"
                f"Publish date: {pub_date_str}\n"
                f"Source:[{source_id}]({url})\n\n"
            )
    except Exception as e:
        print(f"AskNews formatting failed: {e}")

    return formatted_articles


def call_asknews(question_or_details: str | dict) -> str:
    """Synchronous wrapper for AskNews retrieval and filtering.

    If called inside an active event loop, falls back to no-filter mode to avoid loop issues.
    Prefer using `call_asknews_async` in async contexts.
    """
    try:
        loop = asyncio.get_running_loop()
        # We are running inside an event loop; do a no-filter fallback to avoid deadlocks.
        # Use the original fetch/format path without applying the async filter.
        # NOTE: Async paths (e.g., inside_view) should call call_asknews_async instead.
    except RuntimeError:
        # No running loop; safe to run the async version
        return asyncio.run(call_asknews_async(question_or_details))

    # Fallback path (no filtering) when inside loop; replicate minimal original behavior
    if not ASKNEWS_CLIENT_ID or not ASKNEWS_SECRET:
        raise ValueError("ASKNEWS_CLIENT_ID and ASKNEWS_SECRET environment variables must be set")

    ask = AskNewsSDK(
        client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_SECRET, scopes=set(["news"])
    )

    if isinstance(question_or_details, dict):
        question_text = question_or_details.get("title") or question_or_details.get("question") or ""
    else:
        question_text = str(question_or_details)

    hot_articles = _asknews_fetch_articles_with_retries(
        ask,
        query=question_text,
        n_articles=6,
        return_type="both",
        strategy="latest news",
        retries=2,
    )
    time.sleep(10)
    historical_articles = _asknews_fetch_articles_with_retries(
        ask,
        query=question_text,
        n_articles=10,
        return_type="both",
        strategy="news knowledge",
        retries=2,
    )
    def _coerce_article_to_dict_local(a) -> dict:
        if isinstance(a, dict):
            return a
        try:
            return dict(getattr(a, "__dict__", {}))
        except Exception:
            return {"raw": str(a)}
    hot_articles = [_coerce_article_to_dict_local(a) for a in (hot_articles or [])]
    historical_articles = [_coerce_article_to_dict_local(a) for a in (historical_articles or [])]
    kept_articles = hot_articles + historical_articles

    if len(kept_articles) == 0:
        return "Here are the relevant news articles:\n\nNo articles were found.\n\n"

    try:
        kept_articles = sorted(
            kept_articles,
            key=lambda x: x.get("pub_date"),
            reverse=True,
        )
    except Exception:
        pass

    # Debug note: we are inside event loop, filter disabled
    print("AskNews: running in existing event loop; relevance filtering disabled in sync wrapper. Use call_asknews_async for filtering.")

    formatted_articles = "Here are the relevant news articles:\n\n"
    for article in kept_articles:
        pub_date = article.get("pub_date")
        try:
            pub_date_str = pub_date.strftime("%B %d, %Y %I:%M %p") if hasattr(pub_date, "strftime") else str(pub_date or "")
        except Exception:
            pub_date_str = str(pub_date or "")
        title = article.get("eng_title") or article.get("title") or "(no title)"
        summary = article.get("summary") or article.get("eng_summary") or ""
        language = article.get("language") or ""
        source_id = article.get("source_id") or article.get("source") or ""
        url = article.get("article_url") or article.get("url") or ""
        formatted_articles += (
            f"**{title}**\n{summary}\n"
            f"Original language: {language}\n"
            f"Publish date: {pub_date_str}\n"
            f"Source:[{source_id}]({url})\n\n"
        )

    return formatted_articles

async def run_research_async(question_or_details) -> str:
    if ASKNEWS_CLIENT_ID and ASKNEWS_SECRET:
        research = await call_asknews_async(question_or_details)
    else:
        research = "No research done"

    print(f"########################\nResearch Found:\n{research}\n########################")
    return research


def run_research(question_or_details) -> str:
    research = ""
    if ASKNEWS_CLIENT_ID and ASKNEWS_SECRET:
        research = call_asknews(question_or_details)
    else:
        research = "No research done"

    print(f"########################\nResearch Found:\n{research}\n########################")

    return research


# -------------------- AskNews relevance filtering --------------------
async def filter_relevant_asknews_articles(
    question_details: dict,
    articles: list,
    *,
    model: str = "gpt-5-nano",
    temperature: float = 0.3,
) -> list[dict]:
    """Filter AskNews articles by relevance to a forecasting question using an LLM.

    Args:
        question_details: Metaculus question dict (expects keys like 'title', 'description', 'resolution_criteria', 'fine_print').
        articles: A list of AskNews article objects or dicts (from SDK). Mixed types are tolerated.
        model: LLM model name to use.
        temperature: LLM temperature.

    Returns:
        A list of article dicts that are deemed relevant. Each kept article is annotated with
        a 'relevance_reason' field summarizing why it is helpful for the forecast.
    """

    def _coerce_article_to_dict(a) -> dict:
        if isinstance(a, dict):
            return a
        try:
            return dict(getattr(a, "__dict__", {}))
        except Exception:
            return {"raw": str(a)}

    def _format_article_context(ad: dict) -> str:
        title = ad.get("eng_title") or ad.get("title") or ""
        summary = ad.get("summary") or ad.get("eng_summary") or ""
        language = ad.get("language") or ""
        source_id = ad.get("source_id") or ad.get("source") or ""
        url = ad.get("article_url") or ad.get("url") or ""
        pub_date = ad.get("pub_date")
        try:
            # If AskNews returns a datetime, format it; else fallback to str
            if hasattr(pub_date, "strftime"):
                pub_date_str = pub_date.strftime("%Y-%m-%d %H:%M %Z")
            else:
                pub_date_str = str(pub_date or "")
        except Exception:
            pub_date_str = str(pub_date or "")

        lines = []
        if title:
            lines.append(f"Title: {title}")
        if summary:
            lines.append(f"Summary: {summary}")
        if language:
            lines.append(f"Language: {language}")
        if pub_date_str:
            lines.append(f"Published: {pub_date_str}")
        if source_id:
            lines.append(f"Source: {source_id}")
        if url:
            lines.append(f"URL: {url}")
        return "\n".join(lines)

    def _build_prompt(q: dict, article_block: str) -> str:
        title = q.get("title", "")
        background = q.get("description", "")
        resolution_criteria = q.get("resolution_criteria", "")
        fine_print = q.get("fine_print", "")

        return (
            "You are assisting a probabilistic forecaster.\n"
            "Given a Metaculus question and a single news article, decide if the article is likely to help update the forecast (i.e., contains evidence relevant to resolution criteria or key drivers).\n\n"
            "Question:\n"
            f"- Title: {title}\n"
            f"- Background: {background}\n"
            f"- Resolution criteria: {resolution_criteria}\n"
            f"- Fine print: {fine_print}\n\n"
            "Article:\n"
            f"{article_block}\n\n"
            "Respond with strict JSON only, with this schema: {\n"
            "  \"relevant\": true|false,\n"
            "  \"reason\": string (brief, <= 1 sentence)\n"
            "}."
        )

    async def _judge(adict: dict) -> tuple[bool, dict]:
        article_block = _format_article_context(adict)
        prompt = _build_prompt(question_details, article_block)
        try:
            answer = await call_llm(prompt, model, temperature, "medium")
        except Exception:
            return False, adict

        relevant = False
        reason = ""
        try:
            parsed = None
            try:
                parsed = json.loads(answer)
            except Exception:
                # Try to extract JSON substring
                import re as _re
                m = _re.search(r"\{[\s\S]*\}", answer)
                if m:
                    parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                rv = parsed.get("relevant")
                if isinstance(rv, bool):
                    relevant = rv
                else:
                    # Fallback: accept strings like "yes"/"no"
                    if isinstance(rv, str):
                        relevant = rv.strip().lower() in {"yes", "true"}
                rs = parsed.get("reason")
                if isinstance(rs, str):
                    reason = rs.strip()
        except Exception:
            pass

        if relevant:
            # annotate and keep
            adict = dict(adict)
            adict["relevance_reason"] = reason or "Appears helpful for the forecast."
            return True, adict
        return False, adict

    coerced: list[dict] = [_coerce_article_to_dict(a) for a in articles]
    results = await asyncio.gather(*[_judge(a) for a in coerced])
    kept: list[dict] = [ad for ok, ad in results if ok]
    return kept

def extract_probability_from_response_as_percentage_not_decimal(
    forecast_text: str,
) -> float:
    # 1) Prefer a strict line style: "Probability: NN%" or "Probability: NN.N" (without %)
    #    Capture whether a percent sign was present to avoid misinterpreting 1% as 100%.
    strict_iter = re.finditer(
        r"^\s*Probability\s*:\s*([0-9]+(?:\.[0-9]+)?)(\s*%)?",
        forecast_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    candidates: list[float] = []
    for m in strict_iter:
        try:
            number_str = m.group(1)
            has_percent = bool(m.group(2))
            val = float(number_str)
            if has_percent:
                # Explicit percent given; use as-is
                candidates.append(val)
            else:
                # No percent sign — interpret values < 1.0 as decimals, otherwise as percents
                if 0.0 <= val < 1.0:
                    val *= 100.0
                candidates.append(val)
        except Exception:
            continue

    # 2) Any occurrence of a percent anywhere like "NN%" or "NN.N%"
    if not candidates:
        any_pct = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", forecast_text)
        for s in any_pct:
            try:
                val = float(s)
                candidates.append(val)
            except Exception:
                continue

    # 3) Allow textual forms like "NN percent" or "NN pct"
    if not candidates:
        textual = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(?:percent|pct)\b", forecast_text, flags=re.IGNORECASE)
        for s in textual:
            try:
                val = float(s)
                candidates.append(val)
            except Exception:
                continue

    if candidates:
        number = int(round(candidates[-1]))
        number = min(99, max(1, number))  # clamp between 1 and 99
        return number
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")

def extract_percentiles_from_response(forecast_text: str) -> dict:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_percentile_numbers(text) -> dict:
        pattern = r"^.*(?:P|p)ercentile.*$"
        number_pattern = r"-\s*(?:[^\d\-]*\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)|(\d+(?:,\d{3})*(?:\.\d+)?)"
        results = []

        for line in text.split("\n"):
            if re.match(pattern, line):
                numbers = re.findall(number_pattern, line)
                numbers_no_commas = [
                    next(num for num in match if num).replace(",", "")
                    for match in numbers
                ]
                numbers = [
                    float(num) if "." in num else int(num)
                    for num in numbers_no_commas
                ]
                if len(numbers) > 1:
                    first_number = numbers[0]
                    last_number = numbers[-1]
                    # Check if the original line had a negative sign before the last number
                    if "-" in line.split(":")[-1]:
                        last_number = -abs(last_number)
                    results.append((first_number, last_number))

        # Convert results to dictionary
        percentile_values = {}
        for first_num, second_num in results:
            key = first_num
            percentile_values[key] = second_num

        return percentile_values

    percentile_values = extract_percentile_numbers(forecast_text)

    # Fallback: also accept compact forms like "P10: 3.2" or "P95: 100"
    if len(percentile_values) == 0:
        alt: dict = {}
        p_line = re.compile(r"^\s*P(\d{1,2}|100)\s*:\s*(-?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*$", re.IGNORECASE)
        for line in forecast_text.split("\n"):
            m = p_line.match(line)
            if not m:
                continue
            try:
                p_key = int(m.group(1))
                val_str = m.group(2).replace(",", "")
                val = float(val_str) if "." in val_str else int(val_str)
                alt[p_key] = val
            except Exception:
                continue
        if len(alt) > 0:
            percentile_values = alt

    if len(percentile_values) > 0:
        return percentile_values
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")

def generate_continuous_cdf(
    percentile_values: dict,
    question_type: str,
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float,
    lower_bound: float,
    zero_point: float | None,
    cdf_size: int,
) -> list[float]:
    """
    Returns: list[float]: A list of 201 float values representing the CDF.
    """

    percentile_max = max(float(key) for key in percentile_values.keys())
    percentile_min = min(float(key) for key in percentile_values.keys())
    range_min = lower_bound
    range_max = upper_bound
    range_size = range_max - range_min
    buffer = 1 if range_size > 100 else 0.01 * range_size

    # Adjust any values that are exactly at the bounds
    for percentile, value in list(percentile_values.items()):
        if not open_lower_bound and value <= range_min + buffer:
            percentile_values[percentile] = range_min + buffer
        if not open_upper_bound and value >= range_max - buffer:
            percentile_values[percentile] = range_max - buffer

    # Set cdf values outside range
    if open_upper_bound:
        if range_max > percentile_values[percentile_max]:
            percentile_values[int(100 - (0.5 * (100 - percentile_max)))] = range_max
    else:
        percentile_values[100] = range_max

    # Set cdf values outside range
    if open_lower_bound:
        if range_min < percentile_values[percentile_min]:
            percentile_values[int(0.5 * percentile_min)] = range_min
    else:
        percentile_values[0] = range_min

    sorted_percentile_values = dict(sorted(percentile_values.items()))

    # Normalize percentile keys
    normalized_percentile_values = {}
    for key, value in sorted_percentile_values.items():
        percentile = float(key) / 100
        normalized_percentile_values[percentile] = value


    value_percentiles = {
        value: key for key, value in normalized_percentile_values.items()
    }

    # function for log scaled questions
    def generate_cdf_locations(range_min, range_max, zero_point):
        if zero_point is None:
            scale = lambda x: range_min + (range_max - range_min) * x
        else:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            scale = lambda x: range_min + (range_max - range_min) * (
                deriv_ratio**x - 1
            ) / (deriv_ratio - 1)
        return [scale(x) for x in np.linspace(0, 1, cdf_size)]

    cdf_xaxis = generate_cdf_locations(range_min, range_max, zero_point)

    def linear_interpolation(x_values, xy_pairs):
        # Sort the xy_pairs by x-values
        sorted_pairs = sorted(xy_pairs.items())

        # Extract sorted x and y values
        known_x = [pair[0] for pair in sorted_pairs]
        known_y = [pair[1] for pair in sorted_pairs]

        # Initialize the result list
        y_values = []

        for x in x_values:
            # Check if x is exactly in the known x values
            if x in known_x:
                y_values.append(known_y[known_x.index(x)])
            else:
                # Find the indices of the two nearest known x-values
                i = 0
                while i < len(known_x) and known_x[i] < x:
                    i += 1

                list_index_2 = i

                # If x is outside the range of known x-values, use the nearest endpoint
                if i == 0:
                    y_values.append(known_y[0])
                elif i == len(known_x):
                    y_values.append(known_y[-1])
                else:
                    # Perform linear interpolation
                    x0, x1 = known_x[i - 1], known_x[i]
                    y0, y1 = known_y[i - 1], known_y[i]

                    # Linear interpolation formula
                    y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                    y_values.append(y)

        return y_values

    continuous_cdf = linear_interpolation(cdf_xaxis, value_percentiles)
    # --- Enforce Metaculus API constraints on the CDF ---
    # Length is already cdf_size by construction of cdf_xaxis
    n = len(continuous_cdf)
    if n != cdf_size:
        # In the unexpected case lengths diverge, resample by truncating/padding last value
        if n > cdf_size:
            continuous_cdf = continuous_cdf[:cdf_size]
        else:
            last_val = continuous_cdf[-1] if n > 0 else (0.0 if not open_lower_bound else 0.001)
            continuous_cdf = continuous_cdf + [last_val] * (cdf_size - n)
        n = len(continuous_cdf)

    # Clip to [0,1]
    continuous_cdf = [min(1.0, max(0.0, float(v))) for v in continuous_cdf]

    # Enforce endpoint constraints
    lower_min = 0.0 if not open_lower_bound else 0.001
    upper_max = 1.0 if not open_upper_bound else 0.999

    # Set endpoints according to bounds
    continuous_cdf[0] = lower_min if not open_lower_bound else max(continuous_cdf[0], lower_min)
    continuous_cdf[-1] = upper_max if not open_upper_bound else min(continuous_cdf[-1], upper_max)

    # Ensure strict monotonicity with tiny minimal increment
    # Use ~1% total mass within range distributed across steps as a lower bound
    min_delta = max(1e-6, 0.01 / max(1, cdf_size))

    # Forward pass: ensure we can still reach the fixed last value with remaining min steps
    for i in range(1, n - 1):
        remaining_steps = (n - 1) - i
        # Max allowed to leave room for remaining minimal increments up to the final value
        max_allowed_here = continuous_cdf[-1] - remaining_steps * min_delta
        desired = max(continuous_cdf[i], continuous_cdf[i - 1] + min_delta)
        continuous_cdf[i] = min(desired, max_allowed_here)

    # Re-clip to [0,1] and ensure last abides the cap
    continuous_cdf = [min(1.0, max(0.0, float(v))) for v in continuous_cdf]
    continuous_cdf[0] = max(continuous_cdf[0], lower_min)
    continuous_cdf[-1] = min(continuous_cdf[-1], upper_max)

    return continuous_cdf

def extract_option_probabilities_from_response(forecast_text: str, options) -> float:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_option_probabilities(text):

        # Number extraction pattern
        number_pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?"

        results = []

        # Iterate through each line in the text
        for line in text.split("\n"):
            # Extract all numbers from the line
            numbers = re.findall(number_pattern, line)
            numbers_no_commas = [num.replace(",", "") for num in numbers]
            # Convert strings to float or int
            numbers = [
                float(num) if "." in num else int(num) for num in numbers_no_commas
            ]
            # Add the tuple of numbers to results
            if len(numbers) >= 1:
                last_number = numbers[-1]
                results.append(last_number)

        return results

    # Prefer a bracketed list if present: "Probabilities: [x, y, ...]"
    bracket_match = re.findall(r"Probabilities\s*:\s*\[([^\]]+)\]", forecast_text, flags=re.IGNORECASE)
    option_probabilities = []
    if bracket_match:
        nums = re.findall(r"-?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?", bracket_match[-1])
        option_probabilities = [float(n.replace(",", "")) for n in nums]
    else:
        option_probabilities = extract_option_probabilities(forecast_text)

    NUM_OPTIONS = len(options)

    if len(option_probabilities) > 0:
        # return the last NUM_OPTIONS items
        return option_probabilities[-NUM_OPTIONS:]
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")