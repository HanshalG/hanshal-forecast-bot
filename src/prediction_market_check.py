import os
import json
import math
import re
import asyncio
import aiohttp
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

# Load environment variables
load_dotenv()

# --- Configuration ---
LLM_MODEL = os.getenv("SUMMARY_MODEL", "gpt-5-nano")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

POLYMARKET_SCAN_LIMIT = int(os.getenv("POLYMARKET_SCAN_LIMIT", "1500"))
POLYMARKET_PAGE_SIZE = min(int(os.getenv("POLYMARKET_PAGE_SIZE", "100")), 100)
POLYMARKET_MAX_EVENT_CANDIDATES = int(os.getenv("POLYMARKET_MAX_EVENT_CANDIDATES", "80"))
POLYMARKET_MAX_MARKET_CANDIDATES = int(os.getenv("POLYMARKET_MAX_MARKET_CANDIDATES", "40"))

STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "will",
    "more",
    "than",
    "from",
    "that",
    "this",
    "which",
    "what",
    "when",
    "where",
    "who",
    "how",
    "into",
    "onto",
    "over",
    "under",
    "after",
    "before",
    "their",
    "there",
    "about",
    "party",
    "election",
}

# --- Pydantic Models ---


class PolymarketSearchQueries(BaseModel):
    queries: List[str] = Field(
        description="A list of 3-5 distinct search queries to find relevant prediction markets on Polymarket.",
        min_items=1,
        max_items=5,
    )


class Market(BaseModel):
    id: str
    question: str
    probabilities: str
    volume: float
    liquidity: float
    url: str


class SelectedMarketID(BaseModel):
    id: str = Field(description="The ID of the selected market.")


class RelevantMarkets(BaseModel):
    selected_market_ids: List[SelectedMarketID] = Field(
        description="List of IDs for the selected relevant markets.",
    )


# --- Text/Relevance Helpers ---


def _safe_float(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _tokenize(text: str) -> List[str]:
    normalized = _normalize_text(text)
    return [
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in STOP_WORDS and not token.isdigit()
    ]


def _extract_years(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"\b20\d{2}\b", text)


def _extract_event_text(event: dict) -> str:
    tags = event.get("tags", []) or []
    tag_text = " ".join(
        t.get("label", "") if isinstance(t, dict) else str(t)
        for t in tags
    )
    return " ".join(
        [
            str(event.get("title", "") or ""),
            str(event.get("description", "") or ""),
            str(event.get("slug", "") or ""),
            tag_text,
        ]
    )


def _event_relevance_score(
    event: dict,
    query_tokens: set[str],
    question_tokens: set[str],
    query_phrases: List[str],
    years: List[str],
) -> float:
    event_text = _extract_event_text(event)
    event_norm = _normalize_text(event_text)
    event_tokens = set(_tokenize(event_text))

    if not event_tokens and not event_norm:
        return 0.0

    query_overlap = len(query_tokens & event_tokens)
    question_overlap = len(question_tokens & event_tokens)
    phrase_hits = sum(1 for p in query_phrases if p and p in event_norm)
    year_hits = sum(1 for y in years if y in event_norm)
    volume_bonus = math.log1p(_safe_float(event.get("volume", 0))) / 4.0

    return (
        (6.0 * query_overlap)
        + (2.5 * question_overlap)
        + (7.0 * phrase_hits)
        + (3.0 * year_hits)
        + volume_bonus
    )


def _market_relevance_score(
    market: Market,
    query_tokens: set[str],
    question_tokens: set[str],
    query_phrases: List[str],
    years: List[str],
) -> float:
    market_norm = _normalize_text(market.question)
    market_tokens = set(_tokenize(market.question))
    query_overlap = len(query_tokens & market_tokens)
    question_overlap = len(question_tokens & market_tokens)
    phrase_hits = sum(1 for p in query_phrases if p and p in market_norm)
    year_hits = sum(1 for y in years if y in market_norm)
    liquidity_bonus = math.log1p(market.liquidity + market.volume) / 6.0

    return (
        (7.0 * query_overlap)
        + (3.0 * question_overlap)
        + (10.0 * phrase_hits)
        + (2.0 * year_hits)
        + liquidity_bonus
    )


def _rank_events_by_relevance(
    events: List[dict],
    question: str,
    queries: List[str],
    max_candidates: int,
) -> List[dict]:
    question_tokens = set(_tokenize(question))
    query_tokens = set(_tokenize(" ".join(queries)))
    if not query_tokens:
        query_tokens = question_tokens

    query_phrases = [_normalize_text(q) for q in queries]
    years = _extract_years(question + " " + " ".join(queries))

    scored = []
    for event in events:
        score = _event_relevance_score(event, query_tokens, question_tokens, query_phrases, years)
        if score > 0:
            scored.append((score, _safe_float(event.get("volume", 0)), event))

    if not scored:
        by_volume = sorted(events, key=lambda e: _safe_float(e.get("volume", 0)), reverse=True)
        return by_volume[:max_candidates]

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [event for _, _, event in scored[:max_candidates]]


def _rank_markets_by_relevance(
    markets: List[Market],
    question: str,
    queries: List[str],
    max_candidates: int,
) -> List[Market]:
    if not markets:
        return []

    question_tokens = set(_tokenize(question))
    query_tokens = set(_tokenize(" ".join(queries)))
    if not query_tokens:
        query_tokens = question_tokens

    query_phrases = [_normalize_text(q) for q in queries]
    years = _extract_years(question + " " + " ".join(queries))

    scored = []
    for market in markets:
        score = _market_relevance_score(market, query_tokens, question_tokens, query_phrases, years)
        scored.append((score, market.volume + market.liquidity, market))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [market for _, _, market in scored[:max_candidates]]


def _parse_json_list(value) -> List:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


# --- LLM Query Generation ---


async def generate_search_queries(question: str) -> List[str]:
    """
    Generates search queries for Polymarket based on the input question using an LLM.
    """
    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY not set. Using fallback query.")
        return [question]

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.3,
    )

    parser = PydanticOutputParser(pydantic_object=PolymarketSearchQueries)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at finding prediction markets. Your goal is to generate search queries to find markets on Polymarket that are relevant to the user's forecasting question.\n\n"
                "Generate short, specific query phrases (2-5 words) based on entities, geography, office names, election/event type, and year.\n"
                "Avoid broad generic terms.\n"
                "{format_instructions}",
            ),
            ("user", "User Question: {question}"),
        ]
    )

    try:
        prompt_value = await prompt.ainvoke(
            {"question": question, "format_instructions": parser.get_format_instructions()}
        )

        msg = await llm.ainvoke(prompt_value)

        if hasattr(msg, "response_metadata") and "token_usage" in msg.response_metadata:
            usage = msg.response_metadata["token_usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            try:
                from src.token_cost import add_token_usage, calculate_cost

                cost = calculate_cost(LLM_MODEL, prompt_tokens, completion_tokens)
                add_token_usage(
                    "polymarket_queryable", prompt_tokens, completion_tokens, total_tokens, cost
                )
                print(f"  [Token Usage] Queries: {total_tokens} tokens (${cost:.5f})")
            except ImportError:
                pass

        result = parser.invoke(msg)
        deduped = list(dict.fromkeys([q.strip() for q in result.queries if q and q.strip()]))
        return deduped[:5] if deduped else [question]
    except Exception as e:
        print(f"Error generating queries: {e}")
        return [question]


# --- Polymarket Fetching ---


async def _fetch_open_events_page(
    session: aiohttp.ClientSession,
    offset: int,
    limit: int,
) -> List[dict]:
    url = "https://gamma-api.polymarket.com/events"
    params = {
        "limit": limit,
        "offset": offset,
        "closed": "false",
    }

    try:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                print(f"Error fetching Polymarket events at offset {offset}: {response.status}")
                return []
            data = await response.json()
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Exception fetching Polymarket events at offset {offset}: {e}")
        return []


async def fetch_open_events(
    session: aiohttp.ClientSession,
    max_events: int = POLYMARKET_SCAN_LIMIT,
    page_size: int = POLYMARKET_PAGE_SIZE,
) -> List[dict]:
    offsets = list(range(0, max_events, page_size))
    tasks = [_fetch_open_events_page(session, offset, page_size) for offset in offsets]
    pages = await asyncio.gather(*tasks)

    all_events = []
    for page in pages:
        all_events.extend(page)

    deduped = {}
    for event in all_events:
        event_id = str(event.get("id", ""))
        if event_id and event_id not in deduped:
            deduped[event_id] = event

    return list(deduped.values())


# --- Market Parsing ---


def parse_market_data(events: List[dict]) -> List[Market]:
    """
    Parses raw Polymarket event data into a list of Market objects.
    """
    market_objects = []

    for event in events:
        try:
            markets_in_event = event.get("markets", []) or []
            slug = event.get("slug", "")

            for m in markets_in_event:
                if m.get("closed", False):
                    continue

                market_id = m.get("id")
                if market_id is None:
                    continue

                prices = _parse_json_list(m.get("outcomePrices"))
                if not prices:
                    continue

                parsed_prices: List[float] = []
                for p in prices:
                    try:
                        parsed_prices.append(float(p))
                    except (TypeError, ValueError):
                        continue

                if not parsed_prices:
                    continue

                probs_fmt = ", ".join(f"{price * 100:.1f}%" for price in parsed_prices)
                question = m.get("question", event.get("title", "Unknown Question"))

                outcomes = _parse_json_list(m.get("outcomes"))
                if outcomes and len(outcomes) == len(parsed_prices):
                    probs_fmt = ", ".join(
                        f"{out}: {price * 100:.1f}%"
                        for out, price in zip(outcomes, parsed_prices)
                    )

                market_obj = Market(
                    id=str(market_id),
                    question=str(question),
                    probabilities=probs_fmt,
                    volume=_safe_float(m.get("volume", 0)),
                    liquidity=_safe_float(m.get("liquidity", 0)),
                    url=f"https://polymarket.com/event/{slug}",
                )
                market_objects.append(market_obj)

        except Exception:
            continue

    return market_objects


# --- LLM Filtering ---


async def filter_markets_with_llm(question: str, markets: List[Market]) -> List[Market]:
    """
    Filters a list of markets to find those relevant to the question using an LLM.
    """
    if not markets:
        return []

    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY not set. Skipping LLM filtering.")
        return markets[:5]

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.0,
    )

    parser = PydanticOutputParser(pydantic_object=RelevantMarkets)

    candidates_text = ""
    market_map = {}
    for m in markets:
        market_map[m.id] = m
        candidates_text += (
            f"ID: {m.id}\n"
            f"Question: {m.question}\n"
            f"Probabilities: {m.probabilities}\n"
            f"Volume: ${m.volume:,.0f} | Liquidity: ${m.liquidity:,.0f}\n\n"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a forecasting assistant. Select markets that are directly relevant and useful for forecasting the user question.\n"
                "Rules:\n"
                "- Keep only directly related markets.\n"
                "- Prefer liquid/high-volume versions when duplicates exist.\n"
                "- Exclude low-signal tangential markets.\n"
                "Return selected IDs only.\n"
                "{format_instructions}",
            ),
            ("user", "User Question: {question}\n\nCandidate Markets:\n{candidates}"),
        ]
    )

    try:
        prompt_value = await prompt.ainvoke(
            {
                "question": question,
                "candidates": candidates_text,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        msg = await llm.ainvoke(prompt_value)

        if hasattr(msg, "response_metadata") and "token_usage" in msg.response_metadata:
            usage = msg.response_metadata["token_usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            try:
                from src.token_cost import add_token_usage, calculate_cost

                cost = calculate_cost(LLM_MODEL, prompt_tokens, completion_tokens)
                add_token_usage(
                    "polymarket_filtering", prompt_tokens, completion_tokens, total_tokens, cost
                )
                print(f"  [Token Usage] Filtering: {total_tokens} tokens (${cost:.5f})")
            except ImportError:
                pass

        result = parser.invoke(msg)

        selected_markets = []
        for selected in result.selected_market_ids:
            if selected.id in market_map:
                selected_markets.append(market_map[selected.id])

        # If the LLM is too strict, keep top ranked candidates as fallback.
        if not selected_markets:
            return markets[:3]

        return selected_markets
    except Exception as e:
        print(f"Error filtering markets with LLM: {e}")
        return markets[:5]


async def get_prediction_market_data(question: str) -> List[Market]:
    """
    Main function to get prediction market data for a given question.
    """
    try:
        queries = await generate_search_queries(question)
    except Exception as e:
        print(f"Failed to generate queries, falling back to original question. Error: {e}")
        queries = [question]

    print(f"Searching Polymarket with queries: {queries}")

    async with aiohttp.ClientSession() as session:
        all_events = await fetch_open_events(session)

    print(f"Fetched {len(all_events)} open Polymarket events for local relevance ranking.")

    ranked_events = _rank_events_by_relevance(
        all_events,
        question=question,
        queries=queries,
        max_candidates=POLYMARKET_MAX_EVENT_CANDIDATES,
    )

    print(f"Selected {len(ranked_events)} event candidates after lexical ranking.")

    parsed_markets = parse_market_data(ranked_events)

    unique_markets = {}
    for market in parsed_markets:
        if market.id not in unique_markets:
            unique_markets[market.id] = market

    ranked_markets = _rank_markets_by_relevance(
        list(unique_markets.values()),
        question=question,
        queries=queries,
        max_candidates=POLYMARKET_MAX_MARKET_CANDIDATES,
    )

    print("\nDEBUG: Top candidate markets before LLM filtering:")
    for candidate in ranked_markets[:20]:
        print(f" - {candidate.question} (ID: {candidate.id})")
    print("-" * 20 + "\n")

    print(f"Found {len(ranked_markets)} candidates. Filtering with LLM...")
    relevant_markets = await filter_markets_with_llm(question, ranked_markets)
    print(f"Selected {len(relevant_markets)} relevant markets.")

    return relevant_markets


# --- Formatter for Agent ---


def format_semipublic_market_data(markets: List[Market]) -> str:
    if not markets:
        return "No relevant prediction markets found on Polymarket."

    output = ["Found the following relevant prediction markets:"]
    for market in markets:
        output.append(f"- **{market.question}**")
        output.append(f"  - Probabilities: {market.probabilities}")
        output.append(f"  - Volume: ${market.volume:,.0f} | Liquidity: ${market.liquidity:,.0f}")
        output.append(f"  - Link: {market.url}")

    return "\n".join(output)


if __name__ == "__main__":
    test_question = "Will the Tisza Party win more parliamentary seats than Fidesz-KDNP in Hungary's 2026 parliamentary election?"

    async def run_test():
        markets = await get_prediction_market_data(test_question)
        print(format_semipublic_market_data(markets))

    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        pass
