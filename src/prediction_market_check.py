
import os
import sys
import json
import asyncio
import aiohttp
from typing import List, Optional
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

# --- Pydantic Models ---

class PolymarketSearchQueries(BaseModel):
    queries: List[str] = Field(
        description="A list of 3-5 distinct search queries to find relevant prediction markets on Polymarket.",
        min_items=1,
        max_items=5
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

# --- Helper Functions ---

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

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at finding prediction markets. Your goal is to generate search queries to find markets on Polymarket that are relevant to the user's forecasting question.\n\n"
                   "Polymarket search is keyword-based. Generate short, specific queries (2-4 words) that are likely to match the market title or description.\n"
                   "Avoid stop words and overly complex phrasing. Focus on key entities and events.\n"
                   "Example: For 'Will Trump win the 2024 election?', queries might be: 'Trump 2024', 'US Election winner', 'Presidential election'.\n"
                   "{format_instructions}"),
        ("user", "User Question: {question}")
    ])

    try:
        # Generate prompt value
        prompt_value = await prompt.ainvoke({"question": question, "format_instructions": parser.get_format_instructions()})
        
        # Call LLM to get message (with metadata)
        msg = await llm.ainvoke(prompt_value)
        
        # Track Tokens
        if hasattr(msg, "response_metadata") and "token_usage" in msg.response_metadata:
            usage = msg.response_metadata["token_usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            try:
                from src.token_cost import add_token_usage, calculate_cost
                cost = calculate_cost(LLM_MODEL, prompt_tokens, completion_tokens)
                add_token_usage("polymarket_queryable", prompt_tokens, completion_tokens, total_tokens, cost)
                print(f"  [Token Usage] Queries: {total_tokens} tokens (${cost:.5f})")
            except ImportError:
                pass

        # Parse output
        result = parser.invoke(msg)
        return result.queries
    except Exception as e:
        print(f"Error generating queries: {e}")
        return [question] # Fallback to original question

async def fetch_markets_from_polymarket(query: str, session: aiohttp.ClientSession) -> List[dict]:
    """
    Fetches market data from Polymarket API (Gamma) matching the query.
    """
    url = "https://gamma-api.polymarket.com/events"
    params = {
        "limit": 20,
        "closed": "false",
        "q": query
    }
    
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data if isinstance(data, list) else []
            else:
                print(f"Error fetching data from Polymarket for query '{query}': {response.status}")
                return []
    except Exception as e:
        print(f"Exception fetching data from Polymarket for query '{query}': {e}")
        return []

def parse_market_data(events: List[dict]) -> List[Market]:
    """
    Parses raw Polymarket event data into a list of Market objects.
    Extracts the most relevant information (market title, probability, volume, liquidity).
    """
    market_objects = []
    
    for event in events:
        try:
            # Polymarket 'events' contain 'markets'. We need to iterate through them.
            markets_in_event = event.get("markets", [])
            slug = event.get("slug", "")
            
            for m in markets_in_event:
                # Basic validation
                if m.get("closed", False):
                    continue

                # Extract probability
                outcome_prices = m.get("outcomePrices")
                if not outcome_prices:
                    continue
                    
                # Simplify probability representation
                try:
                    probs_str = json.loads(outcome_prices)
                    # Convert to percentage strings for readability
                    if isinstance(probs_str, list):
                         probs_fmt = ", ".join([f"{float(p)*100:.1f}%" for p in probs_str])
                    else:
                        probs_fmt = str(probs_str)
                except:
                    probs_fmt = str(outcome_prices)

                # Question/Title
                question = m.get("question", event.get("title", "Unknown Question"))
                
                # Check for outcomes (Yes/No vs Multiple Choice)
                outcomes = m.get("outcomes")
                if outcomes:
                    try:
                        outcomes_list = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
                        if isinstance(outcomes_list, list) and len(outcomes_list) == len(json.loads(outcome_prices)):
                             # Map outcomes to probabilities
                             prices = json.loads(outcome_prices)
                             probs_fmt = ", ".join([f"{out}: {float(p)*100:.1f}%" for out, p in zip(outcomes_list, prices)])
                    except:
                        pass

                market_obj = Market(
                    id=m.get("id"),
                    question=question,
                    probabilities=probs_fmt,
                    volume=float(m.get("volume", 0) or 0),
                    liquidity=float(m.get("liquidity", 0) or 0),
                    url=f"https://polymarket.com/event/{slug}"
                )
                market_objects.append(market_obj)
                
        except Exception as e:
            # print(f"Error parsing event: {e}")
            continue
            
    return market_objects

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
        temperature=0.0, # Deterministic
    )
    
    parser = PydanticOutputParser(pydantic_object=RelevantMarkets)

    # create a mapping for the LLM to select from
    candidates_text = ""
    market_map = {}
    for i, m in enumerate(markets):
        market_map[m.id] = m
        candidates_text += f"ID: {m.id}\nQuestion: {m.question}\nProbabilities: {m.probabilities}\nVolume: ${m.volume:,.0f} | Liquidity: ${m.liquidity:,.0f}\n\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a forecasting assistant. Your goal is to select prediction markets from the provided list that are **useful** for forecasting the user's question.\n"
                   "Exclude markets that are:\n"
                   "- Irrelevant to the specific question asked.\n"
                   "- Have very low volume/liquidity (unless it's the only option).\n"
                   "- Duplicate or highly correlated (pick the most liquid one).\n"
                   "- Closed or settled, unless they provide useful historical context.\n\n"
                   "Return the IDs of the selected markets.\n"
                   "{format_instructions}"),
        ("user", "User Question: {question}\n\nCandidate Markets:\n{candidates}")
    ])

    try:
        # Generate prompt value
        prompt_value = await prompt.ainvoke({
            "question": question, 
            "candidates": candidates_text,
            "format_instructions": parser.get_format_instructions()
        })
        
        # Call LLM
        msg = await llm.ainvoke(prompt_value)
        
        # Track Tokens
        if hasattr(msg, "response_metadata") and "token_usage" in msg.response_metadata:
            usage = msg.response_metadata["token_usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            try:
                from src.token_cost import add_token_usage, calculate_cost
                cost = calculate_cost(LLM_MODEL, prompt_tokens, completion_tokens)
                add_token_usage("polymarket_filtering", prompt_tokens, completion_tokens, total_tokens, cost)
                print(f"  [Token Usage] Filtering: {total_tokens} tokens (${cost:.5f})")
            except ImportError:
                pass

        # Parse output
        result = parser.invoke(msg)
        
        selected_markets = []
        for selected in result.selected_market_ids:
            if selected.id in market_map:
                selected_markets.append(market_map[selected.id])
                
        return selected_markets
    except Exception as e:
        print(f"Error filtering markets with LLM: {e}")
        # Fallback to volume sort
        return sorted(markets, key=lambda x: x.volume, reverse=True)[:5]

async def get_prediction_market_data(question: str) -> List[Market]:
    """
    Main function to get prediction market data for a given question.
    """
    # 1. Generate Queries
    try:
        queries = await generate_search_queries(question)
    except Exception as e:
        print(f"Failed to generate queries, falling back to original question. Error: {e}")
        queries = [question]
        
    print(f"Searching Polymarket with queries: {queries}")

    # 2. Fetch Data in Parallel
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_markets_from_polymarket(q, session) for q in queries]
        results = await asyncio.gather(*tasks)
    
    # 3. Aggregate and Deduplicate
    all_events = []
    for r in results:
        all_events.extend(r)
        
    parsed_markets = parse_market_data(all_events)
    
    # Deduplicate by ID
    unique_markets = {}
    for m in parsed_markets:
        if m.id not in unique_markets:
            unique_markets[m.id] = m
            
    candidates = list(unique_markets.values())
    
    print("\nDEBUG: All Candidates found:")
    for c in candidates:
        print(f" - {c.question} (ID: {c.id})")
    print("-" * 20 + "\n")
    
    # 4. Filter with LLM
    print(f"Found {len(candidates)} candidates. Filtering with LLM...")
    relevant_markets = await filter_markets_with_llm(question, candidates)
    print(f"Selected {len(relevant_markets)} relevant markets.")
    
    return relevant_markets

# --- Formatter for Agent ---

def format_semipublic_market_data(markets: List[Market]) -> str:
    if not markets:
        return "No relevant prediction markets found on Polymarket."
    
    output = ["Found the following relevant prediction markets:"]
    for m in markets:
        output.append(f"- **{m.question}**")
        output.append(f"  - Probabilities: {m.probabilities}")
        output.append(f"  - Volume: ${m.volume:,.0f} | Liquidity: ${m.liquidity:,.0f}")
        output.append(f"  - Link: {m.url}")
    
    return "\n".join(output)

if __name__ == "__main__":
    # Test cases
    # Test cases
    test_questions = [
        "Will the Tisza Party win more parliamentary seats than Fidesz–KDNP in Hungary’s 2026 parliamentary election?"
    ]

    async def run_tests():
        print("Starting Polymarket Check Tests...\n")
        
        # 1. Debug specific market by slug
        slug = "hungary-parliamentary-election-winner"
        print(f"DEBUG: Fetching specific market by slug: {slug}")
        url = "https://gamma-api.polymarket.com/events"
        params = {"slug": slug}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(json.dumps(data, indent=2))
                else:
                    print(f"Failed to fetch by slug: {resp.status}")
        print("-" * 50)

        # 2. Test specific manual queries to understand search behavior
        manual_queries = [
            "Hungary",
            "Hungary election",
            "Hungarian parliamentary",
            "Tisza",  # Expected to fail
            "Fidesz"  # Expected to fail
        ]
        
        print(f"DEBUG: Testing manual queries to find 'Hungarian parliamentary election winner'")
        for mq in manual_queries:
            print(f"Querying: '{mq}'")
            print("-" * 30)
            async with aiohttp.ClientSession() as session:
                results = await fetch_markets_from_polymarket(mq, session)
                found = False
                for event in results:
                     if "Hungarian parliamentary election winner" in event.get("title", ""):
                         print(f"  SUCCESS! Found event: {event.get('title')}")
                         found = True
                         break
                if not found:
                     print(f"  Failed to find target event. Top 3 results: {[e.get('title') for e in results[:3]]}")
            print("\n")

    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        pass
