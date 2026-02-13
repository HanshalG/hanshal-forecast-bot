import os
import json
from functools import lru_cache
from typing import Any
from copy import deepcopy
from threading import Lock
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, after_model
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
# from langchain_community.tools import BraveSearch
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from exa_py import Exa

# Load environment variables
load_dotenv()

# --- Configuration ---
LLM_MODEL = os.getenv("INSIDE_VIEW_MODEL", "gpt-5-mini")
TOOL_SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-5-nano")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REASONING_EFFORT = "high"

# --- Tool Call Limits (defaults mirror LangChain docs examples) ---
GLOBAL_TOOL_CALL_LIMITS = {"thread_limit": 10, "run_limit": 10}
WEB_SEARCH_TOOL_CALL_LIMITS = {"thread_limit": 5, "run_limit": 5}

# --- Tool Definitions ---

# Initialize Exa client
exa_client = Exa(api_key=os.getenv("EXA_API_KEY"))
_TOOL_RESULT_CACHE: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
_TOOL_RESULT_CACHE_LOCK = Lock()
_TOOL_CACHE_COUNT_LOCK = Lock()
TOOL_CACHE_HIT_COUNTS: dict[str, int] = {}
TOOL_CACHE_MISS_COUNTS: dict[str, int] = {}


def _increment_cache_counter(counter: dict[str, int], tool_name: str) -> None:
    with _TOOL_CACHE_COUNT_LOCK:
        counter[tool_name] = counter.get(tool_name, 0) + 1


def _stable_tool_input_json(tool_input: dict[str, Any]) -> str:
    """Serialize tool input into a stable cache key representation."""
    return json.dumps(tool_input, ensure_ascii=True, sort_keys=True, default=str)


def _get_cached_tool_response(tool_name: str, tool_input: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    key = (tool_name, _stable_tool_input_json(tool_input))
    with _TOOL_RESULT_CACHE_LOCK:
        cached = _TOOL_RESULT_CACHE.get(key)
    if cached is None:
        _increment_cache_counter(TOOL_CACHE_MISS_COUNTS, tool_name)
        return None
    _increment_cache_counter(TOOL_CACHE_HIT_COUNTS, tool_name)
    content, artifact = cached
    return content, deepcopy(artifact)


def _set_cached_tool_response(
    tool_name: str,
    tool_input: dict[str, Any],
    response: tuple[str, dict[str, Any]],
) -> None:
    key = (tool_name, _stable_tool_input_json(tool_input))
    content, artifact = response
    with _TOOL_RESULT_CACHE_LOCK:
        _TOOL_RESULT_CACHE[key] = (content, deepcopy(artifact))


def _normalize_llm_content(content: Any) -> str:
    """Normalize LLM response content into a plain string."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                if block.strip():
                    parts.append(block.strip())
                continue
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                    continue
        return "\n".join(parts).strip()
    return str(content).strip()


@lru_cache(maxsize=1)
def _get_tool_summary_llm() -> ChatOpenAI:
    """Create a shared low-cost model used to summarize tool outputs."""
    return ChatOpenAI(
        model=TOOL_SUMMARY_MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        reasoning={"effort": REASONING_EFFORT},
    )


def _summarize_tool_output(tool_name: str, tool_input: dict[str, Any], raw_result: str) -> str:
    """Generate a concise summary of tool output for agent context."""
    raw_text = str(raw_result).strip()
    if not raw_text:
        return ""

    lowered = raw_text.lower()
    if "tool call limit exceeded" in lowered or lowered.startswith("error"):
        return raw_text

    prompt = (
        "Summarize this tool output for a forecasting agent.\n"
        "Keep it factual and concise. Include only key facts, dates, figures, and named entities.\n"
        "If uncertainty or conflict appears, mention it explicitly.\n"
        "No markdown. No bullet nesting. Maximum 120 words.\n\n"
        f"Tool name: {tool_name}\n"
        f"Tool input JSON: {json.dumps(tool_input, ensure_ascii=True)}\n\n"
        "Tool raw output:\n"
        f"{raw_text[:8000]}"
    )

    try:
        resp = _get_tool_summary_llm().invoke(prompt)
        summary = _normalize_llm_content(getattr(resp, "content", resp)).strip()
        return summary or raw_text
    except Exception:
        return raw_text


def _build_tool_response(tool_name: str, tool_input: dict[str, Any], raw_result: str) -> tuple[str, dict[str, Any]]:
    """Return (content, artifact) where content carries summary for context."""
    summary = _summarize_tool_output(tool_name, tool_input, raw_result)
    content = summary if summary else raw_result
    artifact = {
        "llm_summary": summary,
        "raw_result": raw_result,
        "tool_input": tool_input,
    }
    return content, artifact


@tool(response_format="content_and_artifact")
def exa_answer_tool(question: str) -> tuple[str, dict[str, Any]]:
    """Answer a specific question using Exa's answer endpoint.

    Use this for targeted facts or concise answers where you want citations
    (dates, figures, definitions, authoritative claims). If you need broader
    coverage or multiple sources, use `exa_search` instead.
    """
    tool_input = {"question": question}
    cached = _get_cached_tool_response("exa_answer_tool", tool_input)
    if cached is not None:
        return cached

    raw_result = ""
    try:
        response = exa_client.answer(question, text=True)
        raw_result = f"Answer: {response.answer}\n\nCitations: {response.citations}"
    except Exception as e:
        raw_result = f"Error querying Exa: {e}"
    tool_response = _build_tool_response(
        "exa_answer_tool",
        tool_input,
        raw_result,
    )
    if not raw_result.lower().startswith("error"):
        _set_cached_tool_response("exa_answer_tool", tool_input, tool_response)
    return tool_response


@tool(response_format="content_and_artifact")
def exa_search(search_query: str, guiding_highlights_query: str) -> tuple[str, dict[str, Any]]:
    """Search the web with Exa for breadth, recency, and multiple perspectives.

    Inputs:
    - `search_query`: the main search terms.
    - `guiding_highlights_query`: a refined query to steer highlight extraction.

    Returns a compact list of results with titles, URLs, and key highlights.
    Use this to gather context or find recent developments. For a single
    targeted fact or direct answer, prefer `exa_answer_tool`.
    """
    tool_input = {
        "search_query": search_query,
        "guiding_highlights_query": guiding_highlights_query,
    }
    cached = _get_cached_tool_response("exa_search", tool_input)
    if cached is not None:
        return cached

    raw_result = ""
    try:
        response = exa_client.search_and_contents(
            search_query,
            num_results=3,
            type="auto",  # Uses intelligent combination of neural and other search methods
            text=False,
            highlights={
                "query": guiding_highlights_query,  # Use search query to guide highlights extraction
                "max_characters": 2000  # Limit number of highlight sentences
            },
        )

        results = []
        for i, result in enumerate(response.results, 1):
            result_text = f"{i}. {result.title}\n   URL: {result.url}[:50]\n"
            if hasattr(result, "highlights") and result.highlights:
                result_text += f"   Key highlights: {', '.join(result.highlights)}\n"
            results.append(result_text)

        raw_result = "\n".join(results)
    except Exception as e:
        raw_result = f"Error searching with Exa: {e}"

    tool_response = _build_tool_response(
        "exa_search",
        tool_input,
        raw_result,
    )
    if not raw_result.lower().startswith("error"):
        _set_cached_tool_response("exa_search", tool_input, tool_response)
    return tool_response

# Initialize Brave Search tool
# brave_search = BraveSearch.from_api_key(
#     api_key=os.getenv("BRAVE_API_KEY"),
#     search_kwargs={"count": 3, "summary": True},
# )

# Initialize Python REPL tool
python_repl = PythonREPLTool()

ALL_TOOLS = [exa_search, exa_answer_tool, python_repl]

# --- Tool Tracking ---
TOOL_CALL_COUNTS = {tool.name: 0 for tool in ALL_TOOLS}

def get_tool_call_counts():
    return TOOL_CALL_COUNTS

@after_model
def _count_tool_calls(state, _runtime):
    messages = state.get("messages") or []
    if not messages:
        return None
    last_message = messages[-1]
    tool_calls = last_message.tool_calls if isinstance(last_message, AIMessage) else None
    if tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            if not tool_name:
                continue
            TOOL_CALL_COUNTS[tool_name] = TOOL_CALL_COUNTS.get(tool_name, 0) + 1
    return None


def create_agent_graph(model_name: str = LLM_MODEL):
    """Create and compile the agent with tool-call limits."""
    llm = ChatOpenAI(
        model=model_name,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        reasoning={"effort": "medium"}
    )
    web_search_tool_name = getattr(exa_search, "name", "exa_search")

    middleware = [
        ToolCallLimitMiddleware(**GLOBAL_TOOL_CALL_LIMITS),
        ToolCallLimitMiddleware(tool_name=web_search_tool_name, **WEB_SEARCH_TOOL_CALL_LIMITS),
        _count_tool_calls,
    ]

    app = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )
    return app
