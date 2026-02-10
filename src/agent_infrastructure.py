import os
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
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- Tool Call Limits (defaults mirror LangChain docs examples) ---
GLOBAL_TOOL_CALL_LIMITS = {"thread_limit": 5, "run_limit": 5}
WEB_SEARCH_TOOL_CALL_LIMITS = {"thread_limit": 5, "run_limit": 5}

# --- Tool Definitions ---

# Initialize Exa client
exa_client = Exa(api_key=os.getenv("EXA_API_KEY"))

@tool
def exa_answer_tool(question: str) -> str:
    """Answer a specific question using Exa's answer endpoint.

    Use this for targeted facts or concise answers where you want citations
    (dates, figures, definitions, authoritative claims). If you need broader
    coverage or multiple sources, use `exa_search` instead.
    """
    try:
        response = exa_client.answer(question, text=True)
        return f"Answer: {response.answer}\n\nCitations: {response.citations}"
    except Exception as e:
        return f"Error querying Exa: {e}"

@tool
def exa_search(search_query: str, guiding_highlights_query: str) -> str:
    """Search the web with Exa for breadth, recency, and multiple perspectives.

    Inputs:
    - `search_query`: the main search terms.
    - `guiding_highlights_query`: a refined query to steer highlight extraction.

    Returns a compact list of results with titles, URLs, and key highlights.
    Use this to gather context or find recent developments. For a single
    targeted fact or direct answer, prefer `exa_answer_tool`.
    """
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

        return "\n".join(results)
    except Exception as e:
        return f"Error searching with Exa: {e}"

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
