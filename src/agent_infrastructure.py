import os
import json
from typing import Any
from dotenv import load_dotenv

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import BraveSearch
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt.tool_node import ToolCallRequest

from exa_py import Exa
from langchain_openai import ChatOpenAI
from src.utils import call_llm, SUMMARY_MODEL

# Middleware imports
from langchain.agents import create_agent
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain.agents.middleware.types import AgentMiddleware

# Load environment variables
load_dotenv()

# --- ANSI Colors ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(text, color=Colors.ENDC):
    print(f"{color}{text}{Colors.ENDC}")


# --- Configuration ---
LLM_MODEL = os.getenv("INSIDE_VIEW_MODEL", "gpt-5-mini")
REASONING_EFFORT = "high"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Set to True to use Exa search, False to use Brave search
USE_EXA_SEARCH = True
RUN_TOOL_LIMIT = 5  # Limit per run (single invocation)
THREAD_TOOL_LIMIT = 5 # Limit per thread (accumulated context)

# --- Tool Definitions ---

# Initialize Exa client
exa_client = Exa(api_key=os.getenv("EXA_API_KEY"))

@tool
def exa_answer_tool(question: str) -> str:
    """Use Exa's answer endpoint to answer a question with citations."""
    try:
        response = exa_client.answer(question, text=True)
        return f"Answer: {response.answer}\n\nCitations: {response.citations}"
    except Exception as e:
        return f"Error querying Exa: {e}"

# Web search tool - conditionally use Exa or Brave
if USE_EXA_SEARCH:
    @tool
    def web_search(query: str) -> str:
        """Search the web using Exa. Returns search results with titles, URLs, and key highlights."""
        try:
            response = exa_client.search_and_contents(
                query,
                num_results=3,
                type="auto",  # Uses intelligent combination of neural and other search methods
                text=False,
                highlights={
                    "query": query,  # Use search query to guide highlights extraction
                    "max_characters": 2000  # Limit number of highlight sentences
                    }
                )
            
            results = []
            for i, result in enumerate(response.results, 1):
                result_text = f"{i}. {result.title}\n   URL: {result.url[:50]}\n"
                if hasattr(result, 'highlights') and result.highlights:
                    result_text += f"   Key highlights: {', '.join(result.highlights)}\n"
                results.append(result_text)
            
            return "\n".join(results)
        except Exception as e:
            return f"Error searching with Exa: {e}"
else:
    # Initialize Brave Search tool
    web_search = BraveSearch.from_api_key(
        api_key=os.getenv("BRAVE_API_KEY"),
        search_kwargs={"count": 3, "summary": True}
    )

# Initialize Python REPL tool
python_repl = PythonREPLTool()

ALL_TOOLS = [exa_answer_tool, python_repl]

def get_tool_instructions() -> str:
    """Generate tool usage instructions based on available tools."""
    instructions = [
        " ------------------------------------------------------------------------------------------------",
        " **IMPORTANT: TOOL USAGE INSTRUCTIONS**",
        " You have access to the following tools:"
    ]
    
    # List tools
    for i, tool in enumerate(ALL_TOOLS, 1):
        instructions.append(f" {i}.  **{tool.name}**: {tool.description}")

    instructions.append("")
    instructions.append(" **When to use tools:**")
    
    # Add specific advice based on active tools
    tool_names = [t.name for t in ALL_TOOLS]
    
    # Note: explicit check for common names or substrings to be robust
    has_web = any("search" in t.name.lower() or "web" in t.name.lower() for t in ALL_TOOLS)
    has_exa = any("exa" in t.name.lower() for t in ALL_TOOLS)
    has_python = any("python" in t.name.lower() for t in ALL_TOOLS)

    if has_web:
        instructions.append(" *   **Use `web_search` primarily** for broad searches, recent news, and multiple perspectives.")
    
    if has_exa:
        instructions.append(" *   Use `exa_answer_tool` to verify specific claims, dates, or authoritative data sources if `web_search` is insufficient.")
        
    if has_python:
        instructions.append(" *   **Use Python for all calculations**, including:")
        instructions.append("     - Computing base rates and weighted averages")
        instructions.append("     - Aggregating evidence impacts")
        instructions.append("     - Determining confidence intervals")
        instructions.append(" *   **IMPORTANT**: When using Python, you MUST `print(...)` the final result. Values not printed are lost.")
        
    instructions.append(" *   **Parallel Tool Usage**: You are encouraged to fire multiple search queries in a single turn.")
    instructions.append(" ------------------------------------------------------------------------------------------------")
    
    return "\n".join(instructions)

# --- Tool Tracking ---
TOOL_CALL_COUNTS = {tool.name: 0 for tool in ALL_TOOLS}

def get_tool_call_counts():
    return TOOL_CALL_COUNTS


# --- Custom Summarization Middleware ---
class SummarizationMiddleware(AgentMiddleware):
    """Middleware that summarizes tool outputs to reduce context size."""
    
    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        """Intercept tool calls and summarize the results."""
        # Execute the tool first
        result = await handler(request)
        
        # Skip summarization for Python REPL to maintain precision
        tool_name = request.tool_call.get("name", "")
        if tool_name in ["python_repl", "Python_REPL", "python"]:
            # Track call count
            if tool_name in TOOL_CALL_COUNTS:
                TOOL_CALL_COUNTS[tool_name] += 1
            else:
                TOOL_CALL_COUNTS[tool_name] = 1
            return result
        
        # Get context from state messages
        messages = request.state.get("messages", [])
        context = "No context"
        if messages:
            # Find last AI message for context
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == 'ai':
                    context = msg.content if hasattr(msg, 'content') else str(msg)
                    break
        
        # Summarize if result is a ToolMessage
        if isinstance(result, ToolMessage):
            prompt = (
                f"Context: Agent asked: \"{context}\"\n\n"
                f"Tool Result to Summarize:\n{result.content}\n\n"
                f"Instructions: Extract ONLY the information relevant to the question. "
                f"Remove irrelevant data, boilerplate, and navigation elements. "
                f"If the result is completely irrelevant, return 'Irrelevant'."
            )
            
            summary = await call_llm(prompt, SUMMARY_MODEL, 0.1)
            
            # Update tool call count
            if tool_name in TOOL_CALL_COUNTS:
                TOOL_CALL_COUNTS[tool_name] += 1
            else:
                TOOL_CALL_COUNTS[tool_name] = 1
            
            # Return new ToolMessage with summary
            return ToolMessage(
                content=f"Summary: {summary}",
                tool_call_id=result.tool_call_id,
                name=result.name
            )
        
        return result


# --- Graph Factory ---
def create_agent_graph(model_name: str = LLM_MODEL):
    """Create and compile the agent using create_agent with middleware."""
    
    # Initialize LLM with the specific model
    llm = ChatOpenAI(
        model=model_name,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        reasoning={"effort": REASONING_EFFORT}
    )
    
    # Create middleware instances
    tool_limit_middleware = ToolCallLimitMiddleware(
        run_limit=RUN_TOOL_LIMIT,
        thread_limit=THREAD_TOOL_LIMIT,
        exit_behavior="continue"  # Allow agent to proceed with partial results
    )
    
    summarization_middleware = SummarizationMiddleware()
    
    # Create agent with middleware
    app = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        middleware=[tool_limit_middleware, summarization_middleware],
        debug=True  # Enable debug logging to see execution flow
    )
    
    return app
