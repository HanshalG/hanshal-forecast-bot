import os
import asyncio
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import BraveSearch
from langchain_experimental.tools import PythonREPLTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from exa_py import Exa
from langchain_openai import ChatOpenAI
from src.utils import call_llm, SUMMARY_MODEL

# Load environment variables
load_dotenv()

# --- Configuration ---
LLM_MODEL = os.getenv("INSIDE_VIEW_MODEL", "gpt-5-mini")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Set to True to use Exa search, False to use Brave search
USE_EXA_SEARCH = True

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
                result_text = f"{i}. {result.title}\n   URL: {result.url}[:50]\n"
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

# --- Tool Tracking ---
TOOL_CALL_COUNTS = {tool.name: 0 for tool in ALL_TOOLS}

def get_tool_call_counts():
    return TOOL_CALL_COUNTS

# --- Agent State ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# --- LLM Setup ---

# --- Graph Factory ---

def create_agent_graph(model_name: str = LLM_MODEL):
    """Create and compile the agent state graph with a specific model."""
    
    # Initialize LLM with the specific model
    llm = ChatOpenAI(
        model=model_name,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
        temperature=0
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    # Track cumulative tokens for this agent instance
    token_tracker = {"total": 0, "cost": 0.0}

    async def agent_node(state: AgentState):
        """The agent node that calls the LLM."""
        messages = state["messages"]
        
        # Debug prints for tool outputs
        last_msg_idx = len(messages) - 1
        tool_outputs = []
        while last_msg_idx >= 0 and messages[last_msg_idx].type == "tool":
            tool_outputs.append(messages[last_msg_idx])
            last_msg_idx -= 1
            
        if tool_outputs:
            print(f"\n--- Tool Outputs ({len(tool_outputs)}) ---")
            for msg in reversed(tool_outputs):
                content_preview = str(msg.content)[:500] + "..." if len(str(msg.content)) > 500 else str(msg.content)
                print(f"[Tool: {msg.name if hasattr(msg, 'name') else 'unknown'}]")
                print(content_preview)
                print("-" * 20)

        response = await llm_with_tools.ainvoke(messages)
        
        # Limit parallel tool calls to 5
        if response.tool_calls and len(response.tool_calls) > 5:
            print(f"--- Limiting tool calls from {len(response.tool_calls)} to 5 ---")
            response.tool_calls = response.tool_calls[:5]
        
        # --- Ongoing Token Prints ---
        if hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
            usage = response.response_metadata["token_usage"]
            p = usage.get("prompt_tokens", 0)
            c = usage.get("completion_tokens", 0)
            t = usage.get("total_tokens", 0)
            
            # Calculate cost for this step
            # Calculate cost for this step
            step_cost = 0.0
            try:
                from src.token_cost import calculate_cost
                step_cost = calculate_cost(model_name, p, c)
            except ImportError:
                step_cost = 0.0
            
            # Update cumulative
            token_tracker["total"] += t
            token_tracker["cost"] += step_cost
                
            print(f"\n--- Step Token Usage ({t} tokens) ---")
            print(f"  Prompt: {p}, Completion: {c}")
            print(f"  Est. Cost: ${step_cost:.6f}")
            print(f"  Cumulative Cost: ${token_tracker['cost']:.6f} ({token_tracker['total']} tokens)")
            
        print(f"\n--- Agent Response ---\n{response.content}")
        
        if response.tool_calls:
            print(f"\n--- Tool Calls ---\n{response.tool_calls}")
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                if tool_name in TOOL_CALL_COUNTS:
                    TOOL_CALL_COUNTS[tool_name] += 1
                else:
                    TOOL_CALL_COUNTS[tool_name] = 1

        return {"messages": [response]}

    # ToolNode is a prebuilt node that executes tools
    tool_node = ToolNode(ALL_TOOLS)

    # Summarization Node
    async def summarize_nodes(state: AgentState):
        messages = state["messages"]
        
        # Identify recent tool messages
        tool_messages = []
        i = len(messages) - 1
        while i >= 0 and isinstance(messages[i], ToolMessage):
            tool_messages.append(messages[i])
            i -= 1
        
        if not tool_messages:
            return {"messages": []}
            
        tool_messages.reverse()
        
        # Context for summarization (last AIMessage before tools)
        context_msg = messages[i] if i >= 0 else messages[-1]
        context = context_msg.content if context_msg else "No context"
        
        async def _summarize(msg):
            # Skip summarization for Python REPL to maintain precision
            if msg.name in ["python_repl", "Python_REPL", "python"]:
                return msg

            prompt = (
                f"Context: Agent asked: \"{context}\"\n\n"
                f"Tool Result to Summarize:\n{msg.content}\n\n"
                f"Instructions: Extract ONLY the information relevant to the question. "
                f"Remove irrelevant data, boilerplate, and navigation elements. "
                f"If the result is completely irrelevant, return 'Irrelevant'."
            )
            # Use lower temp for summarization
            summary = await call_llm(prompt, SUMMARY_MODEL, 0.1)
            
            # Return new ToolMessage with SAME ID to update in-place
            return ToolMessage(content=f"Summary: {summary}", tool_call_id=msg.tool_call_id, id=msg.id, name=msg.name)

        # Run summaries in parallel
        new_messages = await asyncio.gather(*[_summarize(msg) for msg in tool_messages])
        
        print(f"\n--- Summarized {len(new_messages)} Tool Outputs ---")
        return {"messages": new_messages}

    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("summarize", summarize_nodes)
    
    # Add edges
    workflow.set_entry_point("agent")
    
    # Conditional edge: check if the agent wants to call a tool
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
    )
    
    # Edge from tools to summarize
    workflow.add_edge("tools", "summarize")
    
    # Edge from summarize back to agent
    workflow.add_edge("summarize", "agent")
    
    # Compile the graph
    app = workflow.compile()
    return app
