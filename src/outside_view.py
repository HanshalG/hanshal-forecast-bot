import os
from typing import TypedDict, Annotated, List, Union, Dict, Any
from operator import add
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools import BraveSearch
from langchain_experimental.tools import PythonREPLTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from exa_py import Exa
from langchain_openai import ChatOpenAI

# Load environment variables (if not already loaded)
load_dotenv()

# --- Configuration & Constants ---

LLM_MODEL = "gpt-5-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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

# Initialize Brave Search tool
brave_tool = BraveSearch.from_api_key(
    api_key=os.getenv("BRAVE_API_KEY"),
    search_kwargs={"count": 3}
)

# Initialize Python REPL tool
python_repl = PythonREPLTool()

tools = [exa_answer_tool, brave_tool, python_repl]

# --- Agent State ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]

# --- Graph Nodes ---

# Initialize LLM with OpenRouter configuration
llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=OPENROUTER_BASE_URL,
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)

async def agent(state: AgentState):
    """The agent node that calls the LLM."""
    messages = state["messages"]
    if messages and messages[-1].type == "tool":
        print(f"\n--- Tool Output ---\n{str(messages[-1].content)[:1000]}...")

    response = await llm_with_tools.ainvoke(messages)
    
    print(f"\n--- Agent Response ---\n{response.content}")
    
    if response.tool_calls:
        print(f"\n--- Tool Calls ---\n{response.tool_calls}")

    return {"messages": [response]}

# ToolNode is a prebuilt node that executes tools
tool_node = ToolNode(tools)

# --- Graph Construction ---

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)

# Add edges
workflow.set_entry_point("agent")

# Conditional edge: check if the agent wants to call a tool
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)

# Edge from tools back to agent
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()


# --- Main Logic ---

async def generate_outside_view(question_details: dict, historical_context: str | None = None, max_searches: int = 10) -> str:
    """Generate an outside view using a LangGraph agent.

    Args:
        question_details: A dict containing keys: "title", "description" (background),
            "resolution_criteria", and "fine_print".
        historical_context: Optional pre-computed context (ignored in this new version unless we want to inject it).
        max_searches: Limit for searches (not strictly enforced by the agent graph, but kept for signature compatibility).

    Returns:
        The outside view analysis produced by the Agent.
    """
    title = question_details.get("title", "")
    background = question_details.get("description", "")
    resolution_criteria = question_details.get("resolution_criteria", "")
    fine_print = question_details.get("fine_print", "")

    # Construct the initial system/user prompt for the agent
    prompt = f"""
You are an expert forecaster tasked with providing an "outside view" analysis for the following question.

Title: {title}

Background:
{background}

Resolution Criteria:
{resolution_criteria}

Fine Print:
{fine_print}

Your goal is to identify a suitable reference class and base rate for this event.
1. Research similar historical events or categories of events.
2. Determine the frequency of the outcome of interest within that reference class.
3. If multiple reference classes are relevant, consider them and weigh their applicability.
4. Provide a final "Outside View" analysis condensing your findings.

Use the available tools (Exa, Brave Search, Python) to gather data and calculate rates if necessary.
Use Exa for questions (its better at answering questions) and Brave for search.
"""

    initial_state = {"messages": [HumanMessage(content=prompt)]}
    
    # We will accumulate the final answer from the agent.
    # For a simple agent loop, we can iterate until the agent stops calling tools.
    # The 'app.invoke' or 'app.astream' can be used.
    
    from langchain_community.callbacks import get_openai_callback
    
    with get_openai_callback() as cb:
        final_output = await app.ainvoke(initial_state)
        print(f"\n--- Token Usage ---\nTotal Tokens: {cb.total_tokens}\nPrompt Tokens: {cb.prompt_tokens}\nCompletion Tokens: {cb.completion_tokens}\nTotal Cost (USD): ${cb.total_cost:.4f}")
    
    # The last message should be the agent's final answer
    last_message = final_output["messages"][-1]
    
    return last_message.content

if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    # Mock get_post_details to avoid dependency if needed, or import if available
    # For this test, we can just use a dummy question or try to import.
    
    #from main import EXAMPLE_QUESTIONS
    from metaculus_utils import get_post_details
    
    # Use the first example question
    #example_question_id, example_post_id = EXAMPLE_QUESTIONS[0]
    post_details = get_post_details(41851) # This ID might need to be valid or we pick one from EXAMPLE_QUESTIONS
    question_details = post_details['question']
    # Hardcoding a sample question for standalone testing if imports fail or IDs are tricky

    async def _run() -> None:
        print(f"Generating outside view for: {question_details['title']}")
        result = await generate_outside_view(question_details)
        print("\n" + "#" * 80)
        print("Outside View\n" + "#" * 80)
        print(result)

    asyncio.run(_run())