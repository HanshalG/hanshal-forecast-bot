import os
import asyncio
import datetime
from dotenv import load_dotenv

# --- Agent Infrastructure Import ---
from src.agent_infrastructure import create_agent_graph
from src.metaculus_utils import get_post_details

from langchain_core.messages import HumanMessage
from langchain_community.callbacks import get_openai_callback

# Load environment variables (if not already loaded)
load_dotenv()

from src.agent_infrastructure import create_agent_graph
from src.metaculus_utils import get_post_details
from src.utils import read_prompt

# Configuration
OUTSIDE_VIEW_MODEL = os.getenv("OUTSIDE_VIEW_MODEL", "gpt-5-mini")

# Initialize Graph
app = create_agent_graph(model_name=OUTSIDE_VIEW_MODEL)

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
    today = datetime.date.today().strftime("%Y-%m-%d")

    # Construct the initial system/user prompt for the agent
    prompt_template = read_prompt("outside_view_agent_prompt.txt")
    prompt = prompt_template.format(
        title=title,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        today=today
    )

    initial_state = {"messages": [HumanMessage(content=prompt)]}
    
    with get_openai_callback() as cb:
        final_output = await app.ainvoke(initial_state)
        from src.token_cost import print_token_usage
        print_token_usage(cb, OUTSIDE_VIEW_MODEL, final_output, component="outside_view")
    
    # The last message should be the agent's final answer
    last_message = final_output["messages"][-1]
    
    return last_message.content

if __name__ == "__main__":
    # Test block
    post_details = get_post_details(41851)
    question_details = post_details['question']

    async def _run() -> None:
        print(f"Generating outside view for: {question_details['title']}")
        result = await generate_outside_view(question_details)
        print("\n" + "#" * 80)
        print("Outside View\n" + "#" * 80)
        print(result)

    asyncio.run(_run())