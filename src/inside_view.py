import asyncio
import uuid
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_community.callbacks import get_openai_callback

# --- Agent Infrastructure Import ---
from src.agent_infrastructure import create_agent_graph, LLM_MODEL, render_tool_call_limits_for_prompt
from src.eval.timebox import today_string_for_prompt
from src.message_utils import message_to_text
from src.metaculus_utils import get_post_details
from src.utils import call_asknews_async, read_prompt, run_agent_with_streaming

# Load environment variables (if not already loaded)
load_dotenv()

# --- Main Logic ---

async def generate_inside_view(
    question_details: dict,
    news_context: str | None = None,
) -> str:
    """Generate an inside view using a LangGraph agent.

    Args:
        question_details: A dict with keys: "title", "description", "resolution_criteria", "fine_print".

    Returns:
        The inside view analysis.
    """
    title = question_details.get("title", "")
    background = question_details.get("description", "")
    resolution_criteria = question_details.get("resolution_criteria", "")
    fine_print = question_details.get("fine_print", "")
    
    today = today_string_for_prompt(question_details.get("as_of_time"))

    # Fetch relevant news unless pre-fetched context is provided.
    if news_context is None:
        print(f"Fetching news for inside view: {title}")
        news_context = await call_asknews_async(
            question_details,
            as_of_time=question_details.get("as_of_time"),
        )
    
    # Load prompt template from file
    prompt_template = read_prompt("inside_view_prompt.txt")
    tool_call_limits = render_tool_call_limits_for_prompt()
    
    prompt = prompt_template.format(
        title=title,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        today=today,
        context=news_context,
        tool_call_limits=tool_call_limits,
    )

    initial_state = {"messages": [HumanMessage(content=prompt)]}
    app = create_agent_graph(model_name=LLM_MODEL)
    question_id = question_details.get("id", "unknown")
    run_config = {
        "configurable": {
            "thread_id": f"inside-{question_id}-{uuid.uuid4().hex}",
        }
    }
    
    with get_openai_callback() as cb:
        final_output = await run_agent_with_streaming(
            app,
            initial_state,
            run_config,
            label="inside_view",
        )
        from src.token_cost import print_token_usage
        print_token_usage(cb, LLM_MODEL, final_output, component="inside_view")
    
    # The last message should be the agent's final answer.
    last_message = final_output["messages"][-1]
    return message_to_text(last_message)


async def generate_inside_view_multiple_choice(
    question_details: dict,
    news_context: str | None = None,
) -> str:
    """Generate an inside view for multiple choice questions."""
    
    title = question_details.get("title", "")
    options = question_details.get("options", [])
    resolution_criteria = question_details.get("resolution_criteria", "")
    fine_print = question_details.get("fine_print", "")
    today = today_string_for_prompt(question_details.get("as_of_time"))

    options_str = ", ".join([str(o) for o in options]) if isinstance(options, (list, tuple)) else str(options)

    # Fetch relevant news unless pre-fetched context is provided.
    if news_context is None:
        print(f"Fetching news for inside view (MC): {title}")
        news_context = await call_asknews_async(
            question_details,
            as_of_time=question_details.get("as_of_time"),
        )

    # Load prompt template from file
    prompt_template = read_prompt("inside_view_multiple_choice_prompt.txt")
    tool_call_limits = render_tool_call_limits_for_prompt()

    prompt = prompt_template.format(
        title=title,
        options=options_str,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        today=today,
        context=news_context,
        tool_call_limits=tool_call_limits,
    )

    initial_state = {"messages": [HumanMessage(content=prompt)]}
    app = create_agent_graph(model_name=LLM_MODEL)
    question_id = question_details.get("id", "unknown")
    run_config = {
        "configurable": {
            "thread_id": f"inside-mc-{question_id}-{uuid.uuid4().hex}",
        }
    }
    
    with get_openai_callback() as cb:
        final_output = await run_agent_with_streaming(
            app,
            initial_state,
            run_config,
            label="inside_view_mc",
        )
        from src.token_cost import print_token_usage
        print_token_usage(cb, LLM_MODEL, final_output, component="inside_view")

    last_message = final_output["messages"][-1]
    return message_to_text(last_message)


async def generate_inside_view_numeric(
    question_details: dict,
    *,
    news_context: str | None = None,
    units: str,
    lower_bound_message: str,
    upper_bound_message: str,
    hint: str = "",
) -> str:
    """Generate an inside view for numeric/discrete questions."""
    
    title = question_details.get("title", "")
    background = question_details.get("description", "")
    resolution_criteria = question_details.get("resolution_criteria", "")
    fine_print = question_details.get("fine_print", "")
    today = today_string_for_prompt(question_details.get("as_of_time"))

    # Fetch relevant news unless pre-fetched context is provided.
    if news_context is None:
        print(f"Fetching news for inside view (Numeric): {title}")
        news_context = await call_asknews_async(
            question_details,
            as_of_time=question_details.get("as_of_time"),
        )

    # Load prompt template from file
    prompt_template = read_prompt("inside_view_numeric_prompt.txt")
    tool_call_limits = render_tool_call_limits_for_prompt()

    prompt = prompt_template.format(
        title=title,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        units=units,
        lower_bound_message=lower_bound_message,
        upper_bound_message=upper_bound_message,
        hint=hint,
        today=today,
        context=news_context,
        tool_call_limits=tool_call_limits,
    )

    initial_state = {"messages": [HumanMessage(content=prompt)]}
    app = create_agent_graph(model_name=LLM_MODEL)
    question_id = question_details.get("id", "unknown")
    run_config = {
        "configurable": {
            "thread_id": f"inside-num-{question_id}-{uuid.uuid4().hex}",
        }
    }
    
    with get_openai_callback() as cb:
        final_output = await run_agent_with_streaming(
            app,
            initial_state,
            run_config,
            label="inside_view_numeric",
        )
        from src.token_cost import print_token_usage
        print_token_usage(cb, LLM_MODEL, final_output, component="inside_view")

    last_message = final_output["messages"][-1]
    return message_to_text(last_message)


if __name__ == "__main__":
    # Test block
    post_details = get_post_details(41851)
    question_details = post_details["question"]

    async def _run() -> None:
        print(f"Generating inside view for: {question_details['title']}")
        # Note: No outside view passed
        result = await generate_inside_view(question_details)
        print("\n" + "#" * 80)
        print("Inside View\n" + "#" * 80)
        print(result)

    asyncio.run(_run())
