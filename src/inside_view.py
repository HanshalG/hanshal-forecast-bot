import datetime
import asyncio
from typing import Dict, List

from .utils import (
    read_prompt,
    call_llm,
    run_research,
    run_research_async,
    get_current_research_questions,
    get_exa_answers,
    get_exa_answers_async,
)


LLM_MODEL = "openai/gpt-5"


INSIDE_VIEW_PROMPT = read_prompt("inside_view_prompt.txt")
CURRENT_QUESTIONS_PROMPT = read_prompt("current_questions_prompt.txt")
INSIDE_VIEW_MULTIPLE_CHOICE_PROMPT = read_prompt("inside_view_multiple_choice_prompt.txt")
INSIDE_VIEW_NUMERIC_PROMPT = read_prompt("inside_view_numeric_prompt.txt")


async def prepare_inside_view_context(
    question_details: dict,
    today: str | None = None,
) -> tuple[str, str]:
    """Prepare news and Exa targeted research contexts once."""
    ts = today or datetime.datetime.now().strftime("%Y-%m-%d")

    title = question_details.get("title", "")
    background = question_details.get("description", "")
    resolution_criteria = question_details.get("resolution_criteria", "")
    fine_print = question_details.get("fine_print", "")

    # Build prompt for Exa question generation
    current_qs_prompt = CURRENT_QUESTIONS_PROMPT.format(
        title=title,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        today=ts,
    )

    # Run AskNews fetch/filter and Exa question generation concurrently
    print("Running AskNews fetch/filter concurrently")
    news_task = run_research_async(question_details)

    questions_task = get_current_research_questions(current_qs_prompt)

    # Gather; if AskNews fails after retries it should raise and propagate up
    news_context, current_questions = await asyncio.gather(news_task, questions_task)
    if not isinstance(news_context, str) or news_context.strip() == "":
        raise RuntimeError("AskNews returned empty context after retries")
    if not isinstance(current_questions, (list, tuple)):
        current_questions = []

    for question in current_questions:
        print("Question: ", question[:100], "...")

    # Fetch Exa answers concurrently as well
    exa_answers_by_q: Dict[str, str] = {}
    if current_questions:
        try:
            print("Running Exa question generation concurrently")
            exa_answers_by_q = await get_exa_answers_async(current_questions)
        except Exception as e:
            print(f"Running Exa question generation synchronously due to error: {e}")
            try:
                exa_answers_by_q = get_exa_answers(current_questions)
            except Exception as e2:
                print(f"Exa sync also failed: {e2}")
                exa_answers_by_q = {}
    else:
        print("No current questions; skipping Exa answers")

    # Filter out any Q/A where the question or answer contains apology text
    filtered_exa_answers_by_q: Dict[str, str] = {}
    i=0
    for q, a in exa_answers_by_q.items():
        q_lower = q.lower()
        a_lower = (a or "").lower()
        if ("i am sorry" in q_lower) or ("i'm sorry" in q_lower) or ("i am sorry" in a_lower) or ("i'm sorry" in a_lower):
            i += 1
            continue
        filtered_exa_answers_by_q[q] = a
    print(f"\nFiltered out {i} questions")

    for question, answer in filtered_exa_answers_by_q.items():
        print("\nQuestion: ", question)
        print("Answer: ", answer)
        print("--------------------------------")

    exa_sections = [f"Question: {q}\nAnswer: {a}" for q, a in filtered_exa_answers_by_q.items()]
    exa_context = "\n\n".join(exa_sections) if exa_sections else "No targeted research found."
    return news_context, exa_context


async def generate_inside_view(
    question_details: dict,
    outside_view: str,
    precomputed_news_context: str | None = None,
    precomputed_exa_context: str | None = None,
) -> str:
    """Generate an inside view using AskNews context and targeted Exa research.

    Args:
        question_details: A dict with keys: "title", "description", "resolution_criteria", "fine_print".

    Returns:
        The inside view analysis.
    """

    today = datetime.datetime.now().strftime("%Y-%m-%d")

    title = question_details.get("title", "")
    background = question_details.get("description", "")
    resolution_criteria = question_details.get("resolution_criteria", "")
    fine_print = question_details.get("fine_print", "")

    # 1) Contexts (use precomputed if provided)
    if precomputed_news_context is not None and precomputed_exa_context is not None:
        news_context = precomputed_news_context
        exa_context = precomputed_exa_context
    else:
        news_context, exa_context = await prepare_inside_view_context(question_details, today)


    # 3) Compose inside view prompt
    # Combine outside view analysis (passed in) + current info into a single {context}
    combined_context = (
        f"Outside View Analysis\n{outside_view}\n\n"
        f"News Context\n{news_context}\n\n"
        f"Targeted Research\n{exa_context}"
    )

    inside_view_content = INSIDE_VIEW_PROMPT.format(
        title=title,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        today=today,
        context=combined_context,
    )

    inside_view = await call_llm(inside_view_content, LLM_MODEL, 0.3)
    return inside_view


async def generate_inside_view_multiple_choice(
    question_details: dict,
    outside_view: str,
    *,
    precomputed_news_context: str | None = None,
    precomputed_exa_context: str | None = None,
) -> str:
    """Generate an inside view for multiple choice questions.

    Returns the inside view analysis expected to end with a line formatted as:
    Probabilities: [p1, p2, ..., pN]
    """

    today = datetime.datetime.now().strftime("%Y-%m-%d")

    title = question_details.get("title", "")
    background = question_details.get("description", "")
    resolution_criteria = question_details.get("resolution_criteria", "")
    fine_print = question_details.get("fine_print", "")
    options = question_details.get("options", [])

    # Contexts
    if precomputed_news_context is not None and precomputed_exa_context is not None:
        news_context = precomputed_news_context
        exa_context = precomputed_exa_context
    else:
        news_context, exa_context = await prepare_inside_view_context(question_details, today)

    combined_context = (
        f"Outside View Analysis\n{outside_view}\n\n"
        f"News Context\n{news_context}\n\n"
        f"Targeted Research\n{exa_context}"
    )

    options_str = ", ".join([str(o) for o in options]) if isinstance(options, (list, tuple)) else str(options)

    content = INSIDE_VIEW_MULTIPLE_CHOICE_PROMPT.format(
        title=title,
        options=options_str,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        today=today,
        context=combined_context,
    )

    return await call_llm(content, LLM_MODEL, 0.3)


async def generate_inside_view_numeric(
    question_details: dict,
    outside_view: str,
    *,
    units: str,
    lower_bound_message: str,
    upper_bound_message: str,
    hint: str = "",
    precomputed_news_context: str | None = None,
    precomputed_exa_context: str | None = None,
) -> str:
    """Generate an inside view for numeric/discrete questions that yields percentiles."""

    today = datetime.datetime.now().strftime("%Y-%m-%d")

    title = question_details.get("title", "")
    background = question_details.get("description", "")
    resolution_criteria = question_details.get("resolution_criteria", "")
    fine_print = question_details.get("fine_print", "")

    # Contexts
    if precomputed_news_context is not None and precomputed_exa_context is not None:
        news_context = precomputed_news_context
        exa_context = precomputed_exa_context
    else:
        news_context, exa_context = await prepare_inside_view_context(question_details, today)

    combined_context = (
        f"Outside View Analysis\n{outside_view}\n\n"
        f"News Context\n{news_context}\n\n"
        f"Targeted Research\n{exa_context}"
    )

    content = INSIDE_VIEW_NUMERIC_PROMPT.format(
        title=title,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        units=units,
        today=today,
        lower_bound_message=lower_bound_message,
        upper_bound_message=upper_bound_message,
        hint=hint,
        context=combined_context,
    )

    return await call_llm(content, LLM_MODEL, 0.3)


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    # Ensure project root is importable to access main.EXAMPLE_QUESTIONS
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from main import EXAMPLE_QUESTIONS  # type: ignore
    from .metaculus_utils import get_post_details

    # Use the first example question
    example_question_id, example_post_id = EXAMPLE_QUESTIONS[0]
    post_details = get_post_details(example_post_id)
    question_details = post_details["question"]

    async def _run() -> None:
        print(f"Generating inside view for: {question_details['title']}")
        # Generate outside view first and pass it as context
        from .outside_view import generate_outside_view
        outside_view_text = await generate_outside_view(question_details)
        result = await generate_inside_view(question_details, outside_view_text)
        print("\n" + "#" * 80)
        print("Inside View\n" + "#" * 80)
        print(result)

    asyncio.run(_run())


