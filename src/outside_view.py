import datetime
from typing import Dict, List

from .utils import (
    read_prompt,
    call_llm,
    get_historical_research_questions,
    get_exa_answers,
)


LLM_MODEL = "openai/gpt-5-mini"


HISTORICAL_QUESTIONS_PROMPT = read_prompt("historical_questions_prompt.txt")
OUTSIDE_VIEW_PROMPT = read_prompt("outside_view_prompt.txt")


def _format_historical_context(qa_by_question: Dict[str, str]) -> str:
    if not qa_by_question:
        return "No historical context was found."

    sections: List[str] = []
    for question, answer in qa_by_question.items():
        sections.append(f"Question: {question}\nAnswer: {answer}")

    return "\n\n".join(sections)


async def prepare_outside_view_context(question_details: dict) -> str:
    """Prepare historical context once by generating questions and Exa answers."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    title = question_details.get("title", "")
    background = question_details.get("description", "")
    resolution_criteria = question_details.get("resolution_criteria", "")
    fine_print = question_details.get("fine_print", "")

    hist_questions_content = HISTORICAL_QUESTIONS_PROMPT.format(
        title=title,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        today=today,
    )
    questions: List[str] = await get_historical_research_questions(hist_questions_content)

    for question in questions:
        print("question", question)
        print("--------------------------------")

    qa_map: Dict[str, str] = get_exa_answers(questions)

    for question, answer in qa_map.items():
        print("question", question)
        print("answer", answer)
        print("--------------------------------")

    context_block = _format_historical_context(qa_map)
    return context_block


async def generate_outside_view(question_details: dict, historical_context: str | None = None) -> str:
    """Generate an outside view using historical research questions and Exa answers.

    Args:
        question_details: A dict containing keys: "title", "description" (background),
            "resolution_criteria", and "fine_print".

    Returns:
        The outside view analysis produced by the LLM.
    """
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    title = question_details.get("title", "")
    background = question_details.get("description", "")
    resolution_criteria = question_details.get("resolution_criteria", "")
    fine_print = question_details.get("fine_print", "")

    # Step 1-3: Use precomputed context if provided, else prepare it now
    context_block = historical_context if historical_context is not None else (
        await prepare_outside_view_context(question_details)
    )

    # Step 4: Build outside view prompt and call LLM
    outside_view_content = OUTSIDE_VIEW_PROMPT.format(
        title=title,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        today=today,
        context=context_block,
    )

    outside_view = await call_llm(outside_view_content, LLM_MODEL, 0.3)
    return outside_view

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
        print(f"Generating outside view for: {question_details['title']}")
        result = await generate_outside_view(question_details)
        print("\n" + "#" * 80)
        print("Outside View\n" + "#" * 80)
        print(result)

    asyncio.run(_run())