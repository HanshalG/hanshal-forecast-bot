from dotenv import load_dotenv
from pathlib import Path
import asyncio
import os
import re
import time
import numpy as np
from asknews_sdk import AskNewsSDK
from openai import AsyncOpenAI
from exa_py import Exa

load_dotenv()

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Constants
CONCURRENT_REQUESTS_LIMIT = 5
llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)

# Environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
EXA_API_KEY = os.getenv("EXA_API_KEY")

exa = Exa(api_key=EXA_API_KEY)

async def get_exa_research_report(content: str) -> str:
    """Get research report by asking questions to Exa."""
    # Get research questions from LLM
    questions_response = await call_llm(content, "gpt-5-mini", 0.3)

    #print("questions_response", questions_response)
    # Extract questions from response
    questions = [q.strip() for q in questions_response.split('Question:') if q.strip()]
    questions = questions[:10] # limit the number of questions to 10
    # for question in questions:
    #     print("question", question)
    #     print("--------------------------------")

    # Get answers for each question
    reports = []
    for question in questions:
        try:
            response = exa.answer(question)

            # Exa Answer API typically returns an object with `answer` and `citations`
            answer_text = None
            if response is not None:
                reports.append("\n\n"+question)
                # Object attribute access
                answer_text = getattr(response, "answer", None)
                # Dict-like fallback
                if answer_text is None and isinstance(response, dict):
                    answer_text = response.get("answer") or response.get("text")

            citation_lines: list[str] = []
            citations = getattr(response, "citations", None)
            if citations and isinstance(citations, list):
                for idx, c in enumerate(citations, start=1):
                    url = getattr(c, "url", None) if hasattr(c, "url") else (
                        c.get("url") if isinstance(c, dict) else None
                    )
                    title = getattr(c, "title", None) if hasattr(c, "title") else (
                        c.get("title") if isinstance(c, dict) else None
                    )
                    if url:
                        if title:
                            citation_lines.append(f"[{idx}] {title} - {url}")
                        else:
                            citation_lines.append(f"[{idx}] {url}")

            if answer_text:
                if citation_lines:
                    reports.append(f"{answer_text}\n\nSources:\n" + "\n".join(citation_lines))
                else:
                    reports.append(answer_text)
        except Exception as e:
            reports.append(f"Error fetching Exa answer: {e}")

    result = "\n\n".join(reports) if reports else "No research results found."
    print("Exa Research Report:\n\n", result)
    # Combine all reports
    return result

def read_prompt(filename: str) -> str:
    """Read a prompt template from the prompts directory."""
    path = PROMPTS_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {path}")

async def call_llm(prompt: str, model: str, temperature: float) -> str:
    """
    Makes a completion request to OpenRouter (OpenAI SDK compatible) with concurrent request limiting.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    async with llm_rate_limiter:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=False,
        )
        answer = response.choices[0].message.content
        if answer is None:
            raise ValueError("No answer returned from LLM")
        return answer

def call_asknews(question: str) -> str:
    """
    Use the AskNews `news` endpoint to get news context for your query.
    The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
    """
    if not ASKNEWS_CLIENT_ID or not ASKNEWS_SECRET:
        raise ValueError("ASKNEWS_CLIENT_ID and ASKNEWS_SECRET environment variables must be set")

    ask = AskNewsSDK(
        client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_SECRET, scopes=set(["news"])
    )

    # get the latest news related to the query (within the past 48 hours)
    hot_response = ask.news.search_news(
        query=question,  # your natural language query
        n_articles=6,  # control the number of articles to include in the context, originally 5
        return_type="both",
        strategy="latest news",  # enforces looking at the latest news only
    )

    time.sleep(10)

    #get context from the "historical" database that contains a news archive going back to 2023
    historical_response = ask.news.search_news(
        query=question,
        n_articles=10,
        return_type="both",
        strategy="news knowledge",  # looks for relevant news within the past 60 days
    )

    hot_articles = hot_response.as_dicts
    historical_articles = historical_response.as_dicts
    formatted_articles = "Here are the relevant news articles:\n\n"

    if hot_articles:
        hot_articles = [article.__dict__ for article in hot_articles]
        hot_articles = sorted(hot_articles, key=lambda x: x["pub_date"], reverse=True)

        for article in hot_articles:
            pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
            formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

    if historical_articles:
        historical_articles = [article.__dict__ for article in historical_articles]
        historical_articles = sorted(
            historical_articles, key=lambda x: x["pub_date"], reverse=True
        )

        for article in historical_articles:
            pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
            formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

    if not hot_articles and not historical_articles:
        formatted_articles += "No articles were found.\n\n"
        return formatted_articles

    return formatted_articles

def extract_probability_from_response_as_percentage_not_decimal(
    forecast_text: str,
) -> float:
    matches = re.findall(r"(\d+)%", forecast_text)
    if matches:
        # Return the last number found before a '%'
        number = int(matches[-1])
        number = min(99, max(1, number))  # clamp the number between 1 and 99
        return number
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")

def extract_percentiles_from_response(forecast_text: str) -> dict:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_percentile_numbers(text) -> dict:
        pattern = r"^.*(?:P|p)ercentile.*$"
        number_pattern = r"-\s*(?:[^\d\-]*\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)|(\d+(?:,\d{3})*(?:\.\d+)?)"
        results = []

        for line in text.split("\n"):
            if re.match(pattern, line):
                numbers = re.findall(number_pattern, line)
                numbers_no_commas = [
                    next(num for num in match if num).replace(",", "")
                    for match in numbers
                ]
                numbers = [
                    float(num) if "." in num else int(num)
                    for num in numbers_no_commas
                ]
                if len(numbers) > 1:
                    first_number = numbers[0]
                    last_number = numbers[-1]
                    # Check if the original line had a negative sign before the last number
                    if "-" in line.split(":")[-1]:
                        last_number = -abs(last_number)
                    results.append((first_number, last_number))

        # Convert results to dictionary
        percentile_values = {}
        for first_num, second_num in results:
            key = first_num
            percentile_values[key] = second_num

        return percentile_values

    percentile_values = extract_percentile_numbers(forecast_text)

    if len(percentile_values) > 0:
        return percentile_values
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")

def generate_continuous_cdf(
    percentile_values: dict,
    question_type: str,
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float,
    lower_bound: float,
    zero_point: float | None,
    cdf_size: int,
) -> list[float]:
    """
    Returns: list[float]: A list of 201 float values representing the CDF.
    """

    percentile_max = max(float(key) for key in percentile_values.keys())
    percentile_min = min(float(key) for key in percentile_values.keys())
    range_min = lower_bound
    range_max = upper_bound
    range_size = range_max - range_min
    buffer = 1 if range_size > 100 else 0.01 * range_size

    # Adjust any values that are exactly at the bounds
    for percentile, value in list(percentile_values.items()):
        if not open_lower_bound and value <= range_min + buffer:
            percentile_values[percentile] = range_min + buffer
        if not open_upper_bound and value >= range_max - buffer:
            percentile_values[percentile] = range_max - buffer

    # Set cdf values outside range
    if open_upper_bound:
        if range_max > percentile_values[percentile_max]:
            percentile_values[int(100 - (0.5 * (100 - percentile_max)))] = range_max
    else:
        percentile_values[100] = range_max

    # Set cdf values outside range
    if open_lower_bound:
        if range_min < percentile_values[percentile_min]:
            percentile_values[int(0.5 * percentile_min)] = range_min
    else:
        percentile_values[0] = range_min

    sorted_percentile_values = dict(sorted(percentile_values.items()))

    # Normalize percentile keys
    normalized_percentile_values = {}
    for key, value in sorted_percentile_values.items():
        percentile = float(key) / 100
        normalized_percentile_values[percentile] = value


    value_percentiles = {
        value: key for key, value in normalized_percentile_values.items()
    }

    # function for log scaled questions
    def generate_cdf_locations(range_min, range_max, zero_point):
        if zero_point is None:
            scale = lambda x: range_min + (range_max - range_min) * x
        else:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            scale = lambda x: range_min + (range_max - range_min) * (
                deriv_ratio**x - 1
            ) / (deriv_ratio - 1)
        return [scale(x) for x in np.linspace(0, 1, cdf_size)]

    cdf_xaxis = generate_cdf_locations(range_min, range_max, zero_point)

    def linear_interpolation(x_values, xy_pairs):
        # Sort the xy_pairs by x-values
        sorted_pairs = sorted(xy_pairs.items())

        # Extract sorted x and y values
        known_x = [pair[0] for pair in sorted_pairs]
        known_y = [pair[1] for pair in sorted_pairs]

        # Initialize the result list
        y_values = []

        for x in x_values:
            # Check if x is exactly in the known x values
            if x in known_x:
                y_values.append(known_y[known_x.index(x)])
            else:
                # Find the indices of the two nearest known x-values
                i = 0
                while i < len(known_x) and known_x[i] < x:
                    i += 1

                list_index_2 = i

                # If x is outside the range of known x-values, use the nearest endpoint
                if i == 0:
                    y_values.append(known_y[0])
                elif i == len(known_x):
                    y_values.append(known_y[-1])
                else:
                    # Perform linear interpolation
                    x0, x1 = known_x[i - 1], known_x[i]
                    y0, y1 = known_y[i - 1], known_y[i]

                    # Linear interpolation formula
                    y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                    y_values.append(y)

        return y_values

    continuous_cdf = linear_interpolation(cdf_xaxis, value_percentiles)
    return continuous_cdf

def extract_option_probabilities_from_response(forecast_text: str, options) -> float:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_option_probabilities(text):

        # Number extraction pattern
        number_pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?"

        results = []

        # Iterate through each line in the text
        for line in text.split("\n"):
            # Extract all numbers from the line
            numbers = re.findall(number_pattern, line)
            numbers_no_commas = [num.replace(",", "") for num in numbers]
            # Convert strings to float or int
            numbers = [
                float(num) if "." in num else int(num) for num in numbers_no_commas
            ]
            # Add the tuple of numbers to results
            if len(numbers) >= 1:
                last_number = numbers[-1]
                results.append(last_number)

        return results

    option_probabilities = extract_option_probabilities(forecast_text)

    NUM_OPTIONS = len(options)

    if len(option_probabilities) > 0:
        # return the last NUM_OPTIONS items
        return option_probabilities[-NUM_OPTIONS:]
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")