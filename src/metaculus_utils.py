import os
import json
import time
import requests
import dotenv

dotenv.load_dotenv()

# Environment / API constants
AUTH_HEADERS = {"headers": {"Authorization": f"Token {os.getenv('METACULUS_TOKEN')}"}}
API_BASE_URL = "https://www.metaculus.com/api"

def _get_json_with_retries(
    url: str,
    params: dict | None = None,
    *,
    method: str = "GET",
    retries: int = 3,
    timeout: float = 10.0,
) -> dict | list:
    """Fetch JSON from Metaculus with tolerant parsing and retries.

    - Adds Accept and User-Agent headers
    - Retries up to `retries` times on network, non-2xx, and JSON decode errors
    - On retries, prints a short message with attempt number
    - Attempts a tolerant parse replacing NaN/Infinity tokens
    - On final failure, raises RuntimeError with status, content-type, and body snippet
    """
    auth_token = os.getenv("METACULUS_TOKEN") or ""
    headers = {
        "Authorization": f"Token {auth_token}",
        "Accept": "application/json",
        "User-Agent": "hanshal-forecast-bot/1.0",
    }

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.request(method, url, headers=headers, params=params, timeout=timeout)
            if not resp.ok:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            try:
                return json.loads(resp.content)
            except json.JSONDecodeError as e:
                # Try tolerant replacements for NaN/Infinity
                text = resp.text or ""
                cleaned = (
                    text.replace("NaN", "null").replace("Infinity", "null").replace("-Infinity", "null")
                )
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    ct = resp.headers.get("content-type", "")
                    snippet = text[:500]
                    raise RuntimeError(
                        f"Failed to parse JSON from {url} (ct={ct}): {e}. Body snippet: {snippet}"
                    )

        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                # Print retry notice and back off
                print(f"Metaculus API retry {attempt + 1}/{retries} for {url}: {e}")
                sleep_seconds = 0.5 * (2 ** attempt)
                time.sleep(sleep_seconds)
                continue
            break

    # Final failure
    if isinstance(last_err, RuntimeError):
        raise last_err
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")

def post_question_comment(post_id: int, comment_text: str) -> None:
    """
    Post a comment on the question page as the bot user.
    """

    response = requests.post(
        f"{API_BASE_URL}/comments/create/",
        json={
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        },
        **AUTH_HEADERS,  # type: ignore
    )
    if not response.ok:
        raise RuntimeError(response.text)

def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    """
    Post a forecast on a question.
    """
    url = f"{API_BASE_URL}/questions/forecast/"
    response = requests.post(
        url,
        json=[
            {
                "question": question_id,
                **forecast_payload,
            },
        ],
        **AUTH_HEADERS,  # type: ignore
    )
    # Status code is checked below; avoid noisy printing in library code
    if not response.ok:
        raise RuntimeError(response.text)


def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict:
    """
    Accepts a forecast and generates the api payload in the correct format.

    If the question is binary, forecast must be a float.
    If the question is multiple choice, forecast must be a dictionary that
      maps question.options labels to floats.
    If the question is numeric, forecast must be a dictionary that maps
      quartiles or percentiles to datetimes, or a 201 value cdf.
    """
    if question_type == "binary":
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    # numeric or date
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }


def list_posts_from_tournament(
    tournament_id: int | str | None = None, offset: int = 0, count: int = 50
) -> list[dict]:
    """
    List (all details) {count} posts from the {tournament_id}
    """
    # determine tournament id: prefer argument, then env, else sensible default
    if tournament_id is None:
        tournament_id = os.getenv("TOURNAMENT_ID") or "spring-aib-2026"

    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join(
            [
                "binary",
                "multiple_choice",
                "numeric",
                "discrete",
            ]
        ),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    data = _get_json_with_retries(url, params=url_qparams)
    return data  # type: ignore[return-value]


def get_open_question_ids_from_tournament(tournament_id: int | str | None = None) -> list[tuple[int, int]]:
    posts = list_posts_from_tournament(tournament_id)

    post_dict = dict()
    for post in posts["results"]:
        if question := post.get("question"):
            # single question post
            post_dict[post["id"]] = [question]

    open_question_id_post_id = []  # [(question_id, post_id)]
    for post_id, questions in post_dict.items():
        for question in questions:
            if question.get("status") == "open":
                # Keep library quiet; caller can log if desired
                open_question_id_post_id.append((question["id"], post_id))

    return open_question_id_post_id


def get_post_details(post_id: int) -> dict:
    """
    Get all details about a post from the Metaculus API.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    details = _get_json_with_retries(url)
    return details  # type: ignore[return-value]


def extract_slug(post_json: dict) -> str:
    """
    Extract the canonical slug for matching across mirrored posts.

    Prefer 'question.slug' (stable across tournaments) and fall back to 'post.slug'.

    Raises RuntimeError if no slug can be found.
    """
    if not isinstance(post_json, dict):
        raise RuntimeError("Post data is not a valid dictionary")

    # Prefer the question slug – this is consistent across public and tournament mirrors
    if "question" in post_json and isinstance(post_json["question"], dict):
        question = post_json["question"]
        if "slug" in question:
            slug = question["slug"]
            if isinstance(slug, str) and slug.strip():
                return slug.strip()
            else:
                raise RuntimeError("Question slug field exists but is empty or invalid")

    # Fall back to the post slug
    if "slug" in post_json:
        slug = post_json["slug"]
        if isinstance(slug, str) and slug.strip():
            return slug.strip()
        else:
            raise RuntimeError("Slug field exists but is empty or invalid")

    raise RuntimeError("No valid slug found in post data (checked both question.slug and post.slug)")


def extract_question_slug(post_json: dict) -> str:
    """
    Extract only the question.slug. Raises if unavailable.
    """
    if not isinstance(post_json, dict):
        raise RuntimeError("Post data is not a valid dictionary")
    question = post_json.get("question")
    if not isinstance(question, dict):
        raise RuntimeError("Post does not contain a valid question object")
    slug = question.get("slug")
    if isinstance(slug, str) and slug.strip():
        return slug.strip()
    raise RuntimeError("Question slug field is missing or invalid")


def strings_equal_ci(a: str | None, b: str | None) -> bool:
    """
    Case-insensitive equality check for strings, safely handling None.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return a.strip().lower() == b.strip().lower()


def is_public_post(post_json: dict) -> bool:
    """
    Heuristic: a public community post generally has no tournaments attached.
    """
    try:
        tournaments = post_json.get("tournaments")
        if isinstance(tournaments, list) and len(tournaments) == 0:
            return True
        return False
    except Exception:
        return False


def search_posts_by_slug(slug: str, limit: int = 50) -> list[dict]:
    """
    Search for posts matching the given slug using Metaculus API.
    Returns list of post summaries.

    Raises RuntimeError if search fails or returns invalid data.
    """
    if not isinstance(slug, str) or not slug.strip():
        raise RuntimeError("Invalid slug provided for search")

    try:
        url = f"{API_BASE_URL}/posts/"
        params = {
            "search": slug,
            "limit": limit,
            "order_by": "-activity"
        }
        data = _get_json_with_retries(url, params=params)

        if not isinstance(data, dict):
            raise RuntimeError("Search response is not a valid JSON object")

        results = data.get("results", [])
        if not isinstance(results, list):
            raise RuntimeError("Search results is not a valid list")

        return results

    except requests.RequestException as e:
        raise RuntimeError(f"Network error during search: {e}")

def forecast_is_already_made(post_details: dict) -> bool:
    """
    Check if a forecast has already been made by looking at my_forecasts in the question data.

    question.my_forecasts.latest.forecast_values has the following values for each question type:
    Binary: [probability for no, probability for yes]
    Numeric: [cdf value 1, cdf value 2, ..., cdf value 201]
    Multiple Choice: [probability for option 1, probability for option 2, ...]
    """
    try:
        forecast_values = post_details["question"]["my_forecasts"]["latest"][
            "forecast_values"
        ]
        return forecast_values is not None
    except Exception:
        return False
