import os
import json
import requests
import dotenv

dotenv.load_dotenv()

# Environment / API constants
AUTH_HEADERS = {"headers": {"Authorization": f"Token {os.getenv('METACULUS_TOKEN')}"}}
API_BASE_URL = "https://www.metaculus.com/api"

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
    print(f"Prediction Post status code: {response.status_code}")
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
        tournament_id = os.getenv("TOURNAMENT_ID") or "fall-aib-2025"

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
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)  # type: ignore
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data


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
                print(
                    f"ID: {question['id']}\nQ: {question['title']}\nCloses: "
                    f"{question['scheduled_close_time']}"
                )
                open_question_id_post_id.append((question["id"], post_id))

    return open_question_id_post_id


def get_post_details(post_id: int) -> dict:
    """
    Get all details about a post from the Metaculus API.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    print(f"Getting details for {url}")
    response = requests.get(
        url,
        **AUTH_HEADERS,  # type: ignore
    )
    if not response.ok:
        raise Exception(response.text)
    details = json.loads(response.content)
    return details


def extract_slug(post_json: dict) -> str:
    """
    Extract the slug from a post JSON. Tries 'slug' first, then 'question.slug'.

    Raises RuntimeError if no slug can be found.
    """
    if not isinstance(post_json, dict):
        raise RuntimeError("Post data is not a valid dictionary")

    if "slug" in post_json:
        slug = post_json["slug"]
        if isinstance(slug, str) and slug.strip():
            return slug.strip()
        else:
            raise RuntimeError("Slug field exists but is empty or invalid")

    if "question" in post_json and isinstance(post_json["question"], dict):
        question = post_json["question"]
        if "slug" in question:
            slug = question["slug"]
            if isinstance(slug, str) and slug.strip():
                return slug.strip()
            else:
                raise RuntimeError("Question slug field exists but is empty or invalid")

    raise RuntimeError("No valid slug found in post data (checked both post.slug and question.slug)")


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
        response = requests.get(url, **AUTH_HEADERS, params=params)  # type: ignore

        if not response.ok:
            raise RuntimeError(f"Search API request failed with status {response.status_code}: {response.text}")

        try:
            data = json.loads(response.content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse search response as JSON: {e}")

        if not isinstance(data, dict):
            raise RuntimeError("Search response is not a valid JSON object")

        results = data.get("results", [])
        if not isinstance(results, list):
            raise RuntimeError("Search results is not a valid list")

        return results

    except requests.RequestException as e:
        raise RuntimeError(f"Network error during search: {e}")


def extract_num_forecasters(post: dict) -> int:
    """
    Extract the number of forecasters from a post dict.

    Returns 0 if no valid forecaster count can be found.
    """
    if not isinstance(post, dict):
        return 0

    try:
        question = post.get("question", {})
        if not isinstance(question, dict):
            return 0

        # Try num_forecasters first (most reliable)
        if "num_forecasters" in question:
            forecasters = question["num_forecasters"]
            if isinstance(forecasters, int) and forecasters >= 0:
                return forecasters

        # Fall back to other possible fields
        for field in ["num_predictions", "forecast_count"]:
            if field in question:
                count = question[field]
                if isinstance(count, int) and count >= 0:
                    return count

        return 0
    except (KeyError, TypeError, ValueError):
        return 0


def extract_recency_weighted_yes_pct(post: dict) -> float:
    """
    Extract the recency-weighted YES percentage from a post.

    Handles various edge cases where community prediction data might be missing:
    - Missing aggregations entirely
    - Missing recency_weighted data
    - Empty or invalid current/history data
    - Missing means/centers fields
    - Invalid data formats
    - Empty history
    """
    try:
        # Check if question exists
        if "question" not in post:
            raise RuntimeError("Post does not contain a question field")

        question = post["question"]

        # Check if aggregations exist
        if "aggregations" not in question:
            raise RuntimeError("Question does not have aggregations data")

        aggs = question["aggregations"]

        # Check if recency_weighted exists
        if "recency_weighted" not in aggs:
            raise RuntimeError("No recency_weighted aggregations found")

        recency_weighted = aggs["recency_weighted"]

        # Ensure recency_weighted is a dict
        if not isinstance(recency_weighted, dict):
            raise RuntimeError("recency_weighted data is not in expected format")

        vec = None

        # Try to get current data first
        if "current" in recency_weighted and recency_weighted["current"]:
            current = recency_weighted["current"]
            if isinstance(current, dict):
                vec = current.get("means") or current.get("centers")

        # Fall back to history if no current data
        if vec is None and "history" in recency_weighted:
            history = recency_weighted["history"]
            if isinstance(history, list) and len(history) > 0:
                last = history[-1]
                if isinstance(last, dict):
                    vec = last.get("means") or last.get("centers")

        # Validate vec exists and is valid
        if vec is None:
            raise RuntimeError("No valid means or centers data found in current or history")

        # Handle empty sequences
        if isinstance(vec, (list, tuple)) and len(vec) == 0:
            raise RuntimeError("Community prediction data is empty")

        # Handle None values in lists
        if isinstance(vec, (list, tuple)):
            if vec[0] is None:
                raise RuntimeError("Community prediction contains None values")
            return float(vec[0] * 100.0)

        # Handle single values
        if vec is None:
            raise RuntimeError("Community prediction value is None")
        return float(vec * 100.0)

    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise RuntimeError(f"Failed to extract community prediction: {e}")


def get_metaculus_community_prediction(post_id: int) -> float:
    """
    Fetch the community prediction from the post with the most forecasters
    that shares the same slug as the given post. Returns the recency-weighted
    mean for YES as a percentage in [0, 100].

    First fetches the original post's slug, searches for matching posts,
    filters to exact slug matches, and selects the post with the most forecasters.
    If multiple posts found, returns the one with most forecasters.
    If only one post found, returns that post's prediction.

    Handles comprehensive error cases:
    - Post not found or API failures
    - Missing slug in post data
    - Search API failures
    - No valid question posts found
    - Missing or invalid community prediction data
    """
    try:
        # Get the original post details and extract slug
        seed_post = get_post_details(post_id)
        slug = extract_slug(seed_post)

        # Search for posts with matching slug
        summaries = search_posts_by_slug(slug)

        # Filter to exact slug matches and get full details
        candidates = []
        for summary in summaries:
            if summary.get("slug") == slug:
                try:
                    full_post = get_post_details(summary["id"])
                    # Only include posts that have questions (forecastable) and are not group posts
                    if "question" in full_post and "group_of_questions" not in full_post:
                        candidates.append(full_post)
                    # Skip group posts and non-question posts silently
                except Exception as e:
                    # Skip posts that can't be fetched
                    continue

        # Handle different numbers of candidates
        if len(candidates) == 0:
            # Check if the original post itself is a valid candidate
            if "question" in seed_post and "group_of_questions" not in seed_post:
                best_post = seed_post
            else:
                raise RuntimeError(f"Found no posts with slug '{slug}' that have questions, and original post is not a valid question post.")
        elif len(candidates) == 1:
            best_post = candidates[0]
        else:
            # Select the post with the most forecasters
            best_post = max(candidates, key=extract_num_forecasters)

        # Extract and return the community prediction
        return extract_recency_weighted_yes_pct(best_post)

    except RuntimeError:
        # Re-raise RuntimeError with original message (from extract_recency_weighted_yes_pct)
        raise
    except Exception as e:
        # Catch any other unexpected errors and provide context
        raise RuntimeError(f"Failed to get community prediction for post {post_id}: {e}")


