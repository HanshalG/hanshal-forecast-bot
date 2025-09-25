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

        # Try aggregations field which might contain forecaster data
        if "aggregations" in question:
            aggregations = question["aggregations"]
            if isinstance(aggregations, dict):
                # Look for forecaster_count in various aggregation types.
                # Prefer unweighted over recency_weighted to avoid inflated RW counts.
                for agg_type in ["unweighted", "recency_weighted"]:
                    if agg_type in aggregations and isinstance(aggregations[agg_type], dict):
                        agg_data = aggregations[agg_type]
                        if "latest" in agg_data and isinstance(agg_data["latest"], dict):
                            latest = agg_data["latest"]
                            if "forecaster_count" in latest:
                                count = latest["forecaster_count"]
                                if isinstance(count, int) and count >= 0:
                                    return count

                # Also check direct fields in aggregations
                for agg_field in ["forecast_count", "num_forecasters", "total_forecasters", "count"]:
                    if agg_field in aggregations:
                        count = aggregations[agg_field]
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

        # Helper to extract a percentage from a single aggregation block
        def _extract_from_agg(agg: dict) -> float | None:
            if not isinstance(agg, dict):
                return None

            # Prefer explicit point-in-time objects first
            for key in ("current", "latest"):
                node = agg.get(key)
                if isinstance(node, dict):
                    vec = node.get("means") or node.get("centers")
                    # Fallback: binary questions sometimes expose forecast_values [p_no, p_yes]
                    if vec is None and isinstance(node.get("forecast_values"), (list, tuple)):
                        fv = node["forecast_values"]
                        if len(fv) >= 2 and isinstance(fv[1], (int, float)):
                            return float(fv[1] * 100.0)

                    if isinstance(vec, (list, tuple)):
                        if len(vec) == 0 or vec[0] is None:
                            return None
                        return float(vec[0] * 100.0)
                    if isinstance(vec, (int, float)):
                        return float(vec * 100.0)

            # Historical fallback
            history = agg.get("history")
            if isinstance(history, list) and len(history) > 0:
                last = history[-1]
                if isinstance(last, dict):
                    vec = last.get("means") or last.get("centers")
                    if isinstance(vec, (list, tuple)):
                        if len(vec) == 0 or vec[0] is None:
                            return None
                        return float(vec[0] * 100.0)
                    if isinstance(vec, (int, float)):
                        return float(vec * 100.0)

            return None

        # Try multiple aggregation types in order of preference
        for agg_key in ("recency_weighted", "unweighted", "metaculus_prediction", "single_aggregation"):
            agg_block = aggs.get(agg_key)
            pct = _extract_from_agg(agg_block) if agg_block is not None else None
            if pct is not None:
                return pct

        raise RuntimeError("No valid community prediction found in aggregations")

    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise RuntimeError(f"Failed to extract community prediction: {e}")


def get_metaculus_community_prediction_and_count(post_id: int) -> tuple[float, int]:
    """
    Return a tuple of (community_prediction_pct, n_forecasters) for the post
    with the most forecasters that shares the same slug as the given post.

    - community_prediction_pct: recency-weighted YES mean as percentage [0, 100]
    - n_forecasters: uses the same selected post; prefers recency-weighted
      latest forecaster_count when available, otherwise falls back via
      extract_num_forecasters.
    """
    try:
        seed_post = get_post_details(post_id)
        # Target the canonical question slug if possible
        try:
            target_slug = extract_question_slug(seed_post)
        except Exception:
            target_slug = extract_slug(seed_post)

        summaries = search_posts_by_slug(target_slug)

        # Allow matching by either question.slug or title as a fallback
        seed_title = seed_post.get("title") if isinstance(seed_post, dict) else None

        candidates = []
        for summary in summaries:
            try:
                full_post = get_post_details(summary["id"])
            except Exception:
                continue
            # Only consider forecastable non-group posts
            if "question" not in full_post or "group_of_questions" in full_post:
                continue
            # Prefer match on question.slug; fallback to title equality if slug differs across mirrors
            matches_slug = False
            try:
                matches_slug = strings_equal_ci(extract_question_slug(full_post), target_slug)
            except Exception:
                matches_slug = False
            matches_title = strings_equal_ci(full_post.get("title"), seed_title)
            if matches_slug or matches_title:
                candidates.append(full_post)

        if len(candidates) == 0:
            if "question" in seed_post and "group_of_questions" not in seed_post:
                best_post = seed_post
            else:
                raise RuntimeError(
                    f"Found no posts with slug '{target_slug}' that have questions, and original post is not a valid question post."
                )
        elif len(candidates) == 1:
            best_post = candidates[0]
        else:
            # Prefer a different post than the seed (likely the public mirror), then most forecasters
            non_seed = [p for p in candidates if p.get("id") != post_id]
            pool = non_seed if len(non_seed) > 0 else candidates
            best_post = max(pool, key=extract_num_forecasters)

        prediction_pct = extract_recency_weighted_yes_pct(best_post)
        n_forecasters = extract_num_forecasters(best_post)
        return prediction_pct, n_forecasters

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to get community prediction and count for post {post_id}: {e}")


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
        # Get the original post details and extract canonical question slug
        seed_post = get_post_details(post_id)
        try:
            target_slug = extract_question_slug(seed_post)
        except Exception:
            target_slug = extract_slug(seed_post)

        # Search for posts with matching slug
        summaries = search_posts_by_slug(target_slug)

        # Filter by question.slug or title match and get full details
        seed_title = seed_post.get("title") if isinstance(seed_post, dict) else None
        candidates = []
        for summary in summaries:
            try:
                full_post = get_post_details(summary["id"])
            except Exception:
                continue
            # Only include posts that have questions (forecastable) and are not group posts
            if "question" not in full_post or "group_of_questions" in full_post:
                continue
            matches_slug = False
            try:
                matches_slug = strings_equal_ci(extract_question_slug(full_post), target_slug)
            except Exception:
                matches_slug = False
            matches_title = strings_equal_ci(full_post.get("title"), seed_title)
            if matches_slug or matches_title:
                candidates.append(full_post)

        # Handle different numbers of candidates
        if len(candidates) == 0:
            # Check if the original post itself is a valid candidate
            if "question" in seed_post and "group_of_questions" not in seed_post:
                best_post = seed_post
            else:
                raise RuntimeError(f"Found no posts with slug '{target_slug}' that have questions, and original post is not a valid question post.")
        elif len(candidates) == 1:
            best_post = candidates[0]
        else:
            # Prefer a different post than the seed (likely the public mirror), then most forecasters
            non_seed = [p for p in candidates if p.get("id") != post_id]
            pool = non_seed if len(non_seed) > 0 else candidates
            best_post = max(pool, key=extract_num_forecasters)

        # Extract and return the community prediction
        return extract_recency_weighted_yes_pct(best_post)

    except RuntimeError:
        # Re-raise RuntimeError with original message (from extract_recency_weighted_yes_pct)
        raise
    except Exception as e:
        # Catch any other unexpected errors and provide context
        raise RuntimeError(f"Failed to get community prediction for post {post_id}: {e}")

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
