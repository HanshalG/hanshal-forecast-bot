from math import log, exp
from typing import Tuple, Dict

# Placeholder for future base rate prior function
# def get_base_rate_prior(question_type: str) -> float:
#     if question_type == "binary":
#         return 0.5  # Default prior
#     else:
#         return 0.5

def clip01(p: float, eps: float = 1e-9) -> float:
    """Clip probability to (0,1) to avoid infinite logits."""
    return min(1.0 - eps, max(eps, float(p)))

def logit(p: float, eps: float = 1e-9) -> float:
    """Log-odds of a probability."""
    p = clip01(p, eps)
    return log(p / (1.0 - p))

def sigmoid(x: float) -> float:
    """Logistic function (inverse logit)."""
    # numerically stable-ish
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = exp(x)
        return z / (1.0 + z)

def metaculus_weight(
    n_forecasters: int,
    community_prediction: float,
) -> float:
    """
    Compute weight to apply to Metaculus community prediction based on number of forecasters.
    A simple heuristic that increases weight with more forecasters up to a cap.
    
    Args:
        n_forecasters: Number of forecasters that contributed to community prediction
        community_prediction: The current community prediction (not used but kept for interface consistency)
        
    Returns:
        float: Weight between 0 and 1 to apply to community prediction
    """
    # Simple heuristic - more weight with more forecasters up to max of 0.8
    w_min = 0.2  # Minimum weight
    w_max = 0.8  # Maximum weight
    n0 = 10.0    # Scale parameter
    
    # Saturation with crowd size
    sat = 1.0 - exp(-max(n_forecasters, 0) / n0)
    
    # Combine and clip
    w = w_min + (w_max - w_min) * sat
    return max(w_min, min(w, w_max))

def integrate_log_odds_shrinkage(
    P0: float,
    pM: float,
    w: float = 0.6,
    eps: float = 1e-9,
) -> Tuple[float, Dict[str, float]]:
    """
    Integrate base-rate prior P0 with Metaculus probability pM using
    log-odds shrinkage toward pM with effective weight w * d_map in [0,1].

    Returns:
        P_int: integrated probability
        steps: dict of intermediates (if return_steps=True)
    """
    P0c = clip01(P0, eps)
    pMc = clip01(pM, eps)
    L0 = logit(P0c, eps)
    LM = logit(pMc, eps)

    L_int = L0 + w * (LM - L0)
    P_int = sigmoid(L_int)

    return P_int

def integrate_base_with_metaculus_weight(
    community_prediction: float,
    n_forecasters: int,
    base_rate_prior: float,
) -> float:
    return integrate_log_odds_shrinkage(base_rate_prior, community_prediction, metaculus_weight(n_forecasters, community_prediction))
# --- Example usage ---
if __name__ == "__main__":
    import asyncio
    from src.base_rate_agent import compute_base_rate_prior
    from src.metaculus_utils import get_post_details, get_metaculus_community_prediction, extract_num_forecasters

    async def test_base_rate_integration():
        # Use the first example question from main.py
        question_id, post_id = (39724, 39724)

        print(f"Testing with question ID {question_id}, post ID {post_id}")
        print("=" * 50)

        # Get question details
        try:
            post_details = get_post_details(post_id)
            question_details = post_details["question"]
            print(f"Question: {question_details['title']}")
            print(f"Type: {question_details['type']}")


        except Exception as e:
            print(f"Error getting post details: {e}")
            return

        # Get base rate prior
        try:
            base_rate_prior = await compute_base_rate_prior(question_details)
            print(f"Base rate prior: {base_rate_prior:.4f}")
        except Exception as e:
            import traceback
            print(f"Error computing base rate prior: {e}")
            print("Full traceback:")
            traceback.print_exc()
            return

        # Get community prediction and number of forecasters
        try:
            from src.metaculus_utils import get_metaculus_community_prediction_and_count
            community_prediction, n_forecasters = get_metaculus_community_prediction_and_count(post_id)
            print(f"Community prediction: {community_prediction:.2f}%")
            print(f"Number of forecasters: {n_forecasters}")
        except Exception as e:
            print(f"Error getting community prediction: {e}")
            return

        # Integrate using bayesian methods
        try:
            integrated_prediction = integrate_base_with_metaculus_weight(
                community_prediction / 100.0,  # Convert to decimal
                n_forecasters,
                base_rate_prior
            )
            print(f"Integrated probability: {integrated_prediction:.4f}")
            print(f"Base rate prior (again): {base_rate_prior:.4f}")

            # Also show the weight used
            weight = metaculus_weight(n_forecasters, community_prediction / 100.0)
            print(f"Applied weight: {weight:.3f}")

        except Exception as e:
            print(f"Error in integration: {e}")
            return

    # Run the test
    asyncio.run(test_base_rate_integration())

