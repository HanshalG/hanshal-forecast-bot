from math import log, exp
from typing import Tuple, Dict, List

from src.base_rate_agent import compute_base_rate_prior_with_research
from src.evidence_agent import generate_evidence
from src.metaculus_utils import get_metaculus_community_prediction_and_count
from src.utils import run_research

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

def probability_to_odds(p: float, eps: float = 1e-9) -> float:
    p = clip01(p, eps)
    return p / (1.0 - p)


def odds_to_probability(o: float) -> float:
    if o <= 0:
        return 0.0
    return o / (1.0 + o)


def crowd_bayes_factor(prior_p: float, crowd_p: float, eps: float = 1e-9) -> float:
    O_prior = probability_to_odds(prior_p, eps)
    O_crowd = probability_to_odds(crowd_p, eps)
    return O_crowd / O_prior


def novel_bayes_factor(evidence_items: List[Dict]) -> float:
    bf = 1.0
    for item in evidence_items:
        try:
            if item.get("priced_in"):
                continue
            lr = float(item.get("likelihood_ratio", 1.0))
            if lr <= 0:
                continue
            if item.get("direction") == "against":
                lr = 1.0 / lr
            bf *= lr
        except Exception:
            continue
    return bf


async def run_bayesian_update(question_details: dict, post_id: int | None = None) -> dict:
    # 0) Shared research once
    summary_report = run_research(question_details.get("title", ""))

    # 1) Base-rate prior using shared research
    prior = await compute_base_rate_prior_with_research(question_details, summary_report)

    # 2) Crowd probability and forecaster count
    crowd_prob = prior
    n_forecasters = 0
    if post_id is not None:
        try:
            crowd_pct, n_forecasters = get_metaculus_community_prediction_and_count(post_id)
            crowd_prob = max(0.01, min(0.99, crowd_pct / 100.0))
        except Exception:
            crowd_prob = prior
            n_forecasters = 0

    # 3) Evidence list
    evidence_payload = await generate_evidence(question_details, summary_report=summary_report)
    items = evidence_payload.get("evidence", [])

    # 4) Compute Bayes factors
    bf_crowd = crowd_bayes_factor(prior, crowd_prob)
    bf_novel = novel_bayes_factor(items)

    # 5) Final odds and posterior (use crowd as baseline then apply novel evidence)
    O_final = probability_to_odds(crowd_prob) * bf_novel
    posterior = odds_to_probability(O_final)
    posterior = max(0.01, min(0.99, posterior))

    return {
        "prior_base_rate": prior,
        "metaculus_probability": crowd_prob,
        "crowd_bayes_factor": bf_crowd,
        "evidence": items,
        "novel_bayes_factor": bf_novel,
        "posterior_probability": posterior,
    }
