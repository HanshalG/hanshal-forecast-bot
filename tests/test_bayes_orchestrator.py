import asyncio
import pytest


@pytest.mark.asyncio
async def test_run_bayesian_update(monkeypatch):
    # Arrange stubs
    async def fake_compute_prior_with_research(q, summary_report):
        return 0.3

    def fake_get_metaculus(post_id: int):
        assert post_id == 123
        return 60.0, 50  # percent and count

    async def fake_generate_evidence(q, summary_report=None):
        return {
            "evidence": [
                {
                    "text": "Strong quarterly report",
                    "direction": "for",
                    "likelihood_ratio": 2.0,
                    "priced_in": True,
                    "confidence": 0.7,
                    "sources": ["https://x"]
                },
                {
                    "text": "Recent sanctions",
                    "direction": "against",
                    "likelihood_ratio": 4.0,
                    "priced_in": False,
                    "confidence": 0.8,
                    "sources": ["https://y"]
                },
                {
                    "text": "New contract signed",
                    "direction": "for",
                    "likelihood_ratio": 2.0,
                    "priced_in": False,
                    "confidence": 0.6,
                    "sources": ["https://z"]
                },
            ]
        }

    # Stub research to avoid network calls/delays
    monkeypatch.setattr("src.utils.run_research", lambda title: "research")
    monkeypatch.setattr("src.base_rate_agent.compute_base_rate_prior_with_research", fake_compute_prior_with_research)
    monkeypatch.setattr("src.metaculus_utils.get_metaculus_community_prediction_and_count", fake_get_metaculus)
    monkeypatch.setattr("src.evidence_agent.generate_evidence", fake_generate_evidence)

    from src import bayes

    qd = {
        "title": "Test Q",
        "description": "Desc",
        "resolution_criteria": "Res",
        "fine_print": "Fine",
    }

    # Act
    result = await bayes.run_bayesian_update(qd, post_id=123)

    # Assert values
    assert pytest.approx(result["prior_base_rate"], rel=1e-6) == 0.3
    assert pytest.approx(result["metaculus_probability"], rel=1e-6) == 0.6
    # Crowd BF: O(0.6)/O(0.3) = 1.5 / (0.3/0.7) = 3.5
    assert pytest.approx(result["crowd_bayes_factor"], rel=1e-6) == 3.5
    # Novel BF: for(2.0) * against(1/4.0) = 0.5 (priced_in True excluded)
    assert pytest.approx(result["novel_bayes_factor"], rel=1e-6) == 0.5
    # Posterior: O_final = O(0.6)*0.5 = 0.75 -> p = 0.75/1.75 = 0.428571...
    assert pytest.approx(result["posterior_probability"], rel=1e-6) == 0.4285714286
    assert isinstance(result["evidence"], list) and len(result["evidence"]) == 3


