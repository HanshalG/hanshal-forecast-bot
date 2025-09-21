import asyncio
import json
import types
import pytest


@pytest.mark.asyncio
async def test_generate_evidence_schema(monkeypatch):
    # Arrange: fake utils and prompt
    from src import base_rate_agent  # just to ensure package path ok

    # Fake prompt content is irrelevant because we stub the LLM response
    def fake_read_prompt(filename: str) -> str:
        assert filename == "evidence_agent_prompt.txt"
        return "PROMPT"

    async def fake_call_llm(prompt: str, model: str, temperature: float) -> str:
        payload = {
            "evidence": [
                {
                    "text": "Recent sanctions reduce revenue",
                    "direction": "against",
                    "likelihood_ratio": 3.0,
                    "priced_in": False,
                    "confidence": 0.8,
                    "sources": ["https://example.com/a"],
                },
                {
                    "text": "Strong quarterly report",
                    "direction": "for",
                    "likelihood_ratio": 2.5,
                    "priced_in": True,
                    "confidence": 0.7,
                    "sources": ["https://example.com/b"],
                },
            ]
        }
        return json.dumps(payload)

    def fake_run_research(title: str) -> str:
        return "research"

    monkeypatch.setattr("src.utils.read_prompt", fake_read_prompt)
    monkeypatch.setattr("src.utils.call_llm", fake_call_llm)
    monkeypatch.setattr("src.utils.run_research", fake_run_research)

    # Lazy import after monkeypatch
    from src import evidence_agent

    question_details = {
        "title": "Test Q",
        "description": "Desc",
        "resolution_criteria": "Res",
        "fine_print": "Fine",
    }

    # Act
    result = await evidence_agent.generate_evidence(question_details)

    # Assert
    assert "evidence" in result
    items = result["evidence"]
    assert isinstance(items, list) and len(items) == 2
    for it in items:
        assert set(["text","direction","likelihood_ratio","priced_in","confidence","sources"]).issubset(it.keys())
        assert it["direction"] in {"for","against"}
        assert it["likelihood_ratio"] > 0
        assert isinstance(it["priced_in"], bool)
        assert 0.0 <= it["confidence"] <= 1.0
        assert isinstance(it["sources"], list)



