from datetime import datetime, timezone

import pytest

from src.eval.metaculus_resolution_loader import ResolutionLoaderError, join_context_and_resolutions
from src.eval.types import ResolvedContextQuestion


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


@pytest.mark.parametrize(
    "resolution,label",
    [
        ("yes", 1),
        ("no", 0),
    ],
)
def test_resolution_loader_normalizes_yes_no(monkeypatch, resolution, label):
    monkeypatch.setenv("METACULUS_TOKEN", "token")

    def _fake_get(url, headers, timeout):
        assert url.endswith("/39523/")
        return _FakeResponse(
            {
                "question": {
                    "id": 123,
                    "status": "resolved",
                    "resolution": resolution,
                    "open_time": "2024-01-01T00:00:00Z",
                    "actual_resolve_time": "2025-01-01T00:00:00Z",
                }
            }
        )

    monkeypatch.setattr("src.eval.metaculus_resolution_loader.requests.get", _fake_get)

    context = {
        39523: ResolvedContextQuestion(
            post_id=39523,
            title="Q",
            type="binary",
            description="d",
            resolution_criteria="r",
            fine_print="f",
            metadata={},
        )
    }

    joined = join_context_and_resolutions(post_ids=[39523], manual_context_by_post_id=context)

    assert len(joined) == 1
    assert joined[0].label == label
    assert joined[0].open_time == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_resolution_loader_fails_when_context_missing(monkeypatch):
    monkeypatch.setenv("METACULUS_TOKEN", "token")

    def _fake_get(url, headers, timeout):
        return _FakeResponse(
            {
                "question": {
                    "id": 999,
                    "status": "resolved",
                    "resolution": "yes",
                    "open_time": "2024-01-01T00:00:00Z",
                    "actual_resolve_time": None,
                }
            }
        )

    monkeypatch.setattr("src.eval.metaculus_resolution_loader.requests.get", _fake_get)

    with pytest.raises(ResolutionLoaderError):
        join_context_and_resolutions(post_ids=[39523], manual_context_by_post_id={})
