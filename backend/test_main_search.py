"""Regression test for main.py's /search accepting every configured description arm.

Guards the defect where ``SearchRequest.description_source`` carried its own
hardcoded ``Literal["original", "llm_direct", "ufd", "sfd"]`` — a second, stale
copy of the arm list living beside ``DESCRIPTION_SOURCE_FIELDS``
(storage/opensearch_client.py), the mapping that actually resolves an arm to
its index field. The mapping had grown two more arms (``profile_only``,
``t_od_s``) that the ``Literal`` never learned about, so Pydantic rejected
those two with a 422 before the request ever reached ``description_fields_for``
— even though the rest of the system (index mapping, field population) fully
supported them. See openspec/changes/fix-stale-arm-literal/proposal.md.

Uses a fake OpenSearch client rather than a live one, since this endpoint's
validation and error-shaping behaviour does not depend on what OpenSearch
returns.

Run:  backend/.venv/bin/python -m test_main_search
(system python3 fails on missing arq / atlas_profiler — see tasks.md 3.1)
"""

from __future__ import annotations

import main
from fastapi.testclient import TestClient
from storage.opensearch_client import DESCRIPTION_SOURCE_FIELDS

client = TestClient(main.app)


class _FakeOpenSearchClient:
    """Stands in for the real client; /search's validation runs before this
    is ever called for the negative case, and the positive case only needs a
    well-formed empty response."""

    def search(self, index, body):
        return {"hits": {"hits": [], "total": {"value": 0}}}


def _use_fake_client(monkeypatch_target) -> None:
    monkeypatch_target.get_client = lambda: _FakeOpenSearchClient()


def test_every_configured_arm_is_accepted() -> None:
    """Every key in DESCRIPTION_SOURCE_FIELDS — driven from the mapping, not a
    written-out list, so a seventh arm is covered the moment it is added."""
    _use_fake_client(main)
    for arm in DESCRIPTION_SOURCE_FIELDS:
        response = client.post("/search", json={"query": "test", "description_source": arm})
        assert response.status_code == 200, (
            f"arm {arm!r} was rejected: {response.status_code} {response.text}"
        )


def test_previously_working_arms_are_unchanged() -> None:
    """The four arms that worked before this change (task 3.2) still do, and
    still reach OpenSearch rather than being short-circuited."""
    _use_fake_client(main)
    for arm in ("original", "llm_direct", "ufd", "sfd"):
        response = client.post("/search", json={"query": "test", "description_source": arm})
        assert response.status_code == 200, (
            f"arm {arm!r} regressed: {response.status_code} {response.text}"
        )
        body = response.json()
        assert body["query"] == "test"
        assert body["results"] == []


def test_unknown_arm_is_refused_with_accepted_values_named() -> None:
    """An arm absent from the mapping is refused with a 400 naming the
    accepted values — not a generic 503, and not a silent fallback to the
    default arm."""
    _use_fake_client(main)
    response = client.post(
        "/search", json={"query": "test", "description_source": "not_a_real_arm"}
    )
    assert response.status_code == 400, (
        f"expected 400, got {response.status_code} {response.text}"
    )
    detail = response.json()["detail"]
    assert "not_a_real_arm" in detail
    for arm in DESCRIPTION_SOURCE_FIELDS:
        assert arm in detail, f"accepted value {arm!r} missing from error: {detail}"


if __name__ == "__main__":
    test_every_configured_arm_is_accepted()
    test_previously_working_arms_are_unchanged()
    test_unknown_arm_is_refused_with_accepted_values_named()
    print("OK: /search accepts every DESCRIPTION_SOURCE_FIELDS arm, refuses unknown ones clearly")
