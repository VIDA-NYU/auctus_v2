"""Offline tests for the id-list ingestion path (corpus-resample-100-with-frame
tasks.md §5). No Redis, no OpenSearch, no network — ``fetch_socrata_update_timestamp``
is stubbed out and a fake ARQ pool / OpenSearch client stand in for the real ones.

Run: python -m test_run_pipeline_ingest_id_list
"""

from __future__ import annotations

import asyncio
from typing import Any

import run_pipeline_ingest as rpi


class _FakeRedisPool:
    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self.fail_ids = fail_ids or set()
        self.enqueued: list[dict[str, Any]] = []
        self.closed = False

    async def enqueue_job(self, task_name: str, dataset_meta: dict[str, Any]) -> None:
        if dataset_meta["dataset_id"] in self.fail_ids:
            raise RuntimeError(f"simulated enqueue failure for {dataset_meta['dataset_id']}")
        self.enqueued.append(dataset_meta)

    async def aclose(self) -> None:
        self.closed = True


class _FakeOsClient:
    def __init__(self, present_ids: set[str]) -> None:
        self.present_ids = present_ids

    def mget(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "docs": [
                {"_id": doc_id, "found": doc_id in self.present_ids}
                for doc_id in body["ids"]
            ]
        }


def _patch_no_network(monkeypatch_target: list) -> None:
    """Stub the one real-network call ingest_id_list makes, so the test never
    touches the network. Restored by the caller via the returned original."""
    original = rpi.fetch_socrata_update_timestamp

    async def _stub(base_url: str, dataset_id: str, http_timeout_seconds: float) -> str | None:
        return None

    rpi.fetch_socrata_update_timestamp = _stub
    monkeypatch_target.append(original)


def _unpatch(monkeypatch_target: list) -> None:
    rpi.fetch_socrata_update_timestamp = monkeypatch_target[0]


def test_every_requested_id_is_enqueued_with_its_agency() -> None:
    saved: list = []
    _patch_no_network(saved)
    try:
        records = [
            {"id": "aaaa-1111", "agency": "Department of Testing (DOT)"},
            {"id": "bbbb-2222", "agency": None},
        ]
        pool = _FakeRedisPool()
        result = asyncio.run(
            rpi.ingest_id_list(
                records, provider_type="socrata", domain_url="data.example.gov",
                redis_pool=pool, run_init_db=False,
            )
        )

        assert result["enqueued_ids"] == ["aaaa-1111", "bbbb-2222"]
        assert result["failed"] == []
        assert not pool.closed, "an injected pool must not be closed by ingest_id_list"

        by_id = {m["dataset_id"]: m for m in pool.enqueued}
        assert by_id["aaaa-1111"]["agency"] == "Department of Testing (DOT)"
        assert by_id["bbbb-2222"]["agency"] is None
        assert by_id["aaaa-1111"]["domain"] == "data.example.gov"
        assert by_id["aaaa-1111"]["provider"] == "socrata"
    finally:
        _unpatch(saved)


def test_a_failed_enqueue_is_reported_not_substituted_or_dropped_silently() -> None:
    saved: list = []
    _patch_no_network(saved)
    try:
        records = [
            {"id": "aaaa-1111", "agency": "Agency A"},
            {"id": "bbbb-2222", "agency": "Agency B"},
            {"id": "cccc-3333", "agency": "Agency C"},
        ]
        pool = _FakeRedisPool(fail_ids={"bbbb-2222"})
        result = asyncio.run(
            rpi.ingest_id_list(
                records, provider_type="socrata", domain_url="data.example.gov",
                redis_pool=pool, run_init_db=False,
            )
        )

        assert result["requested_ids"] == ["aaaa-1111", "bbbb-2222", "cccc-3333"]
        assert result["enqueued_ids"] == ["aaaa-1111", "cccc-3333"]
        assert len(result["failed"]) == 1
        assert result["failed"][0]["id"] == "bbbb-2222"
        # The failure must not silently vanish or be papered over: the total
        # accounted for (enqueued + failed) must equal what was requested.
        assert len(result["enqueued_ids"]) + len(result["failed"]) == len(result["requested_ids"])
    finally:
        _unpatch(saved)


def test_verify_ingested_ids_separates_present_from_missing() -> None:
    os_client = _FakeOsClient(present_ids={"aaaa-1111", "cccc-3333"})
    result = rpi.verify_ingested_ids(os_client, ["aaaa-1111", "bbbb-2222", "cccc-3333"])

    assert result["present"] == ["aaaa-1111", "cccc-3333"]
    assert result["missing"] == ["bbbb-2222"]


def test_verify_ingested_ids_handles_empty_request() -> None:
    result = rpi.verify_ingested_ids(_FakeOsClient(present_ids=set()), [])
    assert result == {"requested": [], "present": [], "missing": []}


def main() -> int:
    test_every_requested_id_is_enqueued_with_its_agency()
    test_a_failed_enqueue_is_reported_not_substituted_or_dropped_silently()
    test_verify_ingested_ids_separates_present_from_missing()
    test_verify_ingested_ids_handles_empty_request()
    print("OK: id-list ingestion enqueues with agency, reports failures without "
          "substitution, and verify_ingested_ids separates present from missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
