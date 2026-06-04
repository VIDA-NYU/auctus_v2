#!/usr/bin/env python3
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

GLOBAL_DOMAINS_ENDPOINT = "https://api.us.socrata.com/api/catalog/v1/domains"
FALLBACK_CATALOG_ENDPOINT = "https://api.us.socrata.com/api/catalog/v1?only=dataset&limit=5000"
MIN_DATASET_COUNT = 10
OUTPUT_FILE = Path(__file__).resolve().parent / "socrata.json"


async def fetch_domains() -> list[dict[str, Any]]:
    print("🌐 Querying global registry...")

    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(GLOBAL_DOMAINS_ENDPOINT)
            response.raise_for_status()
            payload = response.json()

            results = payload.get("results", [])
            if not isinstance(results, list):
                raise ValueError("Unexpected response shape: 'results' is not a list")

            portals: list[dict[str, Any]] = []
            for item in results:
                if not isinstance(item, dict):
                    continue

                domain = item.get("domain")
                count = item.get("count")

                if not domain or count is None:
                    continue

                try:
                    dataset_count = int(count)
                except (TypeError, ValueError):
                    continue

                if dataset_count < MIN_DATASET_COUNT:
                    continue

                portals.append({"url": str(domain), "dataset_count": dataset_count})

            portals.sort(key=lambda p: p["dataset_count"], reverse=True)
            return portals
        except httpx.HTTPStatusError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise
            print("ℹ️ Domain registry endpoint returned 404; falling back to catalog aggregation...")
            response = await client.get(FALLBACK_CATALOG_ENDPOINT)
            response.raise_for_status()
            payload = response.json()

    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Unexpected response shape: 'results' is not a list")

    domain_counts: dict[str, int] = {}
    for item in results:
        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata") or {}
        domain = metadata.get("domain")
        if not domain:
            continue

        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    portals = [
        {"url": domain, "dataset_count": count}
        for domain, count in domain_counts.items()
        if count >= MIN_DATASET_COUNT
    ]

    portals.sort(key=lambda p: p["dataset_count"], reverse=True)
    return portals


async def main() -> None:
    try:
        portals = await fetch_domains()
    except httpx.HTTPError as exc:
        print(f"❌ HTTP error while querying Socrata registry: {exc}")
        raise SystemExit(1) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Failed to discover Socrata domains: {exc}")
        raise SystemExit(1) from exc

    OUTPUT_FILE.write_text(json.dumps(portals, indent=2), encoding="utf-8")
    print(f"✨ Successfully wrote {len(portals)} portals to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    asyncio.run(main())
