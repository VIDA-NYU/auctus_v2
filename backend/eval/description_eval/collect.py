"""Collect the automatic side: AutoDDG output and profiler metadata per file.

Reads the full profile Auctus stored in MinIO for each file (the same object
/api/datasets/{id}/profile serves), so nothing is regenerated.
"""

from __future__ import annotations

import json
import re
from typing import Any

from storage.minio_client import get_heavy_profile_object, get_storage_client

from eval.description_eval.common import COLLECTED_DIR, GOLD_DIR, read_json, write_json

# AutoDDG's semantic profile is prose: "**<column>**: Represents ... ." one line per column.
SEMANTIC_LINE_RE = re.compile(r"^\*\*(.+?)\*\*\s*:\s*(.+)$")


def parse_semantic_profile(text: str | None) -> dict[str, str]:
    columns: dict[str, str] = {}
    for line in (text or "").splitlines():
        match = SEMANTIC_LINE_RE.match(line.strip())
        if match:
            columns[match.group(1).strip()] = match.group(2).strip()
    return columns


def load_profile(storage_client, dataset_id: str) -> dict[str, Any]:
    obj, _ = get_heavy_profile_object(storage_client, dataset_id)
    try:
        return json.loads(obj.read())
    finally:
        obj.close()
        obj.release_conn()


def summarize(profile: dict[str, Any]) -> dict[str, Any]:
    pm = profile.get("profiler_metadata") or {}
    return {
        "title": profile.get("title"),
        "topic": profile.get("autoddg_topic"),
        "ufd": profile.get("autoddg_description"),
        "sfd": profile.get("autoddg_search_description"),
        "semantic_columns": parse_semantic_profile(profile.get("autoddg_semantic_profile")),
        "profiler_columns": {
            c.get("name"): {"structural_type": c.get("structural_type"), "semantic_types": c.get("semantic_types") or []}
            for c in pm.get("columns") or [] if isinstance(c, dict)
        },
        "nb_rows": pm.get("nb_rows"),
        "nb_columns": pm.get("nb_columns"),
        "temporal_coverage": profile.get("temporal_coverage"),
        "spatial_bbox": pm.get("spatial_bbox"),
    }


def main() -> None:
    storage_client = get_storage_client()
    for gold_path in sorted(GOLD_DIR.glob("*.json")):
        gold = read_json(gold_path)
        files = {}
        for f in gold["files"]:
            try:
                files[f["file_key"]] = summarize(load_profile(storage_client, f["dataset_id"]))
            except Exception as exc:  # missing profile: report, don't stop
                files[f["file_key"]] = {"error": repr(exc)}
        write_json(COLLECTED_DIR / gold_path.name, {"record_id": gold["record_id"], "files": files})
        print(f"collected {gold['record_id']}: {len(files)} files")
