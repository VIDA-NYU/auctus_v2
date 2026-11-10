"""FTIR material suggestions vs the conservators' own identifications.

Every NYU conservation FTIR record ships a report (`<ConservationID>_Report.pdf`) that
lists what each sampled object matched in the lab's libraries ("Red plastic sheeting:
PVC, rigid (90% match)"). An LLM (Claude, a different family from anything that made
the suggestions) reads the report, ties each spectrum file to its report entry by the
spectrum's title, and grades Auctus's top suggestion and its top three against it:
same material, related (same family, e.g. "polyvinylchloride" vs "PVC, flexible"), or
different. Spectra the report doesn't identify are left out.

  python -m eval.ftir_materials
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from typing import Any

import httpx

from eval.description_eval.collect import load_profile
from eval.description_eval.common import DATA_DIR, ask_json, write_json

JUDGE_MODEL = "@vertexai/anthropic.claude-sonnet-5-5"
UV = "https://ultraviolet.library.nyu.edu/api/records/"
FTIR_RECORDS = ("6eaks-fvw13", "x1527-n7a37", "tdybb-r2821", "tavfh-hpq12", "fbspx-gaw98", "vje3m-6z249",
                "h1zpq-2fg71", "ss59m-qhy59")

PROMPT = """A conservation lab analysed objects with FTIR and wrote this report:
<<<
{report}
>>>

For each spectrum below (file name and the title the instrument recorded), find the report's
identification of that sample, if the report gives one, and grade the automatic suggestions
against it:
- "same": the same material (PVC = polyvinylchloride; cellophane = regenerated cellulose)
- "related": same family or a component of it (a phthalate plasticizer for flexible PVC;
  nitrocellulose for "cellulose nitrate")
- "different": another material
Grade the top suggestion alone ("top1") and whether any of the three suggestions is same or
related ("top3"). If the report gives no identification for a spectrum, use null for both.

Spectra:
{spectra}

Return JSON only: {{"spectra": [{{"file": "...", "report_identification": "... or null",
  "top1": "same|related|different|null", "top3": "same|related|different|null", "note": "<= 12 words"}}]}}
"""


def report_text(client: httpx.Client, record: dict[str, Any]) -> str:
    from pypdf import PdfReader

    keys = [k for k in record["files"]["entries"] if k.lower().endswith("_report.pdf")]
    if not keys:
        return ""
    data = client.get(f"{UV}{record['id']}/files/{keys[0]}/content").content
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)[:30000]


def main() -> None:
    from crawlers.invenio.http import ssl_context
    from storage.minio_client import get_storage_client

    storage, rows = get_storage_client(), []
    with httpx.Client(verify=ssl_context(), timeout=60, follow_redirects=True) as client:
        for rid in FTIR_RECORDS:
            record = client.get(UV + rid).json()
            report = report_text(client, record)
            profile = load_profile(storage, f"ultraviolet.library.nyu.edu:{rid}:spectra")
            spectra = [r for r in csv.DictReader(io.StringIO(profile["sample"])) if r.get("spectrum") == "sample"]
            if not report or not spectra:
                continue
            listing = "\n".join(
                f"- {r['file']} | title: {r['title']} | suggestions: {r.get('suggested_material')} ({r.get('match_score')}); "
                f"{r.get('other_candidates') or ''}" for r in spectra)
            reply = ask_json(PROMPT.format(report=report, spectra=listing), model=JUDGE_MODEL)
            by_file = {r["file"]: r for r in spectra}
            for g in (reply or {}).get("spectra", []):
                s = by_file.get(g.get("file"), {})
                rows.append({"record": rid, **g, "suggested": s.get("suggested_material"),
                             "score": float(s["match_score"]) if s.get("match_score") else None,
                             "confidence": s.get("match_confidence")})
            print(f"{rid}: {len(spectra)} spectra graded")

    judged = [r for r in rows if r.get("top1") in ("same", "related", "different")]
    summary: dict[str, Any] = {"spectra_with_report_id": len(judged),
                               "top1": Counter(r["top1"] for r in judged), "top3": Counter(r["top3"] for r in judged)}
    for level in ("confident", "possible", "no confident match"):
        sub = [r for r in judged if r.get("confidence") == level]
        summary[f"top1_{level}"] = {"n": len(sub), **Counter(r["top1"] for r in sub)}
    write_json(DATA_DIR / "results" / "ftir-materials.json", {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=1, default=dict))


if __name__ == "__main__":
    main()
