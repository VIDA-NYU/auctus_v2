"""A record's evidence as numbered lines, plus BM25 search over them.

SAFE searches for evidence one query at a time and VeriTrail selects evidence by sentence id, so both
need the evidence cut into small addressable pieces instead of one block of text:
  title     the record title and the dataset names AutoDDG was given
  curator   the curator's description, one sentence per line
  docs      README / codebook text, one sentence or line per line
  data      the whole-file profile of each table: one line per column (type, min-max, mean, distinct
            values), plus the table's size, dates and bounding box, and the first rows
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from eval.description_eval.arms import documentation_text
from eval.description_eval.collect import load_profile
from eval.description_eval.common import COLLECTED_DIR, GOLD_DIR, INSTANCES, read_json
from eval.description_eval.drafts import MAX_TABLES, table_summary
from eval.description_eval.semtypes import data_evidence

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def split_text(text: str, max_len: int = 400) -> list[str]:
    out = []
    for part in _SENT.split(text or ""):
        part = part.strip(" -*\t")
        while len(part) > max_len:  # long README lines (tables, lists) are cut, not lost
            out.append(part[:max_len])
            part = part[max_len:]
        if len(part) > 2:
            out.append(part)
    return out


def gold_path(rid: str):
    """Held-out CaltechDATA gold lives in gold/caltech/ (gold.py), UltraViolet gold at the top."""
    path = GOLD_DIR / f"{rid}.json"
    return path if path.exists() else GOLD_DIR / "caltech" / f"{rid}.json"


def record_evidence(storage, rid: str, with_paper: bool = False) -> dict[str, Any]:
    """title, curator description, the tables' summary text (for prompts) and the numbered lines.
    with_paper: add the data-relevant excerpts of the record's linked papers (papers.py), so a claim the
    paper backs counts as supported."""
    gold = read_json(gold_path(rid))
    instance = next((k for k, v in INSTANCES.items() if v == gold.get("domain")), "ultraviolet")
    cur = gold.get("curator") or {}
    profiles = []
    for f in sorted(gold["files"], key=lambda f: f["file_key"])[:MAX_TABLES]:
        try:
            profiles.append((f["file_key"], load_profile(storage, f["dataset_id"])))
        except Exception:
            continue
    collected = COLLECTED_DIR / f"{rid}.json"
    # AutoDDG is given the dataset's name in the index, not the record title; without it a correct
    # "titled 'COVID Zipcode Rates'" reads as contradicted
    names = sorted({f.get("title") for f in read_json(collected)["files"].values() if f.get("title")}) \
        if collected.exists() else []
    docs = documentation_text(rid, instance)

    lines: list[dict[str, str]] = [{"source": "title", "text": f"Record title: {cur.get('title', '')}"}]
    if names:
        lines.append({"source": "title", "text": "Dataset names in the search index: " + "; ".join(names)})
    lines += [{"source": "curator", "text": s} for s in split_text(cur.get("description", ""))]
    lines += [{"source": "docs", "text": s} for s in split_text(docs)]
    for key, profile in profiles:
        for line in data_evidence(profile, rows=5).splitlines():
            if line.strip() and line.strip() not in ("Column profile:", "First rows:"):
                lines.append({"source": "data", "text": f"[{key}] {line.strip()}"[:400]})
    paper = []
    if with_paper:
        from eval.description_eval.papers import excerpt_lines

        paper = excerpt_lines(rid)
        lines += paper
    for i, line in enumerate(lines):
        line["id"] = f"E{i}"

    tables = "\n\n".join(table_summary(k, p) for k, p in profiles)
    block = "\n\n".join(x for x in (f"Record title: {cur.get('title', '')}",
                                    names and "Dataset names in the search index: " + "; ".join(names),
                                    f"Curator's description:\n{cur.get('description', '')}",
                                    docs and f"Documentation (README / codebook):\n{docs}",
                                    tables and f"Tables:\n{tables}",
                                    paper and "Linked paper excerpts:\n" + "\n\n".join(l["text"] for l in paper)) if x)
    return {"title": cur.get("title", ""), "description": cur.get("description", ""), "tables": tables,
            "docs": docs, "evidence": block, "lines": lines, "has_paper": bool(paper)}


def numbered_lines(lines: list[dict[str, str]]) -> str:
    return "\n".join(f"{l['id']} ({l['source']}): {l['text']}" for l in lines)


class BM25:
    """Plain Okapi BM25 over the evidence lines (k1 1.5, b 0.75); no dependencies."""

    def __init__(self, lines: list[dict[str, str]]):
        self.lines = lines
        self.docs = [_TOKEN.findall(l["text"].lower()) for l in lines]
        self.avg = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def search(self, query: str, k: int = 5, exclude: set[str] | None = None) -> list[dict[str, str]]:
        """Top-k lines for the query; `exclude` (line ids) skips lines an earlier query already returned."""
        q = _TOKEN.findall(query.lower())
        scores = []
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            if exclude and self.lines[i].get("id") in exclude:
                continue
            s = sum(self.idf.get(t, 0) * tf[t] * 2.5 / (tf[t] + 1.5 * (0.25 + 0.75 * len(d) / self.avg))
                    for t in q if t in tf)
            if s > 0:
                scores.append((s, i))
        return [self.lines[i] for _, i in sorted(scores, reverse=True)[:k]]
