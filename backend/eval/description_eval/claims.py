"""Claim-based record-level evals (Claimify-style), next to the question evals of 11-17.

Same 35 UltraViolet records and texts as questions.py:
- curator       the curator's record description (the reference);
- autoddg       AutoDDG's description of the record's first profiled file (what search shows today);
- draft_data    record-level draft from the title + table profiles (drafts.py "data" arm);
- draft_readme  the same + README / codebook (drafts.py "docs" arm).

1. Claim extraction, after Claimify (Metropolitansky & Larson 2025), one sentence at a time with the
   whole description as context:
   - selection       keep the sentence only if it has verifiable content, rewritten to just that part;
   - disambiguation  resolve references from the context and title; a part no reader could pin down
                     is cut (the paper drops the whole sentence: on 2026-11-26 that hid a planted
                     "400 participants" because "binary indicators" in the same sentence was ambiguous);
   - decomposition   split into standalone atomic claims, keeping qualifiers.
   Changes from the paper, for dataset descriptions: one completion per stage (the paper votes over
   several); "the dataset / the study" always means the record, so it is never ambiguous; a statement
   of possible use counts only for the fact it implies about the contents ("can be used to study
   demographics" -> the data has demographic variables); a list of columns named together stays one
   claim, so column-heavy texts don't get dozens of claims.
2. Curator claims tagged data (checkable against the record's tables) / context (the study, its
   findings, material not in the tables) / provenance (who, why, how collected, funding, papers).
   "data" here is narrower than the old "content" tag, which also held findings and report facts.
3. Coverage: each curator claim graded against each candidate text (stated / partial / contradicted /
   not stated).
4. Support: each candidate claim graded against the evidence (title, curator text, README, whole-file
   profile of up to 6 tables + first rows): supported / contradicted / unsupported.

  ./run_in_worker.sh claims [--workers 6] [--records ID ...]

Writes data/claims/results.json and data/claims/hand-check.csv (a sample for a person to grade).
"""

from __future__ import annotations

import csv
import json
import random
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from eval.description_eval.common import EVIDENCE_CHARS, COLLECTED_DIR, DATA_DIR, ask_json, read_json, write_json
from eval.description_eval.evidence import record_evidence
from eval.description_eval.questions import paired, texts
from eval.description_eval.semtypes import JUDGE_MODEL

# Extraction runs once per sentence and stage (~2,000 calls), so it uses the cheap model; a Sonnet
# extraction would cost about $30. Every grading call uses the judge, a different family from the
# generator, as in the other evals.
EXTRACT_MODEL = "@vertexai/gemini-2.5-flash"
OUT = DATA_DIR / "claims"
# The final candidates (2026-11-26, after the drafter loop):
#   autoddg       AutoDDG's description of the record's first file (what a record's top hit shows)
#   autoddg_all   all of the record's AutoDDG file descriptions together (what a user scrolling the
#                 record's datasets sees; the fair "AutoDDG today" at record level)
#   draft_data    the loop's winner without a README (V0b)
#   draft_readme  the loop's winner with the README (V0)
#   draft_paper   the loop's winner with the linked paper's provenance sheet (V10b; 12 records)
CANDIDATES = ("autoddg", "autoddg_all", "draft_data", "draft_readme", "draft_paper")
WINNERS = {"draft_data": ("V0b", "data"), "draft_readme": ("V0", "docs"), "draft_paper": ("V10b", "docs")}

SELECT = """You are extracting verifiable factual claims from a description of a research data record,
one sentence at a time.

Record title: {title}

The whole description, for context:
<<<
{context}
>>>

Target sentence: <<<{sentence}>>>

Does the target sentence contain at least one specific, verifiable proposition: something that is true
or false about the data, what it covers, who made it, or how it was made? Opinions, generic praise and
filler are not verifiable. A statement of possible use ("can be used to...", "valuable for...") counts
only for the fact it implies about the contents: "can be used to study demographics" implies "the data
has demographic variables". If it is verifiable, rewrite the sentence to keep only the verifiable part,
adding nothing.

Return JSON only: {{"verifiable": true|false, "sentence": "<rewritten sentence, or empty>"}}
"""

DISAMBIGUATE = """Record title: {title}

The whole description, for context:
<<<
{context}
>>>

Sentence: <<<{sentence}>>>

Look for ambiguity in the sentence:
- referential: a pronoun or vague reference ("it", "these", "the latter", "the sample");
- structural: a sentence with more than one reasonable reading.
"The dataset", "the data", "the record" and "the study" always mean the record named in the title:
that is never ambiguous, leave it as is. Resolve each ambiguity from the context or the title if most
readers would resolve it the same way. If one can't be resolved with confidence, cut ONLY the ambiguous
part and keep the rest of the sentence; set "resolvable" to false only when nothing clear is left.

Return JSON only: {{"ambiguities": ["..."], "resolvable": true|false, "sentence": "<the sentence with references resolved>"}}
"""

DECOMPOSE = """Record title: {title}

Sentence: <<<{sentence}>>>

Split the sentence into atomic claims about the record. Each claim:
- states one fact that is true or false on its own;
- reads on its own (say "the dataset" or "the study" for the record);
- uses only what the sentence says (context only to clarify, in [brackets]);
- keeps qualifiers such as "approximately", "up to", "selected", "as of" (dropping one changes the fact).
A list of columns, variables or categories named together is ONE claim; don't split it per item.
One action with its manner or method is ONE claim ("choices were recorded by hand on paper with tally
marks", not three). Leave out claims that say nothing specific ("the dataset contains data").

Return JSON only: {{"claims": ["..."]}}
"""

TAG = """These claims come from a curator's description of a research data record. Tag each one:
- "data": could be checked against the record's tables below: a variable or column, its values or
  categories, units, the place or time span the rows cover, the number of rows, files or items;
- "provenance": who made it, why (the study's aim or question), how or where it was collected,
  funding, related publications, license;
- "context": anything else about the study or project: its findings or conclusions, its scope, or
  material not in the tables (reports, other files, instruments the tables don't show).

Record title: {title}

Tables in the record (profiled over every row):
<<<
{tables}
>>>

Claims:
{claims}

Return JSON only: [{{"id": <id>, "kind": "data|provenance|context"}}]
"""

COVER = """Reference claims come from the curator's description of a research data record. The candidate
text is a generated description of the same record. For each claim, grade the candidate text:
- "stated": it says the same fact (wording may differ; a more specific version counts);
- "partial": it says part of the fact, or a vaguer version;
- "contradicted": it says something incompatible with the fact;
- "not_stated": it doesn't say.

Candidate text:
<<<
{candidate}
>>>

Claims:
{claims}

Return JSON only: [{{"id": <id>, "label": "stated|partial|contradicted|not_stated"}}]
"""

SUPPORT = """Check claims from a generated description of a research data record against the evidence.

Evidence:
<<<
{evidence}
>>>

For each claim:
- "supported": the evidence states it, or the data shows it (a value range inside a column's min and
  max, a column or category that exists, a row count, a date span);
- "contradicted": the evidence shows it is wrong (a different year range, a column that doesn't exist,
  a different study design);
- "unsupported": the evidence neither backs nor contradicts it.
For supported and contradicted, name the source: "title", "curator", "docs" (README / codebook), "paper"
(linked paper excerpts) or "data".

Claims:
{claims}

Return JSON only: [{{"id": <id>, "label": "supported|contradicted|unsupported", "source": "title|curator|docs|paper|data|"}}]
"""


ENRICH = """Here is a curator's description of a research data record, and claims from another description
of the same record (each already checked against the record's evidence). For each claim, does the curator's
description state it? "stated" (same fact, wording may differ) or "not_stated".

Curator's description:
<<<
{curator}
>>>

Claims:
{claims}

Return JSON only: [{{"id": <id>, "label": "stated|not_stated"}}]
"""


def sentences(text: str) -> list[str]:
    """Split on sentence ends and line breaks; "e.g. the" stays whole (the next word is lowercase)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+", text or "")
    return [p.strip(" -*\t") for p in parts if len(p.strip(" -*\t")) > 3]


def extract(text: str, title: str, stats: Counter | None = None) -> list[dict[str, str]]:
    """Claimify's three stages, per sentence. Returns [{claim, sentence}]. `stats` counts sentences and
    the ones each stage drops: a vague text loses sentences at disambiguation and so has fewer claims
    that can be graded unsupported, so the drop rate is reported per system."""
    out = []
    stats = stats if stats is not None else Counter()
    context = text[:6000]
    for sent in sentences(text):
        stats["sentences"] += 1
        try:
            out += _extract_sentence(sent, title, context, stats)
        except RuntimeError:  # the model returned no JSON 3 times (e.g. an empty or blocked reply)
            stats["dropped_error"] += 1
    return out


def _extract_sentence(sent: str, title: str, context: str, stats: Counter) -> list[dict[str, str]]:
    sel = ask_json(SELECT.format(title=title, context=context, sentence=sent), model=EXTRACT_MODEL) or {}
    if not sel.get("verifiable") or not str(sel.get("sentence", "")).strip():
        stats["dropped_unverifiable"] += 1
        return []
    dis = ask_json(DISAMBIGUATE.format(title=title, context=context, sentence=sel["sentence"]),
                   model=EXTRACT_MODEL) or {}
    if dis.get("resolvable") is False:
        stats["dropped_ambiguous"] += 1
        return []
    resolved = str(dis.get("sentence") or sel["sentence"])
    dec = ask_json(DECOMPOSE.format(title=title, sentence=resolved), model=EXTRACT_MODEL) or {}
    return [{"claim": c.strip(), "sentence": sent} for c in dec.get("claims") or [] if isinstance(c, str) and c.strip()]


def direct_verify(claims: list[dict[str, Any]], evidence: str, model: str = JUDGE_MODEL) -> list[dict[str, Any]]:
    """Our judge: every claim of one text graded in one call against the whole evidence block."""
    reply = ask_json(SUPPORT.format(evidence=evidence[:EVIDENCE_CHARS], claims=numbered(claims)),
                     model=model) if claims else []
    for c, l, s in zip(claims, by_id(reply, "label", len(claims), "unsupported"),
                       by_id(reply, "source", len(claims), "")):
        c.update(label=l, source=s)
    return claims


def claimify_direct(text: str, title: str, ev: dict[str, Any], stats: Counter | None = None) -> list[dict[str, Any]]:
    """Method interface shared with safe.py / veritrail.py: text -> claims with a label each."""
    return direct_verify(extract(text, title, stats), ev["evidence"])


def numbered(claims: list[dict[str, str]]) -> str:
    return "\n".join(f"{i}. {c['claim']}" for i, c in enumerate(claims))


def by_id(reply: Any, key: str, n: int, default: str) -> list[str]:
    got = {r.get("id"): str(r.get(key) or default) for r in reply if isinstance(r, dict)} if isinstance(reply, list) else {}
    return [got.get(i, default) for i in range(n)]


def rate(labels: list[str], hit: str, partial: str | None = None) -> float | None:
    if not labels:
        return None
    return round(sum(1 if l == hit else 0.5 if l == partial else 0 for l in labels) / len(labels), 3)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


def record_inputs(storage, rid: str) -> dict[str, Any]:
    # paper excerpts in the evidence for every candidate, so a paper-backed claim counts as supported
    # whichever text makes it (records without a paper sheet are unchanged)
    return record_evidence(storage, rid, with_paper=True)


def curator_claims(inp: dict[str, Any], stats: Counter | None = None) -> list[dict[str, Any]]:
    """The frozen reference: Claimify claims from the curator's description, each tagged data / provenance / context."""
    claims = extract(inp["description"], inp["title"], stats)
    kinds: list[str] = []
    # fz2tx-v7x97 (interview transcripts): Sonnet returns an empty reply every time this call carries
    # its table profiles, so retry without them, and as a last resort tag "unknown" (left out of the
    # per-kind recall, counted in stats)
    for tables in (inp["tables"][:15000], "(table profiles not shown)"):
        if not claims:
            break
        try:
            kinds = by_id(ask_json(TAG.format(title=inp["title"], tables=tables, claims=numbered(claims)),
                                   model=JUDGE_MODEL), "kind", len(claims), "context")
            break
        except RuntimeError:
            if stats is not None:
                stats["tag_retry"] += 1
    for i, c in enumerate(claims):
        k = kinds[i] if i < len(kinds) else "unknown"
        c["kind"] = k if k in ("data", "context", "provenance", "unknown") else "context"
    return claims


def score_text(text: str, inp: dict[str, Any], cur_claims: list[dict[str, Any]]) -> dict[str, Any]:
    """Coverage of the curator's claims and support of the text's own claims (Claimify + direct judge)."""
    cover = by_id(ask_json(COVER.format(candidate=text[:8000], claims=numbered(cur_claims)), model=JUDGE_MODEL),
                  "label", len(cur_claims), "not_stated") if cur_claims else []
    stats: Counter = Counter()
    own = claimify_direct(text, inp["title"], inp, stats)
    labels = [c["label"] for c in own]
    res: dict[str, Any] = {"text": text, "coverage": cover, "claims": own,
                           "recall": rate(cover, "stated", "partial"),
                           "contradicts_curator": rate(cover, "contradicted"),
                           "supported": rate(labels, "supported"), "unsupported": rate(labels, "unsupported"),
                           "contradicted": rate(labels, "contradicted"), "n_claims": len(own),
                           "extraction": dict(stats)}
    res["flagged"] = round((res["unsupported"] or 0) + (res["contradicted"] or 0), 3) if own else None
    # enrichment: true claims (supported by README / data / paper) the curator's text doesn't make; a
    # draft that adds true provenance or data detail is adding value, not noise (Cedric, 2026-11-26)
    supported = [c for c in own if c["label"] == "supported"]
    if supported and inp.get("description"):
        enrich = by_id(ask_json(ENRICH.format(curator=inp["description"][:8000], claims=numbered(supported)),
                                model=JUDGE_MODEL), "label", len(supported), "stated")
        for c, l in zip(supported, enrich):
            c["in_curator"] = l == "stated"
        added = [c for c in supported if not c["in_curator"]]
        res["enrichment"] = round(len(added) / len(own), 3)
        res["enrichment_n"] = len(added)
        res["enrichment_sources"] = dict(Counter(c.get("source") or "?" for c in added))
    for kind in ("data", "context", "provenance"):
        res[f"recall_{kind}"] = rate([l for l, c in zip(cover, cur_claims) if c["kind"] == kind], "stated", "partial")
    return res


def run_record(storage, rid: str, t: dict[str, str]) -> dict[str, Any]:
    inp = record_inputs(storage, rid)
    cur_stats: Counter = Counter()
    cur_claims = curator_claims(inp, cur_stats)
    row: dict[str, Any] = {"record": rid, "title": inp["title"], "curator_claims": cur_claims,
                           "curator_extraction": dict(cur_stats), "candidates": {}}
    for cand in CANDIDATES:
        if cand in t:
            row["candidates"][cand] = score_text(t[cand], inp, cur_claims)
    return row


def hand_check_sample(rows: list[dict[str, Any]], n: int = 30) -> None:
    """Half coverage calls, half support calls, half where the judge said yes, for a person to grade."""
    rng = random.Random(20261126)
    cover, support = [], []
    for r in rows:
        for cand, res in r["candidates"].items():
            for c, l in zip(r["curator_claims"], res["coverage"]):
                cover.append({"record": r["record"], "check": "coverage", "candidate": cand, "claim": c["claim"],
                              "judge": l, "text_or_evidence": res["text"]})
            for c in res["claims"]:
                support.append({"record": r["record"], "check": "support", "candidate": cand, "claim": c["claim"],
                                "judge": f"{c['label']} ({c['source']})" if c.get("source") else c["label"],
                                "text_or_evidence": "see the record's README / data"})
    picked = []
    for pool, yes in ((cover, "stated"), (support, "supported")):
        hit = [x for x in pool if x["judge"].startswith(yes)]
        miss = [x for x in pool if not x["judge"].startswith(yes)]
        picked += rng.sample(hit, min(n // 4, len(hit))) + rng.sample(miss, min(n // 4, len(miss)))
    with (OUT / "hand-check.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["record", "check", "candidate", "claim", "judge", "text_or_evidence",
                                           "cedric_grade", "note"])
        w.writeheader()
        w.writerows(picked)


def main(workers: int = 6, only: list[str] | None = None) -> None:
    from storage.minio_client import get_storage_client

    storage = get_storage_client()
    from eval.description_eval.drafter_loop import draft
    from eval.description_eval.papers import sheet_text

    records = texts()
    if only:
        records = {k: v for k, v in records.items() if k in only}
    for rid, t in records.items():
        t.pop("draft_readme", None)  # rebuilt below from the loop's winner (V0 = the same cached draft)
        collected = COLLECTED_DIR / f"{rid}.json"
        if collected.exists():
            ufds = [f["ufd"] for _, f in sorted(read_json(collected)["files"].items()) if f.get("ufd")]
            if ufds:
                t["autoddg_all"] = "\n\n".join(ufds)
        inp = record_evidence(storage, rid)
        for cand, (variant, arm) in WINNERS.items():
            if cand == "draft_paper" and not sheet_text(rid):
                continue
            if arm == "docs" and not inp["docs"]:
                continue
            text = draft(variant, inp, arm, rid)
            if text:
                t[cand] = text
    print(f"{len(records)} UV records; candidates: {dict(Counter(k for t in records.values() for k in t if k != 'curator'))}")

    def safe(item):
        rid, t = item
        try:
            row = run_record(storage, rid, t)
        except Exception as exc:  # one bad record shouldn't lose the paid-for calls of the rest
            return {"record": rid, "error": repr(exc), "candidates": {}, "curator_claims": []}
        print(f"  {rid}: {len(row['curator_claims'])} curator claims; "
              + ", ".join(f"{c} recall {r['recall']} unsupported {r['unsupported']}" for c, r in row["candidates"].items()),
              flush=True)
        return row

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(safe, records.items()))

    ok = [r for r in rows if not r.get("error")]
    kinds = Counter(c["kind"] for r in ok for c in r["curator_claims"])
    summary: dict[str, Any] = {"records": len(ok), "errors": [r["record"] for r in rows if r.get("error")],
                               "extract_model": EXTRACT_MODEL, "judge": JUDGE_MODEL,
                               "curator_claims": sum(kinds.values()), "curator_claim_kinds": dict(kinds),
                               "curator_extraction": dict(sum((Counter(r.get("curator_extraction") or {}) for r in ok),
                                                              Counter())),
                               "candidates": {}}
    questions = {r["record"]: r for r in read_json(DATA_DIR / "questions" / "results.json")["rows"]} \
        if (DATA_DIR / "questions" / "results.json").exists() else {}
    for cand in CANDIDATES:
        mine = [r["candidates"][cand] for r in ok if cand in r["candidates"]]
        if not mine:
            continue
        s: dict[str, Any] = {"records": len(mine)}
        for m in ("recall", "recall_data", "recall_context", "recall_provenance", "contradicts_curator",
                  "supported", "unsupported", "contradicted", "flagged", "n_claims", "enrichment", "enrichment_n"):
            vals = [x[m] for x in mine if x.get(m) is not None]
            s[m] = round(statistics.mean(vals), 3) if vals else None
        ext = sum((Counter(x.get("extraction") or {}) for x in mine), Counter())
        s["extraction"] = {**dict(ext), "drop_rate_unverifiable": round(ext["dropped_unverifiable"] / ext["sentences"], 3)
                           if ext["sentences"] else None,
                           "drop_rate_ambiguous": round(ext["dropped_ambiguous"] / ext["sentences"], 3)
                           if ext["sentences"] else None}
        s["support_labels"] = dict(Counter(c["label"] for x in mine for c in x["claims"]))
        s["support_sources"] = dict(Counter(c["source"] for x in mine for c in x["claims"] if c.get("source")))
        s["enrichment_sources"] = dict(sum((Counter(x.get("enrichment_sources") or {}) for x in mine), Counter()))
        # next to the 11-17 question evals: does claim recall rank records the way question coverage does?
        pairs = [(r["candidates"][cand]["recall"], questions[r["record"]]["coverage"].get(cand))
                 for r in ok if cand in r["candidates"] and r["record"] in questions]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if pairs:
            s["question_coverage"] = round(statistics.mean(b for _, b in pairs), 3)
            s["spearman_vs_question_coverage"] = spearman([a for a, _ in pairs], [b for _, b in pairs])
        summary["candidates"][cand] = s
    flat = [{"recall": {c: r["candidates"][c]["recall"] for c in r["candidates"]},
             "unsupported": {c: r["candidates"][c]["unsupported"] for c in r["candidates"]}} for r in ok]
    summary["draft_readme_vs_autoddg"] = {"recall": paired(flat, "autoddg", "draft_readme", "recall"),
                                          "unsupported": paired(flat, "autoddg", "draft_readme", "unsupported")}
    summary["draft_readme_vs_autoddg_all"] = {"recall": paired(flat, "autoddg_all", "draft_readme", "recall"),
                                              "unsupported": paired(flat, "autoddg_all", "draft_readme", "unsupported")}
    summary["draft_paper_vs_draft_readme"] = {"recall": paired(flat, "draft_readme", "draft_paper", "recall"),
                                              "unsupported": paired(flat, "draft_readme", "draft_paper", "unsupported")}
    summary["draft_readme_vs_draft_data"] = {"recall": paired(flat, "draft_data", "draft_readme", "recall"),
                                             "unsupported": paired(flat, "draft_data", "draft_readme", "unsupported")}
    OUT.mkdir(parents=True, exist_ok=True)
    if only:  # a smoke test or a retry never overwrites the full run
        write_json(OUT / "results-subset.json", {"summary": summary, "rows": rows})
        print(json.dumps(summary, indent=1))
        return
    write_json(OUT / "results.json", {"summary": summary, "rows": rows})
    hand_check_sample(ok)
    print(json.dumps(summary, indent=1))
