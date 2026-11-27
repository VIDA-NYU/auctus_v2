"""Compare the claim-checking methods: Claimify + direct judge, Claimify + VeriTrail, SAFE.

Three checks, on a subset of records (the per-claim methods cost 10-50x the direct judge):
  1. hand-verified claims: 13 claims Cedric's 2026-11-26 review checked against the files (5 false,
     8 true; the true ones include numbers the old judge wrongly called unsupported). Each method's
     verifier grades them directly: does it say "not supported" for the false ones and "supported"
     for the true ones?
  2. planted errors (injected.py): each record's README draft with two planted errors (data, provenance);
     per method, was each error extracted, and was it caught?
  3. the real texts (AutoDDG today, the README draft): claims per text, share unsupported /
     contradicted / inconclusive, and claim-level agreement between direct and VeriTrail (same claims).

  MODULE=eval.description_eval.compare_methods ./run_in_worker.sh [--records ID ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from eval.description_eval import injected, safe as safe_mod, veritrail as vt
from eval.description_eval.claims import DECOMPOSE, EXTRACT_MODEL, claimify_direct, direct_verify, extract
from eval.description_eval.common import DATA_DIR, ask_json, read_json, write_json
from eval.description_eval.evidence import BM25, record_evidence
from eval.description_eval.questions import texts

OUT = DATA_DIR / "methods"
DEFAULT_RECORDS = ["9wnjp-kez15", "9yxx4-fzp66", "wwjkw-vjs44", "7jt4h-p0738", "avp10-a8h86",
                   "1vs56-e3h85", "frv59-hzf81", "dshqk-atn19"]

# Checked by hand against the files on 2026-11-26 (meetings/2026-11-26 notes, doc "Failure cases").
KNOWN = [
    ("wwjkw-vjs44", "The DHL 2022 scores and ranks data spans 20 distinct years.", False),
    ("wwjkw-vjs44", "The year column in the DHL 2022 world trends data covers 2001 to 2020.", False),
    ("7jt4h-p0738", "The DHL 2024 scores and ranks data is available from 2002 to 2022.", False),
    ("9yxx4-fzp66", "The chocolate choices come from a consumer research survey.", False),
    ("avp10-a8h86", "The dataset has demographic variables.", False),
    ("9wnjp-kez15", "The rail projects in the dataset start as early as 1963 and end as late as 2036.", True),
    ("9wnjp-kez15", "The total length of a rail project in the dataset goes up to 200 km.", True),
    ("9wnjp-kez15", "The number of stations on a project goes up to 128.", True),
    ("9wnjp-kez15", "Cost per kilometer reaches up to $5,050.45 million in 2023 dollars.", True),
    ("9yxx4-fzp66", "Each participant in experimental_data.csv has a unique participant_id.", True),
    ("9yxx4-fzp66", "Lindt was the most frequently chosen option in the experiment.", True),
    ("wwjkw-vjs44", "In the DHL 2022 world trends data, depth_index ranges from about 85.32 to 307.55.", True),
    ("wwjkw-vjs44", "The pillar column holds trade, capital, information, people and overall.", True),
]

def direct_sonnet46(text, title, ev, stats=None):
    """Control: the direct judge on VeriTrail's verdict model, same Claimify claims."""
    return direct_verify(extract(text, title, stats), ev["evidence"], model=vt.VERDICT_MODEL)


METHODS = {"direct": claimify_direct, "direct_sonnet46": direct_sonnet46, "veritrail": vt.veritrail, "safe": safe_mod.safe}


def verify_one(method: str, claim: str, title: str, ev: dict[str, Any]) -> str:
    """One claim through a method's verifier only (no extraction)."""
    if method in ("direct", "direct_sonnet46"):
        model = vt.VERDICT_MODEL if method == "direct_sonnet46" else None
        return direct_verify([{"claim": claim}], ev["evidence"], **({"model": model} if model else {}))[0]["label"]
    if method == "veritrail":
        nodes = vt.nodes(ev["lines"])
        subs = (ask_json(DECOMPOSE.format(title=title, sentence=claim), model=EXTRACT_MODEL) or {}).get("claims") or []
        return vt.verdict(claim, vt.select_evidence(claim, [s for s in subs if isinstance(s, str)], nodes), nodes)
    r = safe_mod.rate_fact(claim, claim, f"Describe the research data record titled '{title}'.", BM25(ev["lines"]), standalone=True)
    return r["label"]


def rates(claims: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(claims)
    c = Counter(x.get("label") for x in claims)
    return {"n_claims": n, **{k: round(c[k] / n, 3) if n else None for k in ("supported", "unsupported", "contradicted", "inconclusive", "refused")},
            "flagged": round((c["unsupported"] + c["contradicted"]) / n, 3) if n else None}


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def run_record(storage, rid: str, t: dict[str, str]) -> dict[str, Any]:
    ev = record_evidence(storage, rid)
    row: dict[str, Any] = {"record": rid, "texts": {}, "planted": None}
    planted = injected.plant(t["draft_readme"], ev["evidence"]) if t.get("draft_readme") else None
    row["planted"] = planted
    for name in ("autoddg", "draft_readme"):
        if name in t:
            row["texts"][name] = {"text": t[name], "methods": {}}
    if planted:
        row["texts"]["planted"] = {"text": planted["text"], "methods": {}}
    for name, entry in row["texts"].items():
        for m, fn in METHODS.items():
            stats: Counter = Counter()
            try:
                claims = fn(entry["text"], ev["title"], ev, stats)
            except RuntimeError as exc:  # a refusal the method can't route around: record it, go on
                entry["methods"][m] = {"claims": [], "stats": dict(stats), "error": repr(exc), **rates([])}
                continue
            res = {"claims": claims, "stats": dict(stats), **rates(claims)}
            if name == "planted":
                res["detection"] = injected.detection(planted, claims)
            entry["methods"][m] = res

    def guarded(m, c):
        try:
            return verify_one(m, c, ev["title"], ev)
        except RuntimeError:
            return "refused"

    row["known"] = [{"claim": c, "true": truth, **{m: guarded(m, c) for m in METHODS}} for r, c, truth in KNOWN if r == rid]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", nargs="+", default=DEFAULT_RECORDS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    from storage.minio_client import get_storage_client

    storage = get_storage_client()
    all_texts = texts()

    def safe_run(rid):
        try:
            return run_record(storage, rid, all_texts.get(rid, {}))
        except Exception as exc:
            return {"record": rid, "error": repr(exc)}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(safe_run, args.records))
    ok = [r for r in rows if not r.get("error")]

    summary: dict[str, Any] = {"records": len(ok), "errors": [(r["record"], r["error"]) for r in rows if r.get("error")],
                               "methods": {}}
    known = [k for r in ok for k in r["known"]]
    for m in METHODS:
        s: dict[str, Any] = {}
        # 1. hand-verified claims: a false claim is caught if not "supported"; a true one is right if "supported"
        false_ = [k for k in known if not k["true"]]
        true_ = [k for k in known if k["true"]]
        s["known_false_caught"] = f"{sum(k[m] != 'supported' for k in false_)}/{len(false_)}"
        s["known_true_supported"] = f"{sum(k[m] == 'supported' for k in true_)}/{len(true_)}"
        # 2. planted errors
        det = [d for r in ok if r["texts"].get("planted") for d in r["texts"]["planted"]["methods"][m].get("detection", [])]
        for kind in ("data", "provenance"):
            mine = [d for d in det if d["kind"] == kind]
            s[f"planted_{kind}"] = {"n": len(mine), "extracted": sum(d["extracted"] for d in mine),
                                    "caught": sum(d["caught"] for d in mine)}
        # 3. real texts
        for name in ("autoddg", "draft_readme"):
            per = [r["texts"][name]["methods"][m] for r in ok if name in r["texts"]]
            s[name] = {k: round(statistics.mean(vals), 3) if (vals := [p[k] for p in per if p.get(k) is not None]) else None
                       for k in ("n_claims", "flagged", "contradicted", "inconclusive", "refused")}
        summary["methods"][m] = s
    # direct vs VeriTrail grade the same Claimify claims: claim-level agreement on "supported or not"
    a, b = [], []
    for r in ok:
        for entry in r["texts"].values():
            d, v = entry["methods"]["direct"]["claims"], entry["methods"]["veritrail"]["claims"]
            if d and [x["claim"] for x in d] == [x["claim"] for x in v]:
                pairs = [(x, y) for x, y in zip(d, v) if y["label"] != "refused"]
                a += [x["label"] == "supported" for x, _ in pairs]
                b += [y["label"] == "supported" for _, y in pairs]
    summary["direct_vs_veritrail"] = {"claims": len(a), "agree": round(sum(x == y for x, y in zip(a, b)) / len(a), 3) if a else None,
                                      "kappa": kappa(a, b)}
    for left in ("direct", "direct_sonnet46"):
        a2, b2 = [], []
        for r in ok:
            for entry in r["texts"].values():
                d, v = entry["methods"][left]["claims"], entry["methods"]["veritrail"]["claims"]
                if d and [x["claim"] for x in d] == [x["claim"] for x in v]:
                    pairs = [(x, y) for x, y in zip(d, v) if y["label"] != "refused"]
                    a2 += [x["label"] == "supported" for x, _ in pairs]
                    b2 += [y["label"] == "supported" for _, y in pairs]
        summary[f"{left}_vs_veritrail"] = {"claims": len(a2), "kappa": kappa(a2, b2),
                                           "agree": round(sum(x == y for x, y in zip(a2, b2)) / len(a2), 3) if a2 else None}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "results.json", {"summary": summary, "rows": rows})
    # disagreements for a person to settle
    with (OUT / "disagreements.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["record", "text", "claim", "direct", "veritrail", "cedric_grade", "note"])
        for r in ok:
            for name, entry in r["texts"].items():
                for x, y in zip(entry["methods"]["direct"]["claims"], entry["methods"]["veritrail"]["claims"]):
                    if x["claim"] == y["claim"] and (x["label"] == "supported") != (y["label"] == "supported"):
                        w.writerow([r["record"], name, x["claim"], x["label"], y["label"], "", ""])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
