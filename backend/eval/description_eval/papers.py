"""Provenance from a record's linked papers and reports: full text, data-relevant parts only.

11-17: the abstract alone added little over the README (8 records had one), and Juliana asked to pull
only the data-relevant sections of the full paper, then answer provenance questions from them.

  1. sources    related_identifiers of type publication* (DOI or URL): papers, reports, report PDFs
  2. full text  DOI -> OpenAlex open-access PDF (best_oa_location, then any location); URL -> fetched
                directly (PDF via pypdf, HTML with tags stripped). Paywalled papers are skipped, logged.
  3. selection  ~1,200-character chunks; keep the ones about how the data came about (methods, data,
                participants, measures, sampling, limitations, funding, data availability): BM25 on
                those terms plus section headings, at most 30 chunks
  4. sheet      Claude Sonnet fills a provenance sheet from the selected chunks only, each field with
                the chunk ids it comes from, null when the paper doesn't say

The extractor sees the record title and the tables' column names (to know which data the paper is
about), never the curator's description, so a draft built on the sheet can still be scored against it.

  MODULE=eval.description_eval.papers ./run_in_worker.sh [--records ID ...]
Writes data/papers/<record>.json (+ the fetched text under data/papers/text/).
"""

from __future__ import annotations

import argparse
import io
import json
import re
from typing import Any

from eval.description_eval.common import DATA_DIR, GOLD_DIR, ask_json, read_json, write_json
from eval.description_eval.evidence import BM25
from eval.description_eval.semtypes import JUDGE_MODEL

OUT = DATA_DIR / "papers"
UV_DOI_PREFIX = "10.58153"
CHUNK = 1200
MAX_CHUNKS = 30
HEADERS = {"User-Agent": "auctus-description-eval/0.1 (research evaluation; no crawling)"}
QUERY = ("method methods data dataset collected collection survey participants sample sampling recruited "
         "interview instrument measured measures variables study design procedure protocol sources source "
         "funding funded grant limitation limitations availability available ethics IRB approval period "
         "fieldwork site location coded coding cleaned processing weights questionnaire index indicator")
HEADING = re.compile(r"^\s*(\d+(\.\d+)*\.?\s+)?(methods?|materials and methods|data( and methods)?|data collection|"
                     r"study design|participants|measures|methodology|sample|sampling|procedure|limitations?|"
                     r"funding|acknowledg(e)?ments?|data availability)\b", re.IGNORECASE | re.MULTILINE)

SHEET = """You are an expert research data curator. Below are excerpts from a paper or report linked to a
dataset record in NYU's research data repository. Fill a provenance sheet for the DATA in the record:
how it came about, as the excerpts state it.

Record title: {title}

Tables in the record and their columns (to know which data the paper is about):
{tables}

Excerpts (each with an id):
{chunks}

Rules:
* Use ONLY the excerpts. If they don't say, the field MUST be null. Do NOT fill in what is typical.
* Cite the excerpt ids each field comes from.
* About the data, not the paper's findings (findings go in "findings", briefly).

Return JSON only:
{{"creators": {{"value": "who collected or produced the data", "ids": ["P3"]}},
  "purpose": {{"value": "why: the research question or aim", "ids": []}},
  "collection_method": {{"value": "how the data was collected or compiled", "ids": []}},
  "instruments_measures": {{"value": "instruments, questionnaires, measures, indices", "ids": []}},
  "sampling_participants": {{"value": "who or what was sampled, how many, how recruited", "ids": []}},
  "time_place": {{"value": "when and where the data was collected", "ids": []}},
  "sources": {{"value": "data sources it was compiled from", "ids": []}},
  "processing": {{"value": "cleaning, coding, weighting, derived variables", "ids": []}},
  "assumptions": {{"value": "assumptions the data relies on", "ids": []}},
  "limitations": {{"value": "known limits, gaps, biases of the data", "ids": []}},
  "funding": {{"value": "...", "ids": []}},
  "ethics_access": {{"value": "ethics approval, anonymization, access conditions", "ids": []}},
  "findings": {{"value": "the main finding in one sentence", "ids": []}}}}
Use null in place of any object the excerpts don't support.
"""


def sources(record: dict[str, Any]) -> list[dict[str, str]]:
    out = []
    for rel in (record.get("metadata") or {}).get("related_identifiers") or []:
        rtype = (rel.get("resource_type") or {}).get("id", "")
        ident = rel.get("identifier", "")
        if not rtype.startswith("publication") or not ident:
            continue
        if rel.get("scheme") == "doi" and not ident.startswith(UV_DOI_PREFIX):
            out.append({"kind": "doi", "id": ident, "type": rtype})
        elif rel.get("scheme") == "url":
            out.append({"kind": "url", "id": ident, "type": rtype})
    return out


def _text_from(content: bytes, content_type: str) -> str:
    if content[:5] == b"%PDF-" or "pdf" in content_type:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "") for page in reader.pages[:80])
    html = content.decode("utf-8", errors="replace")
    html = re.sub(r"(?is)<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", text))


def full_text(src: dict[str, str]) -> tuple[str, str]:
    """(text, url used), or ("", reason)."""
    import httpx

    urls = []
    if src["kind"] == "doi":
        try:
            work = httpx.get(f"https://api.openalex.org/works/doi:{src['id']}", timeout=30, headers=HEADERS).json()
        except Exception as exc:
            return "", f"openalex error {exc!r}"
        for loc in [work.get("best_oa_location") or {}] + (work.get("locations") or []):
            for key in ("pdf_url", "landing_page_url"):
                if loc and loc.get(key) and loc.get("is_oa", work.get("open_access", {}).get("is_oa")):
                    urls.append(loc[key])
        if not urls:
            return "", "no open-access copy in OpenAlex"
    else:
        urls = [src["id"]]
    for url in dict.fromkeys(urls):
        try:
            r = httpx.get(url, timeout=60, follow_redirects=True, headers=HEADERS)
        except Exception:
            continue
        if r.status_code != 200:
            continue
        ctype = r.headers.get("content-type", "")
        # a landing page (dhl.com/gci) links to the report: follow PDF links that look like the full
        # report, same site only, before reading the page itself
        if "html" in ctype:
            base = str(r.url)
            links = re.findall(r'href="([^"]+\.pdf[^"]*)"', r.text, re.IGNORECASE)
            host = re.match(r"https?://[^/]+", base).group(0)
            links = [l if l.startswith("http") else host + ("" if l.startswith("/") else "/") + l for l in links]
            links = sorted({l for l in links if l.startswith(host)}, key=lambda l: not re.search(r"report|full|study|index", l, re.I))
            for link in links[:3]:
                try:
                    pr = httpx.get(link, timeout=60, follow_redirects=True, headers=HEADERS)
                except Exception:
                    continue
                if pr.status_code == 200 and pr.content[:5] == b"%PDF-":
                    text = _text_from(pr.content, "application/pdf")
                    if len(text) > 2000:
                        return text, link
        text = _text_from(r.content, ctype)
        if len(text) > 2000:  # a landing page with no paper on it is shorter than this
            return text, url
    return "", "no full text fetched (paywalled, blocked, or a landing page)"


def _pieces(para: str) -> list[str]:
    """A paragraph cut into pieces of at most CHUNK characters, at sentence ends where possible. pypdf
    output has few blank lines, so a "paragraph" can be 30,000 characters (2026-11-26: chunks had a
    median of 3-8k characters, and both the sheet and the judge only saw their starts)."""
    if len(para) <= CHUNK:
        return [para]
    out, buf = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", para):
        while len(sent) > CHUNK:  # a run-on table or list with no sentence end
            out.append(sent[:CHUNK])
            sent = sent[CHUNK:]
        if len(buf) + len(sent) > CHUNK and buf:
            out.append(buf.strip())
            buf = ""
        buf += sent + " "
    if buf.strip():
        out.append(buf.strip())
    return out


def chunks(text: str) -> list[dict[str, str]]:
    text = re.sub(r"[ \t]+", " ", text)
    out, buf = [], ""
    for para in re.split(r"\n\s*\n|\n(?=\s*\d+(?:\.\d+)*\.?\s+[A-Z])", text):
        if not para or not para.strip():
            continue
        for piece in _pieces(para.strip()):
            if len(buf) + len(piece) > CHUNK and buf:
                out.append(buf.strip())
                buf = ""
            buf += piece + "\n"
    if buf.strip():
        out.append(buf.strip())
    return [{"id": f"P{i}", "source": "paper", "text": c} for i, c in enumerate(out)]


def select(cs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Data-relevant chunks: section headings first, then BM25 on provenance terms, in paper order."""
    keep = {c["id"] for c in cs if HEADING.search(c["text"][:200])}
    for c in BM25(cs).search(QUERY, k=MAX_CHUNKS):
        if len(keep) >= MAX_CHUNKS:
            break
        keep.add(c["id"])
    return [c for c in cs if c["id"] in keep][:MAX_CHUNKS]


def provenance(rid: str, record: dict[str, Any]) -> dict[str, Any]:
    gold = read_json(GOLD_DIR / f"{rid}.json")
    title = (gold.get("curator") or {}).get("title", "")
    out: dict[str, Any] = {"record": rid, "sources": [], "sheet": None, "chunks": []}
    texts = []
    # PDFs downloaded by hand (sites that block scripted downloads, e.g. dhl.com): data/papers/manual/<record>*.pdf
    for pdf in sorted((OUT / "manual").glob(f"{rid}*.pdf")) if (OUT / "manual").exists() else []:
        text = _text_from(pdf.read_bytes(), "application/pdf")
        out["sources"].append({"kind": "manual", "id": pdf.name, "fetched": bool(text), "url_or_reason": "downloaded by hand",
                               "chars": len(text)})
        if text:
            texts.append(text)
    for src in sources(record):
        if texts and src["kind"] == "url" and "dhl.com" in src["id"]:
            continue  # the hand-downloaded copy stands in for it
        text, how = full_text(src)
        out["sources"].append({**src, "fetched": bool(text), "url_or_reason": how, "chars": len(text)})
        if text:
            (OUT / "text").mkdir(parents=True, exist_ok=True)
            (OUT / "text" / f"{rid}-{len(texts)}.txt").write_text(text)
            texts.append(text)
    if not texts:
        return out
    cs = chunks("\n\n".join(texts))
    chosen = select(cs)
    out["chunks"] = chosen
    out["n_chunks_total"] = len(cs)
    cols = "\n".join(f"- {f['file_key']}" for f in gold["files"][:12])
    listing = "\n\n".join(f"[{c['id']}] {c['text']}" for c in chosen)
    sheet = ask_json(SHEET.format(title=title, tables=cols, chunks=listing[:60000]), model=JUDGE_MODEL)
    out["sheet"] = {k: v for k, v in (sheet or {}).items() if isinstance(v, dict) and v.get("value")}
    return out


def sheet_text(rid: str) -> str:
    """The sheet as prompt text for the drafter ("" if the record has none)."""
    path = OUT / f"{rid}.json"
    if not path.exists():
        return ""
    sheet = read_json(path).get("sheet") or {}
    return "\n".join(f"- {k.replace('_', ' ')}: {v['value']}" for k, v in sheet.items())


def excerpt_lines(rid: str) -> list[dict[str, str]]:
    """The selected chunks, as evidence lines (source "paper")."""
    path = OUT / f"{rid}.json"
    return [{"source": "paper", "text": c["text"][:1200]} for c in (read_json(path).get("chunks") or [])] if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", nargs="+")
    args = parser.parse_args()
    raw = read_json(DATA_DIR / "records-ultraviolet.json")
    raw = raw if isinstance(raw, list) else raw.get("records") or raw.get("hits") or []
    records = {r["id"]: r for r in raw if isinstance(r, dict) and "id" in r}
    rids = args.records or [read_json(p)["record_id"] for p in sorted(GOLD_DIR.glob("*.json"))
                            if read_json(p).get("domain") == "ultraviolet.library.nyu.edu"]
    summary = []
    for rid in rids:
        if rid not in records or not sources(records[rid]):
            continue
        try:
            res = provenance(rid, records[rid])
        except Exception as exc:
            res = {"record": rid, "error": repr(exc), "sources": []}
        write_json(OUT / f"{rid}.json", res)
        summary.append({"record": rid, "fetched": sum(s.get("fetched", False) for s in res["sources"]),
                        "sources": len(res["sources"]), "fields": sorted((res.get("sheet") or {}).keys()),
                        "error": res.get("error")})
        print(json.dumps(summary[-1]), flush=True)
    write_json(OUT / "summary.json", summary)


if __name__ == "__main__":
    main()
