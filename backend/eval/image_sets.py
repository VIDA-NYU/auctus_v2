"""Collection-level descriptions for image sets (Juliana, 11-10): one per set, not per image.

Each description says what the set is (from the curator's record text) plus what ML people
care about: count, resolution, content type, how varied it is. Three sources, kept apart:
- facts from the index doc and the zip listings (count, sizes, bands, camera, folders, extent);
- the record's own title and description (what the set is, why it was made);
- per-image descriptions (Gemini Flash), sampled across the set's folder groups so the
  variety claim covers the whole set, not one corner of it.
Gemini Flash writes the description in slots; Claude Sonnet 5.5 checks every claim against
the three sources (text only).

  python -m eval.image_sets [--records aqjnq-6c976 1n4p2-ygf77] [--per-set 24]

Reuses eval.image_descriptions' JPEG conversion, prompts and cache, so images described on
11-10 cost nothing again.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from eval.description_eval.common import read_json, write_json
from eval.image_descriptions import (CHECK, CHECKER, DESCRIBE, DESCRIBER, MAX_DOWNLOAD, OUT, UV, ask_image, ask_text,
                                     fetch, to_jpeg)

DEFAULT_RECORDS = ["aqjnq-6c976", "1n4p2-ygf77"]  # Dublin aerial survey, LookUp3D

COMPOSE = """Write a collection-level description of an image set in NYU's research data repository, for
people searching for data (including machine-learning researchers looking for image datasets).

FACTS (from file headers and zip listings; the only source for numbers):
{facts}

CURATOR TEXT (the record's title and description; the only source for what the set is, who made it, why):
{curator}

IMAGE DESCRIPTIONS ({n} images sampled across the set's folder groups; the only source for what images show):
{descriptions}

Fill each slot in plain words. Don't state anything the sources above don't support; leave a slot "" if
they say nothing. Numbers come from FACTS only. Describe the images in this set, not the record's other files
(mention those only in "related_files"). Say what the set is and contains; don't suggest uses or tasks the
sources don't state.

Return JSON only:
{{"what": "1 sentence: what the set is and what it's for",
  "content_type": "kind of images (e.g. aerial orthophotos, calibration patterns)",
  "count": "how many images, how they are packaged",
  "resolution": "pixel sizes, bands, bit depth, camera",
  "variety": "how the images differ across the set (groups, viewpoints, subjects, conditions)",
  "coverage": "place / time covered, if the sources give it",
  "related_files": "other data in the record, briefly, or \\"\\"",
  "caveats": "what a user should know (sampled sizes, missing georeferencing, ...), or \\"\\"",
  "summary": "3-4 sentences for a search result, combining the above"}}
"""

CHECK_SET = """Below are the sources a model was given and the collection description it wrote. Check every
claim in the description against the sources. A claim is supported if FACTS, CURATOR TEXT or the IMAGE
DESCRIPTIONS state it or directly imply it.

FACTS:
{facts}

CURATOR TEXT:
{curator}

IMAGE DESCRIPTIONS:
{descriptions}

DESCRIPTION TO CHECK:
{description}

Return JSON only: {{"claims": <number of distinct claims>, "unsupported": ["<claim> (why)", ...],
"verdict": "accurate|partly|wrong"}}
"""


def group_of(zip_key: str, member: str) -> str:
    """Folder group: zip name + member folder with digits masked (Dublin F_/T_ manifests, LookUp3D positions)."""
    folder = member.rsplit("/", 1)[0] if "/" in member else ""
    zip_part = re.sub(r"\.zip$", "", zip_key.rsplit("/", 1)[-1])
    return re.sub(r"\d+", "#", f"{zip_part}/{folder}" if folder else zip_part)


def set_doc(record_id: str) -> dict[str, Any]:
    from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client

    hits = get_client().search(index=AUCTUS_INDEX_NAME, body={"size": 10, "query": {"bool": {"filter": [
        {"term": {"domain": UV}}, {"prefix": {"id": f"{UV}:{record_id}:"}}]}}})["hits"]["hits"]
    sets = [h for h in hits if h["_id"].endswith(":images")]
    if not sets:
        raise SystemExit(f"{record_id}: no image set in the index")
    return max(sets, key=lambda h: ((h["_source"].get("profiler_metadata") or {}).get("raster") or {}).get("images") or 0)["_source"]


def list_members(record_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(every TIFF member of the record's image sets, the record's files) from the live record."""
    import httpx

    from crawlers.invenio.archives import member_extension, open_remote_zip
    from crawlers.invenio.crawler import expand_archive, file_entries, merge_extras
    from crawlers.invenio.formats import TIFF_EXTENSIONS
    from crawlers.invenio.http import ssl_context

    record = httpx.get(f"https://{UV}/api/records/{record_id}", verify=ssl_context(), timeout=60).json()
    entries = file_entries(record, UV, 200 << 20)
    extras: list[dict[str, Any]] = []
    for e in entries:
        if e["file_extension"] == "zip":
            expand_archive(e, None, extras)
    members = []
    for job in merge_extras([x for x in extras if x.get("image_set")]):
        for z in job["image_set"].get("zips") or [{"url": job["file_url"], "size": job["file_size"], "file_key": job.get("file_key", "")}]:
            with open_remote_zip(z["url"], z["size"]) as zf:
                for i in zf.infolist():
                    if (member_extension(i.filename) in TIFF_EXTENSIONS and not i.filename.rsplit("/", 1)[-1].startswith(".")
                            and i.file_size <= MAX_DOWNLOAD):
                        members.append({"zip": z, "file": i.filename, "group": group_of(z.get("file_key") or z["url"].split("/files/")[-1].split("/")[0], i.filename)})
    files = [{"key": f["key"], "size": f.get("size")} for f in (record.get("files", {}).get("entries") or {}).values()] \
        if isinstance(record.get("files", {}).get("entries"), dict) else []
    return members, files


def pick(members: list[dict[str, Any]], done: set[str], per_set: int, seed: int) -> list[dict[str, Any]]:
    """Spread `per_set` images over the folder groups (at least 2 per group, largest groups first),
    counting images already described."""
    rng = random.Random(seed)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in members:
        by_group[m["group"]].append(m)
    groups = sorted(by_group, key=lambda g: -len(by_group[g]))[: max(1, per_set // 2)]
    quota = {g: 2 for g in groups}
    spare = per_set - 2 * len(groups)
    total = sum(len(by_group[g]) for g in groups)
    for g in groups:  # the rest by group size
        quota[g] += round(spare * len(by_group[g]) / total) if spare > 0 else 0
    out = []
    for g in groups:
        have = [m for m in by_group[g] if m["file"] in done]
        rest = [m for m in by_group[g] if m["file"] not in done]
        need = max(0, quota[g] - len(have))
        out += have[: quota[g]] + rng.sample(rest, min(need, len(rest)))
    return out


def describe(item: dict[str, Any], record_title: str, dataset: str) -> dict[str, Any]:
    import hashlib

    key = hashlib.sha256(f"{dataset}|{item['file']}".encode()).hexdigest()[:16]
    jpeg_path = OUT / "jpeg" / f"{key}.jpg"
    try:
        if not jpeg_path.exists():
            jpeg_path.write_bytes(to_jpeg(fetch({"kind": "set", **item})))
        jpeg = jpeg_path.read_bytes()
        d = ask_image(DESCRIBER, DESCRIBE.format(record=record_title, file=item["file"].rsplit("/", 1)[-1]), jpeg, key)["parsed"]
        c = ask_image(CHECKER, CHECK.format(description=d.get("description", ""), image_type=d.get("image_type", ""),
                                            visible_text=d.get("visible_text", "")), jpeg, key)["parsed"]
        return {"file": item["file"], "group": item["group"], "key": key, **d, "check": c}
    except Exception as exc:
        return {"file": item["file"], "group": item["group"], "key": key, "error": repr(exc)[:300]}


def facts_text(doc: dict[str, Any], members: list[dict[str, Any]], files: list[dict[str, Any]]) -> str:
    r = (doc.get("profiler_metadata") or {}).get("raster") or {}
    groups = Counter(m["group"] for m in members)
    folders = r.get("folders") or {}
    zips = sorted({m["zip"].get("file_key") or m["zip"]["url"] for m in members})
    lines = [f"- images: {r.get('images') or len(members)} TIFF files in {len(zips)} zip file(s)"
             + (f", {len(folders)} folders" if folders and not all(k.endswith(".zip") for k in folders) else ""),
             f"- pixel sizes seen in {r.get('sampled')} sampled headers: {', '.join(r.get('sizes') or [])}",
             f"- bands: {r.get('bands')}, {r.get('bits_per_sample')}-bit {r.get('data_type')}",
             f"- camera / software (from headers): {' '.join(x for x in (r.get('make'), r.get('model')) if x) or 'not given'}"
             + (f"; {r['software']}" if r.get("software") else ""),
             f"- georeferencing in headers: {r.get('crs_name') or r.get('epsg') or 'none found'}",
             "- folder groups (digits masked as #), images per group: "
             + "; ".join(f"{g} {n}" for g, n in groups.most_common(12)) + (f"; and {len(groups) - 12} more" if len(groups) > 12 else "")]
    if doc.get("spatial_coverage"):
        lines.append(f"- map extent (from the index): {json.dumps(doc['spatial_coverage'])[:300]}")
    if doc.get("temporal_coverage"):
        lines.append(f"- time range (from the index): {json.dumps(doc['temporal_coverage'])[:200]}")
    if files:
        exts = Counter(f["key"].rsplit(".", 1)[-1].lower() for f in files)
        lines.append(f"- record files: {len(files)} ({', '.join(f'{n} .{e}' for e, n in exts.most_common(8))})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", nargs="+", default=DEFAULT_RECORDS)
    parser.add_argument("--per-set", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    (OUT / "jpeg").mkdir(parents=True, exist_ok=True)
    earlier = read_json(OUT / "results.json") if (OUT / "results.json").exists() else {"rows": []}
    out = {}
    for rid in args.records:
        doc = set_doc(rid)
        title = doc["source_record"]["title"]
        dataset = f"{UV}:{rid}:images"  # the 11-10 run's label, so its cache keys match
        members, files = list_members(rid)
        done = {r["file"] for r in earlier["rows"] if r.get("dataset") == dataset and not r.get("error")}
        sample = pick(members, done, args.per_set, seed=20261117)
        print(f"{rid}: {len(members)} images, {len(set(m['group'] for m in members))} groups, sampling {len(sample)} "
              f"({sum(m['file'] in done for m in sample)} already described)", flush=True)
        with ThreadPoolExecutor(args.workers) as pool:
            rows = list(pool.map(lambda m: describe(m, title, dataset), sample))
        ok = [r for r in rows if not r.get("error")]
        facts = facts_text(doc, members, files)
        curator = f"Title: {title}\nDescription: {doc['source_record'].get('description') or ''}"[:4000]
        descriptions = "\n".join(f"- [{r['group']}] ({r.get('image_type')}) {r.get('description')}" for r in ok)
        composed = ask_text(DESCRIBER, COMPOSE.format(facts=facts, curator=curator, n=len(ok), descriptions=descriptions))["parsed"]
        text = "\n".join(f"{k}: {v}" for k, v in composed.items() if v)
        checked = ask_text(CHECKER, CHECK_SET.format(facts=facts, curator=curator, descriptions=descriptions, description=text))["parsed"]
        out[rid] = {"title": title, "facts": facts, "curator": curator, "images": rows, "description": composed,
                    "check": checked, "image_checks": dict(Counter((r.get("check") or {}).get("verdict") for r in ok)),
                    "errors": len(rows) - len(ok)}
        print(json.dumps({"description": composed, "check": checked, "image_checks": out[rid]["image_checks"]},
                         indent=1, ensure_ascii=False), flush=True)
    write_json(OUT / "sets.json", out)


if __name__ == "__main__":
    main()
