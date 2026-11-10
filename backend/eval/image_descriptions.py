"""Describe UltraViolet images with a vision model, and check the descriptions.

Images: the loose TIFFs Auctus indexes (photos, scans, GeoTIFF maps) and samples from
its image sets (TIFFs inside zips). Each is shrunk to a JPEG (long side 1024 px,
8-bit) and described by Gemini 2.5 Flash on the NYU gateway: what kind of image it
is, what it shows, visible text. People are "a person", never who. Each image set
also gets a summary from its sampled descriptions (text only).

Claude Sonnet 5.5 then checks every description against the image: accurate, partly
accurate, or wrong, and what in it the picture doesn't show. 30 descriptions go to a
person to grade, which also measures how far the checker can be trusted.

  python -m eval.image_descriptions [--total 100]

Needs tifffile + imagecodecs (16-bit and LZW TIFFs). Calls are cached by image hash.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from eval.description_eval.common import DATA_DIR, llm_client, parse_json, read_json, write_json

OUT = DATA_DIR / "images"
DESCRIBER = "@vertexai/gemini-2.5-flash"
CHECKER = "@vertexai/anthropic.claude-sonnet-5-5"
MAX_SIDE = 1024
MAX_DOWNLOAD = 400 * 1024 * 1024
UV = "ultraviolet.library.nyu.edu"

DESCRIBE = """This image is from the research data record "{record}" in NYU's data repository
(file: {file}). Describe what the image shows, for someone searching the repository.

- image_type: one of photograph, micrograph, aerial/satellite image, map, scanned document,
  chart/plot, 3D render, calibration target, other
- description: 1-2 plain sentences on what is visible. Don't guess beyond what you see; the
  record title is context, not evidence.
- visible_text: any legible text, short, or ""
- people: if people appear, say "a person" / "people" only. Never identify anyone.

Return JSON only: {{"image_type": "...", "description": "...", "visible_text": "..."}}
"""

SUMMARIZE = """These are descriptions of {n} images sampled from a set of {total} images in the research
data record "{record}". Write 2 sentences describing the set as a whole: what kind of images,
what they show, and how they vary. Only what the descriptions support.

{descriptions}

Return JSON only: {{"summary": "..."}}
"""

CHECK = """Here is an image and a description of it written by another model. Check the description
against the image only.

Description: {description}
Image type given: {image_type}
Visible text given: {visible_text}

- verdict: "accurate" (everything stated is visible), "partly" (mostly right, something
  unsupported or wrong), or "wrong"
- unsupported: anything stated that the image doesn't show (empty list if none)

Return JSON only: {{"verdict": "accurate|partly|wrong", "unsupported": ["..."], "note": "<= 15 words"}}
"""


def to_jpeg(data: bytes) -> bytes:
    """A JPEG of at most MAX_SIDE px, 8-bit RGB, from TIFF (incl. 16-bit) or anything PIL reads."""
    from PIL import Image

    try:
        import tifffile

        array = tifffile.imread(io.BytesIO(data), key=0)
        if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.ndim == 3 and array.shape[-1] > 3:
            array = array[..., :3]
        if array.dtype != np.uint8:
            lo, hi = np.percentile(array, (0.5, 99.5))
            array = np.clip((array.astype(float) - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)
    except Exception:
        image = Image.open(io.BytesIO(data))
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def ask_image(model: str, prompt: str, jpeg: bytes, cache_key: str) -> dict[str, Any]:
    path = OUT / "cache" / f"{hashlib.sha256((model + prompt + cache_key).encode()).hexdigest()}.json"
    if path.exists():
        return read_json(path)
    content = [{"type": "text", "text": prompt},
               {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}]
    extra = {} if "anthropic." in model else {"temperature": 0}
    for _ in range(3):
        r = llm_client().chat.completions.create(model=model, messages=[{"role": "user", "content": content}],
                                                 max_tokens=8000, **extra)
        try:
            parsed = parse_json(r.choices[0].message.content or "")
        except ValueError:
            continue
        usage = {"prompt": getattr(r.usage, "prompt_tokens", None), "completion": getattr(r.usage, "completion_tokens", None)}
        write_json(path, {"parsed": parsed, "usage": usage})
        return {"parsed": parsed, "usage": usage}
    raise RuntimeError("no JSON reply")


def ask_text(model: str, prompt: str) -> dict[str, Any]:
    path = OUT / "cache" / f"{hashlib.sha256((model + prompt).encode()).hexdigest()}.json"
    if path.exists():
        return read_json(path)
    r = llm_client().chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                             max_tokens=8000, temperature=0)
    out = {"parsed": parse_json(r.choices[0].message.content or ""),
           "usage": {"prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens}}
    write_json(path, out)
    return out


def image_sources(total: int) -> list[dict[str, Any]]:
    """(source, how to fetch) for the sample: every loose TIFF, then an even share of each set."""
    from crawlers.invenio.archives import member_extension, open_remote_zip
    from crawlers.invenio.formats import TIFF_EXTENSIONS
    from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client

    hits = get_client().search(index=AUCTUS_INDEX_NAME, body={
        "size": 200, "_source": ["id", "title", "download_url", "source_record", "profiler_metadata.raster"],
        "query": {"bool": {"filter": [{"term": {"domain": UV}}, {"terms": {"profiler_metadata.types": ["raster", "image"]}}]}}})["hits"]["hits"]
    loose, sets = [], []
    for h in hits:
        src = h["_source"]
        size = (src.get("source_record") or {}).get("file_size") or 0
        if h["_id"].endswith(":images"):
            sets.append(src)
        elif size <= MAX_DOWNLOAD:
            loose.append({"kind": "loose", "dataset": h["_id"], "record": src["source_record"].get("title", ""),
                          "file": src["source_record"].get("file_key", ""), "url": src["download_url"]})
    # The sets' zips: re-read from discovery (merged sets keep a zip list).
    from crawlers.invenio.crawler import expand_archive, file_entries, merge_extras
    from crawlers.invenio.http import ssl_context
    import httpx

    per_set = max(1, (total - len(loose)) // max(1, len({s["source_record"]["record_id"] for s in sets})))
    out = list(loose)
    rng = random.Random(20261110)
    for rid, src in {s["source_record"]["record_id"]: s for s in sets}.items():  # a record's sets, once
        record = httpx.get(f"https://{UV}/api/records/{rid}", verify=ssl_context(), timeout=60).json()
        extras = []
        for e in file_entries(record, UV, 200 << 20):
            if e["file_extension"] == "zip":
                expand_archive(e, None, extras)
        jobs = [j for j in merge_extras([x for x in extras if x.get("image_set")])]
        members = []
        for job in jobs:
            zips = job["image_set"].get("zips") or [{"url": job["file_url"], "size": job["file_size"]}]
            for z in zips:
                with open_remote_zip(z["url"], z["size"]) as zf:
                    members += [(z, i.filename) for i in zf.infolist() if member_extension(i.filename) in TIFF_EXTENSIONS
                                and not i.filename.rsplit("/", 1)[-1].startswith(".")
                                and i.file_size <= MAX_DOWNLOAD]
        for z, name in rng.sample(members, min(per_set, len(members))):
            out.append({"kind": "set", "dataset": f"{UV}:{rid}:images", "record": src["source_record"].get("title", ""),
                        "file": name, "zip": z})
    return out[:total]


def fetch(item: dict[str, Any]) -> bytes:
    import httpx

    from crawlers.invenio.archives import open_remote_zip
    from crawlers.invenio.http import ssl_context

    if item["kind"] == "loose":
        return httpx.get(item["url"], verify=ssl_context(), timeout=600, follow_redirects=True).content
    with open_remote_zip(item["zip"]["url"], item["zip"]["size"]) as zf:
        return zf.read(item["file"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    items = image_sources(args.total)
    print(f"{len(items)} images: {Counter(i['dataset'].split(':')[1] for i in items)}")
    (OUT / "jpeg").mkdir(parents=True, exist_ok=True)

    def run(item: dict[str, Any]) -> dict[str, Any]:
        key = hashlib.sha256(f"{item['dataset']}|{item['file']}".encode()).hexdigest()[:16]
        jpeg_path = OUT / "jpeg" / f"{key}.jpg"
        try:
            if not jpeg_path.exists():
                jpeg_path.write_bytes(to_jpeg(fetch(item)))
            jpeg = jpeg_path.read_bytes()
            described = ask_image(DESCRIBER, DESCRIBE.format(record=item["record"], file=item["file"].rsplit("/", 1)[-1]), jpeg, key)
            d = described["parsed"]
            checked = ask_image(CHECKER, CHECK.format(description=d.get("description", ""), image_type=d.get("image_type", ""),
                                                      visible_text=d.get("visible_text", "")), jpeg, key)
            return {**{k: v for k, v in item.items() if k != "zip"}, "key": key, **d, "check": checked["parsed"],
                    "usage": {"describe": described["usage"], "check": checked["usage"]}}
        except Exception as exc:
            return {**{k: v for k, v in item.items() if k != "zip"}, "key": key, "error": repr(exc)[:300]}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, items))

    summaries = {}
    for dataset in sorted({r["dataset"] for r in rows if r["kind"] == "set" and not r.get("error")}):
        mine = [r for r in rows if r["dataset"] == dataset and not r.get("error")]
        text = "\n".join(f"- ({r.get('image_type')}) {r.get('description')}" for r in mine)
        summaries[dataset] = ask_text(DESCRIBER, SUMMARIZE.format(n=len(mine), total="many", record=mine[0]["record"],
                                                                   descriptions=text))["parsed"].get("summary")

    ok = [r for r in rows if not r.get("error")]
    tokens = Counter()
    for r in ok:
        for step in ("describe", "check"):
            tokens[f"{step}_in"] += (r["usage"][step] or {}).get("prompt") or 0
            tokens[f"{step}_out"] += (r["usage"][step] or {}).get("completion") or 0
    cost = (tokens["describe_in"] * 0.30 + tokens["describe_out"] * 2.50 + tokens["check_in"] * 2.0
            + tokens["check_out"] * 10.0) / 1e6
    summary = {"images": len(rows), "errors": sum(1 for r in rows if r.get("error")),
               "checker": Counter((r.get("check") or {}).get("verdict") for r in ok),
               "image_types": Counter(r.get("image_type") for r in ok), "tokens": tokens,
               "list_price_usd": round(cost, 2)}
    write_json(OUT / "results.json", {"summary": summary, "rows": rows, "set_summaries": summaries})
    print(json.dumps(summary, indent=1, default=dict))


if __name__ == "__main__":
    main()
