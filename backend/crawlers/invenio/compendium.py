"""Research compendia: one entry per repository record, with its parts and how they relate.

A record is the compendium. Each part is either a dataset Auctus indexed (a table, a map
layer, an image set, model output, code, spectra, ...) or a file it doesn't profile (a
PDF report, a README, a 71 GB tar). Relations come from what Auctus already knows:

  partitions      a table split over many files (member groups)
  join keys       key-like columns shared by tables of the record (tract GEOID, ZIP, id)
  code -> data    the data files each sampled script reads (code collections)
  docs -> tables  codebooks whose value labels were applied to a table's columns
  related records InvenioRDM related_identifiers pointing at other records of the
                  instance ("continues", "isSupplementTo"), and linked publications
  versions        records sharing an InvenioRDM parent

Built after ingestion into its own index (search over files is unchanged):

  python -m crawlers.invenio.compendium ultraviolet.library.nyu.edu

Juliana's four kinds of compendium (2026-10-31 comments) map to: partitions, join keys,
versions, and related records / the record itself.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from typing import Any

COMPENDIUM_INDEX = "auctus_compendia"
COMPENDIUM_MAPPING = {
    "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
    "mappings": {"properties": {
        "record_id": {"type": "keyword"}, "domain": {"type": "keyword"},
        "title": {"type": "text"}, "description": {"type": "text"}, "summary": {"type": "text"},
        "subjects": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "landing_url": {"type": "keyword", "index": False}, "parent_id": {"type": "keyword"},
        "role_counts": {"type": "object"}, "parts": {"type": "object", "enabled": False},
        "relations": {"type": "object", "enabled": False}}},
}
DOC_FILE = re.compile(r"read[_ -]?me|codebook|code[_ -]?book|dictionar|variables?|documentation|report|manual", re.I)
CODE_EXT = {"py", "ipynb", "r", "rmd", "m", "jl", "do", "sas", "sps"}
IMAGE_EXT = {"tif", "tiff", "jpg", "jpeg", "png"}
KEY_LABEL = re.compile(r"\b(id|identifier|code|key|zip|postal|fips|geoid|tract|county|iso|serial)\b", re.I)
KEY_NAME = re.compile(r"(^|_)(id|geoid|fips|zip|zipcode|zcta|tract|iso\d?|code|key)($|_)", re.I)
# Row counters and header-less columns, not keys between tables.
NOT_KEYS = re.compile(r"^(objectid\d*|fid|oid|rowid|index|unnamed.*|id)$", re.I)
MAX_TABLES_FOR_KEYS = 300


def plural(role: str, n: int) -> str:
    if n == 1 or role in ("documentation", "code", "model output", "spectra", "point clouds", "3D meshes",
                          "not profiled", "other"):
        return role
    return {"copy of an indexed layer": "copies of indexed layers", "text corpus": "text corpora"}.get(role, role + "s")


# InvenioRDM relation ids ("issupplementto") as words.
RELATION_WORDS = {"issupplementto": "is a supplement to", "issupplementedby": "is supplemented by",
                  "continues": "continues", "iscontinuedby": "is continued by", "isidenticalto": "is identical to",
                  "references": "references", "isreferencedby": "is referenced by", "cites": "cites",
                  "iscitedby": "is cited by", "isnewversionof": "is a new version of",
                  "ispreviousversionof": "is a previous version of", "ispartof": "is part of", "haspart": "has part",
                  "isdescribedby": "is described by", "describes": "describes", "documents": "documents",
                  "isdocumentedby": "is documented by", "isderivedfrom": "is derived from", "issourceof": "is source of"}


def relation_words(kind: str | None) -> str:
    return RELATION_WORDS.get(str(kind or "").lower(), str(kind or "related to"))


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def base(path: str) -> str:
    return str(path).rsplit("/", 1)[-1].lower()


def role_of_doc(doc: dict[str, Any]) -> str:
    pm = doc.get("profiler_metadata") or {}
    types = pm.get("types") or doc.get("types") or []
    kind = (pm.get("collection") or {}).get("kind")
    if kind:
        return {"texts": "text corpus", "model_output": "model output", "point_clouds": "point clouds",
                "meshes": "3D meshes", "code": "code"}.get(kind, kind)
    if pm.get("spectra"):
        return "spectra"
    if "raster" in types:
        return "raster map" if not doc["id"].endswith(":images") else "image set (georeferenced)"
    if "image" in types:
        return "image set" if doc["id"].endswith(":images") else "image"
    if not pm or (doc.get("errors") and not pm.get("columns")):
        return "not profiled"
    if pm.get("geometry_type"):
        return "map layer"
    if doc.get("member_group") or (doc.get("source_record") or {}).get("member_group"):
        return "split table"
    return "table"


def role_of_file(key: str) -> str:
    ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
    if ext in CODE_EXT:
        return "code"
    if ext in IMAGE_EXT:
        return "image"
    if ext in ("pdf", "md", "txt", "docx", "doc", "html", "rst") or DOC_FILE.search(key):
        return "documentation"
    if ext in ("zip", "tar", "gz", "7z") or re.fullmatch(r"z\d\d", ext):
        return "archive"
    return "other"


def join_keys(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Key-like columns that two or more of the record's tables share (by normalized name)."""
    holders: dict[str, dict[str, Any]] = {}
    for doc in tables[:MAX_TABLES_FOR_KEYS]:
        for col in (doc.get("profiler_metadata") or {}).get("columns") or []:
            name = str(col.get("name", ""))
            label = str(col.get("semantic_type_label") or "")
            semantic = " ".join(col.get("semantic_types") or [])
            if NOT_KEYS.match(norm(name)) or not (KEY_NAME.search(name) or KEY_LABEL.search(label)
                                                  or "identifier" in semantic.lower()):
                continue
            entry = holders.setdefault(norm(name), {"column": name, "label": label, "datasets": []})
            if doc["id"] not in entry["datasets"]:
                entry["datasets"].append(doc["id"])
    keys = [k for k in holders.values() if len(k["datasets"]) >= 2]
    return sorted(keys, key=lambda k: -len(k["datasets"]))[:25]


def code_links(code_rows: dict[str, list[dict[str, str]]], parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sampled scripts' data_files_read matched to the record's parts by file name."""
    by_name: dict[str, list[str]] = defaultdict(list)
    for part in parts:
        for name in {base(part["file_key"]), base(part["file_key"].split(" (", 1)[0])}:
            by_name[name].append(part.get("dataset_id") or part["file_key"])
    links = []
    for dataset_id, rows in code_rows.items():
        for row in rows:
            for data_file in filter(None, (f.strip() for f in str(row.get("data_files_read") or "").split(","))):
                links.append({"code": row.get("file"), "code_dataset": dataset_id, "reads": data_file,
                              "part": (by_name.get(data_file.lower()) or [None])[0]})
    return links


def build_compendium(record: dict[str, Any], docs: list[dict[str, Any]], domain: str,
                     code_rows: dict[str, list[dict[str, str]]] | None = None,
                     instance_records: dict[str, str] | set[str] | None = None) -> dict[str, Any]:
    """`instance_records`: the instance's indexed record ids (a dict maps them to their titles)."""
    titles = instance_records if isinstance(instance_records, dict) else {}
    rid = record["id"]
    metadata = record.get("metadata") or {}
    indexed_keys = set()
    parts = []
    for doc in sorted(docs, key=lambda d: d["id"]):
        src = doc.get("source_record") or {}
        pm = doc.get("profiler_metadata") or {}
        key = src.get("file_key") or doc["id"]
        indexed_keys.add(key.split("/", 1)[0].split(" (", 1)[0])
        # Merged sets / collections name their zips in `folders`; spectra list their files.
        for folders in ((pm.get("collection") or {}).get("folders") or {}, (pm.get("raster") or {}).get("folders") or {}):
            indexed_keys.update(k.split("/", 1)[0] for k in folders)
        indexed_keys.update(s.get("file") for s in pm.get("spectra") or [] if s.get("file"))
        # A split table's partitions are keyed by zip (or folder) name, without ".zip".
        group = src.get("member_group") or {}
        for partition in group.get("partitions") or []:
            if partition.get("key"):
                indexed_keys.update({partition["key"], partition["key"] + ".zip"})
        parts.append({"dataset_id": doc["id"], "file_key": key, "role": role_of_doc(doc),
                      "rows": pm.get("nb_rows"), "columns": pm.get("nb_columns"),
                      "partitions": len(((src.get("member_group") or {}).get("partitions")) or []) or None})
    for key, entry in ((record.get("files") or {}).get("entries") or {}).items():
        if key not in indexed_keys and not any(k.startswith(key) for k in indexed_keys):
            # The Spatial Data Repository's State Plane copy of an indexed WGS84 layer.
            # ... or the original scan of an indexed georeferenced map.
            is_copy = any(k.replace("_WGS84_", "_") == key for k in indexed_keys if "_WGS84_" in k) \
                or key.replace("_original.", "_georeferenced.") in indexed_keys and "_original." in key
            parts.append({"dataset_id": None, "file_key": key, "role": "copy of an indexed layer" if is_copy else role_of_file(key),
                          "size": entry.get("size"),
                          "url": f"https://{domain}/api/records/{rid}/files/{key}/content"})

    tables = [d for d in docs if role_of_doc(d) in ("table", "split table", "map layer")]
    labelled = defaultdict(set)
    for doc in tables:
        for col in (doc.get("profiler_metadata") or {}).get("columns") or []:
            if col.get("value_labels_source"):
                labelled[col["value_labels_source"]].add(doc["id"])
    related, papers = [], []
    for rel in metadata.get("related_identifiers") or []:
        ident, kind = str(rel.get("identifier", "")), (rel.get("relation_type") or {}).get("id")
        target = ident.rsplit("/", 1)[-1] if rel.get("scheme") == "doi" and ident.startswith("10.58153/") else None
        if target:
            related.append({"record_id": target, "relation": kind, "relation_words": relation_words(kind),
                            "indexed": bool(instance_records and target in instance_records),
                            **({"title": titles[target]} if target in titles else {})})
        elif (rel.get("resource_type") or {}).get("id", "").startswith("publication"):
            papers.append({"identifier": ident, "scheme": rel.get("scheme"), "relation": kind,
                           "relation_words": relation_words(kind)})

    relations = {
        "partitions": [{"dataset_id": p["dataset_id"], "parts": p["partitions"]} for p in parts if p.get("partitions")],
        "join_keys": join_keys(tables),
        "code_reads": code_links(code_rows or {}, parts),
        "documentation": [{"doc": name, "labels_tables": sorted(ids)} for name, ids in labelled.items()],
        "related_records": related,
        "publications": papers,
    }
    roles = Counter(p["role"] for p in parts)
    summary_bits = [", ".join(f"{n} {plural(r, n)}" for r, n in roles.most_common())]
    if relations["join_keys"]:
        k = relations["join_keys"][0]
        summary_bits.append(f"tables join on {k['column']} ({len(k['datasets'])} tables)")
    reads = [c for c in relations["code_reads"] if c["part"]]
    if reads:
        summary_bits.append(f"code reads {len({c['part'] for c in reads})} of the record's datasets")
    if related:
        kinds = Counter(r["relation"] for r in related)
        summary_bits.append("; ".join(f"{relation_words(k)} {n} other record{'s' if n > 1 else ''}" for k, n in kinds.items()))
    if papers:
        summary_bits.append(f"{len(papers)} linked publication{'s' if len(papers) > 1 else ''}")
    return {
        "record_id": rid, "domain": domain, "title": metadata.get("title", rid),
        "description": re.sub(r"<[^>]+>", " ", metadata.get("description") or "")[:5000],
        "subjects": [s.get("subject") for s in metadata.get("subjects") or [] if s.get("subject")],
        "landing_url": f"https://{domain}/records/{rid}", "parent_id": (record.get("parent") or {}).get("id"),
        "role_counts": dict(roles), "parts": parts, "relations": relations, "summary": "; ".join(summary_bits),
    }


def build_all(domain: str) -> dict[str, Any]:
    import httpx

    from crawlers.invenio.http import ssl_context
    from storage.minio_client import get_heavy_profile_object, get_storage_client
    from storage.opensearch_client import AUCTUS_INDEX_NAME, get_client

    os_client, storage = get_client(), get_storage_client()
    if not os_client.indices.exists(index=COMPENDIUM_INDEX):
        os_client.indices.create(index=COMPENDIUM_INDEX, body=COMPENDIUM_MAPPING)
    docs_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    after = None
    while True:
        body = {"size": 500, "sort": [{"_id": "asc"}], "query": {"term": {"domain": domain}},
                "_source": ["title", "types", "errors", "source_record", "member_group", "archive_dataset_id",
                            "profiler_metadata.types", "profiler_metadata.nb_rows", "profiler_metadata.nb_columns",
                            "profiler_metadata.geometry_type", "profiler_metadata.collection",
                            "profiler_metadata.spectra.file", "profiler_metadata.columns",
                            "profiler_metadata.raster.folders"]}
        if after:
            body["search_after"] = after
        hits = os_client.search(index=AUCTUS_INDEX_NAME, body=body)["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            rid = (h["_source"].get("source_record") or {}).get("record_id")
            if rid:
                docs_by_record[rid].append({"id": h["_id"], **h["_source"]})
        after = hits[-1]["sort"]
    records = {rid: ((docs[0].get("source_record") or {}).get("title") or rid) for rid, docs in docs_by_record.items()}
    built, parents = 0, defaultdict(list)
    with httpx.Client(verify=ssl_context(), timeout=60, follow_redirects=True) as client:
        compendia = []
        for rid, docs in sorted(docs_by_record.items()):
            record = client.get(f"https://{domain}/api/records/{rid}").json()
            code_rows = {}
            for doc in docs:
                if ((doc.get("profiler_metadata") or {}).get("collection") or {}).get("kind") == "code":
                    try:
                        obj, _ = get_heavy_profile_object(storage, doc["id"])
                        sample = json.loads(obj.read()).get("sample") or ""
                        obj.close(); obj.release_conn()
                        code_rows[doc["id"]] = list(csv.DictReader(io.StringIO(sample)))
                    except Exception:
                        pass
            compendium = build_compendium(record, docs, domain, code_rows, records)
            parents[compendium["parent_id"]].append(rid)
            compendia.append(compendium)
        for compendium in compendia:
            others = [r for r in parents.get(compendium["parent_id"], []) if r != compendium["record_id"]]
            compendium["relations"]["versions"] = others
            os_client.index(index=COMPENDIUM_INDEX, id=f"{domain}:{compendium['record_id']}", body=compendium)
            built += 1
    os_client.indices.refresh(index=COMPENDIUM_INDEX)
    return {"domain": domain, "compendia": built}


if __name__ == "__main__":
    print(build_all(sys.argv[1] if len(sys.argv) > 1 else "ultraviolet.library.nyu.edu"))
