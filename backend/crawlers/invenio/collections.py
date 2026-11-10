"""Collections of non-tabular files inside zips: OCR texts, model output, point clouds,
3D meshes, code.

Repositories ship these by the thousand (4,596 OCR pages of a newspaper, 4,047 MITgcm
output fields, 570 LiDAR tiles). Each zip's members of one kind become one dataset,
like a tile set: the member names give the totals and the structure (folders,
variables, time steps), and a spread-out sample of members is read for what's
inside. The sample becomes a table, one row per sampled file, so the dataset is
profiled, described (AutoDDG) and typed like any other table.

Images (TIFF / JPEG) have their own path (crawler.image_set_job) since they came first.
"""

from __future__ import annotations

import io
import json
import re
import struct
from typing import Any, Callable

KINDS: dict[str, set[str]] = {
    "texts": {"txt"},
    "model_output": {"meta"},  # MITgcm: a .meta text header per binary .data field
    "point_clouds": {"las", "laz"},
    "meshes": {"ply", "obj"},
    "code": {"py", "ipynb", "r", "rmd", "m", "jl"},
}
# A few .txt files in a zip are READMEs or notes; a text collection is a corpus.
MIN_MEMBERS = {"texts": 20, "model_output": 5, "point_clouds": 1, "meshes": 1, "code": 3}
SAMPLES = {"texts": 20, "model_output": 20, "point_clouds": 12, "meshes": 12, "code": 40}
MAX_MEMBER_BYTES = {"texts": 2 << 20, "model_output": 1 << 16, "meshes": 0, "point_clouds": 0, "code": 1 << 20}
LABELS = {"texts": "text files", "model_output": "model output fields", "point_clouds": "LiDAR point clouds",
          "meshes": "3D meshes", "code": "code files"}
DATA_REF = re.compile(r"""['"]([^'"\n]{1,200}\.(?:csv|tsv|xlsx|xls|parquet|json|geojson|shp|dta|sav|rds|nc|h5|mat|txt|zip))['"]""",
                      re.IGNORECASE)


def kind_of(ext: str) -> str | None:
    return next((k for k, exts in KINDS.items() if ext in exts), None)


# --- texts (OCR pages, transcripts) -----------------------------------------------------

_GAELIC = re.compile("[ḃċḋḟġṁṗṡṫ]", re.IGNORECASE)
_ENGLISH = re.compile(r"\b(the|and|of|to|in|is|that|was|for)\b", re.IGNORECASE)


def read_text(fh, name: str) -> dict[str, Any]:
    text = fh.read(MAX_MEMBER_BYTES["texts"]).decode("utf-8-sig", errors="replace")
    words = re.findall(r"\w+", text)
    english = len(_ENGLISH.findall(text)) / max(1, len(words))
    script = "Gaelic script" if _GAELIC.search(text) else ""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20]
    return {"file": name, "characters": len(text), "words": len(words), "lines": text.count("\n") + 1,
            "share_english_function_words": round(english, 3), "script": script,
            "excerpt": " ".join(lines[:3])[:300]}


# --- MITgcm model output -------------------------------------------------------------------

_META_FIELD = re.compile(r"(\w+)\s*=\s*\[(.*?)\];", re.DOTALL)
_MITGCM_NAME = re.compile(r"^(?P<var>.+?)\.(?P<step>\d{6,})\.meta$")


def read_mitgcm_meta(fh, name: str) -> dict[str, Any]:
    text = fh.read(MAX_MEMBER_BYTES["model_output"]).decode("latin-1")
    fields = {k: " ".join(v.replace("'", "").split()) for k, v in _META_FIELD.findall(text)}
    dims = [int(n) for n in re.findall(r"\d+", fields.get("dimList", ""))]
    base = name.rsplit("/", 1)[-1]
    match = _MITGCM_NAME.match(base)
    variable = fields.get("fldList") or (match.group("var") if match else base.rsplit(".", 1)[0])  # grid files: DRF.meta
    return {"file": name, "variable": variable.strip(),
            "time_step": int(fields["timeStepNumber"].split()[0]) if fields.get("timeStepNumber") else
            (int(match.group("step")) if match else None),
            "grid": " x ".join(str(n) for n in dims[0::3]) if dims else "",
            "dimensions": fields.get("nDims", ""), "precision": fields.get("dataprec", ""),
            "records": fields.get("nrecords", ""), "case": name.rsplit("/", 1)[0] if "/" in name else ""}


# --- LAS / LAZ point clouds --------------------------------------------------------------------


def read_las(fh, name: str) -> dict[str, Any]:
    """The public header block (and its GeoKey VLR for the CRS); LAZ shares the header."""
    head = fh.read(375)
    if head[:4] != b"LASF":
        raise ValueError("not a LAS file")
    major, minor = head[24], head[25]
    header_size, point_offset, n_vlrs = struct.unpack("<HII", head[94:104])
    points = struct.unpack("<I", head[107:111])[0]
    if (major, minor) >= (1, 4) and len(head) >= 255:
        points = struct.unpack("<Q", head[247:255])[0] or points
    max_x, min_x, max_y, min_y, max_z, min_z = struct.unpack("<6d", head[179:227])
    epsg = None
    vlrs = (head + fh.read(max(0, point_offset - len(head))))[header_size:point_offset]
    pos = 0
    for _ in range(n_vlrs):
        if pos + 54 > len(vlrs):
            break
        record_id, length = struct.unpack("<HH", vlrs[pos + 18:pos + 22])
        body = vlrs[pos + 54:pos + 54 + length]
        if record_id == 34735 and len(body) >= 8:  # GeoKeyDirectory, as in GeoTIFF
            from crawlers.invenio.formats import tiff_epsg
            epsg = tiff_epsg({"geokeys": list(struct.unpack(f"<{len(body) // 2}H", body[:len(body) // 2 * 2]))})
        pos += 54 + length
    return {"file": name, "points": points, "las_version": f"{major}.{minor}", "point_format": head[104],
            "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "min_z": min_z, "max_z": max_z,
            "epsg": epsg}


# --- PLY / OBJ meshes ---------------------------------------------------------------------------


def read_mesh(fh, name: str) -> dict[str, Any]:
    head = fh.read(4096).decode("latin-1")
    row: dict[str, Any] = {"file": name, "format": name.rsplit(".", 1)[-1].lower()}
    if head.startswith("ply"):
        for kind in ("vertex", "face"):
            m = re.search(rf"element {kind} (\d+)", head)
            row[f"{kind}_count"] = int(m.group(1)) if m else None
        m = re.search(r"format (\w+)", head)
        row["encoding"] = m.group(1) if m else ""
        row["properties"] = ", ".join(re.findall(r"property \S+(?: \S+)? (\w+)", head.split("end_header")[0]))[:200]
    return row


# --- code -----------------------------------------------------------------------------------------

_PY_IMPORT = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w., ]+))", re.MULTILINE)
_R_LIBRARY = re.compile(r"(?:library|require)\(\s*['\"]?([\w.]+)", re.MULTILINE)


def read_code(fh, name: str) -> dict[str, Any]:
    raw = fh.read(MAX_MEMBER_BYTES["code"]).decode("utf-8", errors="replace")
    ext = name.rsplit(".", 1)[-1].lower()
    text = raw
    if ext == "ipynb":
        try:
            cells = json.loads(raw).get("cells", [])
            text = "\n".join("".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code")
        except ValueError:
            pass
    if ext in ("py", "ipynb"):
        mods = set()
        for a, b in _PY_IMPORT.findall(text):
            for mod in (a or b).split(","):
                if mod.strip():
                    mods.add(mod.strip().split()[0].split(".")[0])
        language = "Python"
    elif ext in ("r", "rmd"):
        mods, language = set(_R_LIBRARY.findall(text)), "R"
    else:
        mods, language = set(), {"m": "MATLAB", "jl": "Julia"}.get(ext, ext)
    data = sorted({m.rsplit("/", 1)[-1] for m in DATA_REF.findall(text)})
    return {"file": name, "language": language, "lines": text.count("\n") + 1,
            "libraries": ", ".join(sorted(mods))[:300], "data_files_read": ", ".join(data)[:300]}


READERS: dict[str, Callable] = {"texts": read_text, "model_output": read_mitgcm_meta, "point_clouds": read_las,
                                "meshes": read_mesh, "code": read_code}


def summary_from_names(kind: str, names: list[str]) -> dict[str, Any]:
    """What the member names alone say: MITgcm variables and time steps, top folders."""
    out: dict[str, Any] = {}
    if kind == "model_output":
        variables: dict[str, int] = {}
        steps = []
        for name in names:
            m = _MITGCM_NAME.match(name.rsplit("/", 1)[-1])
            if m:
                variables[m.group("var")] = variables.get(m.group("var"), 0) + 1
                steps.append(int(m.group("step")))
        out["variables"] = dict(sorted(variables.items(), key=lambda kv: -kv[1])[:30])
        if steps:
            out["time_steps"] = {"first": min(steps), "last": max(steps), "distinct": len(set(steps))}
    return out
