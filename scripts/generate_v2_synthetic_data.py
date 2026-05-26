#!/usr/bin/env python3
"""Generate v2 synthetic datasets from the existing baseline mock file.

Writes `backend/data/synthetic_datasets.json` containing an array
of 10 dataset objects using the modernized v2 layout with an embedded
`profiler_metadata` payload modeled on the atlas-profiler reference.
"""
import json
import os
import random
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
IN_PATH = os.path.join(BASE_DIR, "data", "synthetic_datasets.json")
OUT_PATH = os.path.join(BASE_DIR, "data", "synthetic_datasets.json")


def read_baseline():
    with open(IN_PATH, "r") as fh:
        return json.load(fh)


def envelope_from_bbox(bbox):
    # input bbox is [min_lat, min_lon, max_lat, max_lon]
    min_lat, min_lon, max_lat, max_lon = bbox
    return {
        "type": "envelope",
        "coordinates": [[min_lon, max_lat], [max_lon, min_lat]],
    }


def pick_columns_for_dataset(ds):
    """Return a list of (name, kind) pairs where kind is 'numerical','categorical','spatial','temporal' or 'text'.
    Use the dataset id or title to choose a reasonable set of columns.
    """
    mapping = {
        "ds-001": [
            ("trip_id", "numerical"),
            ("pickup_datetime", "temporal"),
            ("dropoff_datetime", "temporal"),
            ("pickup_lon", "numerical"),
            ("pickup_lat", "numerical"),
            ("fare_amount", "numerical"),
            ("passenger_count", "numerical"),
            ("payment_type", "categorical"),
        ],
        "ds-002": [
            ("year_month", "temporal"),
            ("anomaly_celsius", "numerical"),
            ("reference_baseline", "text"),
        ],
        "ds-003": [
            ("tree_id", "numerical"),
            ("species_common", "text"),
            ("health", "categorical"),
            ("wkt_geom", "spatial"),
        ],
        "ds-004": [
            ("period", "temporal"),
            ("cpi_all_items", "numerical"),
            ("area_name", "text"),
        ],
        "ds-005": [
            ("country", "categorical"),
            ("year", "temporal"),
            ("pop_growth_pct", "numerical"),
        ],
        "ds-006": [
            ("fire_id", "numerical"),
            ("start_date", "temporal"),
            ("end_date", "temporal"),
            ("perimeter_wkt", "spatial"),
        ],
        "ds-007": [
            ("listing_id", "numerical"),
            ("room_type", "categorical"),
            ("price_usd", "numerical"),
            ("location", "spatial"),
        ],
        "ds-008": [
            ("date", "temporal"),
            ("eur_usd", "numerical"),
        ],
        "ds-009": [
            ("counter_id", "numerical"),
            ("timestamp", "temporal"),
            ("bicycle_count", "numerical"),
            ("point_wkt", "spatial"),
        ],
        "ds-010": [
            ("country", "categorical"),
            ("date", "temporal"),
            ("people_vaccinated", "numerical"),
            ("daily_vaccinations", "numerical"),
        ],
    }
    return mapping.get(ds.get("id"), [("col1", "text"), ("col2", "numerical")])


def make_column_entry(name, kind):
    entry = {"name": name}
    if kind == "numerical":
        entry.update(
            {
                "structural_type": "http://schema.org/Number",
                "semantic_types": [],
                "unclean_values_ratio": round(random.uniform(0.0, 0.05), 3),
                "num_distinct_values": random.randint(10, 500),
                "mean": round(random.uniform(10.0, 5000.0), 2),
                "stddev": round(random.uniform(1.0, 2000.0), 2),
                "coverage": [{"range": {"gte": 0.0, "lte": round(random.uniform(100.0, 10000.0), 1)}}],
                "plot": {"type": "histogram_numerical", "data": [{"count": random.randint(1, 20), "bin_start": 0.0, "bin_end": 100.0}]},
            }
        )
    elif kind == "categorical":
        entry.update(
            {
                "structural_type": "http://schema.org/Text",
                "semantic_types": [],
                "num_distinct_values": random.randint(2, 30),
                "plot": {"type": "histogram_categorical", "data": [{"bin": "A", "count": random.randint(1, 50)}]},
            }
        )
    elif kind == "spatial":
        entry.update(
            {
                "structural_type": "http://schema.org/GeoCoordinates",
                "semantic_types": [],
                "geo_classifier": {"label": "point", "confidence": round(random.uniform(0.8, 0.99), 4), "source": "ml+validated"},
            }
        )
    elif kind == "temporal":
        entry.update({"structural_type": "http://schema.org/Date", "semantic_types": []})
    else:
        entry.update({"structural_type": "http://schema.org/Text", "semantic_types": []})
    return entry


def make_profiler_metadata(ds, columns):
    nb_rows = random.choice([100, 500, 1000, 5000])
    nb_profiled_rows = min(nb_rows, 100)
    cols = [make_column_entry(n, k) for n, k in columns]
    attribute_keywords = [c[0] for c in columns]
    # craft a short sample string consistent with header
    header = ",".join(attribute_keywords)
    sample_row_values = []
    for n, k in columns:
        if k == "numerical":
            sample_row_values.append(str(random.randint(1, 10000)))
        elif k == "temporal":
            sample_row_values.append(ds.get("temporal_coverage", {}).get("start", "2020-01-01"))
        elif k == "spatial":
            # embed a mock WKT point using bbox center
            bbox = ds.get("spatial_coverage", {}).get("bbox", [0, 0, 0, 0])
            if bbox and len(bbox) == 4:
                min_lat, min_lon, max_lat, max_lon = bbox
                lat = round((min_lat + max_lat) / 2, 5)
                lon = round((min_lon + max_lon) / 2, 5)
                sample_row_values.append(f"POINT ({lon} {lat})")
            else:
                sample_row_values.append("POINT (0 0)")
        else:
            # text or categorical
            if "country" in n.lower():
                sample_row_values.append(ds.get("spatial_coverage", {}).get("label", "Global"))
            elif "boro" in n.lower() or "borough" in n.lower():
                sample_row_values.append(ds.get("spatial_coverage", {}).get("label", "Unknown"))
            else:
                sample_row_values.append("sample_text")

    sample = header + "\r\n" + ",".join(sample_row_values)

    profiler = {
        "nb_rows": nb_rows,
        "nb_profiled_rows": nb_profiled_rows,
        "nb_columns": len(columns),
        "columns": cols,
        "nb_spatial_columns": sum(1 for _, k in columns if k == "spatial"),
        "nb_categorical_columns": sum(1 for _, k in columns if k == "categorical"),
        "nb_numerical_columns": sum(1 for _, k in columns if k == "numerical"),
        "types": list({k for _, k in columns}),
        "attribute_keywords": attribute_keywords,
        "sample": sample,
        "_profiling_times": {"steps": {"1_load_data": round(random.uniform(0.001, 0.01), 4), "2_geo_batch_predict": round(random.uniform(0.01, 0.5), 4)}, "total": round(random.uniform(0.5, 3.0), 3)},
    }
    return profiler


def make_download_url(ds):
    src = ds.get("source", "example").lower()
    if "socrata" in src or "nyc" in ds.get("spatial_coverage", {}).get("label", "").lower():
        return f"https://data.cityofnewyork.us/api/views/{ds['id']}/rows.csv"
    if "zenodo" in src:
        return f"https://zenodo.org/records/{ds['id']}/files/data.csv"
    if "world bank" in src.lower():
        return f"https://api.worldbank.org/v2/en/country/all/indicator/{ds['id']}.csv"
    if "our world" in src.lower() or "our world in data" in src.lower():
        return f"https://ourworldindata.org/grapher/{ds['id']}.csv"
    # fallback
    return f"https://example.org/datasets/{ds['id']}/download.csv"


def transform():
    baseline = read_baseline()
    out = []
    for ds in baseline:
        columns = pick_columns_for_dataset(ds)
        profiler = make_profiler_metadata(ds, columns)
        bbox = ds.get("spatial_coverage", {}).get("bbox", [0, 0, 0, 0])
        spatial = {
            "label": ds.get("spatial_coverage", {}).get("label", ""),
            "bbox": envelope_from_bbox(bbox),
        }

        v2 = {
            "id": ds.get("id"),
            "title": ds.get("title"),
            "description": ds.get("description"),
            "source": ds.get("source"),
            "download_url": make_download_url(ds),
            "types": ds.get("types", []),
            "temporal_coverage": ds.get("temporal_coverage", {}),
            "spatial_coverage": spatial,
            "profiler_metadata": profiler,
        }
        out.append(v2)

    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"Wrote {len(out)} records to {OUT_PATH}")


if __name__ == "__main__":
    transform()
