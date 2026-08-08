"""Regression tests for what build_profile_text() is allowed to put in a profile.

Guards the rule that every value in the profile is derived from the dataset's own
data, so an absent field reads as "the data has none" rather than as a defaulted
placeholder. Two defects motivated it:

  * ``spatial_coverage`` was copied from the record's top-level block, which
    ``transformer.build_validation_record`` writes for *every* dataset with a
    portal-level fallback bbox. A dataset whose own ``types`` is ["categorical"]
    was handed a profile asserting it covers New York City.
  * ``temporal_coverage`` never reached the profile at all, so the spatial extent
    went in and the temporal extent did not.

Also covers ``_merge_profiler_output`` leaving the profiler's nb_*_columns counts
alone; it used to recompute them against the wrong fields.

Run:  python -m storage.test_profile_content
"""

from __future__ import annotations

import json
from pathlib import Path

from crawlers.socrata.transformer import SampleStats, _merge_profiler_output
from run_pipeline_ingest import DEFAULT_FALLBACK_BBOX, isolate_search_payload
from storage.arq_worker import build_profile_text

# The shape transformer.build_validation_record always emits, fallback or not.
FALLBACK_SPATIAL_BLOCK = {
    "label": "data.cityofnewyork.us",
    "bbox": {"type": "envelope", "coordinates": [[-74.259, 40.917], [-73.7, 40.477]]},
}


def _range(gte: float, lte: float) -> dict:
    return {"range": {"gte": gte, "lte": lte}}


def _non_spatial_record() -> dict:
    """A purely categorical dataset that still carries the fallback bbox."""
    return {
        "spatial_coverage": FALLBACK_SPATIAL_BLOCK,
        "profiler_metadata": {
            "nb_rows": 1000,
            "nb_columns": 2,
            "types": ["categorical"],
            "columns": [
                {"name": "Title", "structural_type": "http://schema.org/Text"},
                {
                    "name": "Agency",
                    "structural_type": "http://schema.org/Text",
                    "semantic_types": ["http://schema.org/Enumeration"],
                },
            ],
        },
    }


def test_spatial_coverage_omitted_when_profiler_found_no_coordinates() -> None:
    profile = json.loads(build_profile_text(_non_spatial_record()))
    assert "spatial_coverage" not in profile, profile
    # The rest of the profile must survive; this is an omission, not a bail-out.
    assert profile["nb_rows"] == 1000 and len(profile["columns"]) == 2


def test_spatial_coverage_kept_when_the_data_has_coordinates() -> None:
    record = _non_spatial_record()
    record["profiler_metadata"]["spatial_coverage"] = [
        {"type": "latlong", "column_names": ["Latitude", "Longitude"]}
    ]
    profile = json.loads(build_profile_text(record))
    assert profile["spatial_coverage"]["bbox"] == FALLBACK_SPATIAL_BLOCK["bbox"]
    # The portal label is identical for every dataset, so it says nothing here.
    assert "label" not in profile["spatial_coverage"], profile["spatial_coverage"]


def test_temporal_interval_present_when_the_profiler_parsed_dates() -> None:
    record = _non_spatial_record()
    record["profiler_metadata"]["temporal_coverage"] = [
        {"type": "datetime", "column_names": ["Created Date"],
         "ranges": [_range(1420070400.0, 1451606400.0)]}  # 2015-01-01 .. 2016-01-01
    ]
    profile = json.loads(build_profile_text(record))
    assert profile["temporal_coverage"] == {"start": "2015-01-01", "end": "2016-01-01"}


def test_temporal_interval_omitted_when_there_are_no_datetime_columns() -> None:
    profile = json.loads(build_profile_text(_non_spatial_record()))
    assert "temporal_coverage" not in profile, profile


def test_several_datetime_columns_collapse_to_the_outer_envelope() -> None:
    record = _non_spatial_record()
    record["profiler_metadata"]["temporal_coverage"] = [
        {"type": "datetime", "column_names": ["Closed Date"],
         "ranges": [_range(1451606400.0, 1483228800.0)]},   # 2016-01-01 .. 2017-01-01
        {"type": "datetime", "column_names": ["Created Date"],
         "ranges": [_range(1420070400.0, 1435708800.0),     # 2015-01-01 .. 2015-07-01
                    _range(1443657600.0, 1446336000.0)]},   # 2015-10-01 .. 2015-11-01
    ]
    profile = json.loads(build_profile_text(record))
    assert profile["temporal_coverage"] == {"start": "2015-01-01", "end": "2017-01-01"}


def test_both_range_shapes_are_accepted() -> None:
    """The installed profiler wraps its bounds; ingested corpus data was recorded
    unwrapped. Reading only one shape would silently drop every date range."""
    record = _non_spatial_record()
    for ranges in ([{"range": {"gte": 1420070400.0, "lte": 1451606400.0}}],
                   [{"gte": 1420070400.0, "lte": 1451606400.0}]):
        record["profiler_metadata"]["temporal_coverage"] = [
            {"type": "datetime", "column_names": ["Created Date"], "ranges": ranges}
        ]
        profile = json.loads(build_profile_text(record))
        assert profile["temporal_coverage"] == {"start": "2015-01-01", "end": "2016-01-01"}, ranges


def test_malformed_temporal_coverage_is_ignored_not_crashed_on() -> None:
    record = _non_spatial_record()
    for junk in ({"not": "a list"}, [], [{"ranges": []}], [{"ranges": [{"range": {}}]}]):
        record["profiler_metadata"]["temporal_coverage"] = junk
        profile = json.loads(build_profile_text(record))
        assert "temporal_coverage" not in profile, (junk, profile)


def test_gate_fields_survive_into_the_indexed_document() -> None:
    """Not every caller passes a full crawler record.

    ``eval/backfill_static_arms.py`` and ``eval/generate_queries.py`` both call
    build_profile_text with an **OpenSearch document**. The new gates key on
    ``profiler_metadata.spatial_coverage`` / ``.temporal_coverage``, which are bulky
    profiler fields (geohash grids, per-column ranges) and exactly the kind of thing
    an indexing trim removes. If ``isolate_search_payload`` ever starts stripping
    them, the gate would silently blank spatial coverage for genuinely spatial
    datasets and temporal would never fire — the inverse of the intended behaviour,
    and invisible to every other test here, which builds records by hand.
    """
    record = json.loads(
        (Path(__file__).resolve().parents[1] / "catalog_record.json").read_text()
    )
    record["profiler_metadata"]["temporal_coverage"] = [
        {"type": "datetime", "column_names": ["Created Date"],
         "ranges": [_range(1420070400.0, 1451606400.0)]}
    ]

    indexed = isolate_search_payload(record)
    profile = json.loads(build_profile_text(indexed))

    assert "spatial_coverage" in profile, profile
    assert profile["temporal_coverage"] == {"start": "2015-01-01", "end": "2016-01-01"}
    # And it is the dataset's own bbox, not the portal fallback.
    assert profile["spatial_coverage"]["bbox"] == record["spatial_coverage"]["bbox"]


def test_merge_leaves_the_profilers_column_counts_alone() -> None:
    """The committed sample record is the case the old predicate got wrong.

    Its three spatial columns (Latitude, Longitude, Location) were counted as one,
    because the deleted code looked for GeoCoordinates in structural_type only and
    for AdministrativeArea in semantic_types only.
    """
    record = json.loads(
        (Path(__file__).resolve().parents[1] / "catalog_record.json").read_text()
    )
    profiler_output = dict(record["profiler_metadata"])
    profiler_output["nb_spatial_columns"] = 3  # what determine_dataset_type gives

    merged = _merge_profiler_output(
        profiler_output,
        SampleStats(sample_path=Path("/tmp/x.csv"), bytes_written=0,
                    newline_count=0, truncated_by_rows=False, truncated_by_bytes=False),
        None,
        DEFAULT_FALLBACK_BBOX,
    )
    assert merged["nb_spatial_columns"] == 3, merged["nb_spatial_columns"]
    assert merged["nb_profiled_rows"] == profiler_output["nb_profiled_rows"]


if __name__ == "__main__":
    test_spatial_coverage_omitted_when_profiler_found_no_coordinates()
    test_spatial_coverage_kept_when_the_data_has_coordinates()
    test_temporal_interval_present_when_the_profiler_parsed_dates()
    test_temporal_interval_omitted_when_there_are_no_datetime_columns()
    test_several_datetime_columns_collapse_to_the_outer_envelope()
    test_both_range_shapes_are_accepted()
    test_malformed_temporal_coverage_is_ignored_not_crashed_on()
    test_gate_fields_survive_into_the_indexed_document()
    test_merge_leaves_the_profilers_column_counts_alone()
    print("OK: the profile only claims what the dataset's own data supports")
