"""Regression test for ingest-settings resolution (no pytest dependency).

Guards the defect where ``load_runtime_config()`` — which returns the whole
config dict — was unpacked into three names:

    active_domain, portal_cfg, pipeline_settings = load_runtime_config()

That binds the dict's KEYS (plain strings), so every ``portal_cfg.get(...)``
fallback raised ``AttributeError: 'str' object has no attribute 'get'``. It was
invisible during normal ingest because the dispatcher always populates
``base_url`` / ``fallback_bbox`` / ``spatial_label``, making the ``or``
fallbacks short-circuit; it only fired on a hand-enqueued partial payload.

Run:  python -m storage.test_resolve_ingest_settings
"""

from __future__ import annotations

from storage.arq_worker import DEFAULT_PORTAL_DOMAIN, resolve_ingest_settings


def test_minimal_payload_resolves_every_fallback() -> None:
    """The bug's trigger: only dataset_id, so every fallback must be exercised."""
    settings = resolve_ingest_settings({"dataset_id": "abcd-1234"})

    # Would have raised AttributeError before the fix.
    assert settings["base_url"].startswith("https://"), settings["base_url"]
    assert isinstance(settings["fallback_bbox"], list) and len(settings["fallback_bbox"]) == 4
    assert isinstance(settings["spatial_label"], str)
    assert isinstance(settings["max_sample_rows"], int)
    assert isinstance(settings["max_sample_bytes"], int)
    assert isinstance(settings["http_timeout_seconds"], float)
    assert settings["provider_type"] == "socrata"
    assert settings["socrata_updated_at"] is None
    # With no domain in the payload and no active_portal in config, the
    # base_url must fall back to the default portal domain.
    assert DEFAULT_PORTAL_DOMAIN in settings["base_url"], settings["base_url"]
    assert settings["domain_url"] and "://" not in settings["domain_url"]


def test_payload_values_win_over_config() -> None:
    """A fully-populated payload (the dispatcher's normal case) is respected."""
    settings = resolve_ingest_settings({
        "dataset_id": "abcd-1234",
        "base_url": "https://data.cityofchicago.org",
        "domain": "data.cityofchicago.org",
        "fallback_bbox": [-1.0, -2.0, 3.0, 4.0],
        "spatial_label": "Chicago",
        "max_sample_rows": 7,
        "max_sample_bytes": 999,
        "http_timeout_seconds": 1.5,
        "provider": "ckan",
        "socrata_updated_at": "2026-01-01T00:00:00Z",
    })

    assert settings["base_url"] == "https://data.cityofchicago.org"
    assert settings["domain_url"] == "data.cityofchicago.org"
    assert settings["fallback_bbox"] == [-1.0, -2.0, 3.0, 4.0]
    assert settings["spatial_label"] == "Chicago"
    assert settings["max_sample_rows"] == 7
    assert settings["max_sample_bytes"] == 999
    assert settings["http_timeout_seconds"] == 1.5
    assert settings["provider_type"] == "ckan"
    assert settings["socrata_updated_at"] == "2026-01-01T00:00:00Z"


def test_domain_only_payload_derives_base_url() -> None:
    """A payload naming only the domain still yields a usable base_url."""
    settings = resolve_ingest_settings({
        "dataset_id": "abcd-1234",
        "domain": "data.cityofchicago.org",
    })
    assert "data.cityofchicago.org" in settings["base_url"]
    assert settings["domain_url"] == "data.cityofchicago.org"


if __name__ == "__main__":
    test_minimal_payload_resolves_every_fallback()
    test_payload_values_win_over_config()
    test_domain_only_payload_derives_base_url()
    print("OK: resolve_ingest_settings falls back without AttributeError")
