"""HTTP settings shared by the InvenioRDM crawler and transformer."""

from __future__ import annotations

import ssl
from pathlib import Path

import certifi
import httpx

# Zenodo answers plain application/json with its legacy (pre-2023) serialization,
# which is what broke the Auctus v1 Zenodo discoverer. This media type returns the
# standard InvenioRDM shape on every instance, so one parser covers all of them.
INVENIORDM_JSON = "application/vnd.inveniordm.v1+json"
USER_AGENT = "auctus-v2 (VIDA NYU)"

# Some instances (NYU UltraViolet) serve only their leaf certificate. Browsers fill
# in the missing intermediate themselves; certifi-based clients do not.
EXTRA_INTERMEDIATES = Path(__file__).with_name("extra_intermediates.pem")


def ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context(cafile=certifi.where())
    if EXTRA_INTERMEDIATES.exists():
        context.load_verify_locations(cafile=str(EXTRA_INTERMEDIATES))
    return context


def async_client(timeout_seconds: float = 30.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds, connect=15.0),
        follow_redirects=True,
        verify=ssl_context(),
        headers={"Accept": INVENIORDM_JSON, "User-Agent": USER_AGENT},
    )
