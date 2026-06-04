"""CKAN crawler placeholder.

This module exposes the same async discovery interface as other providers,
while intentionally returning no datasets until CKAN crawling is implemented.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


async def discover_ckan_datasets(
    domain: str | None = None,
    limit: int = 10,
    offset: int = 0,
) -> list[str]:
    """Placeholder CKAN discovery function.

    Parameters are accepted to mirror provider crawler signatures so the
    dispatcher can plug this engine in without future interface changes.
    """
    _ = domain, limit, offset
    LOGGER.warning("CKAN crawler is not implemented yet. Returning no datasets.")
    return []
