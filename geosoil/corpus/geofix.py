"""Coordinate corrections for sources whose published coordinates are known to be wrong.

Flanders (Aardewerk, 1947-74 survey): the authoritative service (DOV, Flemish government) publishes Belgian
Lambert 72 coordinates in EPSG:31370. WoSIS (dataset BE-VASPDB) and the OpenLandMap compendium ("Vlaanderen")
converted the same X/Y as EPSG:31300, which places every profile ~730 m west of its true position (checked
2026-11-18: profile 001E/01 at X 158297, Y 240419 is 4.48817 E with 31370; our copies held 4.47762 E, the 31300
conversion; 99.1% of profiles reproduce within 10 m). Corrected sites get a new site key suffix so their features are
re-extracted at the right place instead of reusing features sampled at the wrong one.
"""
from __future__ import annotations

import numpy as np

SUFFIX = "~lb72"


def flanders_31300_to_31370(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undo a Lambert 72 conversion done with EPSG:31300 and redo it with EPSG:31370."""
    from pyproj import Transformer
    to_xy = Transformer.from_crs("EPSG:4326", "EPSG:31300", always_xy=True)
    to_ll = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
    x, y = to_xy.transform(np.asarray(lon, float), np.asarray(lat, float))
    return to_ll.transform(x, y)
