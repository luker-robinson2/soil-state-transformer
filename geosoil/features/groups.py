"""One definition of the feature groups, shared by the baseline and the feature-selection report.

A group is a modality, identified by column prefix. Adding a feature source means adding its
prefix here; join.py merges any source directory automatically.
"""
from __future__ import annotations

import re

from .. import config as C
from . import terrain

MONTHLY = re.compile(r"^clim_[a-z]+_\d{2}$")
PREFIXES = {                                  # group -> column prefix
    "terrain_multiscale": "tms_",
    "radiometrics": "rad_",
    "lithology": "litho_",
    "surficial": "surf_",
    "bare_soil": "bs_",
    "alphaearth_context": "aectx",
    "thermal": "lst_",
    "regolith": "regolith_",
    "hydrology": "hydro_",
    "land_use_history": "hist_",
}


def feature_groups(columns, include_location: bool = True) -> dict[str, list[str]]:
    cols = list(columns)
    g = {}
    if include_location:
        g["location"] = [c for c in ("lon", "lat") if c in cols]
        g["depth"] = [c for c in ("depth_mid",) if c in cols]
    g["terrain"] = [c for c in terrain.FEATURES if c in cols]
    g["alphaearth"] = [c for c in C.AE_COLS if c in cols]
    g["climate_annual"] = [c for c in cols if c.startswith("clim_") and not MONTHLY.match(c)]
    g["climate_monthly"] = [c for c in cols if MONTHLY.match(c)]
    for name, prefix in PREFIXES.items():
        g[name] = [c for c in cols if c.startswith(prefix)]
    return {k: v for k, v in g.items() if v}
