"""Canonical long-format label table shared by every corpus source.

One row = one measured value of one target on one layer at one site. Values are
kept per analytical method (``method_id`` / ``method_family``) instead of being
converted up front; harmonization happens downstream (CORPUS-PLAN.md §E).
"""
from __future__ import annotations

import pandas as pd

from .. import config as C

COLUMNS = {
    "site_id": "string",           # "<source>:<native key>", unique per site
    "layer_id": "string",          # "<source>:<native layer key>"
    "source": "string",            # registry id in data/corpus/sources.yaml
    "lon": "float64",
    "lat": "float64",
    "coord_source": "string",      # which native coordinate field was used
    "coord_precision_m": "float64",  # NaN = unknown
    "depth_top": "float64",        # cm
    "depth_bot": "float64",        # cm
    "target": "string",            # one of config.TARGETS
    "value": "float64",            # in config.TARGET_UNITS[target]
    "unit": "string",
    "method_id": "string",         # source-native method code, e.g. "kssl/6A1c"
    "method_family": "string",     # cross-source family, e.g. "walkley_black"
    "year": "Int64",               # sampling year, if known
    "license": "string",
}

# Physically plausible ranges in canonical units; rows outside are dropped.
VALID_RANGE = {
    "soc": (0.0, 600.0),   # g/kg; organic soils reach ~580
    "ph": (2.0, 11.0),
    "sand": (0.0, 100.0),
    "clay": (0.0, 100.0),
    "bd": (0.1, 2.65),     # g/cm3
    "om": (0.0, 1000.0),   # g/kg
    "cec": (0.0, 250.0),   # cmol(+)/kg; organic soils reach ~200
    "k": (0.0, 25.0),      # cmol(+)/kg exchangeable
    "ca": (0.0, 150.0),    # NH4OAc dissolves some carbonate, so calcareous soils read high
    "mg": (0.0, 80.0),
}


def empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in COLUMNS.items()})


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Order and type columns; raise if any are missing."""
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"label table missing columns: {sorted(missing)}")
    return df[list(COLUMNS)].astype(COLUMNS).reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """Raise on structural problems that would poison training."""
    bad_target = set(df["target"].dropna()) - set(C.TARGETS)
    if bad_target:
        raise ValueError(f"unknown targets: {sorted(bad_target)}")
    if df[["site_id", "target", "value", "method_id"]].isna().any().any():
        raise ValueError("site_id, target, value and method_id must be non-null")
    if not df["lon"].dropna().between(-180, 180).all() or not df["lat"].dropna().between(-90, 90).all():
        raise ValueError("coordinates out of range")
    if (df["depth_bot"] <= df["depth_top"]).any():
        raise ValueError("depth_bot must exceed depth_top")
    for t, (lo, hi) in VALID_RANGE.items():
        v = df.loc[df["target"] == t, "value"]
        if not v.between(lo, hi).all():
            raise ValueError(f"{t} values outside [{lo}, {hi}]")
    for t, unit in C.TARGET_UNITS.items():
        if (df.loc[df["target"] == t, "unit"] != unit).any():
            raise ValueError(f"{t} must be in {unit}")
    if df.duplicated(["layer_id", "target", "method_id"]).any():
        raise ValueError("duplicate (layer_id, target, method_id) rows")
