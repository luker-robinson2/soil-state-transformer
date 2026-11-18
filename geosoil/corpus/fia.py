"""USFS Forest Inventory and Analysis (FIA) Phase-3 soils lab → canonical labels.

Source: FIA DataMart, table SOILS_LAB (ENTIRE_SOILS_LAB.csv, 10 MB), joined to plot coordinates on
SOILS_LAB.PLT_CN = PLOT.CN. Licence: US Government work, public domain.

Only the mineral-soil layers are used: MIN_1 = 0–10.16 cm and MIN_2 = 10.16–20.32 cm (0–4 in and 4–8 in,
FIA P3 soils manual). Forest-floor (FF_TOTAL, FF, L_ORG) and organic (ORG_1/2) layers are skipped. Rows with
QASTATCD 7 are QA re-measurements and are dropped.

Units and conversions (checked on the data, 2026-11-18 data scout pass 3):
  C_ORG_PCT %                → ×10 g/kg, dry_combustion
  PH_H2O / PH_CACL2          → ph_h2o / ph_cacl2
  BULK_DENSITY g/cm3         → bd_oven_dry (a few values above 2.65 are dropped by finalize)
  EXCHNG_K / _CA / _MG mg/kg → ÷391.0 / ÷200.4 / ÷121.5 cmol(+)/kg. The extractant is 1 M NH4Cl (unbuffered),
                               so the family is unknown (method_id fia/nh4cl). Proof of units: the summed bases
                               (+ Na, Al) divided by ECEC is 1.000.
ECEC is skipped: it is computed as the sum of those cations, so it adds no measurement.

Keys: site_key = PLT_CN (one plot visit), layer_key = CN (one sample per subplot and layer). Revisits of a plot
share coordinates and are merged downstream (dedup.merge_same_coordinates). Year = MEASYEAR.

Coordinates: public FIA plot coordinates are fuzzed (within 0.5 mile, up to 1 mile for some plots) and up to
20% of private plots are swapped within the county, so coord_precision_m = 1600. ENTIRE_PLOT.csv is 476 MB,
so coordinates come from the per-state PLOTGEOM tables (same CN, LAT, LON; about 60% of the PLOT size). Each
state file is downloaded through common.fetch, reduced to the soil plots' CN/LAT/LON, and deleted; the reduced
table is cached as data/raw/fia/soil_plot_coords.csv.

    python -m geosoil.corpus.fia
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .. import config as C
from . import common

SOURCE = "fia"
BASE_URL = "https://apps.fs.usda.gov/fia/datamart/CSV/"
LAB_URL = BASE_URL + "ENTIRE_SOILS_LAB.csv"
RAW = common.RAW / "fia"
LICENSE = "US Government work, public domain (USDA Forest Service FIA)"
COORD_SOURCE = "FIA public PLOT LAT/LON (fuzzed+swapped)"
COORD_PRECISION_M = 1600.0
LAYERS = {"MIN_1": (0.0, 10.16), "MIN_2": (10.16, 20.32)}
MG_PER_CMOL = {"k": ("EXCHNG_K", 391.0), "ca": ("EXCHNG_CA", 200.4), "mg": ("EXCHNG_MG", 121.5)}
QA_REMEASURE = 7
STATE_ABBR = {1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE", 11: "DC", 12: "FL",
              13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME",
              24: "MD", 25: "MA", 26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV", 33: "NH",
              34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI",
              45: "SC", 46: "SD", 47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
              56: "WY", 72: "PR", 78: "VI"}


def mineral(lab: pd.DataFrame) -> pd.DataFrame:
    """Mineral 0–20 cm layers, QA re-measurements removed."""
    return lab[lab["LAYER_TYPE"].isin(LAYERS) & (common.num(lab["QASTATCD"]) != QA_REMEASURE)]


def build(lab: pd.DataFrame, plots: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """lab: SOILS_LAB rows; plots: CN, LAT, LON of the plot visits."""
    df = mineral(lab).copy()
    site_key = df["PLT_CN"].astype("int64").astype(str)
    depth = df["LAYER_TYPE"].map(LAYERS)
    base = pd.DataFrame({"site_key": site_key, "layer_key": df["CN"].astype("int64").astype(str),
                         "depth_top": depth.str[0].astype(float), "depth_bot": depth.str[1].astype(float)})
    col = lambda c: common.num(df[c])
    parts = [
        common.long_rows(base, "soc", (col("C_ORG_PCT") * 10.0).where(col("C_ORG_PCT") > 0), "fia/C_ORG_PCT",
                         "dry_combustion"),
        common.long_rows(base, "ph", col("PH_H2O"), "fia/PH_H2O", "ph_h2o"),
        common.long_rows(base, "ph", col("PH_CACL2"), "fia/PH_CACL2", "ph_cacl2"),
        common.long_rows(base, "bd", col("BULK_DENSITY"), "fia/BULK_DENSITY", "bd_oven_dry"),
        *[common.long_rows(base, t, (col(c) / f).where(col(c) >= 0), "fia/nh4cl", "unknown")
          for t, (c, f) in MG_PER_CMOL.items()],
    ]
    vals = pd.concat(parts, ignore_index=True)
    p = plots.assign(site_key=plots["CN"].astype("int64").astype(str)).drop_duplicates("site_key")
    sites = pd.DataFrame({"site_key": site_key, "year": col("MEASYEAR")}).drop_duplicates("site_key")
    sites = sites.merge(p[["site_key", "LAT", "LON"]], on="site_key", how="left")
    sites = sites.rename(columns={"LAT": "lat", "LON": "lon"}).assign(
        lat=lambda s: common.num(s["lat"]), lon=lambda s: common.num(s["lon"]),
        coord_source=COORD_SOURCE, coord_precision_m=COORD_PRECISION_M)
    return common.finalize(vals, sites, source=SOURCE, prefix="fia", license=LICENSE)


def _state_coords(abbr: str, cns: set[int]) -> pd.DataFrame:
    path = common.fetch(BASE_URL + f"{abbr}_PLOTGEOM.csv", RAW / f"{abbr}_PLOTGEOM.csv")
    got = pd.read_csv(path, usecols=["CN", "LAT", "LON"])
    got = got[got["CN"].isin(cns)]
    path.unlink()                                 # keep disk modest; the url+sha256 sidecar stays
    return got


def plot_coords(lab: pd.DataFrame) -> pd.DataFrame:
    out = RAW / "soil_plot_coords.csv"
    m = mineral(lab)
    cns = set(m["PLT_CN"].astype("int64"))
    if out.exists():
        have = pd.read_csv(out)
        if cns <= set(have["CN"]) or (RAW / "soil_plot_coords.done").exists():
            return have
    states = sorted(STATE_ABBR[s] for s in m["STATECD"].astype(int).unique() if s in STATE_ABBR)
    with ThreadPoolExecutor(4) as ex:
        coords = pd.concat(ex.map(lambda a: _state_coords(a, cns), states), ignore_index=True)
    coords.to_csv(out, index=False)
    (RAW / "soil_plot_coords.done").write_text(",".join(states) + "\n")
    return coords


def load() -> tuple[pd.DataFrame, dict]:
    lab = pd.read_csv(common.fetch(LAB_URL, RAW / "ENTIRE_SOILS_LAB.csv"), low_memory=False)
    return build(lab, plot_coords(lab))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
