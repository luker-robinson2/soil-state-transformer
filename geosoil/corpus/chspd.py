"""Chile Soil Profile Database ChSPD v3 (Zenodo 22648780, CC-BY 4.0) → canonical labels.

A compilation of Chilean soil profiles. Two of its sources are already in the corpus through WoSIS and are
skipped: CHLSOC (99.8% of its locations are within 100 m of WoSIS sites) and the WoSIS subset itself. The rest
(CIREN, UChile, INIA, PUC, UACh, Otros) is kept. CIREN profiles are mostly WoSIS (CHLSOC) locations, but WoSIS
has no clay or bulk density there, so this source is a gap-fill source in dedup (REPUBLISHES + GAP_FILL).

Units and mapping:
  clay, sand %          → family unknown; kept only when clay + silt + sand is within 90–110 %
                          (or silt is missing and clay + sand ≤ 100)
  bulk_density g/cm3    → bd_method Cylinder or Excavation: bd_oven_dry; Clod (CIREN; the matric potential is
                          not stated) or missing: unknown
Skipped: organic_carbon (units are mixed: the non-CHLSOC rows look like g/kg, CHLSOC like %), and pH (the
method is not recorded, and harmonize keeps only water and CaCl2 pH).

Coordinates `x` = lon, `y` = lat (WGS84). Their precision varies by source, so coord_precision_m is set from
the number of decimals (2 dp ≈ 1.1 km, 3 dp ≈ 110 m, ...). Depth `top`/`bottom` in cm. Year = `year` (null for
CIREN and INIA). Keys: site_key = ID, layer_key = ID/top-bottom/row number within the profile.

    python -m geosoil.corpus.chspd
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import common

SOURCE = "chspd"
URL = "https://zenodo.org/api/records/22648780/files/ChSPD_V3.parquet/content"
RAW = common.RAW / "chspd" / "ChSPD_V3.parquet"
LICENSE = "CC-BY-4.0 (ChSPD v3, Zenodo 22648780)"
SKIP_SOURCES = {"CHLSOC", "WoSIS"}
BD_FAMILY = {"Cylinder": "bd_oven_dry", "Excavation": "bd_oven_dry", "Clod": "unknown"}
M_PER_DEGREE = 111_320.0


def decimals(v: pd.Series) -> pd.Series:
    """Number of decimals a coordinate was written with (trailing zeros and float noise ignored)."""
    s = v.round(8).map(lambda x: np.nan if pd.isna(x) else f"{x:.8f}".rstrip("0").split(".")[1])
    return s.map(lambda d: np.nan if not isinstance(d, str) else len(d))


def precision_m(lon: pd.Series, lat: pd.Series) -> pd.Series:
    dp = pd.concat([decimals(lon), decimals(lat)], axis=1).min(axis=1)
    return (M_PER_DEGREE * 10.0 ** -dp).clip(lower=1.0)


def texture_ok(clay: pd.Series, silt: pd.Series, sand: pd.Series) -> pd.Series:
    with_silt = (clay + silt + sand).between(90, 110)
    no_silt = silt.isna() & (clay + sand).between(1, 100)
    return with_silt | no_silt


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df[~df["source"].isin(SKIP_SOURCES)].copy()
    site_key = df["ID"].astype(str)
    top, bot = common.num(df["top"]), common.num(df["bottom"])
    layer_key = site_key + "/" + top.astype(str) + "-" + bot.astype(str) + "/" + \
        df.groupby(site_key).cumcount().astype(str)
    base = pd.DataFrame({"site_key": site_key, "layer_key": layer_key, "depth_top": top, "depth_bot": bot})
    clay, silt, sand = common.num(df["clay"]), common.num(df["silt"]), common.num(df["sand"])
    tex = texture_ok(clay, silt, sand)
    bd_method = df["bd_method"].fillna("NA").astype(str)
    parts = [
        common.long_rows(base, "sand", sand.where(tex), "chspd/sand", "unknown"),
        common.long_rows(base, "clay", clay.where(tex), "chspd/clay", "unknown"),
        common.long_rows(base, "bd", common.num(df["bulk_density"]), "chspd/bd_" + bd_method,
                         bd_method.map(BD_FAMILY).fillna("unknown")),
    ]
    vals = pd.concat(parts, ignore_index=True)
    lon, lat = common.num(df["x"]), common.num(df["y"])
    sites = pd.DataFrame({
        "site_key": site_key, "lon": lon, "lat": lat, "coord_source": "x/y",
        "coord_precision_m": precision_m(lon, lat), "year": common.num(df["year"]),
        "license": LICENSE + " [" + df["source"].astype(str) + "]",
    })
    return common.finalize(vals, sites, source=SOURCE, prefix="chspd", license=LICENSE)


def load() -> tuple[pd.DataFrame, dict]:
    return build(pd.read_parquet(common.fetch(URL, RAW)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
