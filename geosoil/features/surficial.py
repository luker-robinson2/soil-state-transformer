"""Surficial materials for every labelled site: what the soil formed in, and how coarse it is (Phase 2).

Bedrock lithology (lithology.py) says little where soils sit on transported deposits: glacial till,
loess, dune sand, alluvium. Those deposits set the starting texture directly, and most US pedons are
on them. Parent-material texture class is a standard covariate in digital soil mapping for that
reason (McBratney et al. 2003, "scorpan" p).

Source: USGS DS-425, Surficial Materials in the Conterminous United States (Soller, Reheis, Garrity &
Van Sistine 2009), 1:5M, 49 map units, public domain. Each unit name encodes origin, grain size
and thickness, which are parsed into three readable encodings instead of 49 sparse columns:

  surf_<origin>     one-hot origin family (alluvial, eolian_loess, glacial_till, residual, ...)
  surf_grain        0 fine (clayey / fine-grained / lacustrine / playa), 1 medium or silty
                    (loess, silty till, medium coastal), 2 coarse (sand, gravel, dune sand); NaN unknown
  surf_thick        0 discontinuous, 1 thin (< 100 ft), 2 thick (> 100 ft); NaN unknown
NaN in every column outside the conterminous US.

    python -m geosoil.features.static --source surficial
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..corpus import common
from . import shards
from .lithology import point_in_polygon

URL = "https://pubs.usgs.gov/ds/425/USGS_DS_425_SHAPES.zip"
MEMBER = "USGS_DS_425_SHAPES/Surficial_materials"
LICENCE = "USGS public domain (Soller et al. 2009, USGS Data Series 425)"
ORIGINS = ("alluvial", "biological", "coastal", "eolian_loess", "eolian_sand", "glacial_till", "glaciofluvial",
           "lacustrine", "colluvial", "organic", "proglacial", "residual", "volcanic", "water")


def origin(code: int) -> str:
    """Origin family from the DS-425 unit code (hundreds digit, with eolian and colluvial split)."""
    c = int(code)
    if c < 100:
        return "alluvial"
    if c == 999:
        return "water"
    if 300 <= c < 320:
        return "eolian_loess"
    if 320 <= c < 400:
        return "eolian_sand"                      # dune sand and the southern High Plains cover sands
    if 450 <= c < 500:
        return "glaciofluvial"
    if 980 <= c < 999:
        return "volcanic"
    return {1: "biological", 2: "coastal", 4: "glacial_till", 5: "lacustrine", 6: "colluvial", 7: "organic",
            8: "proglacial", 9: "residual"}.get(c // 100, "residual")


def grain(name: str) -> float:
    n = name.lower()
    if any(k in n for k in ("clayey", "fine-grained", "fine grained", "lacustrine", "playa")):
        return 0.0
    if any(k in n for k in ("sandy", "sand and gravel", "dune sand", "coarse-grained", "high plains")):
        return 2.0
    if any(k in n for k in ("loess", "silty", "medium-grained")):
        return 1.0
    return np.nan


def thickness(name: str) -> float:
    n = name.lower()
    return 0.0 if "discontinuous" in n else 2.0 if "thick" in n else 1.0 if "thin" in n else np.nan


def encode(codes: pd.Series, names: pd.Series) -> pd.DataFrame:
    """Per-site encodings from the containing polygon's unit code and name (NaN = no polygon)."""
    has = codes.notna()
    fam = codes.where(has).map(lambda c: origin(c) if pd.notna(c) else np.nan)
    out = pd.DataFrame({f"surf_{o}": (fam == o).astype(float) for o in ORIGINS}, index=codes.index)
    out["surf_grain"] = names.map(lambda s: grain(s) if isinstance(s, str) else np.nan)
    out["surf_thick"] = names.map(lambda s: thickness(s) if isinstance(s, str) else np.nan)
    out[~has] = np.nan
    return out


def extract(sites: pd.DataFrame, out_root: Path = C.DATA_PROC / "features") -> dict:
    import pyogrio
    t0 = time.time()
    z = common.fetch(URL, common.RAW / "surficial" / "USGS_DS_425_SHAPES.zip")
    shp = z.parent / f"{MEMBER}.shp"
    if not shp.exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(z.parent, [m for m in f.namelist() if m.startswith(MEMBER + ".")])
    polys = pyogrio.read_dataframe(shp, columns=["UNIT_CODE", "UNIT_NAME"])
    s = sites.reset_index(drop=True)
    code = point_in_polygon(s, polys, "UNIT_CODE")
    name = point_in_polygon(s, polys, "UNIT_NAME")
    df = pd.concat([pd.DataFrame({"site_id": s["site_id"].to_numpy()}), encode(code, name)], axis=1)
    cover = round(float(df["surf_water"].notna().mean()), 4)
    shards.write(df, out_root / "surficial", 0, 1, {
        "source": "surficial", "url": URL, "licence": LICENCE, "file_sha256": {z.name: shards.sha256(z)},
        "coverage": cover, "failed_blocks": [], "sites": int(len(df)), "seconds": round(time.time() - t0, 1)})
    return {"source": "surficial", "sites": len(df), "coverage": cover, "seconds": round(time.time() - t0, 1)}
