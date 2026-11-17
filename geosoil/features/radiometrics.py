"""Airborne gamma-ray radiometrics (K, eU, eTh) for every labelled site (Phase 2).

Radiometrics sense the top ~30 cm of mineral soil and parent material directly, which is the
texture / parent-material signal that surface-reflectance embeddings miss.
No global grid exists, so regional grids are coalesced into one set of columns:

  US (CONUS)  USGS NURE Bayesian grids, 2 km, Albers (Goldman & Ellefsen 2020, doi:10.5066/P9YEAFHI),
              public domain. Uses the concentration grids (K_Y_mean_per, eU/eTh_Y_mean_ppm), NOT the
              log-ratio *_Y_mean files.
  Australia   Geoscience Australia Radmap v4 (2019), filtered, 0.001 deg (~100 m), GDA94, CC-BY 4.0
              (c) Commonwealth of Australia (Geoscience Australia) 2020. ~2 GB per element, staged once.

Not included yet: Canada/Alaska (USGS OFR 2005-1413). Its spherical Transverse Mercator radius is
ambiguous to ~15 km and must first be validated against the NURE grid where they overlap.

Columns: rad_k (%), rad_u (ppm eU), rad_th (ppm eTh), rad_th_k, rad_u_th (ratios),
rad_src (1 = NURE, 2 = Radmap; NaN = no coverage).

    python -m geosoil.features.static --source radiometrics
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rasters import Layer, Source

SB = "https://www.sciencebase.gov/catalog/file/get/5dd2f264e4b0695797628351?f=__disk__"
NURE = {
    "k": SB + "da%2Ff7%2F47%2Fdaf747f6a8a515b9a529a81d419bd1d46142313c",
    "u": SB + "83%2F8b%2F0c%2F838b0c3ee2776f108c1e9a8066aab7a696dd0b28",
    "th": SB + "f8%2F4e%2F6c%2Ff84e6c9e66efd3b1b86df5ab63cdee30cb4ce9db",
}
NURE_FILES = {"k": "K_Y_mean_per.tif", "u": "eU_Y_mean_ppm.tif", "th": "eTh_Y_mean_ppm.tif"}
RADMAP = ("https://thredds.nci.org.au/thredds/fileServer/iv65/Geoscience_Australia_Geophysics_Reference_Data_Collection/"
          "national_geophysical_compilations/Radmap2019/Radmap2019-grid-{el}_conc-Filtered-AWAGS_RAD_2019.nc")
RANGES = {"k": (0.0, 10.0), "u": (0.0, 50.0), "th": (0.0, 150.0)}     # physically plausible, else NaN


def derive(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    src = pd.Series(np.nan, index=df.index)
    for el in ("k", "u", "th"):
        nure, au = df.get(f"nure_{el}"), df.get(f"radmap_{el}")
        out[f"rad_{el}"] = nure.where(nure.notna(), au) if nure is not None and au is not None else (nure if nure is not None else au)
    if "nure_k" in df:
        src[df["nure_k"].notna()] = 1
    if "radmap_k" in df:
        src[src.isna() & df["radmap_k"].notna()] = 2
    out["rad_th_k"] = out["rad_th"] / out["rad_k"].where(out["rad_k"] > 0)
    out["rad_u_th"] = out["rad_u"] / out["rad_th"].where(out["rad_th"] > 0)
    out["rad_src"] = src
    return pd.concat([df[["site_id"]], out], axis=1)


def source() -> Source:
    layers = [Layer(f"nure_{el}", NURE[el], filename=NURE_FILES[el], valid_range=RANGES[el]) for el in NURE]
    layers += [Layer(f"radmap_{el}", RADMAP.format(el=el), valid_range=RANGES[el])
               for el in ("k", "u", "th")]
    return Source("radiometrics", layers,
                  licence="NURE grids: USGS public domain; Radmap: CC-BY 4.0 (c) Commonwealth of Australia "
                          "(Geoscience Australia) 2020",
                  citation="Goldman & Ellefsen (2020) doi:10.5066/P9YEAFHI; Geoscience Australia Radmap v4 (2019)",
                  derive=derive)
