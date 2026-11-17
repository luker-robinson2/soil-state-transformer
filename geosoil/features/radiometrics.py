"""Airborne gamma-ray radiometrics (K, eU, eTh) for every labelled site (Phase 2).

Radiometrics sense the top ~30 cm of mineral soil and parent material directly, which is the
texture / parent-material signal that surface-reflectance embeddings miss.
No global grid exists, so regional grids are coalesced into one set of columns:

  US (CONUS)  USGS NURE Bayesian grids, 2 km, Albers (Goldman & Ellefsen 2020, doi:10.5066/P9YEAFHI),
              public domain. Uses the concentration grids (K_Y_mean_per, eU/eTh_Y_mean_ppm), NOT the
              log-ratio *_Y_mean files.
  Australia   Geoscience Australia Radmap v4 (2019), filtered, 0.001 deg (~100 m), GDA94, CC-BY 4.0
              (c) Commonwealth of Australia (Geoscience Australia) 2020. ~2 GB per element, staged once.

  N. America  USGS OFR 2005-1413 (Duval et al. 2005), 2 km, Canada + Alaska + CONUS, public domain.
              Used only where NURE and Radmap have no value. The grid's metadata claims a Clarke-1866
              sphere (R = 6378206.4 m) in spherical Transverse Mercator (lon_0 -100, k 0.926), but
              that places it ~4 km off. Validated against NURE at 5,000 CONUS sites (2026-11-17):
              R = 6370997 m with no datum shift aligns at zero offset, r = 0.91 K / 0.86 eTh /
              0.81 eU, median bias +0.03 % K / +0.2 ppm eTh / +0.15 ppm eU, so no recalibration.

Columns: rad_k (%), rad_u (ppm eU), rad_th (ppm eTh), rad_th_k, rad_u_th (ratios),
rad_src (1 = NURE, 2 = Radmap, 3 = OFR 2005-1413; NaN = no coverage).

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
NAMRAD = "https://mrdata.usgs.gov/radiometric/data/NAMrad_{el}.zip"
NAMRAD_CRS = "+proj=tmerc +lat_0=0 +lon_0=-100 +k=0.926 +x_0=0 +y_0=0 +R=6370997 +units=m +no_defs"
NAMRAD_EL = {"k": "K", "u": "U", "th": "Th"}
SOURCES = (("nure", 1), ("radmap", 2), ("namrad", 3))   # coalesce order: first non-NaN wins
RANGES = {"k": (0.0, 10.0), "u": (0.0, 50.0), "th": (0.0, 150.0)}     # physically plausible, else NaN


def derive(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    src = pd.Series(np.nan, index=df.index)
    for el in ("k", "u", "th"):
        v = pd.Series(np.nan, index=df.index)
        for name, _ in SOURCES:
            if f"{name}_{el}" in df:
                v = v.where(v.notna(), df[f"{name}_{el}"])
        out[f"rad_{el}"] = v
    for name, code in SOURCES:
        if f"{name}_k" in df:
            src[src.isna() & df[f"{name}_k"].notna()] = code
    out["rad_th_k"] = out["rad_th"] / out["rad_k"].where(out["rad_k"] > 0)
    out["rad_u_th"] = out["rad_u"] / out["rad_th"].where(out["rad_th"] > 0)
    out["rad_src"] = src
    return pd.concat([df[["site_id"]], out], axis=1)


def source() -> Source:
    layers = [Layer(f"nure_{el}", NURE[el], filename=NURE_FILES[el], valid_range=RANGES[el]) for el in NURE]
    layers += [Layer(f"radmap_{el}", RADMAP.format(el=el), valid_range=RANGES[el])
               for el in ("k", "u", "th")]
    layers += [Layer(f"namrad_{el}", NAMRAD.format(el=E), member=f"NAMrad_{E}.flt", crs=NAMRAD_CRS,
                     nodata=-9999.0, valid_range=RANGES[el]) for el, E in NAMRAD_EL.items()]
    return Source("radiometrics", layers,
                  licence="NURE and OFR 2005-1413 grids: USGS public domain; Radmap: CC-BY 4.0 (c) Commonwealth of Australia "
                          "(Geoscience Australia) 2020",
                  citation="Goldman & Ellefsen (2020) doi:10.5066/P9YEAFHI; Geoscience Australia Radmap v4 (2019); Duval et al. (2005) USGS OFR 2005-1413",
                  derive=derive)
