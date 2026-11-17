"""Climate normals for every labelled site: TerraClimate 1991-2020 monthly climatology (Phase 2).

Source: TerraClimate (Abatzoglou et al. 2018), 1/24 degree (~4 km), global land, CC0. One NetCDF per
variable, 12 monthly bands, scale/offset declared in the files (applied by the sampler).
Files are staged once into data/raw/climate/ (16-154 MB each).

Columns (all prefixed clim_):
  monthly   clim_<var>_<mm>  for ppt, tmax, tmin, pet, aet, def, soil, vpd, srad (108 columns)
  annual    clim_map (mm), clim_mat (degC), clim_pet, clim_aet, clim_def (mm/yr), clim_soil (mm, mean),
            clim_vpd (kPa, mean), clim_srad (W/m2, mean), clim_trange (degC, warmest - coldest month),
            clim_ai (aridity index = map / pet), clim_ppt_cv (precipitation seasonality, CV of months)

    python -m geosoil.features.static --source climate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rasters import Layer, Source

URL = ("http://thredds.northwestknowledge.net:8080/thredds/fileServer/TERRACLIMATE_ALL/climatology/"
       "TerraClimate_19912020_{var}.nc")
VARS = ("ppt", "tmax", "tmin", "pet", "aet", "def", "soil", "vpd", "srad")
SUMS = ("ppt", "pet", "aet", "def")              # annual totals; the rest are annual means


def _months(df: pd.DataFrame, var: str) -> pd.DataFrame:
    return df[[f"clim_{var}_{m:02d}" for m in range(1, 13)]]


def derive(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for v in VARS:
        if v in ("tmax", "tmin"):
            continue
        m = _months(df, v)
        out[{"ppt": "clim_map"}.get(v, f"clim_{v}")] = m.sum(axis=1, min_count=12) if v in SUMS else m.mean(axis=1)
    tmean = (_months(df, "tmax").to_numpy() + _months(df, "tmin").to_numpy()) / 2
    out["clim_mat"] = np.nanmean(tmean, axis=1) if np.isfinite(tmean).any() else np.nan
    out["clim_trange"] = np.nanmax(tmean, axis=1) - np.nanmin(tmean, axis=1)
    ppt = _months(df, "ppt")
    out["clim_ai"] = out["clim_map"] / out["clim_pet"].where(out["clim_pet"] > 0)
    out["clim_ppt_cv"] = ppt.std(axis=1) / ppt.mean(axis=1).where(ppt.mean(axis=1) > 0)
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


def source() -> Source:
    layers = [Layer(f"clim_{v}_{m:02d}", URL.format(var=v), band=m, subdataset=v)
              for v in VARS for m in range(1, 13)]
    return Source("climate", layers, licence="CC0 (TerraClimate)",
                  citation="Abatzoglou, J.T. et al. (2018) TerraClimate, Scientific Data 5:170191",
                  derive=derive, meta={"normals": "1991-2020", "resolution_deg": 1 / 24})
