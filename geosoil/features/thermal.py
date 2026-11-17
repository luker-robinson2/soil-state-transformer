"""Land-surface temperature climatology for every labelled site (Phase 2).

The day-night range of surface temperature tracks thermal inertia, which depends on soil moisture and
texture (sandy soils swing more), so it helps sand and clay where vegetation is sparse.

Source: OpenLandMap MODIS MOD11A2 LST climatology 2000-2017, monthly day and night means at 1 km,
CC-BY-SA 4.0 (Hengl et al.). Int16 x 0.02 = kelvin. Files are one-row strips (not tiled COGs), so they
are staged once (~8 GB for all 24).

Columns: lst_day_<mm>, lst_night_<mm> (degC), lst_day_mean, lst_night_mean, lst_diurnal_mean
(day - night), lst_diurnal_max (largest monthly day - night), lst_day_amp (warmest - coldest month).

    python -m geosoil.features.static --source thermal
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rasters import Layer, Source

URL = ("https://s3.openlandmap.org/arco/lst_mod11a2.{mon}.{dn}_m_1km_s_20000101_20171231_go_epsg.4326_v1.0.tif")
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def derive(df: pd.DataFrame) -> pd.DataFrame:
    day = df[[f"lst_day_{m:02d}" for m in range(1, 13)]].to_numpy()
    night = df[[f"lst_night_{m:02d}" for m in range(1, 13)]].to_numpy()
    with np.errstate(all="ignore"):
        out = pd.DataFrame({
            "lst_day_mean": np.nanmean(day, axis=1),
            "lst_night_mean": np.nanmean(night, axis=1),
            "lst_diurnal_mean": np.nanmean(day - night, axis=1),
            "lst_diurnal_max": np.nanmax(day - night, axis=1),
            "lst_day_amp": np.nanmax(day, axis=1) - np.nanmin(day, axis=1),
        }, index=df.index)
    return pd.concat([df, out], axis=1)


def source() -> Source:
    layers = [Layer(f"lst_{dn}_{i + 1:02d}", URL.format(mon=m, dn=dn), scale=0.02, offset=-273.15,
                    nodata=-32767, valid_range=(-80, 80))
              for dn in ("day", "night") for i, m in enumerate(MONTHS)]
    return Source("thermal", layers, licence="CC-BY-SA 4.0 (OpenLandMap MOD11A2 LST climatology)",
                  citation="Hengl et al., OpenLandMap LST monthly 1 km 2000-2017 (zenodo 6458406)", derive=derive)
