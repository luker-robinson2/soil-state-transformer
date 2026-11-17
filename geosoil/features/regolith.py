"""Depth to bedrock for every labelled site (Phase 2).

Regolith thickness reflects weathering history and sediment accumulation, which shape texture and
bulk density. Source: Shangguan et al. 2017 (J. Adv. Model. Earth Syst. 9:65), absolute depth to
bedrock (BDTICM) at 250 m, released with SoilGrids 2017 by ISRIC. Int32 cm. The file is one-row strips
(8.5 GB), so it is staged once.

Note: this layer is a model product trained on soil profiles and wells to predict *bedrock depth*,
not on any of GeoSoil's targets, so it is a covariate here, not leakage. (SoilGrids predictions of
SOC/pH/texture are never used: they were trained on WoSIS, i.e. our own labels.)

Columns: regolith_bdticm_cm, regolith_log_bdticm.

    python -m geosoil.features.static --source regolith
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rasters import Layer, Source

URL = "https://files.isric.org/soilgrids/former/2017-03-10/data/BDTICM_M_250m_ll.tif"


def derive(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(regolith_log_bdticm=np.log1p(df["regolith_bdticm_cm"].clip(lower=0)))


def source() -> Source:
    return Source("regolith", [Layer("regolith_bdticm_cm", URL, scale=1.0, offset=0.0, nodata=-32768,
                                     valid_range=(0, 1e6))],
                  licence="ISRIC SoilGrids 2017 release (check current terms; ISRIC data are CC-BY 4.0)",
                  citation="Shangguan et al. 2017, doi:10.1002/2016MS000686", derive=derive)
