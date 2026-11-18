"""LUCAS Soil 2018 measured topsoil bulk density (Chen et al. 2024; Zenodo 10211884, CC-BY 4.0) → canonical labels.

Chen et al. (2024, ESSD 16:2367) compiled fine-earth bulk density (Bdfine) for the LUCAS 2018 topsoil points
in the EU and UK. Of the 18,945 rows only 5,163 are measured (`BDfine  method` == "Measurement"); the rest are
pedotransfer predictions ("Local-Prediction", "PTF-4-Prediction") and are dropped. The record holds two CSVs
that differ only in which PTF fills the gaps; their measured rows are identical, so one file is enough.

Columns: POINTID (LUCAS grid point id), `Bdfine (g cm-3)`, `SOCS (kg cm-2)` (not used: a stock, not a target),
coarse_vol, GPS_LAT / GPS_LONG (WGS84, the surveyor's GPS reading at the sampled point), `BDfine  method `
(note the double space and trailing space in the header; matched after whitespace normalisation).
Depth: the published value is for 0–20 cm (LUCAS 2018 cored 0–10 and 10–20 cm; only the 0–20 cm value is in
this record). Family bd_fine_earth: oven-dry mass of the < 2 mm fraction over core volume minus the coarse
fragment volume. Year 2018 (the survey round). coord_precision_m is NaN (GPS fix, accuracy not published).
Licence: CC-BY 4.0 on the Zenodo record; the upstream LUCAS microdata are published by the JRC.
LUCAS 2018 revisits points of the 2009/2015 rounds under the same POINTID, so any later LUCAS loader shares
site keys with this one (prefix "lucas").

    python -m geosoil.corpus.lucas_bd2018
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import common

SOURCE = "lucas_bd2018"
FILE = "LUCAS SOIL 2018 BD SOCS T-PTF4.csv"
URL = "https://zenodo.org/api/records/10211884/files/LUCAS%20SOIL%202018%20BD%20SOCS%20T-PTF4.csv/content"
RAW = common.RAW / "lucas_bd2018" / "LUCAS_SOIL_2018_BD_SOCS_T-PTF4.csv"
LICENSE = "CC-BY-4.0 (Chen et al. 2024, Zenodo 10211884; LUCAS Soil 2018, JRC)"
MEASURED = "measurement"
DEPTH = (0.0, 20.0)
YEAR = 2018


def _norm(c) -> str:
    return re.sub(r"\s+", " ", str(c)).strip()


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.rename(columns=_norm)
    n_all = len(df)
    measured = df["BDfine method"].astype(str).str.strip().str.lower() == MEASURED
    df = df[measured].copy()
    site_key = df["POINTID"].astype(str).str.strip()
    base = pd.DataFrame({"site_key": site_key, "layer_key": site_key + f"/{DEPTH[0]:g}-{DEPTH[1]:g}",
                         "depth_top": DEPTH[0], "depth_bot": DEPTH[1]})
    vals = common.long_rows(base, "bd", common.num(df["Bdfine (g cm-3)"]), "lucas2018/Bdfine_measured",
                            "bd_fine_earth")
    sites = pd.DataFrame({
        "site_key": site_key,
        "lon": common.num(df["GPS_LONG"]),
        "lat": common.num(df["GPS_LAT"]),
        "coord_source": "GPS_LAT/GPS_LONG (WGS84)",
        "coord_precision_m": np.nan,
        "year": YEAR,
    })
    labels, drops = common.finalize(vals, sites, source=SOURCE, prefix="lucas", license=LICENSE)
    return labels, {"ptf_predicted_rows": int(n_all - measured.sum()), **drops}


def load() -> tuple[pd.DataFrame, dict]:
    return build(pd.read_csv(common.fetch(URL, RAW)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
