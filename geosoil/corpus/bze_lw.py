"""German Agricultural Soil Inventory (BZE-LW) rows of GSBDDB (Zenodo 7075158, CC-BY 4.0) → canonical labels.

GSBDDB (global soil bulk density database) is a CSV compilation; only its `Source == "BZE_LW"` rows are used
(15,665 layers, 2,878 profiles on the 8 km BZE-LW grid; Poeplau et al. 2020, doi:10.1002/jpln.202000113).
The other GSBDDB sources repeat FIA, AfSP, WISE, NCSS and NatSoil, which we load natively or through WoSIS.
The native Thünen dataset (doi:10.3220/DATA20200203151139) sits behind a bot challenge, so GSBDDB is the
pullable copy. BZE-LW is not in WoSIS or the compendium loader (nearest C1 site median 12.9 km), so dedup
treats it as an independent survey.

Gotchas: the file is Latin-1 (UTF-8 decoding fails). Coordinates are WGS84 decimal degrees; whether they are
shifted for privacy is unknown, so coord_precision_m is NaN. SamplingYear (2011–2018) is read when present.
Licence: CC-BY 4.0 on Zenodo; the Thünen upstream licence cannot be verified by script.
Units (no conversion needed): UpperDepth/LowerDepth cm, Clay/Sand %, SOC g/kg, BD g/cm3.
Families: SOC dry_combustion (elemental analyser, total C minus carbonate C); clay/sand unknown (DIN pipette
and sieve, German limits); BD unknown (the paper fills missing fine-soil BD with a pedotransfer function
for some samples and GSBDDB does not say which).

    python -m geosoil.corpus.bze_lw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import common

SOURCE = "bze_lw"
URL = "https://zenodo.org/api/records/7075158/files/GSBDDB.csv/content"
RAW = common.RAW / "bze_lw" / "GSBDDB.csv"
LICENSE = "CC-BY-4.0 (GSBDDB, Zenodo 7075158; BZE-LW, Thünen Institute)"
KEEP_SOURCE = "BZE_LW"


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n_all = len(df)
    df = df[df["Source"].astype(str).str.strip() == KEEP_SOURCE].copy()
    site_key = df["ProfileID"].astype(str)
    top, bot = common.num(df["UpperDepth(cm)"]), common.num(df["LowerDepth(cm)"])
    base = pd.DataFrame({"site_key": site_key,
                         "layer_key": site_key + "/" + top.astype(str) + "-" + bot.astype(str),
                         "depth_top": top, "depth_bot": bot})
    parts = [
        common.long_rows(base, "soc", common.num(df["SOC(g/kg)"]), "bze_lw/SOC_g_kg", "dry_combustion"),
        common.long_rows(base, "clay", common.num(df["Clay(%)"]), "bze_lw/Clay_pct", "unknown"),
        common.long_rows(base, "sand", common.num(df["Sand(%)"]), "bze_lw/Sand_pct", "unknown"),
        common.long_rows(base, "bd", common.num(df["BD(g/cm3)"]), "bze_lw/BD_g_cm3", "unknown"),
    ]
    vals = pd.concat(parts, ignore_index=True)
    year = pd.to_numeric(df["SamplingYear"].astype(str).str[:4], errors="coerce")
    sites = pd.DataFrame({
        "site_key": site_key,
        "lon": common.num(df["Longitude"]),
        "lat": common.num(df["Latitude"]),
        "coord_source": "Longitude/Latitude (GSBDDB)",
        "coord_precision_m": np.nan,
        "year": year,
    })
    labels, drops = common.finalize(vals, sites, source=SOURCE, prefix="bze", license=LICENSE)
    return labels, {"other_gsbddb_rows": n_all - len(df), **drops}


def load() -> tuple[pd.DataFrame, dict]:
    return build(pd.read_csv(common.fetch(URL, RAW), encoding="latin-1", low_memory=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
