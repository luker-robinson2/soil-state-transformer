"""Canada National Pedon Database v3 (open subset) → canonical label table.

Units: CARB_ORG %, T_SAND/T_CLAY %, BULK_DEN g/cm3. Canadian surface organic (LFH)
horizons are recorded with negative depths above the mineral surface, so each pedon's
depths are shifted so that its shallowest layer starts at 0 cm.

    python -m geosoil.corpus.npdb
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from .. import config as C
from . import common

SOURCE = "canada_npdb"
URL = ("https://agriculture.canada.ca/atlas/data_donnees/nationalPedonDatabase/"
       "data_donnees/csv/npdb_v3_csv.zip")
RAW = common.RAW / "npdb" / "npdb_v3_csv.zip"
LICENSE = "Open Government Licence - Canada"


def _shift_depths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["U_DEPTH"], df["L_DEPTH"] = common.num(df["U_DEPTH"]), common.num(df["L_DEPTH"])
    offset = df.groupby("PEDON_ID")["U_DEPTH"].transform("min").clip(upper=0)
    df["U_DEPTH"] -= offset
    df["L_DEPTH"] -= offset
    return df


def build(chem: pd.DataFrame, phys: pd.DataFrame, info: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    parts = []
    for tbl, cols in ((chem, [("soc", "CARB_ORG", 10.0, "carb_org", "unknown"),
                              ("ph", "PH_H2O", 1.0, "ph_h2o", "ph_h2o"),
                              ("ph", "PH_CACL2", 1.0, "ph_cacl2", "ph_cacl2")]),
                      (phys, [("sand", "T_SAND", 1.0, "t_sand", "unknown"),
                              ("clay", "T_CLAY", 1.0, "t_clay", "unknown"),
                              ("bd", "BULK_DEN", 1.0, "bulk_den", "unknown")])):
        t = _shift_depths(tbl)
        base = pd.DataFrame({"site_key": t["PEDON_ID"].astype(str),
                             "layer_key": t["PEDON_ID"].astype(str) + "/" + t["LAYER_ID"].astype(str),
                             "depth_top": t["U_DEPTH"], "depth_bot": t["L_DEPTH"]})
        for target, col, scale, mid, fam in cols:
            parts.append(common.long_rows(base, target, common.num(t[col]) * scale, f"npdb/{mid}", fam))
    sites = pd.DataFrame({
        "site_key": info["PEDON_ID"].astype(str),
        "lon": common.num(info["DD_LONG_N"]),
        "lat": common.num(info["DD_LAT_N"]),
        "coord_source": "DD_LAT_N/DD_LONG_N",
        "coord_precision_m": common.num(info["CONF_METRS"]),
        "year": common.num(info["CAL_YEAR"]),
    })
    return common.finalize(pd.concat(parts, ignore_index=True), sites,
                           source=SOURCE, prefix="npdb", license=LICENSE)


def read(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    z = zipfile.ZipFile(path)
    rd = lambda n: pd.read_csv(z.open(n), encoding="utf-8-sig", low_memory=False)
    return rd("Chemical.csv"), rd("Physical.csv"), rd("Info.csv")


def load() -> tuple[pd.DataFrame, dict]:
    return build(*read(common.fetch(URL, RAW)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
