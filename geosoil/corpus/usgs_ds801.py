"""USGS DS 801, soils of the conterminous US → canonical label table (SOC only).

A generalized-random-tessellation sample of 4,857 sites, which makes it useful for
design-based accuracy checks. Only the A horizon has carbon data (the 0-5 cm file has
none). A_C_Org = total C - carbonate C, in wt. %; A_Depth is a "top-bottom" string in cm.

    python -m geosoil.corpus.usgs_ds801
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .. import config as C
from . import common

SOURCE = "usgs_ds801"
URL = "https://pubs.usgs.gov/ds/801/downloads/Appendix_3b_Ahorizon_18Sept2013.txt"
RAW = common.RAW / "usgs_ds801" / "Appendix_3b_Ahorizon_18Sept2013.txt"
LICENSE = "public domain"


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df[df["A_LabID"].notna() & (df["A_Depth"].astype(str) != "cm")]   # drop the units row
    depth = df["A_Depth"].astype(str).str.extract(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$").astype(float)
    base = pd.DataFrame({"site_key": df["SiteID"].astype(str), "layer_key": df["A_LabID"].astype(str),
                         "depth_top": depth[0], "depth_bot": depth[1]})
    vals = common.long_rows(base, "soc", common.num(df["A_C_Org"]) * 10.0, "usgs/A_C_Org", "dry_combustion")
    sites = pd.DataFrame({
        "site_key": df["SiteID"].astype(str),
        "lon": common.num(df["Longitude"]), "lat": common.num(df["Latitude"]),
        "coord_source": "Latitude/Longitude",
        "year": pd.to_datetime(df["CollDate"], format="%m/%d/%y", errors="coerce").dt.year,
    })
    return common.finalize(vals, sites, source=SOURCE, prefix="usgs801", license=LICENSE)


def load() -> tuple[pd.DataFrame, dict]:
    return build(pd.read_csv(common.fetch(URL, RAW), sep="\t", skiprows=12, low_memory=False, encoding="latin-1"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
