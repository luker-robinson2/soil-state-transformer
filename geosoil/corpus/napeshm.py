"""Soil Health Institute NAPESHM → canonical label table.

Per the data dictionary: b_soc = total C - inorganic C by dry combustion (%), b_ph = 1:2
water, b_sand / b_clay = sieve and pipette (%), bd_2mm = fine-earth bulk density (g/cm3).
Coordinates are per site (several plots share one point). Depth is 0-15 cm for every
sample in the published data.

    python -m geosoil.corpus.napeshm
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from .. import config as C
from . import common

SOURCE = "napeshm"
URL = "https://ndownloader.figshare.com/files/46794877"
RAW = common.RAW / "napeshm" / "napeshm.zip"
LICENSE = "CC-BY-4.0"
DEPTH = (0.0, 15.0)


def build(soil: pd.DataFrame, sites: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    base = pd.DataFrame({"site_key": soil["siteid"].astype(str) + "/" + soil["euid"].astype(str),
                         "layer_key": soil["euid"].astype(str),
                         "depth_top": DEPTH[0], "depth_bot": DEPTH[1]})
    vals = pd.concat([
        common.long_rows(base, "soc", common.num(soil["b_soc"]) * 10.0, "napeshm/b_soc", "dry_combustion"),
        common.long_rows(base, "ph", common.num(soil["b_ph"]), "napeshm/b_ph_1to2", "ph_h2o"),
        common.long_rows(base, "sand", common.num(soil["b_sand"]), "napeshm/b_sand", "pipette"),
        common.long_rows(base, "clay", common.num(soil["b_clay"]), "napeshm/b_clay", "pipette"),
        common.long_rows(base, "bd", common.num(soil["bd_2mm"]), "napeshm/bd_2mm", "bd_fine_earth"),
    ], ignore_index=True)
    s = soil[["siteid", "euid"]].merge(sites, on="siteid", how="left")
    site_tbl = pd.DataFrame({
        "site_key": s["siteid"].astype(str) + "/" + s["euid"].astype(str),
        "lon": common.num(s["site_longitude"]), "lat": common.num(s["site_latitude"]),
        "coord_source": "site_latitude/site_longitude (site centroid)",
        "year": common.num(s["start_year"]),
    })
    return common.finalize(vals, site_tbl, source=SOURCE, prefix="napeshm", license=LICENSE)


def load() -> tuple[pd.DataFrame, dict]:
    z = zipfile.ZipFile(common.fetch(URL, RAW))
    return build(pd.read_csv(z.open("CSV Files/soildata.csv"), low_memory=False),
                 pd.read_csv(z.open("CSV Files/sites.csv"), low_memory=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
