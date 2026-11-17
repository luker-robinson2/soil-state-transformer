"""USDA Rapid Carbon Assessment (RaCA) → canonical label table. The SOC hold-out set.

Only lab measurements are used: SOC = c_tot_ncs - 0.12 * caco3 (dry combustion, %),
with the same carbonate rule as KSSL, and bulk density only where BDmeasured ==
"measured". SOC_pred1 (predicted from spectra) is ignored. Site coordinates are
published rounded to 0.01 degrees (~1 km), so coord_precision_m = 1000.

    python -m geosoil.corpus.raca
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .. import config as C
from . import common

SOURCE = "raca"
BOX = "https://nrcs.box.com/index.php?rm=box_download_shared_file&shared_name=s9bcdroihv1vyre70bkl336cdr13up7q&file_id="
URLS = {"samples": BOX + "f_1031424800228", "locations": BOX + "f_1031428176175"}
LICENSE = "CC0"
CARBONATE_FREE_PH = 7.0   # not available in RaCA; carbonate is treated as 0 only when measured absent


def build(samples: pd.DataFrame, locations: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    base = pd.DataFrame({"site_key": samples["rcasiteid"].astype(str),
                         "layer_key": samples["samp"].astype(str),
                         "depth_top": common.num(samples["TOP"]), "depth_bot": common.num(samples["BOT"])})
    tc, caco3 = common.num(samples["c_tot_ncs"]), common.num(samples["caco3"])
    # Without a carbonate measurement, total C counts as organic only if the sample did not
    # effervesce in 1N HCl (no free carbonates).
    no_fizz = samples["efferv_1nhcl"].astype(str).str.lower().isin(["none", "nan"]) & caco3.isna()
    soc = ((tc - common.CACO3_C_FRACTION * caco3.fillna(0.0)) * 10.0).where(caco3.notna() | no_fizz)
    bd = common.num(samples["Bulkdensity"]).where(samples["BDmeasured"] == "measured")
    vals = pd.concat([
        common.long_rows(base, "soc", soc, "raca/c_tot_ncs-caco3", "dry_combustion"),
        common.long_rows(base, "bd", bd, "raca/bd_measured", "unknown"),
    ], ignore_index=True)
    sites = pd.DataFrame({
        "site_key": locations["RaCA_Id"].astype(str),
        "lon": common.num(locations["Gen_long"]), "lat": common.num(locations["Gen_lat"]),
        "coord_source": "Gen_lat/Gen_long (rounded 0.01 deg)", "coord_precision_m": 1000.0,
        "year": 2011,
    })
    return common.finalize(vals, sites, source=SOURCE, prefix="raca", license=LICENSE)


def load() -> tuple[pd.DataFrame, dict]:
    paths = {k: common.fetch(u, common.RAW / "raca" / f"raca_{k}.csv") for k, u in URLS.items()}
    return build(pd.read_csv(paths["samples"], low_memory=False),
                 pd.read_csv(paths["locations"], low_memory=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
