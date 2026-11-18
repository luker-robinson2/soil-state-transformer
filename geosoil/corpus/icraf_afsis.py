"""ICRAF AfSIS Phase-I wet chemistry, doi:10.34725/DVN/66BFOB (CC-BY 4.0) → canonical labels.

AfSIS Phase-I topsoil and subsoil samples (2009–2013, 2,002 samples in sub-Saharan Africa) re-analysed at
Rothamsted in 2017. The plots are mostly the AfSIS sites WoSIS already holds, with offset coordinates (27% within
100 m, 97% within 1 km of a WoSIS AfSIS site), but WoSIS has none of these exchangeable bases there, so this is a
gap-fill source in dedup (REPUBLISHES + GAP_FILL). The file is a CSV despite the .tab name on Dataverse.

Depth: `Depth` Topsoil = 0–20 cm, Subsoil = 20–50 cm (documented). Coordinates: decimal `Latitude`/`Longitude`
(the variable doc swaps the X/Y labels, but the values are right).

Units and mapping:
  Caex / Kex / Mgex cmol(+)/kg, cobalt hexamine (Cohex) extraction → family unknown, method_id icraf/cohex
  eCEC cmol(+)/kg (Cohex, calculated)                          → cec family ecec; values ≤ 0 are dropped
  Percent_Org_C % (LECO total C minus inorganic C)              → ×10 g/kg, dry_combustion
Skipped: pH (water 1:2.5) duplicates WoSIS AfSIS pH at the same plots. The variable doc swaps the descriptions
of Percent_N and Percent_C; the values confirm Percent_C is carbon.

Year: samples were taken 2009–2013 by country, not recorded per row, so the year is left empty.
Keys: site = Country/Site/Cluster/Plot, layer = SSN (sample).

    python -m geosoil.corpus.icraf_afsis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import common

SOURCE = "icraf_afsis"
URL = "https://data.worldagroforestry.org/api/access/datafile/11218?format=original"
RAW = common.RAW / "icraf_afsis" / "afsis_phase1_wetchem.csv"
LICENSE = "CC-BY-4.0 (ICRAF AfSIS Phase I wet chemistry, doi:10.34725/DVN/66BFOB)"
DEPTHS = {"Topsoil": (0.0, 20.0), "Subsoil": (20.0, 50.0)}


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    site_key = df[["Country", "Site", "Cluster", "Plot"]].astype(str).agg("/".join, axis=1)
    depth = df["Depth"].map(DEPTHS)
    base = pd.DataFrame({"site_key": site_key, "layer_key": df["SSN"].astype(str),
                         "depth_top": depth.str[0].astype(float), "depth_bot": depth.str[1].astype(float)})
    col = lambda c: common.num(df[c])
    parts = [
        common.long_rows(base, "soc", (col("Percent_Org_C") * 10.0).where(col("Percent_Org_C") > 0),
                         "icraf/Percent_Org_C", "dry_combustion"),
        common.long_rows(base, "cec", col("eCEC").where(col("eCEC") > 0), "icraf/eCEC_cohex", "ecec"),
        *[common.long_rows(base, t, col(c).where(col(c) >= 0), "icraf/cohex", "unknown")
          for t, c in (("k", "Kex"), ("ca", "Caex"), ("mg", "Mgex"))],
    ]
    vals = pd.concat(parts, ignore_index=True)
    sites = pd.DataFrame({"site_key": site_key, "lon": col("Longitude"), "lat": col("Latitude"),
                          "coord_source": "Latitude/Longitude", "coord_precision_m": np.nan, "year": pd.NA})
    return common.finalize(vals, sites, source=SOURCE, prefix="icraf", license=LICENSE)


def load() -> tuple[pd.DataFrame, dict]:
    return build(pd.read_csv(common.fetch(URL, RAW)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
