"""SODES v1, Spanish soil profile database (Zenodo 5094759, CC-BY 4.0) → canonical labels.

An MS Access file compiling 1,683 profiles from published Spanish soil monographs (CSIC, ICONA ...), read
with the pure-Python `access_parser`. Tables `DB2020_1683_PROFILES` (coordinates) and `HORIZON DATA`
(6,300 horizons), joined on PROF_CODE. An independent survey compilation: not in WoSIS (0% of its sites
within 100 m of C1), so dedup never drops it as a republisher.

Coordinates: every profile is in EPSG:25830 (ETRS89 / UTM 30N, extended into zones 29 and 31 for Galicia and
Catalonia/Baleares; verified per province), converted with pyproj. X/Y are seldom rounded:
coord_precision_m = 100. No Canary Islands. No sampling year (REFERENCE names the monograph).
Depth: top = M_DEPTH_UPP cm, bottom = top + M_THICKNESS. Thickness 0 marks an open last horizon (R/C);
finalize drops those as bad_depth.
Units and conversions:
  * clay A_CLAY %, family unknown (pipette in the monographs, not recorded per row).
  * sand = A_COARSE_SAND + A_FINE_SAND (%), family unknown (silt limit 20 or 50 µm varies by monograph).
    A_SAND_TOT is NOT used: it differs from CS + FS by a median of 34 points, so it is another quantity.
  * pH A_PH, method not stated; Spanish surveys report water pH (1:2.5), so it is loaded as ph_h2o under
    method id `sodes/A_PH_assumed_h2o`.
  * SOC = A_OM / 1.724 × 10 g/kg (A_OM %; OM in these monographs is Walkley-Black OC × 1.724), walkley_black
    under method id `sodes/A_OM_div1.724`.
  * CEC A_CEC cmol(+)/kg, family unknown. At ~2.3k horizons (mostly calcareous, base saturation 100%) A_CEC
    equals the sum of bases A_SUM_BASES_S: those values get method id `sodes/A_CEC_eq_sum_bases` so they can
    be separated downstream. A_SUM_BASES_S itself is never used as CEC.
  * Ca/Mg/K A_EXCH_* cmol(+)/kg, family unknown.
  * BD A_BDENSITY is skipped: rounded to 0.1 (49 distinct values), typical of estimated BD.
Several columns are stored as text with blanks; they are coerced to numbers.

    python -m geosoil.corpus.sodes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from .. import config as C
from . import common

SOURCE = "sodes"
URL = "https://zenodo.org/api/records/5094759/files/SODES_v1_Final.accdb/content"
RAW = common.RAW / "sodes" / "SODES_v1_Final.accdb"
LICENSE = "CC-BY-4.0 (SODES, Zenodo 5094759)"
PROFILES, HORIZONS = "DB2020_1683_PROFILES", "HORIZON DATA"
OM_PER_SOC = 1.724
CRS = "EPSG:25830"


def build(profiles: pd.DataFrame, horizons: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    h = horizons.copy()
    n = lambda c: common.num(h[c].astype(str).str.strip().replace("", np.nan))
    site_key = h["PROF_CODE"].astype(str).str.strip()
    top = n("M_DEPTH_UPP")
    base = pd.DataFrame({"site_key": site_key, "layer_key": h["HOR_CODE"].astype(str).str.strip(),
                         "depth_top": top, "depth_bot": top + n("M_THICKNESS")})
    om, cec, sb = n("A_OM"), n("A_CEC"), n("A_SUM_BASES_S")
    cec_id = pd.Series("sodes/A_CEC", index=h.index).where((cec - sb).abs() >= 0.05, "sodes/A_CEC_eq_sum_bases")
    parts = [
        common.long_rows(base, "clay", n("A_CLAY"), "sodes/A_CLAY", "unknown"),
        common.long_rows(base, "sand", n("A_COARSE_SAND") + n("A_FINE_SAND"), "sodes/A_COARSE_SAND+A_FINE_SAND",
                         "unknown"),
        common.long_rows(base, "ph", n("A_PH"), "sodes/A_PH_assumed_h2o", "ph_h2o"),
        common.long_rows(base, "soc", om / OM_PER_SOC * 10.0, "sodes/A_OM_div1.724", "walkley_black"),
        common.long_rows(base, "cec", cec, cec_id, "unknown"),
        *[common.long_rows(base, t, n(f"A_EXCH_{t.upper()}"), f"sodes/A_EXCH_{t.upper()}", "unknown")
          for t in ("ca", "mg", "k")],
    ]
    vals = pd.concat(parts, ignore_index=True)
    p = profiles.drop_duplicates("PROF_CODE")
    x, y = common.num(p["X_UTM_ETRS"]), common.num(p["Y_UTM_ETRS"])
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(x.to_numpy(), y.to_numpy())
    sites = pd.DataFrame({
        "site_key": p["PROF_CODE"].astype(str).str.strip(),
        "lon": lon, "lat": lat,
        "coord_source": "X_UTM_ETRS/Y_UTM_ETRS (EPSG:25830)",
        "coord_precision_m": 100.0,
        "year": pd.NA,
    })
    return common.finalize(vals, sites, source=SOURCE, prefix="sodes", license=LICENSE)


def load() -> tuple[pd.DataFrame, dict]:
    from access_parser import AccessParser          # pure-Python MS Access reader (pip install access-parser)
    db = AccessParser(str(common.fetch(URL, RAW)))
    return build(pd.DataFrame(db.parse_table(PROFILES)), pd.DataFrame(db.parse_table(HORIZONS)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
