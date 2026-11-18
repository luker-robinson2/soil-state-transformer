"""OpenLandMap SoilSamples v1.1 (Hengl et al. 2026; Zenodo 19731002, CC-BY 4.0) → canonical labels.

A compilation of lab-measured soil horizons from many national and research databases (not model predictions).
Only the source databases we do not already ingest are kept (SOURCES), so nothing is counted twice; sources that
republish ours (USDA NCSS, RaCA, SISLAC, NatSoil, AfSP, NPDB, iSDA, WISE, Brazil, Chile SOCDB, INEGI, ISCN …)
are skipped, as is China SOTER (76% of its sites sit within 1 km of a WoSIS site: the same profiles with shifted
coordinates), and LandPKS (app users' hand-texture estimates, not lab data), GLanCE (pseudo-samples), LUCAS
(JRC terms) and Russia EGRPR (the compendium warns its locations may be synthetic). Dedup still runs against
WoSIS and the others (dedup.REPUBLISHES), because some national sets also reached WoSIS.

Units in the compendium: organic_carbon g/kg, clay/sand %, pH, CEC cmol(+)/kg, ca/mg/k_ext mg/kg (→ cmol(+)/kg
by 200.4, 121.5, 391), bulk density g/cm3. Extraction methods are not recorded per value, so the bases and CEC
go in as unknown-method families; pH-7 CEC is kept as its own family. China SOCS rows keep texture and pH only,
because 5k of its SOC/BD values were filled by a pedotransfer function. Licences are recorded per source; where
the upstream licence is unclear the rows are train-only and never committed.

    python -m geosoil.corpus.olm_compendium
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import common

SOURCE = "olm_compendium"
URL = "https://zenodo.org/api/records/19731002/files/sol_chem.pnts_horizons.rds/content"
RAW = common.RAW / "olm_compendium"
CC = "CC-BY-4.0 (OpenLandMap SoilSamples)"
UNCLEAR = CC + "; upstream licence unclear, train-only"
SOURCES = {   # source_db -> licence text
    "Vlaanderen": CC + "; Flemish open data licence", "QuebecTEX": CC, "CNSOCS": CC, "Mexico-SAGARPA": CC,
    "Portugal.INFOSOLO": CC, "Slovenia.SPDB": UNCLEAR, "Croatian_Soil_Pedon": UNCLEAR, "ScotlandNSIS1": CC + "; OS open data",
    "Iran_SPDB": UNCLEAR, "Canada_CUFS": CC, "NAMSOTER": CC, "CostaRica": CC,
    "Netherlands.BIS": CC, "Canada_subarctic": CC, "Alaska_interior": CC, "Guatemala_GTMSOC": CC, "GEMAS_2009": CC,
    "USGS.NGS": CC,
}
TEXTURE_PH_ONLY = {"CNSOCS"}
MG_PER_CMOL = {"k": 391.0, "ca": 200.4, "mg": 121.5}


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df[df["source_db"].isin(SOURCES)].copy()
    lon, lat = common.num(df["longitude_decimal_degrees"]), common.num(df["latitude_decimal_degrees"])
    # some sources carry no usable site key (one key for every row): key sites by coordinates instead
    coord_key = df["source_db"] + "/" + lon.round(5).astype(str) + "_" + lat.round(5).astype(str)
    weak = df.groupby("source_db")["site_key"].transform("nunique") <= 1
    site_key = np.where(weak | df["site_key"].isna(), coord_key, df["source_db"] + "/" + df["site_key"].astype(str))
    site_key = pd.Series(site_key, index=df.index)
    layer_key = site_key + "/" + df["hzn_top"].astype(str) + "-" + df["hzn_bot"].astype(str) + "/" + \
        df.groupby(site_key).cumcount().astype(str)
    base = pd.DataFrame({"site_key": site_key, "layer_key": layer_key,
                         "depth_top": common.num(df["hzn_top"]), "depth_bot": common.num(df["hzn_bot"])})
    full = ~df["source_db"].isin(TEXTURE_PH_ONLY)
    col = lambda c: common.num(df[c]) if c in df else pd.Series(np.nan, index=df.index)
    parts = [
        common.long_rows(base, "sand", col("sand_total"), "olm/sand_total", "unknown"),
        common.long_rows(base, "clay", col("clay_total"), "olm/clay_total", "unknown"),
        common.long_rows(base, "ph", col("ph_h2o"), "olm/ph_h2o", "ph_h2o"),
        common.long_rows(base, "ph", col("ph_cacl2"), "olm/ph_cacl2", "ph_cacl2"),
        common.long_rows(base, "soc", col("organic_carbon").where(full & (col("organic_carbon") > 0)), "olm/organic_carbon", "unknown"),
        common.long_rows(base, "bd", col("bulk_density_oven_dry").where(full), "olm/bulk_density_oven_dry", "bd_oven_dry"),
        common.long_rows(base, "cec", col("cec_nh4_ph_7").where(full), "olm/cec_nh4_ph_7", "cec_ph7"),
        common.long_rows(base, "cec", col("sum_of_cations_cec_pH_8_2").where(full), "olm/cec_ph8.2", "cec_sum_cations_ph82"),
        *[common.long_rows(base, el, (col(f"{el}_ext") / f).where(full & (col(f"{el}_ext") > 0)), f"olm/{el}_ext", "unknown")
          for el, f in MG_PER_CMOL.items()],
    ]
    vals = pd.concat(parts, ignore_index=True)
    year = pd.to_numeric(df["site_obsdate"].astype(str).str[:4], errors="coerce")
    sites = pd.DataFrame({
        "site_key": site_key, "lon": lon, "lat": lat, "coord_source": "longitude/latitude_decimal_degrees",
        "coord_precision_m": np.nan, "year": year.where(year.between(1900, 2026)),
        "license": df["source_db"].map(SOURCES) + " [" + df["source_db"] + "]",
    })
    return common.finalize(vals, sites, source=SOURCE, prefix="olm", license="see row")


def _csv(rds: Path) -> Path:
    out = rds.parent / "olm_horizons.csv.gz"
    if not out.exists():
        subprocess.run(["Rscript", str(C.REPO / "data" / "extraction" / "olm_compendium_export.R"), str(rds), str(out),
                        ",".join(SOURCES)], check=True)
    return out


def load() -> tuple[pd.DataFrame, dict]:
    return build(pd.read_csv(_csv(common.fetch(URL, RAW / "sol_chem.pnts_horizons.rds")), low_memory=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
