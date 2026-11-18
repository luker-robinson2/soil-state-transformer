"""Open Soil Spectral Library v1.2 lab values → canonical label table.

Only the datasets not covered elsewhere are kept: KSSL.SSL duplicates the KSSL Lab Data
Mart, and LUCAS*.SSL rows carry the JRC licence (no redistribution). What remains is
ICRAF-ISRIC, AfSIS, CAF, SCHIEDUNG, GARRETT and SERBIA.

    python -m geosoil.corpus.ossl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .. import config as C
from . import common

SOURCE = "ossl_v1_2"
BASE = "https://storage.googleapis.com/soilspec4gg-public"
FILES = {"site": "ossl_soilsite_L0_v1.2.csv.gz", "lab": "ossl_soillab_L1_v1.2.csv.gz"}
EXCLUDE_PREFIXES = ("KSSL", "LUCAS")


def build(site: pd.DataFrame, lab: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ds = site["dataset.code_ascii_txt"].astype(str)
    site = site[~ds.str.startswith(EXCLUDE_PREFIXES)]
    df = site.merge(lab.drop(columns=["dataset.code_ascii_txt"], errors="ignore"),
                    on="id.layer_uuid_txt", how="inner")
    layer_key = df["id.layer_uuid_txt"].astype(str)
    # OSSL has one row per layer; the site is the georeferenced point within a dataset.
    lon, lat = common.num(df["longitude.point_wgs84_dd"]), common.num(df["latitude.point_wgs84_dd"])
    site_key = df["dataset.code_ascii_txt"].astype(str) + "/" + lon.round(5).astype(str) + "_" + lat.round(5).astype(str)
    base = pd.DataFrame({"site_key": site_key, "layer_key": layer_key,
                         "depth_top": common.num(df["layer.upper.depth_usda_cm"]),
                         "depth_bot": common.num(df["layer.lower.depth_usda_cm"])})
    col = lambda c: common.num(df[c]) if c in df else pd.Series(float("nan"), index=df.index)
    vals = pd.concat([
        # CEC and exchangeable bases: NH4OAc pH 7 (USDA a722-a725, cmol(+)/kg) and Mehlich-3 (a1059/a1065/a1066, mg/kg)
        common.long_rows(base, "cec", col("cec_usda.a723_cmolc.kg"), "ossl/cec_usda.a723", "cec_ph7"),
        common.long_rows(base, "ca", col("ca.ext_usda.a722_cmolc.kg"), "ossl/ca.ext_usda.a722", "nh4oac_ph7"),
        common.long_rows(base, "mg", col("mg.ext_usda.a724_cmolc.kg"), "ossl/mg.ext_usda.a724", "nh4oac_ph7"),
        common.long_rows(base, "k", col("k.ext_usda.a725_cmolc.kg"), "ossl/k.ext_usda.a725", "nh4oac_ph7"),
        common.long_rows(base, "k", col("k.ext_usda.a1065_mg.kg") / 391.0, "ossl/k.ext_usda.a1065", "mehlich3"),
        common.long_rows(base, "ca", col("ca.ext_usda.a1059_mg.kg") / 200.4, "ossl/ca.ext_usda.a1059", "mehlich3"),
        common.long_rows(base, "mg", col("mg.ext_usda.a1066_mg.kg") / 121.5, "ossl/mg.ext_usda.a1066", "mehlich3"),
        common.long_rows(base, "soc", common.num(df["oc_usda.c729_w.pct"]) * 10.0, "ossl/oc_usda.c729", "dry_combustion"),
        common.long_rows(base, "ph", common.num(df["ph.h2o_usda.a268_index"]), "ossl/ph.h2o_usda.a268", "ph_h2o"),
        common.long_rows(base, "ph", common.num(df["ph.cacl2_usda.a481_index"]), "ossl/ph.cacl2_usda.a481", "ph_cacl2"),
        common.long_rows(base, "sand", common.num(df["sand.tot_usda.c60_w.pct"]), "ossl/sand.tot_usda.c60", "unknown"),
        common.long_rows(base, "clay", common.num(df["clay.tot_usda.a334_w.pct"]), "ossl/clay.tot_usda.a334", "unknown"),
        common.long_rows(base, "bd", common.num(df["bd_usda.a4_g.cm3"]), "ossl/bd_usda.a4", "unknown"),
    ], ignore_index=True)
    sites = pd.DataFrame({
        "site_key": site_key, "lon": lon, "lat": lat,
        "coord_source": "longitude/latitude.point_wgs84_dd",
        "coord_precision_m": common.num(df["location.point.error_any_m"]),
        "year": pd.to_datetime(df["observation.date.begin_iso.8601_yyyy.mm.dd"], errors="coerce").dt.year,
        "license": df["dataset.license.title_ascii_txt"].astype(str),
    })
    return common.finalize(vals, sites, source=SOURCE, prefix="ossl", license="see row")


def load() -> tuple[pd.DataFrame, dict]:
    paths = {k: common.fetch(f"{BASE}/{f}", common.RAW / "ossl" / f) for k, f in FILES.items()}
    return build(pd.read_csv(paths["site"], low_memory=False), pd.read_csv(paths["lab"], low_memory=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
