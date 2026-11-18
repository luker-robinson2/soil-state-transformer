"""KSSL Lab Data Mart → canonical label table (EXECUTION-PLAN step 1.2).

Input: the two CSVs written by ``data/extraction/kssl_export.R`` (sites + layers).
Output: ``data/processed/labels_kssl.parquet`` in the schema of ``schema.py``.

Method handling follows the KSSL method codes, not the column names: the
``organic_carbon_walkley_black`` column also holds dry-combustion results
(6A2*), so each value is classified by its own method code.

    python -m geosoil.corpus.kssl --raw data/raw/kssl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import common

SOURCE = "kssl_ldm"
LICENSE = "public domain"
CACO3_C_FRACTION = common.CACO3_C_FRACTION
# Without a carbonate measurement, total C is only trusted as organic below this pH.
CARBONATE_FREE_PH = 7.0


def _family_carbon(code) -> str:
    code = "" if pd.isna(code) else str(code)
    if code.startswith("6A1"):
        return "walkley_black"
    if code.startswith("6A2") or code.startswith("4H2"):
        return "dry_combustion"
    return "unknown"


def _family_texture(code) -> str:
    code = "" if pd.isna(code) else str(code)
    if code.startswith("3A1") or code.startswith("PSDA"):
        return "pipette"
    if code.lower().startswith("hyd"):
        return "hydrometer"
    return "unknown"


def _method_id(code) -> str:
    return "kssl/" + ("NA" if pd.isna(code) or str(code) == "" else str(code))


def sites_table(sites: pd.DataFrame) -> pd.DataFrame:
    """One row per pedon with the best available coordinates and year."""
    out = pd.DataFrame({"site_key": sites["pedon_key"]})
    std_ok = sites["latitude_std_decimal_degrees"].notna() & sites["longitude_std_decimal_degrees"].notna()
    out["lat"] = np.where(std_ok, sites["latitude_std_decimal_degrees"], sites["latitude_decimal_degrees"])
    out["lon"] = np.where(std_ok, sites["longitude_std_decimal_degrees"], sites["longitude_decimal_degrees"])
    out["coord_source"] = np.where(std_ok, "latitude_std_decimal_degrees", "latitude_decimal_degrees")
    ok = (out["lat"].between(-90, 90) & out["lon"].between(-180, 180)
          & ~((out["lat"] == 0) & (out["lon"] == 0)))
    out.loc[~ok, ["lat", "lon"]] = np.nan
    out.loc[out["lat"].isna(), "coord_source"] = pd.NA
    year = pd.to_datetime(sites["site_obsdate"], errors="coerce", utc=True).dt.year.astype("Int64")
    out["year"] = year.where(year.between(1900, pd.Timestamp.now().year))  # some records carry typo years
    return out


def _rows(layers: pd.DataFrame, target: str, value: pd.Series, code: pd.Series, family) -> pd.DataFrame:
    base = pd.DataFrame({"site_key": layers["pedon_key"], "layer_key": layers["layer_key"],
                         "depth_top": layers["hzn_top"], "depth_bot": layers["hzn_bot"]})
    return common.long_rows(base, target, value, code.map(_method_id), code.map(family) if callable(family) else family)


def layer_values(layers: pd.DataFrame) -> pd.DataFrame:
    """Long table of every target value per layer, one row per method."""
    L = layers[layers["prep_code"].isna() | (layers["prep_code"] == "S")].copy()
    parts = []

    # SOC from total carbon (dry combustion) minus carbonate carbon, in g/kg.
    caco3 = L["caco3_lt_2_mm"]
    carbonate_known = caco3.notna() | (L["ph_h2o"] < CARBONATE_FREE_PH)
    soc_dc = (L["total_carbon_ncs"] - CACO3_C_FRACTION * caco3.fillna(0.0)) * 10.0
    soc_dc = soc_dc.where(carbonate_known & L["total_carbon_ncs"].notna())
    code_dc = L["total_carbon_ncs_method"].astype("string").fillna("total_c") + "-caco3"
    parts.append(_rows(L, "soc", soc_dc, code_dc, "dry_combustion"))

    # Organic carbon reported directly (Walkley-Black or legacy dry combustion), in g/kg.
    oc = L["organic_carbon_walkley_black"] * 10.0
    parts.append(_rows(L, "soc", oc, L["oc_walkley_black_method"], _family_carbon))

    parts.append(_rows(L, "ph", L["ph_h2o"], "h2o:" + L["ph_h2o_method"].astype("string").fillna("NA"), "ph_h2o"))
    parts.append(_rows(L, "ph", L["ph_cacl2"], "cacl2:" + L["ph_cacl2_method"].astype("string").fillna("NA"), "ph_cacl2"))
    parts.append(_rows(L, "sand", L["sand_total"], L["particle_size_method"], _family_texture))
    parts.append(_rows(L, "clay", L["clay_total"], L["particle_size_method"], _family_texture))
    parts.append(_rows(L, "bd", L["bulk_density_third_bar"],
                       "third_bar:" + L["bulk_density_third_bar_method"].astype("string").fillna("NA"), "bd_third_bar"))
    parts.append(_rows(L, "bd", L["bulk_density_oven_dry"],
                       "oven_dry:" + L["bulk_density_oven_dry_method"].astype("string").fillna("NA"), "bd_oven_dry"))
    # CEC and exchangeable bases (cmol(+)/kg). NH4OAc pH 7 is canonical; CEC at pH 8.2 (sum of cations) and
    # ECEC are kept as their own families so harmonize.py can fit conversions on layers measured both ways.
    if "cec_nh4_ph_7" in L:
        parts.append(_rows(L, "cec", L["cec_nh4_ph_7"],
                           "nh4oac7:" + L["cec_nh4_ph_7_method"].astype("string").fillna("NA"), "cec_ph7"))
        code = lambda c: pd.Series(c, index=L.index, dtype="string")
        parts.append(_rows(L, "cec", L["sum_of_cations_cec_pH_8_2"], code("sum_cations_8.2"), "cec_sum_cations_ph82"))
        parts.append(_rows(L, "cec", L["ecec_base_plus_aluminum"], code("ecec"), "ecec"))
        for el in ("ca", "mg", "k"):
            parts.append(_rows(L, el, L[f"{el}_nh4_ph_7"],
                               "nh4oac7:" + L[f"{el}_nh4_ph_7_method"].astype("string").fillna("NA"), "nh4oac_ph7"))
        # Mehlich-3 K (mg/kg) -> cmol(+)/kg
        parts.append(_rows(L, "k", L["potassium_mehlich3_extractable"] / 391.0, code("mehlich3"), "mehlich3"))
    return pd.concat(parts, ignore_index=True)


def build(sites: pd.DataFrame, layers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the canonical label table and a count of what was dropped and why."""
    return common.finalize(layer_values(layers), sites_table(sites),
                           source=SOURCE, prefix="kssl", license=LICENSE)


MIRROR = "https://github.com/brownag/labtaxa/releases/download/2026.07/cached-LDM-SPC.rds"


def load(raw: Path = common.RAW / "kssl") -> tuple[pd.DataFrame, dict]:
    """Read the exported CSVs, creating them from the labtaxa mirror if needed (needs Rscript)."""
    stale = (raw / "kssl_layers.csv.gz").exists() and "cec_nh4_ph_7" not in \
        pd.read_csv(raw / "kssl_layers.csv.gz", nrows=0).columns       # exported before CEC/bases were added
    if stale or not (raw / "kssl_layers.csv.gz").exists():
        import subprocess
        rds = common.fetch(MIRROR, raw / "cached-LDM-SPC.rds")
        subprocess.run(["Rscript", str(C.REPO / "data" / "extraction" / "kssl_export.R"), str(rds), str(raw)],
                       check=True)
    return build(pd.read_csv(raw / "kssl_sites.csv.gz", low_memory=False),
                 pd.read_csv(raw / "kssl_layers.csv.gz", low_memory=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=C.REPO / "data" / "raw" / "kssl")
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / "labels_kssl.parquet")
    a = ap.parse_args()
    common.write(*load(a.raw), a.out)


if __name__ == "__main__":
    main()
