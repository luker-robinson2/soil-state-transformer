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
from . import schema

SOURCE = "kssl_ldm"
LICENSE = "public domain"
# Carbon in CaCO3 is 12/100 of its mass; KSSL SOC = total C - 0.12 * CaCO3 (%).
CACO3_C_FRACTION = 0.12
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
    out = pd.DataFrame({"pedon_key": sites["pedon_key"]})
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
    keep = value.notna()
    fam = code[keep].map(family) if callable(family) else family
    return pd.DataFrame({
        "pedon_key": layers.loc[keep, "pedon_key"],
        "layer_key": layers.loc[keep, "layer_key"],
        "depth_top": layers.loc[keep, "hzn_top"],
        "depth_bot": layers.loc[keep, "hzn_bot"],
        "target": target,
        "value": value[keep],
        "method_id": code[keep].map(_method_id),
        "method_family": fam,
    })


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

    parts.append(_rows(L, "ph", L["ph_h2o"], "h2o:" + L["ph_h2o_method"].astype("string").fillna("NA"), "ph_h2o_1to1"))
    parts.append(_rows(L, "ph", L["ph_cacl2"], "cacl2:" + L["ph_cacl2_method"].astype("string").fillna("NA"), "ph_cacl2_0.01M"))
    parts.append(_rows(L, "sand", L["sand_total"], L["particle_size_method"], _family_texture))
    parts.append(_rows(L, "clay", L["clay_total"], L["particle_size_method"], _family_texture))
    parts.append(_rows(L, "bd", L["bulk_density_third_bar"],
                       "third_bar:" + L["bulk_density_third_bar_method"].astype("string").fillna("NA"), "bd_third_bar"))
    parts.append(_rows(L, "bd", L["bulk_density_oven_dry"],
                       "oven_dry:" + L["bulk_density_oven_dry_method"].astype("string").fillna("NA"), "bd_oven_dry"))
    return pd.concat(parts, ignore_index=True)


def build(sites: pd.DataFrame, layers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the canonical label table and a count of what was dropped and why."""
    vals = layer_values(layers)
    n0 = len(vals)
    drops = {}

    bad_depth = vals["depth_top"].isna() | vals["depth_bot"].isna() | (vals["depth_bot"] <= vals["depth_top"])
    drops["bad_depth"] = int(bad_depth.sum())
    vals = vals[~bad_depth]

    in_range = pd.Series(True, index=vals.index)
    for t, (lo, hi) in schema.VALID_RANGE.items():
        m = vals["target"] == t
        in_range &= ~m | vals["value"].between(lo, hi)
    drops["out_of_range"] = int((~in_range).sum())
    vals = vals[in_range]

    site = sites_table(sites)
    vals = vals.merge(site, on="pedon_key", how="left")
    no_coord = vals["lat"].isna()
    drops["no_coordinates"] = int(no_coord.sum())
    vals = vals[~no_coord]

    vals = vals.drop_duplicates(["layer_key", "target", "method_id"])
    out = pd.DataFrame({
        "site_id": "kssl:" + vals["pedon_key"].astype(str),
        "layer_id": "kssl:" + vals["layer_key"].astype(str),
        "source": SOURCE,
        "lon": vals["lon"],
        "lat": vals["lat"],
        "coord_source": vals["coord_source"],
        "coord_precision_m": np.nan,
        "depth_top": vals["depth_top"],
        "depth_bot": vals["depth_bot"],
        "target": vals["target"],
        "value": vals["value"],
        "unit": vals["target"].map(C.TARGET_UNITS),
        "method_id": vals["method_id"],
        "method_family": vals["method_family"],
        "year": vals["year"],
        "license": LICENSE,
    })
    out = schema.conform(out)
    schema.validate(out)
    drops["kept"] = len(out)
    drops["input_values"] = n0
    return out, drops


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=C.REPO / "data" / "raw" / "kssl")
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / "labels_kssl.parquet")
    a = ap.parse_args()
    sites = pd.read_csv(a.raw / "kssl_sites.csv.gz", low_memory=False)
    layers = pd.read_csv(a.raw / "kssl_layers.csv.gz", low_memory=False)
    labels, drops = build(sites, layers)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(a.out, index=False)
    print(f"{a.out}: {len(labels):,} values, {labels['site_id'].nunique():,} sites")
    print("dropped:", drops)
    print(labels.groupby(["target", "method_family"]).size().to_string())


if __name__ == "__main__":
    main()
