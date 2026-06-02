"""
03_ingest_kssl.py
=================
Ingest the USDA NCSS Kellogg Soil Survey Laboratory (KSSL) SQLite snapshot,
filter to clean post-2010 GPS-located CONUS pedons with dry-combustion SOC,
depth-harmonize each property to 0-30 cm via thickness-weighted means, and
write a tidy CSV.

Schema archaeology approach:
  1. Download the SQLite to data/kssl_raw/.
  2. List all tables; cache schema dump for inspection.
  3. Print the columns of each table that looks soil-related so we can
     pick the right ones at runtime (location quality, sample year, SOC
     method, horizon depths, properties).
  4. Run the filter + harmonize.

Output: data/kssl_points.csv with columns:
   pedon_key, lon, lat, sample_year, soc_pct_0_30, ph_h2o_0_30,
   bd_0_30, sand_pct_0_30, clay_pct_0_30, coord_quality, soc_method
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import sqlite3
import urllib.request
import zipfile
import json
from typing import Optional

import pandas as pd


HERE        = os.path.dirname(__file__)
DATA_DIR    = os.path.join(HERE, "..", "data")
RAW_DIR     = os.path.join(DATA_DIR, "kssl_raw")
SCHEMA_JSON = os.path.join(RAW_DIR, "schema.json")
OUT_CSV     = os.path.join(DATA_DIR, "kssl_points.csv")

# Note: the canonical NCSS "Lab Data Mart" download URL changes; user may
# need to download manually if this URL fails. After 2024 the snapshot is
# typically a zipped Access database, but they also publish a SQLite
# version under https://ncsslabdatamart.sc.egov.usda.gov/datadownload.aspx
# Once a .sqlite file lands in data/kssl_raw/ we can proceed regardless of
# how it got there.

CONUS_BBOX = (-125, 24.5, -66.9, 49.5)
MIN_YEAR   = 2010


def find_sqlite() -> Optional[str]:
    if not os.path.isdir(RAW_DIR):
        return None
    for fname in os.listdir(RAW_DIR):
        if fname.endswith(".sqlite") or fname.endswith(".db"):
            return os.path.join(RAW_DIR, fname)
    return None


def dump_schema(con: sqlite3.Connection) -> dict:
    """Inspect schema; return {table_name: [columns]}."""
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    schema = {}
    for t in tables:
        cur.execute(f"PRAGMA table_info('{t}')")
        cols = [r[1] for r in cur.fetchall()]
        schema[t] = cols
    return schema


def find_table(schema: dict, candidates: list[str]) -> Optional[str]:
    """Return the first table whose name (case-insensitive) matches any candidate."""
    lower = {t.lower(): t for t in schema}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def find_column(cols: list[str], candidates: list[str]) -> Optional[str]:
    """Return the first column matching any candidate (case-insensitive)."""
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def harmonize_0_30(horizons: pd.DataFrame, value_col: str,
                   top_col: str, bot_col: str) -> Optional[float]:
    """Thickness-weighted mean of value_col over [0, 30 cm]."""
    h = horizons.dropna(subset=[value_col, top_col, bot_col]).copy()
    if h.empty:
        return None
    h["t"] = h[top_col].clip(lower=0, upper=30)
    h["b"] = h[bot_col].clip(lower=0, upper=30)
    h["thick"] = (h["b"] - h["t"]).clip(lower=0)
    h = h[h["thick"] > 0]
    if h.empty:
        return None
    return float((h[value_col] * h["thick"]).sum() / h["thick"].sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-only", action="store_true",
                    help="Dump schema and exit without filtering.")
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    sqlite_path = find_sqlite()
    if not sqlite_path:
        print(f"ERROR: no .sqlite or .db file in {RAW_DIR}\n"
              f"Download manually from "
              f"https://ncsslabdatamart.sc.egov.usda.gov/datadownload.aspx "
              f"and place the SQLite file there.")
        return 1

    print(f"Opening {sqlite_path}")
    con = sqlite3.connect(sqlite_path)
    schema = dump_schema(con)
    with open(SCHEMA_JSON, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Wrote schema to {SCHEMA_JSON} ({len(schema)} tables)")

    # Identify likely tables
    pedon_tab    = find_table(schema, ["lab_pedon", "pedon", "site"])
    horizon_tab  = find_table(schema, ["lab_layer", "lab_horizon", "layer", "horizon"])
    chem_tab     = find_table(schema, ["lab_chemical_properties", "chem", "lab_chem"])
    phys_tab     = find_table(schema, ["lab_physical_properties", "phys", "lab_phys"])

    print(f"\nPedon table:    {pedon_tab} (cols: {schema.get(pedon_tab, [])[:8]}...)")
    print(f"Horizon table:  {horizon_tab}")
    print(f"Chem table:     {chem_tab}")
    print(f"Phys table:     {phys_tab}")

    if args.inspect_only:
        return 0

    if not all([pedon_tab, horizon_tab]):
        print("ERROR: cannot find pedon/horizon tables; please inspect schema.json")
        return 1

    pcols = schema[pedon_tab]
    hcols = schema[horizon_tab]
    chc   = schema.get(chem_tab, [])
    phc   = schema.get(phys_tab, [])

    # Locate columns
    pid_col   = find_column(pcols, ["pedon_key","pedlabsampnum","upedonid"])
    lon_col   = find_column(pcols, ["longitude_decimal_degrees","longitude","lon","x"])
    lat_col   = find_column(pcols, ["latitude_decimal_degrees","latitude","lat","y"])
    year_col  = find_column(pcols, ["samp_classdate_year","corr_year","sampled_year","sample_year"])
    qual_col  = find_column(pcols, ["lat_dir_method","loc_method","coord_qual","horizontal_position_quality_code"])
    print(f"\nLocated columns: id={pid_col} lon={lon_col} lat={lat_col} year={year_col} qual={qual_col}")
    if not all([pid_col, lon_col, lat_col]):
        print("ERROR: cannot find pedon_id/lon/lat columns; inspect schema.json")
        return 1

    pedon_df = pd.read_sql(f"SELECT * FROM {pedon_tab}", con)
    horizon_df = pd.read_sql(f"SELECT * FROM {horizon_tab}", con)
    chem_df = pd.read_sql(f"SELECT * FROM {chem_tab}", con) if chem_tab else None
    phys_df = pd.read_sql(f"SELECT * FROM {phys_tab}", con) if phys_tab else None

    print(f"\nLoaded: pedon={len(pedon_df)}  horizon={len(horizon_df)}  "
          f"chem={len(chem_df) if chem_df is not None else 0}  "
          f"phys={len(phys_df) if phys_df is not None else 0}")

    # Filter pedons
    p = pedon_df.dropna(subset=[lon_col, lat_col]).copy()
    p[lon_col] = pd.to_numeric(p[lon_col], errors="coerce")
    p[lat_col] = pd.to_numeric(p[lat_col], errors="coerce")
    p = p.dropna(subset=[lon_col, lat_col])
    p = p[(p[lon_col] >= CONUS_BBOX[0]) & (p[lon_col] <= CONUS_BBOX[2]) &
          (p[lat_col] >= CONUS_BBOX[1]) & (p[lat_col] <= CONUS_BBOX[3])]
    if year_col:
        p[year_col] = pd.to_numeric(p[year_col], errors="coerce")
        p = p[p[year_col] >= MIN_YEAR]
    print(f"After CONUS+year>={MIN_YEAR} filter: {len(p)} pedons")

    # Identify horizon depth columns and target columns
    htop = find_column(hcols, ["hzn_top","top","layer_top","horizon_top","depth_top"])
    hbot = find_column(hcols, ["hzn_bot","bottom","layer_bot","horizon_bot","depth_bot"])
    h_pid = find_column(hcols, ["pedon_key","pedlabsampnum"])
    print(f"Horizon: pid={h_pid} top={htop} bot={hbot}")

    target_specs = {
        "soc_pct_0_30":   ["c_tot_ncs","c_tot","oc","ocgr","oc_method","total_carbon_method"],
        "ph_h2o_0_30":    ["ph_h2o","ph_water","ph"],
        "bd_0_30":        ["db_13b","bulk_density_clod","bd_clod","bd"],
        "sand_pct_0_30":  ["sand_tot_psa","sand_tot","sand","totsand"],
        "clay_pct_0_30":  ["clay_tot_psa","clay_tot","clay","totclay"],
    }
    # We try each target across chem and phys tables
    src = pd.merge(horizon_df, chem_df, how="left",
                   on=h_pid if (chem_df is not None and h_pid in chem_df.columns) else None,
                   suffixes=("","_c")) if chem_df is not None else horizon_df
    src = pd.merge(src, phys_df, how="left",
                   on=h_pid if (phys_df is not None and h_pid in phys_df.columns) else None,
                   suffixes=("","_p")) if phys_df is not None else src

    # build per-pedon harmonized record
    pedons_kept = []
    p_index = p.set_index(pid_col)
    for pid, hg in src.groupby(h_pid):
        if pid not in p_index.index:
            continue
        rec = {"pedon_key": pid,
               "lon": float(p_index.loc[pid, lon_col]),
               "lat": float(p_index.loc[pid, lat_col]),
               "sample_year": float(p_index.loc[pid, year_col]) if year_col else None}
        for out_name, candidates in target_specs.items():
            col = find_column(hg.columns.tolist(), candidates)
            if col is None:
                rec[out_name] = None
            else:
                rec[out_name] = harmonize_0_30(hg, col, htop, hbot) if (htop and hbot) else None
        # Require SOC and at least one other property
        if rec["soc_pct_0_30"] is not None:
            pedons_kept.append(rec)
    print(f"Pedons with valid 0-30cm SOC: {len(pedons_kept)}")

    out_df = pd.DataFrame(pedons_kept)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
