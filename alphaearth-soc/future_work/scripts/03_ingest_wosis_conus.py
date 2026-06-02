"""
03_ingest_wosis_conus.py
========================
Use the pre-processed WoSIS parquet files from the foundation-model
service (../../foundation-model/data/processed) as ground-truth
soil pedons. Filter to CONUS bounding box and write a tidy CSV with
columns mirroring `kssl_points.csv` for downstream Phase E/F scripts.

WoSIS schema (per Hengl 2022):
  profile_id    (unique)
  latitude / longitude (decimal degrees)
  upper_depth / lower_depth (cm)
  clay, sand, silt (% by mass)
  phaq    pH in water
  orgc    organic carbon (g/kg)
  nitkjd  total N Kjeldahl (g/kg)
  cecph7  CEC at pH 7 (cmol/kg)
  bdfi33  bulk density (g/cm^3)

Output: future_work/data/wosis_conus.csv
  pedon_key, lon, lat, soc_g_kg, ph_h2o, sand_pct, clay_pct, silt_pct,
  bd_g_cm3, depth_interval
"""
from __future__ import annotations

import os
import sys

import pandas as pd


HERE = os.path.dirname(__file__)
SRC_DIR = "../../foundation-model/data/processed"
OUT_CSV = os.path.join(HERE, "..", "data", "wosis_conus.csv")
CONUS_BBOX = (-125, 24.5, -66.9, 49.5)


def main() -> int:
    parts = []
    for split in ("train", "val", "test"):
        p = os.path.join(SRC_DIR, f"wosis_{split}.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p)
            df["split"] = split
            parts.append(df)
            print(f"  loaded {split}: {len(df):,} rows")
    full = pd.concat(parts, ignore_index=True)
    print(f"Total: {len(full):,} rows")

    # CONUS filter
    conus = full[
        (full["longitude"] >= CONUS_BBOX[0]) & (full["longitude"] <= CONUS_BBOX[2]) &
        (full["latitude"]  >= CONUS_BBOX[1]) & (full["latitude"]  <= CONUS_BBOX[3])
    ].copy()
    print(f"CONUS rows: {len(conus):,}")

    # Keep 0-30 cm interval (the file already harmonizes; double-check)
    if "depth_interval" in conus.columns:
        depth_counts = conus["depth_interval"].value_counts()
        print("Depth intervals:"); print(depth_counts)
    conus_0_30 = conus[
        ((conus["upper_depth"] == 0) & (conus["lower_depth"] == 30)) |
        (conus.get("depth_interval", "") == "0-30cm")
    ].copy()
    print(f"After 0-30cm filter: {len(conus_0_30):,}")

    # Require SOC and at least lon/lat
    conus_0_30 = conus_0_30.dropna(subset=["orgc", "longitude", "latitude"])
    print(f"After dropping rows missing SOC/coords: {len(conus_0_30):,}")

    # De-dup by profile_id (one row per profile)
    conus_0_30 = conus_0_30.drop_duplicates(subset=["profile_id"], keep="first")
    print(f"After de-dup by profile_id: {len(conus_0_30):,}")

    out = pd.DataFrame({
        "pedon_key":  conus_0_30["profile_id"].astype(str),
        "lon":        conus_0_30["longitude"],
        "lat":        conus_0_30["latitude"],
        "soc_g_kg":   conus_0_30["orgc"],
        "ph_h2o":     conus_0_30["phaq"],
        "sand_pct":   conus_0_30["sand"],
        "clay_pct":   conus_0_30["clay"],
        "silt_pct":   conus_0_30["silt"],
        "bd_g_cm3":   conus_0_30["bdfi33"],
        "depth_interval": conus_0_30.get("depth_interval", "0-30cm"),
        "split":      conus_0_30["split"],
    })

    print("\nProperty coverage in CONUS 0-30 cm:")
    for c in ["soc_g_kg","ph_h2o","sand_pct","clay_pct","silt_pct","bd_g_cm3"]:
        n = out[c].notna().sum()
        print(f"  {c:12s}  {n:>6} ({100*n/len(out):.1f}%)")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out):,} rows -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
