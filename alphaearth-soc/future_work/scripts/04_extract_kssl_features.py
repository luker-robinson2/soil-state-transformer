"""
04_extract_kssl_features.py
============================
Pair WoSIS-CONUS pedons (kssl-equivalent ground truth) with AlphaEarth
+ OpenLandMap features. Subsamples to 3000 random points (seed=42) to
keep extraction tractable.

Output: future_work/data/kssl_with_features.csv
"""
from __future__ import annotations

import csv
import os
import sys
import time

import ee
import google.auth
import pandas as pd


PROJECT_ID = "your-gcp-project"
YEAR = 2024
CHUNK = 200
N_SAMPLE = 3000
SEED = 42
HERE = os.path.dirname(__file__)
IN_CSV  = os.path.join(HERE, "..", "data", "wosis_conus.csv")
OUT_CSV = os.path.join(HERE, "..", "data", "kssl_with_features.csv")


def init_ee() -> None:
    creds, _ = google.auth.default()
    ee.Initialize(creds, project=PROJECT_ID)


def build_ae_stack(year: int) -> ee.Image:
    ae = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .mosaic()
    )
    soc_olm  = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02").select(["b0"], ["soc_olm"])
    return ae.addBands(soc_olm)


def main() -> int:
    init_ee()
    print("EE initialized.", flush=True)

    df_full = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df_full):,} WoSIS-CONUS pedons.", flush=True)

    df = df_full.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)
    print(f"Subsampled to {len(df):,} (seed={SEED}).", flush=True)

    stack = build_ae_stack(YEAR)
    band_cols = [f"A{ix:02d}" for ix in range(64)] + ["soc_olm"]

    rows: list[dict | None] = [None] * len(df)
    for start in range(0, len(df), CHUNK):
        sub = df.iloc[start:start + CHUNK]
        feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                            {"row_id": int(r.Index)})
                 for r in sub.itertuples()]
        fc = ee.FeatureCollection(feats)
        t0 = time.time()
        try:
            out = stack.sampleRegions(collection=fc, scale=10, geometries=False, tileScale=4).getInfo()["features"]
        except ee.ee_exception.EEException as e:
            print(f"  chunk {start}-{start+CHUNK}: EE error '{e}', retrying...", flush=True)
            time.sleep(3)
            try:
                out = stack.sampleRegions(collection=fc, scale=10, geometries=False, tileScale=4).getInfo()["features"]
            except ee.ee_exception.EEException as e2:
                print(f"  chunk {start}-{start+CHUNK}: skip ({e2})", flush=True)
                continue
        for f in out:
            props = f.get("properties") or {}
            rid = props.get("row_id")
            if rid is None:
                continue
            base = df.iloc[rid].to_dict()
            row = {**base}
            for col in band_cols:
                row[col] = props.get(col)
            rows[rid] = row
        valid = sum(1 for r in rows if r is not None)
        print(f"  chunk {start:>5}-{start+CHUNK:<5}  valid {valid:>5}/{len(df)}  {time.time() - t0:5.1f}s", flush=True)

    rows_out = [r for r in rows if r is not None]
    fieldnames = list(df.columns) + band_cols
    fieldnames = list(dict.fromkeys(fieldnames))
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    print(f"Writing {len(rows_out)} rows -> {OUT_CSV}", flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
