"""
02_extract_s2_features.py
=========================
Extract a compact 10-feature Sentinel-2 stack at the same lon/lat as
soil_unified.csv. We use a simple annual-median composite (Cloud Score+
masked) — this is the lightest defensible S2 baseline and stays under
GEE memory limits when chunked properly.

Features (10):
  Annual-median spectral bands: B2, B4, B6, B8, B11, B12  (6)
  Annual-median indices:        NDVI, EVI, NDWI, NBR        (4)

We drop the bare-soil-composite from the literature recipe because it
exceeds GEE per-call memory limits when sampled across CONUS. The
analysis Rmd notes this as a limitation; bare-soil compositing remains
a common step in soil-property mapping pipelines.
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
CHUNK = 50
HERE = os.path.dirname(__file__)
IN_CSV  = os.path.join(HERE, "..", "data", "soil_unified.csv")
OUT_CSV = os.path.join(HERE, "..", "data", "s2_features.csv")

BAND_NAMES = ["B2","B4","B6","B8","B11","B12"]
INDEX_NAMES = ["NDVI","EVI","NDWI","NBR"]
ALL_NAMES = BAND_NAMES + INDEX_NAMES
assert len(ALL_NAMES) == 10


def init_ee() -> None:
    creds, _ = google.auth.default()
    ee.Initialize(creds, project=PROJECT_ID)


def scale_refl(img: ee.Image) -> ee.Image:
    bands = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
    return ee.Image(img.select(bands).divide(10000)).addBands(img.select(["cs"]))


def mask_clouds(img: ee.Image) -> ee.Image:
    return img.updateMask(img.select("cs").gte(0.6))


def s2_year_collection(year: int, geom: ee.Geometry) -> ee.ImageCollection:
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
    )
    csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = s2.linkCollection(csp, ["cs"]).map(scale_refl).map(mask_clouds)
    return s2


def build_median_image(coll: ee.ImageCollection) -> ee.Image:
    """Median of bands then derive indices from the median image."""
    bands = coll.select(BAND_NAMES).median()
    B2 = bands.select("B2")
    B4 = bands.select("B4")
    B8 = bands.select("B8")
    B11 = bands.select("B11")
    B12 = bands.select("B12")
    ndvi = B8.subtract(B4).divide(B8.add(B4)).rename("NDVI")
    evi  = B8.subtract(B4).multiply(2.5).divide(
              B8.add(B4.multiply(6)).subtract(B2.multiply(7.5)).add(1)).rename("EVI")
    ndwi = B8.subtract(B11).divide(B8.add(B11)).rename("NDWI")
    nbr  = B8.subtract(B12).divide(B8.add(B12)).rename("NBR")
    return bands.addBands([ndvi, evi, ndwi, nbr])


def chunk_bbox(sub: pd.DataFrame, pad: float = 0.2) -> ee.Geometry:
    minx, miny = sub["lon"].min(), sub["lat"].min()
    maxx, maxy = sub["lon"].max(), sub["lat"].max()
    return ee.Geometry.Rectangle([minx - pad, miny - pad, maxx + pad, maxy + pad])


def main() -> int:
    init_ee()
    print("EE initialized.", flush=True)
    df = pd.read_csv(IN_CSV)
    n = len(df)
    print(f"Loaded {n} points from {IN_CSV}", flush=True)
    df["orig_row"] = df.index.astype(int)
    df = df.sort_values(["lat", "lon"]).reset_index(drop=True)
    print(f"Sampling {len(ALL_NAMES)} S2 features per point, CHUNK={CHUNK}, location-sorted.", flush=True)

    rows: list[dict | None] = [None] * n
    for start in range(0, n, CHUNK):
        sub = df.iloc[start:start + CHUNK]
        bbox = chunk_bbox(sub, pad=0.2)
        coll = s2_year_collection(YEAR, bbox)
        stack = build_median_image(coll)

        feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                            {"row_id": int(r.orig_row)})
                 for r in sub.itertuples()]
        fc = ee.FeatureCollection(feats)

        t0 = time.time()
        try:
            out = stack.sampleRegions(collection=fc, scale=20, geometries=False, tileScale=4).getInfo()["features"]
        except ee.ee_exception.EEException as e:
            print(f"  chunk {start}-{start+CHUNK}: EE '{e}', retry...", flush=True)
            time.sleep(5)
            try:
                out = stack.sampleRegions(collection=fc, scale=20, geometries=False, tileScale=4).getInfo()["features"]
            except ee.ee_exception.EEException as e2:
                print(f"  chunk {start}-{start+CHUNK}: EE '{e2}', skipping.", flush=True)
                continue

        for f in out:
            props = f.get("properties") or {}
            rid = props.get("row_id")
            if rid is None:
                continue
            row = {"lon": float(df[df["orig_row"] == rid].iloc[0]["lon"]),
                   "lat": float(df[df["orig_row"] == rid].iloc[0]["lat"])}
            for nm in ALL_NAMES:
                row[nm] = props.get(nm)
            rows[rid] = row
        valid = sum(1 for r in rows if r is not None)
        print(f"  chunk {start:>5}-{start+CHUNK:<5}  valid {valid:>4}/{n}  {time.time() - t0:5.1f}s", flush=True)

    rows_out = []
    for i in range(n):
        r = rows[i]
        if r is None:
            r = {"lon": None, "lat": None}
            r.update({nm: None for nm in ALL_NAMES})
        rows_out.append(r)
    fieldnames = ["lon", "lat"] + ALL_NAMES
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    print(f"Writing {len(rows_out)} rows -> {OUT_CSV}", flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
