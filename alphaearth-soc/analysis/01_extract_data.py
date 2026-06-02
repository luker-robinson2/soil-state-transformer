"""
01_extract_data.py
==================
Pull n=3000 random sample points across the contiguous United States and
extract:
  * AlphaEarth Foundations 2024 annual embedding (64 bands A00..A63)
  * OpenLandMap Soil Organic Carbon at 0 cm (g/kg)
  * MODIS IGBP land cover (LC_Type1)
  * Latitude / longitude

Writes the result to soil_samples.csv. Runs sampleRegions in 250-point
chunks to stay under GEE per-call computation limits.

Run from your venv:
    source venv/bin/activate
    python 01_extract_data.py
"""

from __future__ import annotations

import csv
import os
import sys
import time

import ee
import google.auth


PROJECT_ID = "your-gcp-project"
N_TARGET = 3000
N_OVERSAMPLE = 4500
SEED = 42
YEAR = 2024
CHUNK = 250                          # points per sampleRegions call
OUT_CSV = os.path.join(os.path.dirname(__file__), "soil_samples.csv")


def init_ee() -> None:
    creds, _ = google.auth.default()
    ee.Initialize(creds, project=PROJECT_ID)


def conus_geometry() -> ee.Geometry:
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    usa = countries.filter(ee.Filter.eq("country_co", "US")).geometry()
    bbox = ee.Geometry.Rectangle([-125, 24.5, -66.9, 49.5])
    return usa.intersection(bbox, maxError=1000)


def build_stack(year: int) -> ee.Image:
    ae = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .mosaic()
    )
    soc = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02").select(
        ["b0"], ["soc"]
    )
    lc = (
        ee.ImageCollection("MODIS/061/MCD12Q1")
        .filterDate("2022-01-01", "2022-12-31")
        .first()
        .select(["LC_Type1"], ["land_cover"])
    )
    return ae.addBands(soc).addBands(lc)


def sample_chunk(stack: ee.Image, points: ee.FeatureCollection) -> list[dict]:
    sampled = stack.sampleRegions(
        collection=points,
        scale=250,
        geometries=True,
        tileScale=4,
    )
    return sampled.getInfo()["features"]


def main() -> int:
    init_ee()
    print("EE initialized.")

    geom = conus_geometry()
    stack = build_stack(YEAR)

    print(f"Generating {N_OVERSAMPLE} random points (seed={SEED})...")
    all_points = ee.FeatureCollection.randomPoints(region=geom, points=N_OVERSAMPLE, seed=SEED)
    pts_list = all_points.toList(N_OVERSAMPLE)

    band_cols = [f"A{ix:02d}" for ix in range(64)] + ["soc", "land_cover"]
    fieldnames = ["lon", "lat"] + band_cols

    rows: list[dict] = []
    for i in range(0, N_OVERSAMPLE, CHUNK):
        if len(rows) >= N_TARGET:
            break
        sub_fc = ee.FeatureCollection(pts_list.slice(i, i + CHUNK))
        t0 = time.time()
        try:
            feats = sample_chunk(stack, sub_fc)
        except ee.ee_exception.EEException as e:
            print(f"  chunk {i}-{i+CHUNK}: EE error '{e}', retrying once...")
            time.sleep(3)
            feats = sample_chunk(stack, sub_fc)

        kept_before = len(rows)
        for f in feats:
            props = f.get("properties") or {}
            if props.get("soc") is None:
                continue
            geom_coords = (f.get("geometry") or {}).get("coordinates")
            if not geom_coords:
                continue
            if any(props.get(f"A{ix:02d}") is None for ix in range(64)):
                continue
            row = {"lon": geom_coords[0], "lat": geom_coords[1]}
            for col in band_cols:
                row[col] = props.get(col)
            rows.append(row)
            if len(rows) >= N_TARGET:
                break
        print(
            f"  chunk {i:>5}-{i + CHUNK:<5}  raw {len(feats):>3}  "
            f"kept +{len(rows) - kept_before:<3}  total {len(rows):>4}  "
            f"{time.time() - t0:5.1f}s"
        )

    print(f"Writing {len(rows)} rows -> {OUT_CSV}")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
