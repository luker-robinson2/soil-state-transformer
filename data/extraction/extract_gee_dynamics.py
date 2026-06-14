"""Dynamics & micro-event modalities — what makes the latent capture *field
alterations* and *micro-events* rather than only static soil state:

  - USDA Cropland Data Layer (CDL) crop sequence 2018-2022  -> rotation / management change
  - SMAP soil moisture, monthly 2022 (12-step sequence)     -> wetting/drying dynamics
  - CHIRPS daily precipitation -> event features (totals, wet days, max-1-day, p95, dry spell)

Writes data/processed/gee_dynamics.parquet (auto-joined by geosoil/data.py).
Usage: .venv/bin/python -m data.extraction.extract_gee_dynamics --project geo-soil-foundational-model
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MASTER = REPO / "data" / "processed" / "master.parquet"
OUTDIR = REPO / "data" / "processed"
CDL_YEARS = [2018, 2019, 2020, 2021, 2022]


def smap_monthly(ee, year=2022):
    ic = ee.ImageCollection("NASA/SMAP/SPL4SMGP/007").select("sm_surface")
    imgs = []
    for m in range(12):
        start = ee.Date.fromYMD(year, m + 1, 1)
        imgs.append(ic.filterDate(start, start.advance(1, "month")).mean().rename(f"sm_{m:02d}"))
    return ee.Image.cat(imgs)


def precip_events(ee, year=2022):
    ic = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation").filterDate(f"{year}-01-01", f"{year}-12-31")
    total = ic.sum().rename("precip_total")
    wet = ic.map(lambda im: im.gt(1.0)).sum().rename("precip_wetdays")
    mx = ic.max().rename("precip_max1d")
    p95 = ic.reduce(ee.Reducer.percentile([95])).rename("precip_p95")
    return ee.Image.cat([total, wet, mx, p95])


def cdl_sequence(ee):
    imgs = []
    for y in CDL_YEARS:
        c = ee.Image(f"USDA/NASS/CDL/{y}").select("cropland")   # CDL is addressable by year id
        imgs.append(c.rename(f"cdl_{y}"))
    return ee.Image.cat(imgs)


def sample_image(ee, image, fc, scale):
    out = image.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=scale)
    return [f["properties"] for f in out.getInfo()["features"]]


def main():
    import ee
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--chunk", type=int, default=400)
    args = ap.parse_args()
    try:
        import google.auth
        creds, _ = google.auth.default(scopes=[
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform"])
        ee.Initialize(creds, project=args.project)
    except Exception:
        ee.Initialize(project=args.project)

    pts = pd.read_parquet(MASTER)[["lon", "lat"]].drop_duplicates().reset_index(drop=True)
    print(f"{len(pts)} points; SMAP soil moisture + CHIRPS events + CDL rotation")
    sm = smap_monthly(ee)
    pe = precip_events(ee)
    cdl = cdl_sequence(ee)

    merged = {}
    pts_list = list(pts.itertuples(index=False))
    for ci in range(0, len(pts_list), args.chunk):
        chunk = pts_list[ci:ci + args.chunk]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {"lon": float(lon), "lat": float(lat)})
            for lon, lat in chunk])
        t0 = time.time()
        for im, sc in ((sm, 11000), (pe, 5000), (cdl, 30)):
            for r in sample_image(ee, im, fc, sc):
                merged.setdefault((r["lon"], r["lat"]), {"lon": r["lon"], "lat": r["lat"]}).update(r)
        print(f"  chunk {ci // args.chunk}: {len(chunk)} pts in {time.time()-t0:.1f}s")

    df = pd.DataFrame(list(merged.values()))
    # derive rotation features from the CDL sequence
    cdl_cols = [f"cdl_{y}" for y in CDL_YEARS]
    present = [c for c in cdl_cols if c in df]
    if present:
        seq = df[present]
        df["cdl_n_crops"] = seq.nunique(axis=1)                       # rotation diversity
        df["cdl_changes"] = (seq.diff(axis=1) != 0).iloc[:, 1:].sum(axis=1)  # # of year-to-year changes
        # fractions of the major commodity crops (1=corn, 5=soy, 24=winter wheat)
        for code, name in ((1, "corn"), (5, "soy"), (24, "wwheat")):
            df[f"cdl_frac_{name}"] = (seq == code).mean(axis=1)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTDIR / "gee_dynamics.parquet", index=False)
    print(f"wrote gee_dynamics.parquet {df.shape}")


if __name__ == "__main__":
    main()
