"""Soil-sensing remote modalities from GEE — the layers most likely to carry the
texture/BD signal AlphaEarth misses:
  - bare-soil Sentinel-2 composite (vegetation-masked multi-year median reflectance)
  - Sentinel-1 SAR (VV/VH annual median + ratio) — roughness/moisture/texture sensitive

Writes data/processed/gee_soil.parquet (auto-joined by geosoil/data.py).
Usage: .venv/bin/python -m data.extraction.extract_gee_soil --project geo-soil-foundational-model
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MASTER = REPO / "data" / "processed" / "master.parquet"
OUTDIR = REPO / "data" / "processed"

BS_BANDS = ["B2", "B4", "B6", "B8", "B11", "B12"]


def bare_soil_composite(ee, years=("2022", "2022")):
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(f"{years[0]}-01-01", f"{years[1]}-12-31")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)))

    def mask_veg(img):
        ndvi = img.normalizedDifference(["B8", "B4"])
        nbr2 = img.normalizedDifference(["B11", "B12"])
        bare = ndvi.lt(0.25).And(nbr2.lt(0.075))            # bare-soil pixels (Demattê et al.)
        scl = img.select("SCL")
        clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.updateMask(bare.And(clear))

    comp = s2.map(mask_veg).select(BS_BANDS).median().multiply(1e-4)
    return comp.rename([f"BS_{b}" for b in BS_BANDS])


def sar_composite(ee, year="2022"):
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterDate(f"{year}-01-01", f"{year}-12-31")
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    vv = s1.select("VV").median().rename("S1_VV")
    vh = s1.select("VH").median().rename("S1_VH")
    ratio = vv.subtract(vh).rename("S1_ratio")              # dB difference
    return ee.Image.cat([vv, vh, ratio])


def sample_image(ee, image, fc, scale):
    out = image.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=scale)
    return [f["properties"] for f in out.getInfo()["features"]]


def main():
    import ee
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--chunk", type=int, default=200)
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
    print(f"{len(pts)} points; building bare-soil S2 (100m) + S1 SAR (30m) composites")
    bs = bare_soil_composite(ee)
    sar = sar_composite(ee)

    merged = {}
    pts_list = list(pts.itertuples(index=False))
    for ci in range(0, len(pts_list), args.chunk):
        chunk = pts_list[ci:ci + args.chunk]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {"lon": float(lon), "lat": float(lat)})
            for lon, lat in chunk])
        t0 = time.time()
        # SAR is cheap and reliable — always sample it
        for r in sample_image(ee, sar, fc, 30):
            merged.setdefault((r["lon"], r["lat"]), {"lon": r["lon"], "lat": r["lat"]}).update(r)
        # bare-soil S2 is memory-heavy server-side; attempt it but never let it
        # kill the run (skip the chunk's bare-soil on a memory error)
        try:
            for r in sample_image(ee, bs, fc, 250):
                merged.setdefault((r["lon"], r["lat"]), {"lon": r["lon"], "lat": r["lat"]}).update(r)
            bs_ok = "+BS"
        except Exception as e:
            bs_ok = f"(BS skipped: {str(e)[:40]})"
        print(f"  chunk {ci // args.chunk}: {len(chunk)} pts SAR {bs_ok} in {time.time()-t0:.1f}s")

    df = pd.DataFrame(list(merged.values()))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTDIR / "gee_soil.parquet", index=False)
    print(f"wrote gee_soil.parquet {df.shape}")


if __name__ == "__main__":
    main()
