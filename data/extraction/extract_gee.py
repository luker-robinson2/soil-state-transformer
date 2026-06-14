"""Extract temporal + terrain covariates from Google Earth Engine at every point
in the master table, writing parquet files that geosoil/data.py auto-joins.

Outputs:
  data/processed/gee_temporal.parquet  -> ERA5 climate (12mo) + MODIS veg (12mo)
  data/processed/gee_static.parquet    -> Copernicus DEM terrain

Usage (after creating your own Earth-Engine-enabled GCP project):
  .venv/bin/earthengine authenticate
  .venv/bin/python -m data.extraction.extract_gee --project YOUR_GEE_PROJECT --year 2022

This is the Phase-0 step that unlocks the temporal modalities (and therefore the
Mamba branch) and richer signal. Written to match geosoil/config column names.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MASTER = REPO / "data" / "processed" / "master.parquet"
OUTDIR = REPO / "data" / "processed"

# config column names (kept in sync with geosoil/config.py)
ERA5_BANDS = {  # source band -> short name
    "temperature_2m": "t2m",
    "total_precipitation_sum": "tprate",
    "soil_temperature_level_1": "stl1",
    "volumetric_soil_water_layer_1": "swvl1",
}
VEG_BANDS = {"NDVI": "ndvi", "EVI": "evi"}


def monthly_cat(ic, bands_map, year, scale_factor=1.0):
    import ee
    imgs = []
    for m in range(12):
        start = ee.Date.fromYMD(year, m + 1, 1)
        comp = ic.filterDate(start, start.advance(1, "month")).mean()
        src = list(bands_map.keys())
        dst = [f"{bands_map[b]}_{m:02d}" for b in src]
        img = comp.select(src).rename(dst)
        if scale_factor != 1.0:
            img = img.multiply(scale_factor).rename(dst)
        imgs.append(img)
    return ee.Image.cat(imgs)


def sample_image(ee, image, pts_fc, scale):
    fc = image.reduceRegions(collection=pts_fc, reducer=ee.Reducer.first(), scale=scale)
    rows = []
    for f in fc.getInfo()["features"]:
        p = f["properties"]
        rows.append(p)
    return rows


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    import ee
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="your Earth-Engine-enabled GCP project id")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--chunk", type=int, default=800)
    args = ap.parse_args()
    # Use Application Default Credentials explicitly (avoids a stale
    # ~/.config/earthengine/credentials file shadowing valid ADC).
    try:
        import google.auth
        creds, _ = google.auth.default(scopes=[
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform",
        ])
        ee.Initialize(creds, project=args.project)
    except Exception:
        ee.Initialize(project=args.project)
    print(f"EE initialized on project={args.project}; sampling year {args.year}")

    master = pd.read_parquet(MASTER)
    pts = master[["lon", "lat"]].drop_duplicates().reset_index(drop=True)
    print(f"{len(pts)} unique points")

    # Build images
    era5 = monthly_cat(ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR"), ERA5_BANDS, args.year)
    modis = monthly_cat(ee.ImageCollection("MODIS/061/MOD13Q1"), VEG_BANDS, args.year, scale_factor=1e-4)
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().rename("elevation")
    terr = ee.Terrain.products(dem)            # elevation, slope, aspect, hillshade
    aspect = terr.select("aspect").multiply(3.14159 / 180.0)
    static = ee.Image.cat([
        terr.select("elevation"),
        terr.select("slope"),
        aspect.sin().rename("aspect_sin"),
        aspect.cos().rename("aspect_cos"),
        # slope-based wetness proxy (documented TWI proxy; not flow-accumulated)
        terr.select("slope").multiply(3.14159 / 180.0).tan().add(0.01).pow(-1).log().rename("twi"),
    ])

    temporal_rows, static_rows = [], []
    for ci, chunk in enumerate(chunks(list(pts.itertuples(index=False)), args.chunk)):
        import ee as _ee
        fc = _ee.FeatureCollection([
            _ee.Feature(_ee.Geometry.Point([float(lon), float(lat)]), {"lon": float(lon), "lat": float(lat)})
            for lon, lat in chunk
        ])
        t0 = time.time()
        e_rows = sample_image(ee, era5, fc, 10000)
        m_rows = sample_image(ee, modis, fc, 250)
        s_rows = sample_image(ee, static, fc, 30)
        # merge era5 + modis per point by (lon,lat)
        em = {(r["lon"], r["lat"]): r for r in e_rows}
        for r in m_rows:
            em.setdefault((r["lon"], r["lat"]), {"lon": r["lon"], "lat": r["lat"]}).update(r)
        temporal_rows.extend(em.values())
        static_rows.extend(s_rows)
        print(f"  chunk {ci}: {len(chunk)} pts in {time.time()-t0:.1f}s")

    tdf = pd.DataFrame(temporal_rows)
    sdf = pd.DataFrame(static_rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tdf.to_parquet(OUTDIR / "gee_temporal.parquet", index=False)
    sdf.to_parquet(OUTDIR / "gee_static.parquet", index=False)
    print(f"wrote gee_temporal.parquet {tdf.shape} and gee_static.parquet {sdf.shape}")
    print("Re-run: python -m geosoil.data --build  (to fold these into master.parquet)")


if __name__ == "__main__":
    main()
