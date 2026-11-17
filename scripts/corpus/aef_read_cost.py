"""Measure what it costs to read AlphaEarth embeddings from the free Source Cooperative mirror.

EXECUTION-PLAN step 0.6. Compares two access patterns:
  scattered  one pixel at each of N labelled sites (how C1-C3 would be built)
  dense      every pixel of one internal COG block (how C4 would be built)

The COGs are band-interleaved with 1024x1024 zstd blocks, so a single-pixel read
pulls one compressed block per band (64). Bytes are taken exactly from the TIFF
block-size metadata, not from network counters; seconds are wall-clock.

    python scripts/corpus/aef_read_cost.py --points data/processed/labels_kssl.parquet \
        --n 40 --year 2020 --out geosoil/results/aef_read_cost.json
    # --points also takes a CSV (site_id, lon, lat), e.g. scripts/corpus/aef_sample_points.csv

On Alpine, run it as a short `acpu` job; the projection at the end scales the
measured numbers to the C2 / C4 corpus tiers in SCALING-PLAN.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

# Direct S3 HTTPS is ~10-60x faster than the data.source.coop proxy (66 MB/s vs <4 MB/s from
# Alpine, 2026-11-16); pass --base to compare.
S3_BASE = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/tge-labs/aef/v1/annual"
PROXY_BASE = "https://data.source.coop/tge-labs/aef/v1/annual"
BASE = S3_BASE
INDEX_URL = f"{BASE}/aef_index.parquet"
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff", GDAL_HTTP_MAX_RETRY="3")
N_BANDS = 64


def load_index(path: Path) -> pd.DataFrame:
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(INDEX_URL, path)
    cols = ["year", "crs", "utm_zone", "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north",
            "utm_west", "utm_south", "utm_east", "utm_north"]
    idx = pd.read_parquet(path)
    name_col = next(c for c in idx.columns if c.lower() in ("path", "filename", "file", "name", "location", "url"))
    return idx[cols + [name_col]].rename(columns={name_col: "path"})


def tile_url(path: str) -> str:
    """Index rows give a bucket path or URL; normalise to an HTTPS /vsicurl/ path."""
    path = str(path)
    if path.startswith("s3://") or path.startswith("gs://"):
        path = path.split("/aef/v1/annual/", 1)[1]
    elif "/aef/v1/annual/" in path:
        path = path.split("/aef/v1/annual/", 1)[1]
    if path.endswith(".vrt"):
        path = path[:-4] + ".tiff"
    return f"/vsicurl/{BASE}/{path}"


def find_tile(idx: pd.DataFrame, lon: float, lat: float, year: int):
    """The first tile whose UTM pixel bounds contain the point (exact, unlike the WGS84 bbox)."""
    cand = idx[(idx.year == year) & (idx.wgs84_west <= lon) & (idx.wgs84_east >= lon)
               & (idx.wgs84_south <= lat) & (idx.wgs84_north >= lat)]
    for _, t in cand.iterrows():
        x, y = Transformer.from_crs("EPSG:4326", t.crs, always_xy=True).transform(lon, lat)
        if t.utm_west <= x < t.utm_east and t.utm_south <= y < t.utm_north:
            return t, x, y
    return None, None, None


def block_bytes(ds, bx: int, by: int) -> int:
    """Compressed bytes of block (bx, by) summed over all bands."""
    total = 0
    for b in range(1, ds.count + 1):
        v = ds.get_tag_item(f"BLOCK_SIZE_{bx}_{by}", "TIFF", bidx=b)
        total += int(v) if v else 0
    return total


def dequantize(raw: np.ndarray) -> np.ndarray:
    out = ((raw.astype(np.float32) / 127.5) ** 2) * np.sign(raw)
    out[raw == -128] = np.nan
    return out


def measure_scattered(idx, pts: pd.DataFrame, year: int) -> list[dict]:
    rows = []
    for p in pts.itertuples():
        t, x, y = find_tile(idx, p.lon, p.lat, year)
        if t is None:
            rows.append(dict(site_id=p.site_id, found=False))
            continue
        t0 = time.perf_counter()
        with rasterio.open(tile_url(t.path)) as ds:
            r, c = ds.index(x, y)
            bh, bw = ds.block_shapes[0]
            nbytes = block_bytes(ds, c // bw, r // bh)
            raw = ds.read(window=Window(c, r, 1, 1))[:, 0, 0]
        dt = time.perf_counter() - t0
        emb = dequantize(raw)
        rows.append(dict(site_id=p.site_id, found=True, seconds=dt, bytes=nbytes,
                         valid=bool(np.isfinite(emb).all()), norm=float(np.linalg.norm(emb))))
    return rows


def measure_dense(idx, lon: float, lat: float, year: int) -> dict:
    t, x, y = find_tile(idx, lon, lat, year)
    t0 = time.perf_counter()
    with rasterio.open(tile_url(t.path)) as ds:
        r, c = ds.index(x, y)
        bh, bw = ds.block_shapes[0]
        bx, by = c // bw, r // bh
        nbytes = block_bytes(ds, bx, by)
        raw = ds.read(window=Window(bx * bw, by * bh, bw, bh))
    dt = time.perf_counter() - t0
    valid = int((raw[0] != -128).sum())
    return dict(seconds=dt, bytes=nbytes, pixels=bw * bh, valid_pixels=valid,
                bytes_per_valid_pixel=nbytes / max(valid, 1))


def project(scat: pd.DataFrame, dense: dict) -> dict:
    """Scale to SCALING-PLAN tiers: C2 = 2M points x 9 years scattered; C4 = 1e9 pixel-years dense."""
    per_pt_b, per_pt_s = scat.bytes.median(), scat.seconds.median()
    c2 = 2e6 * 9
    c4 = 1e9
    return {
        "C2_scattered": {"location_years": c2, "TB_read": c2 * per_pt_b / 1e12,
                         "serial_hours": c2 * per_pt_s / 3600},
        "C4_dense": {"location_years": c4, "TB_read": c4 * dense["bytes_per_valid_pixel"] / 1e12,
                     "serial_hours": c4 / dense["valid_pixels"] * dense["seconds"] / 3600},
        "note": "serial_hours assumes one reader; divide by concurrent readers. "
                "Scattered points that share a block share its bytes, so C2 TB is an upper bound.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", type=Path, default=Path("data/processed/labels_kssl.parquet"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index", type=Path, default=Path("data/raw/aef/aef_index.parquet"))
    ap.add_argument("--out", type=Path, default=Path("geosoil/results/aef_read_cost.json"))
    ap.add_argument("--base", default=S3_BASE, help="tile URL prefix (S3 direct or the source.coop proxy)")
    a = ap.parse_args()
    global BASE
    BASE = a.base.rstrip("/")

    idx = load_index(a.index)
    read = pd.read_csv if a.points.suffix == ".csv" else pd.read_parquet
    sites = read(a.points)[["site_id", "lon", "lat"]].drop_duplicates("site_id")
    sites = sites[sites.lon.between(-125, -66) & sites.lat.between(24, 50)]   # CONUS
    pts = sites.sample(min(a.n, len(sites)), random_state=a.seed)

    with rasterio.Env(**GDAL_ENV):
        scat = pd.DataFrame(measure_scattered(idx, pts, a.year))
        found = scat[scat.found]
        dense = measure_dense(idx, found.merge(pts, on="site_id").lon.iloc[0],
                              found.merge(pts, on="site_id").lat.iloc[0], a.year)

    result = {
        "base": BASE, "year": a.year, "n_points": a.n, "n_found": int(len(found)),
        "n_valid": int(found.valid.sum()),
        "scattered": {"median_bytes": float(found.bytes.median()), "median_seconds": float(found.seconds.median()),
                      "p90_seconds": float(found.seconds.quantile(0.9))},
        "dense": dense,
        "projection": project(found, dense),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
