"""Terrain features for every labelled site from the Copernicus DEM GLO-30 (Phase 2).

Source: 1-degree COG tiles in EPSG:4326 on AWS Open Data (anonymous HTTPS), 1 arc-second
(~30 m; tile width shrinks north of 50 degrees). Licence: Copernicus DEM GLO-30, free; attribution
"(c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under
COPERNICUS by the European Union and ESA".

Unit of work = one 1-degree tile. A tile holding at least WHOLE_TILE_MIN sites is read
once (~40 MB); otherwise each site gets a boundless window read. Features come from a
(2R+1)^2 window around each site, with pixel spacing converted to metres at the site's latitude:

    elevation (m), slope (deg), aspect_sin, aspect_cos      Horn's method, 3x3
    curvature (1/100 m)                                      Laplacian, 3x3
    tpi_100, tpi_500 (m)       centre minus mean within ~100 m / ~500 m
    roughness_100 (m)          std within ~100 m
    relief_500 (m)             max - min within ~500 m

    python -m geosoil.features.terrain extract --task 0 --ntasks 16 --threads 8
    python -m geosoil.features.terrain status --ntasks 16
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import shards

BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
OUT = C.DATA_PROC / "features" / "terrain"
R = 17                     # window half-width in pixels (~500 m)
R_SMALL = 3                # ~100 m
WHOLE_TILE_MIN = 10
M_PER_DEG = 111_320.0
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2")
FEATURES = ("elevation", "slope", "aspect_sin", "aspect_cos", "curvature", "tpi_100", "tpi_500",
            "roughness_100", "relief_500")
LICENCE = ("Copernicus DEM GLO-30: (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH "
           "2014-2018 provided under COPERNICUS by the European Union and ESA")


def tile_name(lon: float, lat: float) -> str:
    la, lo = math.floor(lat), math.floor(lon)
    ns = f"{'N' if la >= 0 else 'S'}{abs(la):02d}"
    ew = f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def tile_url(name: str) -> str:
    return f"/vsicurl/{BASE}/{name}/{name}.tif"


def features(win: np.ndarray, dx: float, dy: float) -> dict:
    """Terrain features from a (2R+1)^2 elevation window centred on the site; dx, dy in metres."""
    c = win.shape[0] // 2
    z = win[c - 1:c + 2, c - 1:c + 2]
    out = dict.fromkeys(FEATURES, np.nan)
    if not np.isfinite(win[c, c]):
        return out
    out["elevation"] = float(win[c, c])
    if np.isfinite(z).all():
        # Horn (1981); rows run north -> south, columns west -> east
        dzdx = ((z[0, 2] + 2 * z[1, 2] + z[2, 2]) - (z[0, 0] + 2 * z[1, 0] + z[2, 0])) / (8 * dx)
        dzdy = ((z[0, 0] + 2 * z[0, 1] + z[0, 2]) - (z[2, 0] + 2 * z[2, 1] + z[2, 2])) / (8 * dy)  # +north
        out["slope"] = float(np.degrees(np.arctan(np.hypot(dzdx, dzdy))))
        if dzdx or dzdy:
            aspect = np.arctan2(-dzdx, -dzdy)                  # direction of steepest descent, 0 = north
            out["aspect_sin"], out["aspect_cos"] = float(np.sin(aspect)), float(np.cos(aspect))
        else:
            out["aspect_sin"] = out["aspect_cos"] = 0.0
        lap = (z[1, 0] + z[1, 2] - 2 * z[1, 1]) / dx ** 2 + (z[0, 1] + z[2, 1] - 2 * z[1, 1]) / dy ** 2
        out["curvature"] = float(lap * 100)
    small = win[c - R_SMALL:c + R_SMALL + 1, c - R_SMALL:c + R_SMALL + 1]
    if np.isfinite(small).sum() > small.size // 2:
        out["tpi_100"] = float(win[c, c] - np.nanmean(small))
        out["roughness_100"] = float(np.nanstd(small))
    if np.isfinite(win).sum() > win.size // 2:
        out["tpi_500"] = float(win[c, c] - np.nanmean(win))
        out["relief_500"] = float(np.nanmax(win) - np.nanmin(win))
    return out


def _windows_from_array(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> list[np.ndarray]:
    pad = np.pad(arr, R, constant_values=np.nan)
    return [pad[r:r + 2 * R + 1, c:c + 2 * R + 1] for r, c in zip(rows, cols)]


def extract_tile(name: str, sites: pd.DataFrame) -> pd.DataFrame:
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(tile_url(name)) as ds:
        from rasterio.transform import rowcol
        rows, cols = rowcol(ds.transform, sites["lon"].to_numpy(), sites["lat"].to_numpy())   # vectorised
        rows, cols = np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
        if len(sites) >= WHOLE_TILE_MIN:
            arr = ds.read(1).astype(np.float32)
            if ds.nodata is not None:
                arr[arr == ds.nodata] = np.nan
            wins = _windows_from_array(arr, rows, cols)
        else:
            wins = []
            for r, c in zip(rows, cols):
                w = ds.read(1, window=Window(c - R, r - R, 2 * R + 1, 2 * R + 1), boundless=True,
                            fill_value=np.nan).astype(np.float32)
                wins.append(w)
        res_lon, res_lat = ds.transform.a, -ds.transform.e
    recs = []
    for w, lat in zip(wins, sites["lat"].to_numpy()):
        recs.append(features(w, res_lon * M_PER_DEG * math.cos(math.radians(lat)), res_lat * M_PER_DEG))
    out = pd.DataFrame(recs, columns=list(FEATURES))
    out.insert(0, "site_id", sites["site_id"].to_numpy())
    return out


def extract(sites: pd.DataFrame, task: int, ntasks: int, threads: int, out: Path = OUT) -> dict:
    import rasterio
    if shards.shard_path(out, task, ntasks).exists():
        return {"task": task, "skipped": True}
    sites = sites.assign(tile=[tile_name(lo, la) for lo, la in zip(sites["lon"], sites["lat"])])
    mine = sites[sites["tile"].map(lambda t: shards.shard_of(t, ntasks)) == task]
    groups = list(mine.groupby("tile", sort=True))
    t0, parts, failed, missing = time.time(), [], [], []
    with rasterio.Env(**GDAL_ENV):
        with ThreadPoolExecutor(threads) as ex:
            futs = {ex.submit(extract_tile, k, g): k for k, g in groups}
            for f, k in futs.items():
                try:
                    parts.append(f.result())
                except Exception as e:
                    msg = repr(e)
                    # ocean / unpublished tiles simply do not exist: that is not a failure
                    (missing if ("404" in msg or "does not exist" in msg or "No such file" in msg) else failed).append(
                        {"block": k, "error": msg[:300]})
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["site_id", *FEATURES])
    shards.write(df, out, task, ntasks, {
        "source": BASE, "licence": LICENCE, "tiles": len(groups), "sites": int(len(df)),
        "failed_blocks": failed, "missing_tiles": [m["block"] for m in missing],
        "seconds": round(time.time() - t0, 1), "threads": threads, "window_half_px": R})
    return {"task": task, "tiles": len(groups), "sites": int(len(df)), "failed": len(failed),
            "missing_tiles": len(missing), "seconds": round(time.time() - t0, 1)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("extract")
    pe.add_argument("--task", type=int, default=0)
    pe.add_argument("--ntasks", type=int, default=1)
    pe.add_argument("--threads", type=int, default=8)
    pe.add_argument("--labels", type=Path, default=C.DATA_PROC / "labels_c1.parquet")
    pe.add_argument("--limit-sites", type=int, default=0)
    ps = sub.add_parser("status")
    ps.add_argument("--ntasks", type=int, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print(json.dumps(shards.status(a.ntasks, OUT), indent=2))
        return
    s = pd.read_parquet(a.labels, columns=["site_id", "lon", "lat"]).drop_duplicates("site_id")
    if a.limit_sites:
        s = s.sample(a.limit_sites, random_state=0)
    print(json.dumps(extract(s, a.task, a.ntasks, a.threads), indent=2))


if __name__ == "__main__":
    main()
