"""Multi-scale terrain context for every labelled site (Phase 2).

Texture is set by landscape position at hillslope-to-basin scale; 30 m windows of a few hundred
metres miss it. Multi-scale terrain was the strongest single texture lever in digital soil mapping
(Behrens et al. 2018, Geoderma 310:128: clay R2 0.38 -> 0.68). This module computes, at radii
0.5, 1, 2, 5, 10 and 20 km:

    tms_dev_<s>     deviation from mean elevation, (z - mean) / std  (Wilson & Gallant 2000)
    tms_tpi_<s>     z - mean (m)
    tms_relief_<s>  std of elevation (m)
    tms_slope_<s>   slope (deg) of the elevation surface smoothed at that scale
    tms_dev_max     the signed DEV with the largest magnitude across scales
    tms_dev_scale   the radius (km) where it occurs

Source: Copernicus DEM GLO-90 (90 m) 1-degree COGs on AWS Open Data, with its tileList.txt
(ocean tiles do not exist). Unit of work = one 1-degree tile with sites, mosaicked with its 8
neighbours so 20 km windows never run off the edge. Window statistics come from summed-area
tables, so each (site, scale) is O(1). Tiles are cached in memory and centres are processed in
spatial order, so neighbours are mostly reused.

    python -m geosoil.features.terrain_ms extract --task 0 --ntasks 32
    python -m geosoil.features.terrain_ms status --ntasks 32
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import shards
from .terrain import M_PER_DEG

BASE = "https://copernicus-dem-90m.s3.amazonaws.com"
OUT = C.DATA_PROC / "features" / "terrain_ms"
SCALES_KM = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
LICENCE = ("Copernicus DEM GLO-90: (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH "
           "2014-2018 provided under COPERNICUS by the European Union and ESA")
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2")


def tile_name(la: int, lo: int) -> str:
    return (f"Copernicus_DSM_COG_30_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_"
            f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM")


def columns() -> list[str]:
    cols = []
    for s in SCALES_KM:
        k = f"{s:g}km".replace(".", "p")
        cols += [f"tms_dev_{k}", f"tms_tpi_{k}", f"tms_relief_{k}", f"tms_slope_{k}"]
    return cols + ["tms_dev_max", "tms_dev_scale"]


# ---------------------------------------------------------------- summed-area statistics
def integral(a: np.ndarray) -> np.ndarray:
    """Zero-padded 2-D cumulative sum, so box sums are I[r1,c1]-I[r0,c1]-I[r1,c0]+I[r0,c0]."""
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    np.cumsum(np.cumsum(a, axis=0, dtype=np.float64), axis=1, out=out[1:, 1:])
    return out


def box(I: np.ndarray, r: np.ndarray, c: np.ndarray, ry: int, rx: int) -> np.ndarray:
    H, W = I.shape[0] - 1, I.shape[1] - 1
    r0, r1 = np.clip(r - ry, 0, H), np.clip(r + ry + 1, 0, H)
    c0, c1 = np.clip(c - rx, 0, W), np.clip(c + rx + 1, 0, W)
    return I[r1, c1] - I[r0, c1] - I[r1, c0] + I[r0, c0]


class Stats:
    """NaN-aware window mean / std of an elevation grid via summed-area tables."""

    def __init__(self, z: np.ndarray):
        ok = np.isfinite(z)
        z0 = np.where(ok, z, 0.0)
        self.I1, self.I2, self.N = integral(z0), integral(z0 * z0), integral(ok.astype(np.float64))

    def mean_std(self, r, c, ry, rx):
        n = box(self.N, r, c, ry, rx)
        s1, s2 = box(self.I1, r, c, ry, rx), box(self.I2, r, c, ry, rx)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(n > 0, s1 / n, np.nan)
            v = np.where(n > 1, s2 / n - m * m, np.nan)
        return m, np.sqrt(np.clip(v, 0, None)), n / ((2 * ry + 1) * (2 * rx + 1))


def multiscale(z: np.ndarray, r: np.ndarray, c: np.ndarray, dx: float, dy: float,
               scales_km=SCALES_KM, min_cover: float = 0.5) -> pd.DataFrame:
    """Features at each site pixel (r, c) of grid z with pixel size dx, dy metres."""
    st = Stats(z)
    zc = z[r, c].astype(np.float64)
    out, devs = {}, []
    for s in scales_km:
        k = f"{s:g}km".replace(".", "p")
        ry, rx = max(1, round(s * 1000 / dy)), max(1, round(s * 1000 / dx))
        m, sd, cover = st.mean_std(r, c, ry, rx)
        good = cover >= min_cover
        with np.errstate(invalid="ignore", divide="ignore"):
            dev = np.where(good & (sd > 0), (zc - m) / sd, np.nan)
        out[f"tms_dev_{k}"] = dev
        out[f"tms_tpi_{k}"] = np.where(good, zc - m, np.nan)
        out[f"tms_relief_{k}"] = np.where(good, sd, np.nan)
        # slope of the surface smoothed at this scale: central differences of window means, step = radius
        mN, _, _ = st.mean_std(r - ry, c, ry, rx)
        mS, _, _ = st.mean_std(r + ry, c, ry, rx)
        mE, _, _ = st.mean_std(r, c + rx, ry, rx)
        mW, _, _ = st.mean_std(r, c - rx, ry, rx)
        gx, gy = (mE - mW) / (2 * rx * dx), (mN - mS) / (2 * ry * dy)
        out[f"tms_slope_{k}"] = np.where(good, np.degrees(np.arctan(np.hypot(gx, gy))), np.nan)
        devs.append(dev)
    D = np.vstack(devs).T
    absD = np.where(np.isfinite(D), np.abs(D), -1)
    best = absD.argmax(axis=1)
    has = absD.max(axis=1) >= 0
    out["tms_dev_max"] = np.where(has, D[np.arange(len(D)), best], np.nan)
    out["tms_dev_scale"] = np.where(has, np.asarray(scales_km)[best], np.nan)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- tiles and mosaics
class TileCache:
    def __init__(self, existing: set[str], maxsize: int = 96):
        self.existing, self.maxsize, self.c = existing, maxsize, OrderedDict()

    def get(self, la: int, lo: int):
        """(array float32 with NaN, (res_lon, res_lat)) or None for a tile that does not exist."""
        lo = ((lo + 180) % 360) - 180
        name = tile_name(la, lo)
        if name in self.c:
            self.c.move_to_end(name)
            return self.c[name]
        val = None
        if name in self.existing:
            import rasterio
            with rasterio.open(f"/vsicurl/{BASE}/{name}/{name}.tif") as ds:
                a = ds.read(1).astype(np.float32)
                if ds.nodata is not None:
                    a[a == ds.nodata] = np.nan
                val = (a, (ds.transform.a, -ds.transform.e))
        self.c[name] = val
        if len(self.c) > self.maxsize:
            self.c.popitem(last=False)
        return val


def mosaic(cache: TileCache, la: int, lo: int):
    """3x3 tiles around (la, lo) on the centre tile's pixel grid; returns (z, H, W, res) or None."""
    from scipy.ndimage import zoom
    centre = cache.get(la, lo)
    if centre is None:
        return None
    a, res = centre
    H, W = a.shape
    z = np.full((3 * H, 3 * W), np.nan, dtype=np.float32)
    for i, dla in enumerate((1, 0, -1)):              # rows run north -> south
        for j, dlo in enumerate((-1, 0, 1)):
            t = cache.get(la + dla, lo + dlo)
            if t is None:
                continue
            b = t[0]
            if b.shape != (H, W):                        # width changes across latitude bands
                b = zoom(np.nan_to_num(b, nan=np.nan), (H / b.shape[0], W / b.shape[1]), order=1)[:H, :W]
            z[i * H:i * H + b.shape[0], j * W:j * W + b.shape[1]] = b
    return z, H, W, res


def extract(sites: pd.DataFrame, task: int, ntasks: int, out: Path = OUT) -> dict:
    if shards.shard_path(out, task, ntasks).exists():
        return {"task": task, "skipped": True}
    import rasterio
    import urllib.request
    existing = set(urllib.request.urlopen(f"{BASE}/tileList.txt").read().decode().split())
    s = sites.assign(la=np.floor(sites["lat"]).astype(int), lo=np.floor(sites["lon"]).astype(int))
    centres = s[["la", "lo"]].drop_duplicates().sort_values(["la", "lo"]).to_numpy()
    mine = np.array_split(centres, ntasks)[task]           # contiguous chunks keep neighbours in cache
    cache, parts, failed, missing, t0 = TileCache(existing), [], [], 0, time.time()
    with rasterio.Env(**GDAL_ENV):
        for la, lo in mine:
            g = s[(s["la"] == la) & (s["lo"] == lo)]
            try:
                m = mosaic(cache, int(la), int(lo))
                if m is None:
                    missing += 1
                    continue
                z, H, W, (rlon, rlat) = m
                r = H + np.clip(((la + 1 - g["lat"].to_numpy()) / rlat).astype(int), 0, H - 1)
                c = W + np.clip(((g["lon"].to_numpy() - lo) / rlon).astype(int), 0, W - 1)
                lat_c = la + 0.5
                f = multiscale(z, r, c, rlon * M_PER_DEG * math.cos(math.radians(lat_c)), rlat * M_PER_DEG)
                f.insert(0, "site_id", g["site_id"].to_numpy())
                parts.append(f)
            except Exception as e:
                failed.append({"block": tile_name(int(la), int(lo)), "error": repr(e)[:300]})
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["site_id", *columns()])
    shards.write(df, out, task, ntasks, {"source": BASE, "licence": LICENCE, "tiles": int(len(mine)),
                                         "sites": int(len(df)), "failed_blocks": failed, "missing_tiles": missing,
                                         "scales_km": list(SCALES_KM), "seconds": round(time.time() - t0, 1)})
    return {"task": task, "tiles": int(len(mine)), "sites": int(len(df)), "failed": len(failed),
            "missing_tiles": missing, "seconds": round(time.time() - t0, 1)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("extract")
    pe.add_argument("--task", type=int, default=0)
    pe.add_argument("--ntasks", type=int, default=1)
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
    print(json.dumps(extract(s, a.task, a.ntasks), indent=2))


if __name__ == "__main__":
    main()
