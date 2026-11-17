"""Height above nearest drainage (HAND) for every labelled site (Phase 2).

HAND separates alluvial / valley-floor soils (fine, wet, carbon-rich) from uplands, which affects
texture, SOC and bulk density. Source: ASF GLO-30 HAND (derived from Copernicus DEM GLO-30), 1-degree
Float32 COGs on AWS Open Data, CC0. MERIT Hydro's HAND needs registration, so it is not used.

Unit of work = one 1-degree tile (same grouping as terrain.py): whole-tile read when it holds
>= 10 sites, otherwise windowed reads.

Columns: hydro_hand (m, containing 30 m cell), hydro_hand_270m (mean over a 9x9 window),
hydro_hand_min_270m (lowest HAND nearby, i.e. how close a channel is).

    python -m geosoil.features.static --source hand
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import shards

BASE = "https://glo-30-hand.s3.amazonaws.com/v1/2021"
R = 4                                   # 9x9 window at 30 m
WHOLE_TILE_MIN = 10
COLUMNS = ("hydro_hand", "hydro_hand_270m", "hydro_hand_min_270m")
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2")


def tile_name(lon: float, lat: float) -> str:
    la, lo = math.floor(lat), math.floor(lon)
    return (f"Copernicus_DSM_COG_10_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_"
            f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_HAND")


def window_features(w: np.ndarray) -> list[float]:
    c = w.shape[0] // 2
    if not np.isfinite(w[c, c]):
        return [np.nan] * len(COLUMNS)
    with np.errstate(all="ignore"):
        return [float(w[c, c]), float(np.nanmean(w)), float(np.nanmin(w))]


def extract_tile(name: str, g: pd.DataFrame) -> pd.DataFrame:
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.windows import Window
    with rasterio.open(f"/vsicurl/{BASE}/{name}.tif") as ds:
        rows, cols = rowcol(ds.transform, g["lon"].to_numpy(), g["lat"].to_numpy())
        rows, cols = np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
        nod = ds.nodata
        if len(g) >= WHOLE_TILE_MIN:
            a = np.pad(ds.read(1).astype(np.float32), R, constant_values=np.nan)
            wins = [a[r:r + 2 * R + 1, c:c + 2 * R + 1] for r, c in zip(rows, cols)]
        else:
            wins = [ds.read(1, window=Window(c - R, r - R, 2 * R + 1, 2 * R + 1), boundless=True,
                            fill_value=np.nan).astype(np.float32) for r, c in zip(rows, cols)]
    if nod is not None:
        wins = [np.where(w == nod, np.nan, w) for w in wins]
    out = pd.DataFrame([window_features(w) for w in wins], columns=list(COLUMNS))
    out.insert(0, "site_id", g["site_id"].to_numpy())
    return out


def extract(sites: pd.DataFrame, out_root: Path = C.DATA_PROC / "features", threads: int = 8) -> dict:
    import rasterio
    s = sites.assign(tile=[tile_name(lo, la) for lo, la in zip(sites["lon"], sites["lat"])])
    groups, t0, parts, failed, missing = list(s.groupby("tile")), time.time(), [], [], []
    with rasterio.Env(**GDAL_ENV):
        with ThreadPoolExecutor(threads) as ex:
            futs = {ex.submit(extract_tile, k, g): k for k, g in groups}
            for f, k in futs.items():
                try:
                    parts.append(f.result())
                except Exception as e:
                    msg = repr(e)
                    (missing if ("404" in msg or "does not exist" in msg or "No such file" in msg) else failed).append(
                        {"block": k, "error": msg[:300]})
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["site_id", *COLUMNS])
    shards.write(df, out_root / "hand", 0, 1, {"source": BASE, "licence": "CC0 1.0 (ASF GLO-30 HAND)",
                                               "tiles": len(groups), "sites": int(len(df)), "failed_blocks": failed,
                                               "missing_tiles": len(missing), "seconds": round(time.time() - t0, 1)})
    return {"source": "hand", "tiles": len(groups), "sites": int(len(df)), "failed": len(failed),
            "missing_tiles": len(missing), "seconds": round(time.time() - t0, 1)}
