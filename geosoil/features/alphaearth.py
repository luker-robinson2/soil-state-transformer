"""AlphaEarth embeddings for every labelled site, read one COG block at a time (Phase 2).

Why blocks: the AlphaEarth COGs are band-interleaved with 1024x1024 zstd blocks, so one
pixel costs as much to read as its whole block (~38 MB over 64 bands). Sites are grouped by
(tile, block), each block is read once, and every site pixel in it is kept. Optionally a
deterministic sample of unlabelled pixels is kept from the same block, which seeds the
Phase 3 pretraining corpus at no extra read cost.

Two steps:
    python -m geosoil.features.alphaearth plan    --years 2017-2025      # site -> block plan
    python -m geosoil.features.alphaearth extract --task 0 --ntasks 64 --threads 8
    python -m geosoil.features.alphaearth status --ntasks 64     # done / missing / failed blocks
`extract` is idempotent: a shard that exists is skipped, so array jobs can simply be re-run.
Every shard carries a JSON manifest with provenance (source URL, index sha256, plan sha256,
code version) and the list of failed blocks; `status` turns those into a resubmit list.
Tiles are read over direct S3 HTTPS (66 MB/s from Alpine; the data.source.coop proxy is <4 MB/s).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import shards

S3 = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/tge-labs/aef/v1/annual"
INDEX_URL = f"{S3}/aef_index.parquet"
BLOCK = 1024
RES = 10.0
N_BANDS = 64
OUT = C.DATA_PROC / "features" / "alphaearth"
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff",
                GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2", GDAL_HTTP_MULTIPLEX="YES",
                GDAL_HTTP_VERSION="2", VSI_CACHE="FALSE")
MAX_COORD_PRECISION_M = 1000.0   # a 10 m embedding is meaningless for sites placed to >1 km


def dequantize(raw: np.ndarray) -> np.ndarray:
    """int8 -> [-1, 1] per the dataset README; -128 is no-data."""
    out = ((raw.astype(np.float32) / 127.5) ** 2) * np.sign(raw)
    out[raw == -128] = np.nan
    return out


def tile_url(path: str) -> str:
    return f"/vsicurl/{S3}/{path.split('/aef/v1/annual/', 1)[1]}"


def plan(sites: pd.DataFrame, index: pd.DataFrame, years) -> pd.DataFrame:
    """Map each (site, year) to tile path, block (bx, by) and pixel within the block (px, py).

    Tiles are bottom-up (origin at utm_south, +10 m y-resolution), so pixel row counts up from
    the tile's southern edge; the extractor reads windows in that same frame.
    """
    from pyproj import Transformer
    rows = []
    idx = index[index["year"].isin(years)]
    for crs, tiles in idx.groupby("crs"):
        west, east = tiles["wgs84_west"].min(), tiles["wgs84_east"].max()
        south, north = tiles["wgs84_south"].min(), tiles["wgs84_north"].max()
        s = sites[sites["lon"].between(west, east) & sites["lat"].between(south, north)]
        if s.empty:
            continue
        x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(s["lon"].to_numpy(), s["lat"].to_numpy())
        for t in tiles.itertuples():
            m = (x >= t.utm_west) & (x < t.utm_east) & (y >= t.utm_south) & (y < t.utm_north)
            if not m.any():
                continue
            col = ((x[m] - t.utm_west) // RES).astype(int)
            row = ((y[m] - t.utm_south) // RES).astype(int)
            rows.append(pd.DataFrame({"site_id": s["site_id"].to_numpy()[m], "year": t.year, "path": t.path,
                                      "bx": col // BLOCK, "by": row // BLOCK, "px": col % BLOCK, "py": row % BLOCK,
                                      "utm_x0": t.utm_west, "utm_y0": t.utm_south, "crs": crs}))
    p = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    # a site near a UTM-zone edge can fall in two tiles; keep one per (site, year)
    return p.drop_duplicates(["site_id", "year"]).reset_index(drop=True)


def _groups(p: pd.DataFrame, task: int, ntasks: int):
    p = p.assign(key=p["path"] + "#" + p["bx"].astype(str) + "_" + p["by"].astype(str))
    mine = p[p["key"].map(lambda k: shards.shard_of(k, ntasks)) == task]
    return list(mine.groupby("key", sort=True))


_local = threading.local()


def _dataset(path: str):
    import rasterio
    cache = getattr(_local, "ds", None)
    if cache is None:
        cache = _local.ds = {}
    if path not in cache:
        if len(cache) > 8:
            for d in cache.values():
                d.close()
            cache.clear()
        cache[path] = rasterio.open(tile_url(path))
    return cache[path]


def read_block(path: str, bx: int, by: int) -> np.ndarray:
    """All 64 bands of one block, shape (64, h, w), int8, in the tile's bottom-up row order."""
    from rasterio.windows import Window
    ds = _dataset(path)
    w = Window(bx * BLOCK, by * BLOCK, min(BLOCK, ds.width - bx * BLOCK), min(BLOCK, ds.height - by * BLOCK))
    return ds.read(window=w)


def extract_group(g: pd.DataFrame, neighbors: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    r = g.iloc[0]
    raw = read_block(r.path, int(r.bx), int(r.by))
    emb = dequantize(raw[:, g["py"].to_numpy(), g["px"].to_numpy()].T)            # (n_sites, 64)
    sites = pd.DataFrame(emb, columns=list(C.AE_COLS))
    sites.insert(0, "site_id", g["site_id"].to_numpy())
    sites.insert(1, "year", g["year"].to_numpy())
    unl = None
    if neighbors:
        valid = np.argwhere(raw[0] != -128)
        if len(valid):
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{r.path}{r.bx}{r.by}{seed}".encode()).digest()[:4], "big"))
            pick = valid[rng.choice(len(valid), size=min(neighbors, len(valid)), replace=False)]
            yy, xx = pick[:, 0], pick[:, 1]
            unl = pd.DataFrame(raw[:, yy, xx].T, columns=list(C.AE_COLS))     # keep int8; 64 B per pixel
            unl.insert(0, "year", int(r.year))
            unl.insert(1, "crs", r.crs)
            unl.insert(2, "utm_x", r.utm_x0 + (int(r.bx) * BLOCK + xx + 0.5) * RES)
            unl.insert(3, "utm_y", r.utm_y0 + (int(r.by) * BLOCK + yy + 0.5) * RES)
    return sites, unl


def extract(p: pd.DataFrame, task: int, ntasks: int, threads: int, neighbors: int, out: Path) -> dict:
    import rasterio
    out.mkdir(parents=True, exist_ok=True)
    shard = out / f"sites-{task:04d}-of-{ntasks:04d}.parquet"
    if shard.exists():
        return {"task": task, "skipped": True}
    groups = _groups(p, task, ntasks)
    t0 = time.time()
    site_parts, unl_parts, failed = [], [], []
    with rasterio.Env(**GDAL_ENV):
        with ThreadPoolExecutor(threads) as ex:
            futs = {ex.submit(extract_group, g, neighbors): k for k, g in groups}
            for f, k in futs.items():
                try:
                    s, u = f.result()
                    site_parts.append(s)
                    if u is not None:
                        unl_parts.append(u)
                except Exception as e:                                  # fail loud, keep the rest
                    failed.append({"block": k, "error": repr(e)[:300]})
    sites = pd.concat(site_parts, ignore_index=True) if site_parts else pd.DataFrame()
    tmp = shard.with_suffix(".tmp")
    sites.to_parquet(tmp, index=False)
    if unl_parts:
        pd.concat(unl_parts, ignore_index=True).to_parquet(out / f"unlabelled-{task:04d}-of-{ntasks:04d}.parquet", index=False)
    tmp.rename(shard)
    meta = {"task": task, "ntasks": ntasks, "blocks": len(groups), "failed_blocks": failed,
            "site_years": int(len(sites)), "unlabelled_pixels": int(sum(len(u) for u in unl_parts)),
            "seconds": round(time.time() - t0, 1), "threads": threads, "source": S3,
            "plan_sha256": shards.sha256(OUT / "plan.parquet"),
            "index_sha256": shards.sha256(C.REPO / "data" / "raw" / "aef" / "aef_index.parquet"),
            "code_version": shards.code_version(), "licence": "CC-BY-4.0, The AlphaEarth Foundations Satellite "
            "Embedding dataset is produced by Google and Google DeepMind"}
    (out / f"sites-{task:04d}-of-{ntasks:04d}.json").write_text(json.dumps(meta, indent=2))
    return meta


def status(ntasks: int, out: Path = OUT) -> dict:
    return shards.status(ntasks, out)


def _years(spec: str) -> list[int]:
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("plan")
    pp.add_argument("--years", default="2017-2025")
    pp.add_argument("--labels", type=Path, default=C.DATA_PROC / "labels_c1.parquet")
    pe = sub.add_parser("extract")
    pe.add_argument("--task", type=int, default=0)
    pe.add_argument("--ntasks", type=int, default=1)
    pe.add_argument("--threads", type=int, default=8)
    pe.add_argument("--neighbors", type=int, default=0, help="unlabelled pixels kept per block (Phase 3 seed)")
    pe.add_argument("--limit-blocks", type=int, default=0, help="for smoke tests")
    pe.add_argument("--years", default="", help="restrict to these years, e.g. 2020 or 2017-2025")
    ps = sub.add_parser("status")
    ps.add_argument("--ntasks", type=int, required=True)
    ps.add_argument("--years", default="")
    a = ap.parse_args(argv)
    run_dir = OUT / (f"y{a.years}" if getattr(a, "years", "") and a.cmd != "plan" else "all")
    if a.cmd == "status":
        print(json.dumps(status(a.ntasks, run_dir), indent=2))
        return
    plan_path = OUT / "plan.parquet"
    if a.cmd == "plan":
        cols = ["site_id", "lon", "lat", "coord_precision_m"]
        s = pd.read_parquet(a.labels, columns=cols).drop_duplicates("site_id")
        s = s[s["coord_precision_m"].isna() | (s["coord_precision_m"] <= MAX_COORD_PRECISION_M)]
        idx_path = C.REPO / "data" / "raw" / "aef" / "aef_index.parquet"
        if not idx_path.exists():
            import urllib.request
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(INDEX_URL, idx_path)
        p = plan(s, pd.read_parquet(idx_path), _years(a.years))
        OUT.mkdir(parents=True, exist_ok=True)
        p.to_parquet(plan_path, index=False)
        nb = p.groupby(["path", "bx", "by"]).ngroups
        print(json.dumps({"sites": int(s["site_id"].nunique()), "site_years": len(p), "blocks": nb,
                          "est_TB_read": round(nb * 38e6 / 1e12, 2)}, indent=2))
    else:
        p = pd.read_parquet(plan_path)
        if a.years:
            p = p[p["year"].isin(_years(a.years))]
        if a.limit_blocks:
            first = p.drop_duplicates(["path", "bx", "by"]).head(a.limit_blocks)[["path", "bx", "by"]]
            p = p.merge(first, on=["path", "bx", "by"])
        print(json.dumps(extract(p, a.task, a.ntasks, a.threads, a.neighbors, run_dir), indent=2))


if __name__ == "__main__":
    main()
