"""Earth Engine features for every labelled site: bare-soil reflectance and AlphaEarth landscape context.

Two roadmap items (I5, I3), both sampled server-side so nothing large is downloaded:

  bare_soil   Sentinel-2 L2A 2018-2024, Cloud Score+ masked, then only bare-soil observations kept
              (NDVI < 0.25 and NBR2 < 0.075, snow removed with NDSI < 0.4; Demattê et al. 2018 RSE 212:161, 2020 RSE 239:111610).
              Median reflectance of the bare observations, mean over a 60 m disc. Bare soil is the one
              optical view of the soil surface itself, and texture/SOC/iron drive its spectrum.
              Columns bs_<band> (reflectance 0-1), bs_frac (share of clear observations that were bare),
              bs_clay_idx (B11/B12, clay-mineral absorption), bs_ndvi (residual greenness).
              NaN where the site was never bare in 7 years (forest, permanent grassland).

  alphaearth_context
              AlphaEarth (GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL) 2017-2023 mean embedding, averaged over
              discs of 500 m and 3 km around the site. 2024 is excluded, matching the temporal hold-out.
              Columns aectx500_A00..A63, aectx3k_A00..A63, plus aectx_sim500 / aectx_sim3k: cosine
              similarity between the site pixel and each context mean, i.e. how typical the site is of
              its surroundings.

Auth: application-default credentials, or a token from the gcloud CLI (refreshed automatically).
Project: GEOSOIL_EE_PROJECT (default geo-soil-foundational-model, registered for noncommercial use).

Runs as Earth Engine batch exports (interactive requests time out on 7-year composites), one table asset
per chunk of sites under projects/<project>/assets/geosoil, then downloads them into shards:

    python -m geosoil.features.gee_context --part bare_soil submit      # start the exports
    python -m geosoil.features.gee_context --part bare_soil collect     # rerun until all are collected
    python -m geosoil.features.gee_context --part alphaearth_context interactive   # light parts: direct requests
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import shards

PROJECT = os.environ.get("GEOSOIL_EE_PROJECT", "geo-soil-foundational-model")
S2_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
YEARS = (2018, 2024)                      # bare soil: Sentinel-2 L2A is global from 2018
AE_YEARS = (2017, 2023)                   # context: 2024 stays out (temporal hold-out)
RADII = {"500": (500, 50), "3k": (3000, 200)}   # name -> (disc radius m, reduction scale m)
CHUNK = 5000                              # sites per Earth Engine batch export
PARTS = ("bare_soil", "alphaearth_context")


def init_ee():
    """Initialise Earth Engine with ADC if present, else with a self-refreshing gcloud token."""
    import ee
    try:
        import google.auth
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/earthengine",
                                               "https://www.googleapis.com/auth/cloud-platform"])
    except Exception:
        from google.oauth2.credentials import Credentials

        def refresh(request, scopes):
            tok = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True,
                                 check=True).stdout.strip()
            return tok, dt.datetime.utcnow() + dt.timedelta(minutes=50)
        tok, exp = refresh(None, None)
        creds = Credentials(tok, expiry=exp, refresh_handler=refresh)
    ee.Initialize(creds, project=PROJECT)
    return ee


def bare_soil_image(ee):
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(f"{YEARS[0]}-01-01", f"{YEARS[1]}-12-31")
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = s2.linkCollection(cs, ["cs_cdf"])

    def prep(img):
        clear = img.select("cs_cdf").gte(0.6)
        refl = img.select(list(S2_BANDS)).divide(10000)
        ndvi = refl.normalizedDifference(["B8", "B4"])
        nbr2 = refl.normalizedDifference(["B11", "B12"])
        ndsi = refl.normalizedDifference(["B3", "B11"])     # snow is also low-NDVI and bright: drop it
        bare = ndvi.lt(0.25).And(nbr2.lt(0.075)).And(ndsi.lt(0.4)).And(clear)
        return refl.updateMask(bare).addBands(bare.rename("bare").updateMask(clear))

    col = s2.map(prep)
    med = col.select(list(S2_BANDS)).median()
    frac = col.select("bare").mean().rename("frac")
    return med.addBands(frac)


def alphaearth_image(ee):
    ae = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate(
        f"{AE_YEARS[0]}-01-01", f"{AE_YEARS[1] + 1}-01-01")
    return ae.mean()


def _features(ee, chunk: pd.DataFrame, radius: float):
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]).buffer(radius), {"site_id": str(r.site_id)})
        for r in chunk.itertuples(index=False)])


def derive_bare_soil(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    for b in S2_BANDS:
        out[f"bs_{b.lower()}"] = pd.to_numeric(raw.get(b), errors="coerce")
    out["bs_frac"] = pd.to_numeric(raw.get("frac"), errors="coerce")
    b11, b12 = out["bs_b11"], out["bs_b12"]
    out["bs_clay_idx"] = b11 / b12.where(b12 > 0)
    out["bs_ndvi"] = (out["bs_b8"] - out["bs_b4"]) / (out["bs_b8"] + out["bs_b4"]).where(lambda s: s > 0)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.nansum(a * b, axis=1)
    den = np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def derive_context(ctx: dict[str, pd.DataFrame], site: pd.DataFrame) -> pd.DataFrame:
    """ctx: radius name -> DataFrame of A00..A63 means; site: the site pixel's A00..A63."""
    parts = []
    for name, df in ctx.items():
        m = df.reindex(columns=list(C.AE_COLS)).apply(pd.to_numeric, errors="coerce")
        p = m.add_prefix(f"aectx{name}_")
        p[f"aectx_sim{name}"] = cosine(site.reindex(columns=list(C.AE_COLS)).to_numpy(float), m.to_numpy(float))
        parts.append(p)
    return pd.concat(parts, axis=1)


ASSET_DIR = f"projects/{PROJECT}/assets/geosoil"


def chunk_image(ee, part: str, images: dict, chunk: pd.DataFrame):
    """The server-side table for one chunk: one feature per site, all columns this part needs."""
    if part == "bare_soil":
        return images["bs"].reduceRegions(collection=_features(ee, chunk, 60), reducer=ee.Reducer.mean(),
                                          scale=20, tileScale=8)
    pts = _features(ee, chunk, 5)
    fc = images["ae"].reduceRegions(collection=pts, reducer=ee.Reducer.mean(), scale=10, tileScale=4)
    for name, (r, sc) in RADII.items():
        ctx = images["ae"].rename([f"c{name}_{b}" for b in C.AE_COLS])
        fc = ctx.reduceRegions(collection=fc.map(lambda f, r=r: f.setGeometry(f.geometry().centroid(1).buffer(r))), reducer=ee.Reducer.mean(),
                               scale=sc, tileScale=4)
    return fc.map(lambda f: f.setGeometry(f.geometry().centroid(1)))   # asset exports need a geometry


def to_frame(part: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Downloaded table (site_id + raw band means) -> final feature columns."""
    raw = raw.set_index(raw["site_id"].astype(str))
    if part == "bare_soil":
        out = derive_bare_soil(raw)
    else:
        site = raw.reindex(columns=list(C.AE_COLS))
        ctx = {n: raw.reindex(columns=[f"c{n}_{b}" for b in C.AE_COLS]).set_axis(list(C.AE_COLS), axis=1)
               for n in RADII}
        out = derive_context(ctx, site)
    out.insert(0, "site_id", out.index.to_numpy())
    return out.reset_index(drop=True)


def _load_state(out: Path) -> dict:
    f = out / "ee_tasks.json"
    state = json.loads(f.read_text()) if f.exists() else {}
    # older runs keyed tasks by index only; key by "<ntasks>:<task>" so several plans can share a directory
    return {(k if ":" in k else f"{v['ntasks']}:{k}"): v for k, v in state.items()}


def _plans(out: Path) -> dict[int, pd.DataFrame]:
    """Every submitted plan in a run directory: ntasks -> (site_id, task)."""
    return {int(f.stem.rsplit("_", 1)[1]): pd.read_parquet(f) for f in sorted(out.glob("ee_plan_*.parquet"))}


def submit(sites: pd.DataFrame, part: str, out_root: Path, chunk: int, limit: int = 0) -> dict:
    """Start Earth Engine batch exports: retry failed tasks of earlier plans, then plan any new sites.

    Each plan records which sites each task covers (ee_plan_<ntasks>.parquet), so adding labels later
    exports only the new sites, and shard names never collide between plans.
    """
    ee = init_ee()
    images = {"bs": bare_soil_image(ee)} if part == "bare_soil" else {"ae": alphaearth_image(ee)}
    out = out_root / part
    out.mkdir(parents=True, exist_ok=True)
    state, plans = _load_state(out), _plans(out)
    for folder in (f"projects/{PROJECT}/assets", ASSET_DIR):
        try:
            ee.data.createAsset({"type": "FOLDER"}, folder)
        except Exception:
            pass                                       # exists
    sites = sites.assign(site_id=sites["site_id"].astype(str))
    covered = set().union(*(set(p["site_id"]) for p in plans.values())) if plans else set()
    new = sites[~sites["site_id"].isin(covered)]
    if len(new):
        n = max(1, int(np.ceil(len(new) / chunk)))
        while n in plans:                              # distinct ntasks -> distinct shard names
            n += 1
        plan = pd.DataFrame({"site_id": new["site_id"].to_numpy(),
                             "task": new["site_id"].map(lambda k: shards.shard_of(k, n)).to_numpy()})
        plan.to_parquet(out / f"ee_plan_{n}.parquet", index=False)
        plans[n] = plan
    live = {t["id"]: t["state"] for t in ee.data.getTaskList()} if state else {}
    todo = [(n, t) for n, plan in plans.items() for t in sorted(plan["task"].unique())
            if not shards.shard_path(out, int(t), n).exists()
            and live.get(state.get(f"{n}:{t}", {}).get("task"), "FAILED") in ("FAILED", "CANCELLED")]
    if limit:
        todo = todo[:limit]
    coords = sites.set_index("site_id")[["lon", "lat"]]
    for n, t in todo:
        ids = plans[n].loc[plans[n]["task"] == t, "site_id"]
        sub = coords.reindex(ids).dropna().reset_index()
        asset = f"{ASSET_DIR}/{part}_{n}_{int(t):04d}"
        try:
            ee.data.deleteAsset(asset)                 # a failed earlier attempt may have left a partial
        except Exception:
            pass
        task = ee.batch.Export.table.toAsset(collection=chunk_image(ee, part, images, sub),
                                             description=f"geosoil_{part}_{n}_{int(t):04d}", assetId=asset)
        task.start()
        state[f"{n}:{t}"] = {"task": task.id, "asset": asset, "ntasks": n}
        (out / "ee_tasks.json").write_text(json.dumps(state, indent=1))
    return {"part": part, "new_sites": int(len(new)), "submitted": len(todo), "plans": sorted(plans)}


def collect(part: str, out_root: Path) -> dict:
    """Download every finished export into its shard; report what is still running or failed."""
    import urllib.request
    ee = init_ee()
    out = out_root / part
    state = _load_state(out)
    tasks = {t["id"]: t for t in ee.data.getTaskList()}
    counts, errors = {}, []
    for key, info in state.items():
        ntasks, t = info["ntasks"], int(key.split(":")[1])
        if shards.shard_path(out, t, ntasks).exists():
            counts["collected"] = counts.get("collected", 0) + 1
            continue
        st = tasks.get(info["task"], {}).get("state", "UNKNOWN")
        counts[st] = counts.get(st, 0) + 1
        if st == "FAILED":
            errors.append({"block": key, "error": tasks[info["task"]].get("error_message", "")[:200]})
        if st != "COMPLETED":
            continue
        url = ee.FeatureCollection(info["asset"]).getDownloadURL(filetype="csv")
        raw = pd.read_csv(urllib.request.urlopen(url))
        df = to_frame(part, raw.drop(columns=[c for c in ("system:index", ".geo") if c in raw]))
        shards.write(df, out, t, ntasks, {
            "source": part, "project": PROJECT, "asset": info["asset"], "ee_task": info["task"],
            "years": YEARS if part == "bare_soil" else AE_YEARS,
            "licence": "Sentinel-2: Copernicus open licence; AlphaEarth: CC-BY 4.0 (Google)",
            "sites": int(len(df)), "failed_blocks": [],
            "coverage": round(float(df.iloc[:, 1:].notna().any(axis=1).mean()), 4)})
        counts["collected"] = counts.get("collected", 0) + 1
    return {"part": part, **counts, "errors": errors[:3]}


def interactive(sites: pd.DataFrame, part: str, out_root: Path, workers: int = 3, sub: int = 250) -> dict:
    """Fill missing shards of the existing plans with direct requests instead of batch exports.

    Only for parts light enough to finish inside the interactive time limit (alphaearth_context:
    ~35 s per 250 sites). Bare-soil composites over 7 years must use batch exports.
    Interactive requests do not queue behind the project's four concurrent batch tasks.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ee = init_ee()
    images = {"ae": alphaearth_image(ee)} if part == "alphaearth_context" else {"bs": bare_soil_image(ee)}
    out = out_root / part
    coords = sites.assign(site_id=sites["site_id"].astype(str)).set_index("site_id")[["lon", "lat"]]
    todo = [(n, int(t), plan.loc[plan["task"] == t, "site_id"]) for n, plan in _plans(out).items()
            for t in sorted(plan["task"].unique()) if not shards.shard_path(out, int(t), n).exists()]

    def work(n, t, ids):
        chunk = coords.reindex(ids).dropna().reset_index()
        parts = []
        for i in range(0, len(chunk), sub):
            for attempt in range(8):                   # restricted-mode projects hit 429s often
                try:
                    fc = chunk_image(ee, part, images, chunk.iloc[i:i + sub])
                    parts.append(pd.DataFrame([f["properties"] for f in fc.getInfo()["features"]]))
                    break
                except Exception:
                    if attempt == 7:
                        raise
                    import random
                    import time
                    time.sleep(min(300, 15 * 2 ** attempt) * random.uniform(0.5, 1.5))
        df = to_frame(part, pd.concat(parts, ignore_index=True))
        shards.write(df, out, t, n, {
            "source": part, "project": PROJECT, "mode": "interactive",
            "years": YEARS if part == "bare_soil" else AE_YEARS,
            "licence": "Sentinel-2: Copernicus open licence; AlphaEarth: CC-BY 4.0 (Google)",
            "sites": int(len(df)), "failed_blocks": [],
            "coverage": round(float(df.iloc[:, 1:].notna().any(axis=1).mean()), 4)})
        return t

    done, failed = 0, []
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(work, n, t, ids): (n, t) for n, t, ids in todo}
        for f in as_completed(futs):
            try:
                f.result()
                done += 1
            except Exception as e:
                failed.append({"block": f"{futs[f][0]}:{futs[f][1]}", "error": repr(e)[:200]})
            print(f"{part}: {done}/{len(todo)} shards, {len(failed)} failed", flush=True)
    return {"part": part, "written": done, "failed": len(failed), "errors": failed[:3]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=PARTS, required=True)
    ap.add_argument("action", choices=("submit", "collect", "interactive"))
    ap.add_argument("--labels", type=Path, default=C.DATA_PROC / "labels_c1.parquet")
    ap.add_argument("--out-root", type=Path, default=C.DATA_PROC / "features")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--limit", type=int, default=0, help="submit only this many chunks (trial runs)")
    ap.add_argument("--workers", type=int, default=3,
                    help="interactive concurrency; keep low for projects in Earth Engine restricted mode")
    a = ap.parse_args(argv)
    if a.action == "submit":
        sites = pd.read_parquet(a.labels, columns=["site_id", "lon", "lat"]).drop_duplicates("site_id")
        res = submit(sites, a.part, a.out_root, a.chunk, a.limit)
    elif a.action == "interactive":
        sites = pd.read_parquet(a.labels, columns=["site_id", "lon", "lat"]).drop_duplicates("site_id")
        res = interactive(sites, a.part, a.out_root, workers=a.workers)
    else:
        res = collect(a.part, a.out_root)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
