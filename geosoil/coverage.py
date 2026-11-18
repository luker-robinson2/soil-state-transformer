"""Where would new labels help most? Area of applicability of the training data (Meyer & Pebesma 2021, MEE 12:1620).

The zone experiment (LOCAL-MODELS.md) showed the model's real weakness is unseen environments: holding out a
pedo-climatic zone costs 0.09-0.18 R². This module measures, for the world's land, how far each place is from the
training data in feature space, so data acquisition can target the gaps:

  1. sample   uniform random points over land (equal-area: latitude drawn by arcsin), keep those with climate data
  2. features the same climate normals, land-surface temperature and GLiM lithology the models use, extracted with
              the existing extractors (alphaearth is left out: it needs block reads per point; the pedo-climatic
              covariates already define the zones)
  3. DI       dissimilarity index: standardise the features on the training sites, project to principal components
              (removes redundancy among the 150+ correlated climate/LST columns), then
              DI = distance to the nearest training site / mean nearest-neighbour distance among training sites
  4. AOA      a place is inside the area of applicability if its DI is below the training data's own cross-validated
              DI (each training site vs the nearest training site in OTHER spatial folds), upper whisker
              Q3 + 1.5 IQR as in CAST
  5. report   share of land outside the AOA and mean DI by country and continent (Natural Earth 1:110m), largest
              uncovered areas first

    python -m geosoil.coverage --n 30000        # -> geosoil/results/runs/coverage_<time>.json (+ points parquet)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .corpus import common

N_PCS = 20
COUNTRIES = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
PREFIXES = ("clim_", "lst_", "litho_glim_")


def sample_points(n: int, seed: int = 0) -> pd.DataFrame:
    """Uniform on the sphere between 56°S and 78°N (excludes Antarctica and the high Arctic)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.sin(np.radians(-56)), np.sin(np.radians(78))
    lat = np.degrees(np.arcsin(rng.uniform(lo, hi, n)))
    lon = rng.uniform(-180, 180, n)
    return pd.DataFrame({"site_id": [f"cov:{i}" for i in range(n)], "lon": lon, "lat": lat})


def extract_features(pts: pd.DataFrame, root: Path) -> pd.DataFrame:
    from .features import climate, lithology, rasters, thermal
    from .features.join import read_shards
    rasters.extract(climate.source(), pts, out_root=root)
    land = read_shards(root / "climate")
    land = land[land[[c for c in land.columns if c.startswith("clim_")]].notna().any(axis=1)]
    sites = pts[pts["site_id"].isin(land["site_id"])]
    rasters.extract(thermal.source(), sites, out_root=root)
    lithology.extract(sites, out_root=root)
    df = sites[["site_id", "lon", "lat"]]
    for d in ("climate", "thermal", "lithology"):
        df = df.merge(read_shards(root / d), on="site_id", how="left")
    return df


def feature_columns(columns) -> list[str]:
    return [c for c in columns if c.startswith(PREFIXES)]


def dissimilarity(train: pd.DataFrame, query: pd.DataFrame, cols: list[str], folds: np.ndarray | None = None) -> dict:
    """DI for query points and the cross-validated DI threshold of the training data."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    Xt = train[cols].to_numpy(float)
    mu, sd = np.nanmean(Xt, axis=0), np.nanstd(Xt, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = lambda X: np.nan_to_num((X - mu) / sd, nan=0.0).clip(-10, 10)
    pca = PCA(n_components=min(N_PCS, len(cols)), random_state=0).fit(z(Xt))
    Pt, Pq = pca.transform(z(Xt)), pca.transform(z(query[cols].to_numpy(float)))
    nn = NearestNeighbors(n_neighbors=2).fit(Pt)
    d_tt = nn.kneighbors(Pt)[0][:, 1]                 # nearest other training site
    dbar = float(np.mean(d_tt))
    di_q = nn.kneighbors(Pq, n_neighbors=1)[0][:, 0] / dbar
    if folds is not None:                             # training DI against other spatial folds only
        di_t = np.empty(len(Pt))
        for k in np.unique(folds):
            m = folds == k
            nnk = NearestNeighbors(n_neighbors=1).fit(Pt[~m])
            di_t[m] = nnk.kneighbors(Pt[m])[0][:, 0] / dbar
    else:
        di_t = d_tt / dbar
    q1, q3 = np.percentile(di_t, [25, 75])
    thr = float(q3 + 1.5 * (q3 - q1))
    return {"di": di_q, "threshold": thr, "train_di": di_t, "dbar": dbar,
            "explained_variance": float(pca.explained_variance_ratio_.sum())}


def by_country(pts: pd.DataFrame) -> pd.DataFrame:
    import geopandas as gpd
    path = common.fetch(COUNTRIES, common.RAW / "naturalearth" / "ne_110m_admin_0_countries.zip")
    countries = gpd.read_file(f"zip://{path}")[["NAME", "CONTINENT", "geometry"]]
    g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["lon"], pts["lat"]), crs="EPSG:4326")
    j = gpd.sjoin(g, countries, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    return pd.DataFrame(j.drop(columns=["geometry", "index_right"], errors="ignore"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=30000, help="random points before the land mask (~30% are land)")
    ap.add_argument("--table", default=str(C.DATA_PROC / "train_c1.parquet"))
    a = ap.parse_args(argv)
    t0 = time.time()
    root = C.DATA_PROC / "coverage_sample"
    pts = extract_features(sample_points(a.n), root)
    train = pd.read_parquet(a.table, filters=[("interval", "in", ["0-5", "5-15", "15-30"])])
    cols = [c for c in feature_columns(train.columns) if c in pts.columns]
    train = train.drop_duplicates("site_id")
    train = train[train[cols].notna().mean(axis=1) > 0.5]
    res = dissimilarity(train, pts, cols, folds=train["fold"].to_numpy())
    pts["di"], pts["outside_aoa"] = res["di"], res["di"] > res["threshold"]
    pts = by_country(pts)
    agg = lambda g: pd.Series({"land_points": len(g), "share_outside": float(g["outside_aoa"].mean()),
                               "mean_di": float(g["di"].mean()), "outside_points": int(g["outside_aoa"].sum())})
    cont = pts.groupby("CONTINENT").apply(agg).sort_values("outside_points", ascending=False)
    ctry = pts.groupby(["CONTINENT", "NAME"]).apply(agg).sort_values("outside_points", ascending=False)
    runs = C.RESULTS / "runs"
    runs.mkdir(exist_ok=True)
    stamp = f"{pd.Timestamp.now():%Y%m%dT%H%M}"
    pts[["site_id", "lon", "lat", "di", "outside_aoa", "NAME", "CONTINENT"]].to_parquet(runs / f"coverage_points_{stamp}.parquet")
    out = {"features": len(cols), "pcs_explained_variance": res["explained_variance"], "threshold": res["threshold"],
           "train_sites": int(len(train)), "land_points": int(len(pts)), "share_land_outside": float(pts["outside_aoa"].mean()),
           "by_continent": cont.reset_index().to_dict("records"),
           "top_countries": ctry.reset_index().head(40).to_dict("records"), "seconds": round(time.time() - t0, 1)}
    (runs / f"coverage_{stamp}.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("features", "threshold", "land_points", "share_land_outside")}, indent=2))
    print(cont.head(8).to_string())
    print(ctry.head(20).to_string())


if __name__ == "__main__":
    main()
