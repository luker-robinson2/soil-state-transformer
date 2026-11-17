"""Noise floor: how much of each target's variance no map can explain (EXECUTION-PLAN 1.11).

KSSL does not publish analytical replicates in the tables we ingest, so the floor is
estimated from co-located pedons: distinct KSSL pedons within MAX_DIST_M of each other,
same standard depth interval, same method family. Half the mean squared difference of
such pairs is the nugget (lab error + sub-30 m soil variation). Against the total
variance of the same values it gives the best R² a model with ~10 m inputs could reach:

    r2_ceiling = 1 - nugget / total_variance

Coordinates shared by more than MAX_SITES_PER_POINT pedons are treated as placeholders
(e.g. county centroids) and skipped, since their pairs would not be co-located.

Output: geosoil/results/noise_floor.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .. import config as C
from .harmonize import to_intervals

MAX_DIST_M = 30.0
MAX_SITES_PER_POINT = 3
EARTH_R = 6_371_000.0
TOPSOIL = (0.0, 5.0, 15.0)          # interval tops pooled as "topsoil" (0-30 cm)


def colocated_pairs(sites: pd.DataFrame, max_dist_m: float = MAX_DIST_M,
                    max_sites_per_point: int = MAX_SITES_PER_POINT) -> pd.DataFrame:
    """Pairs of distinct sites within max_dist_m; sites: site_id, lon, lat."""
    key = sites["lon"].round(6).astype(str) + "_" + sites["lat"].round(6).astype(str)
    crowd = key.map(key.value_counts())
    s = sites[crowd <= max_sites_per_point].reset_index(drop=True)
    if len(s) < 2:
        return pd.DataFrame({"a": [], "b": [], "dist_m": []})
    X = np.radians(s[["lat", "lon"]].to_numpy())
    nbrs, dists = BallTree(X, metric="haversine").query_radius(X, r=max_dist_m / EARTH_R, return_distance=True)
    i = np.repeat(np.arange(len(s)), [len(n) for n in nbrs])
    j = np.concatenate(nbrs) if len(nbrs) else np.array([], int)
    d = np.concatenate(dists) * EARTH_R if len(dists) else np.array([])
    keep = i < j
    return pd.DataFrame({"a": s["site_id"].to_numpy()[i[keep]], "b": s["site_id"].to_numpy()[j[keep]],
                         "dist_m": d[keep]})


def floor_table(iv: pd.DataFrame, pairs: pd.DataFrame, log_targets=C.LOG_TARGETS) -> dict:
    """Nugget, total variance and R² ceiling per target (topsoil and all depths)."""
    key = ["target", "method_family", "depth_top"]
    va = iv.rename(columns={"site_id": "a", "value": "va"})[["a", "va"] + key]
    vb = iv.rename(columns={"site_id": "b", "value": "vb"})[["b", "vb"] + key]
    m = pairs.merge(va, on="a").merge(vb, on=["b"] + key)
    out = {}
    for t in sorted(iv["target"].unique()):
        for scope, tops in (("topsoil_0_30", TOPSOIL), ("all_depths", None)):
            mt = m[(m["target"] == t) & (m["depth_top"].isin(tops) if tops else True)]
            it = iv[(iv["target"] == t) & (iv["depth_top"].isin(tops) if tops else True)]
            if len(mt) < 30:
                continue
            for transform in ("raw", "log1p") if t in log_targets else ("raw",):
                f = np.log1p if transform == "log1p" else (lambda x: x)
                nugget = 0.5 * float(np.mean((f(mt["va"]) - f(mt["vb"])) ** 2))
                total = float(np.var(f(it["value"])))
                out[f"{t}/{scope}/{transform}"] = dict(
                    n_pairs=int(len(mt)), nugget=nugget, total_variance=total,
                    r2_ceiling=1 - nugget / total, median_pair_dist_m=float(mt["dist_m"].median()))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    k = pd.read_parquet(C.DATA_PROC / "labels_kssl_ldm.parquet")
    iv = to_intervals(k)
    pairs = colocated_pairs(k.drop_duplicates("site_id")[["site_id", "lon", "lat"]])
    table = floor_table(iv, pairs)
    result = {"method": __doc__.split("\n\n")[1].replace("\n", " "), "max_dist_m": MAX_DIST_M,
              "max_sites_per_point": MAX_SITES_PER_POINT, "n_site_pairs": int(len(pairs)), "targets": table}
    (C.RESULTS / "noise_floor.json").write_text(json.dumps(result, indent=2))
    for k_, v in table.items():
        print(f"{k_:28s} pairs {v['n_pairs']:6d}  R2 ceiling {v['r2_ceiling']:.3f}")


if __name__ == "__main__":
    main()
