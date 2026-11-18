"""Cross-source deduplication of label tables → the C1 label set (EXECUTION-PLAN step 1.9).

1. Within each source, sites with identical coordinates are merged into one site_id
   (their layers are kept; they are distinct samples at one point).
2. Across sources, a site within RADIUS_M of an already-kept site, with overlapping depth
   span, is a duplicate, but only when the two sources are known to share records
   (REPUBLISHES: WoSIS and OSSL are compilations of national data). Independent surveys
   (USGS DS 801, NAPESHM) are never dropped for being near another source's site; those are
   different samples. Native sources outrank the compilations that copy them.
3. Hold-out sources (RaCA) are never deduplicated; any training site within
   HOLDOUT_RADIUS_M of a hold-out site is flagged near_holdout so the split step can
   keep it out of training folds that are scored on the hold-out.

Every dropped or merged site is logged with its keeper and the distance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

PRIORITY = ("kssl_ldm", "canada_npdb", "brazil_soildata", "rmqs", "wosis_2023", "sislac", "olm_compendium", "ossl_v1_2", "napeshm",
            "usgs_ds801", "fia", "nwca", "sodes", "bze_lw", "chspd", "icraf_afsis", "qld_sali", "nspd", "wosis_latest", "lucas_bd2018", "icrisat_karnataka")
# source -> higher-priority sources whose records it may contain
REPUBLISHES = {
    "wosis_2023": {"kssl_ldm", "canada_npdb", "brazil_soildata", "rmqs"},   # WoSIS holds RMQS1 without its bases
    "ossl_v1_2": {"kssl_ldm", "wosis_2023", "brazil_soildata"},
    "sislac": {"wosis_2023", "brazil_soildata"},    # WoSIS holds some of the same national surveys
    # a compilation: only sources we lack are loaded, but some of them also reached WoSIS (e.g. SOTER China/Iran)
    "olm_compendium": {"kssl_ldm", "wosis_2023", "brazil_soildata", "canada_npdb", "sislac"},
    "nwca": {"kssl_ldm"},                           # KSSL ran the NWCA analyses
    "chspd": {"wosis_2023"},
    "icraf_afsis": {"wosis_2023"},
    "qld_sali": {"wosis_2023", "olm_compendium"},  # NatSoil (via WoSIS and the compendium) holds Queensland profiles
    "nspd": {"wosis_2023", "olm_compendium"},
    "wosis_latest": {"wosis_2023"},
}
# Gap-fill sources: a duplicate site keeps the targets its keeper lacks instead of being dropped whole. The
# compendium carries exchangeable K/Ca/Mg and CEC at ~30k sites whose WoSIS copy has neither (2026-11-18).
# A kept remainder is co-located with its keeper, so both always fall in the same 1° block and fold.
GAP_FILL = {"olm_compendium", "chspd", "icraf_afsis", "qld_sali", "nspd"}
HOLDOUT = ("raca",)
RADIUS_M = 100.0
HOLDOUT_RADIUS_M = 1000.0
EARTH_R = 6_371_000.0


def _sites(labels: pd.DataFrame) -> pd.DataFrame:
    return labels.groupby("site_id", sort=False).agg(
        source=("source", "first"), lon=("lon", "first"), lat=("lat", "first"),
        top=("depth_top", "min"), bot=("depth_bot", "max")).reset_index()


def _rad(df: pd.DataFrame) -> np.ndarray:
    return np.radians(df[["lat", "lon"]].to_numpy())


def merge_same_coordinates(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within each source, give sites at identical coordinates one site_id."""
    s = _sites(labels)
    key = s["source"] + "|" + s["lon"].round(6).astype(str) + "|" + s["lat"].round(6).astype(str)
    canon = s.groupby(key)["site_id"].transform("first")
    merged = s[canon != s["site_id"]]
    log = pd.DataFrame({"site_id": merged["site_id"], "source": merged["source"], "action": "merged",
                        "keeper_site_id": canon[merged.index], "keeper_source": merged["source"],
                        "distance_m": 0.0, "reason": "same coordinates within source"})
    mapping = dict(zip(s["site_id"], canon))
    out = labels.copy()
    out["site_id"] = out["site_id"].map(mapping)
    return out, log


def drop_cross_source(labels: pd.DataFrame, priority=PRIORITY, radius_m: float = RADIUS_M,
                      republishes=REPUBLISHES, gap_fill=GAP_FILL):
    """Keep the highest-priority copy of sites that appear in more than one source."""
    s = _sites(labels)
    rank = {src: i for i, src in enumerate(priority)}
    s["rank"] = s["source"].map(rank).fillna(len(priority))
    s = s.sort_values(["rank", "site_id"]).reset_index(drop=True)
    tree = BallTree(_rad(s), metric="haversine")
    nbrs, dists = tree.query_radius(_rad(s), r=radius_m / EARTH_R, return_distance=True)
    kept = np.zeros(len(s), dtype=bool)
    rows = []
    targets = {}
    if gap_fill:   # explicit sets: groupby().agg(set) returns lists for pandas' string dtype
        pairs = labels.loc[labels["source"].isin(set(gap_fill) | {k for v in republishes.values() for k in v}),
                           ["site_id", "target"]].drop_duplicates()
        for sid, t in zip(pairs["site_id"].astype(str), pairs["target"].astype(str)):
            targets.setdefault(sid, set()).add(t)
    partial = set()                                       # (site_id, target) rows dropped from gap-fill sites
    src, top, bot = s["source"].to_numpy(), s["top"].to_numpy(), s["bot"].to_numpy()
    for i in range(len(s)):
        hit = None
        for j, d in zip(nbrs[i], dists[i]):
            if (j != i and kept[j] and src[j] in republishes.get(src[i], ())
                    and top[i] < bot[j] and top[j] < bot[i]):
                if hit is None or d < hit[1]:
                    hit = (j, d)
        if hit is None:
            kept[i] = True
        else:
            j, d = hit
            sid, kid = s.at[i, "site_id"], s.at[j, "site_id"]
            extra = targets.get(sid, set()) - targets.get(kid, set()) if src[i] in gap_fill else set()
            if extra:
                partial |= {(sid, t) for t in targets[sid] - extra}
                rows.append(dict(site_id=sid, source=src[i], action="gap_fill", keeper_site_id=kid, keeper_source=src[j],
                                 distance_m=float(d * EARTH_R), reason=f"kept {sorted(extra)}, which the keeper lacks"))
                continue
            rows.append(dict(site_id=sid, source=src[i], action="dropped",
                             keeper_site_id=kid, keeper_source=src[j],
                             distance_m=float(d * EARTH_R), reason=f"within {radius_m:g} m of a site in a source it republishes"))
    log = pd.DataFrame(rows, columns=["site_id", "source", "action", "keeper_site_id", "keeper_source",
                                      "distance_m", "reason"])
    gone = set(log.loc[log["action"] == "dropped", "site_id"])
    keep = ~labels["site_id"].isin(gone)
    if partial:
        keep &= ~pd.Series(list(zip(labels["site_id"], labels["target"])), index=labels.index).isin(partial)
    return labels[keep], log


def flag_near_holdout(train: pd.DataFrame, holdout: pd.DataFrame, radius_m: float = HOLDOUT_RADIUS_M) -> pd.Series:
    """Boolean per training row: is its site within radius_m of any hold-out site?"""
    if holdout.empty or train.empty:
        return pd.Series(False, index=train.index)
    hs, ts = _sites(holdout), _sites(train)
    d, _ = BallTree(_rad(hs), metric="haversine").query(_rad(ts), k=1)
    near = dict(zip(ts["site_id"], d[:, 0] * EARTH_R <= radius_m))
    return train["site_id"].map(near).astype(bool)


def build_c1(tables: list[pd.DataFrame], priority=PRIORITY, holdout=HOLDOUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine per-source label tables into C1. Adds `role` (train/holdout) and `near_holdout`."""
    labels = pd.concat(tables, ignore_index=True)
    hold = labels[labels["source"].isin(holdout)].copy()
    train = labels[~labels["source"].isin(holdout)]
    train, log_merge = merge_same_coordinates(train)
    train, log_drop = drop_cross_source(train, priority)
    train = train.copy()
    train["role"] = "train"
    train["near_holdout"] = flag_near_holdout(train, hold)
    hold["role"] = "holdout"
    hold["near_holdout"] = False
    log = pd.concat([log_merge, log_drop], ignore_index=True)
    return pd.concat([train, hold], ignore_index=True), log
