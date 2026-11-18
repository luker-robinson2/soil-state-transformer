"""Harmonize C1 labels to standard depths and one method per target (EXECUTION-PLAN step 1.8).

Two steps, both recorded so the model can still learn per-method offsets:

1. Depth. Each site's layers are averaged onto standard intervals, weighting by the
   thickness of overlap (mass-preserving; no extrapolation). An interval is kept only
   where layers cover at least MIN_COVERAGE of it.
2. Method. Where a site has several method families for one target and interval, the
   preferred family wins (PRIORITY). Families with a fitted conversion to the canonical
   one (CONVERSIONS) are converted and flagged `converted`. The conversions are fitted
   on KSSL layers measured both ways, by `fit_conversions`.

Output: data/processed/labels_harmonized.parquet, one row per (site, target, interval),
and geosoil/results/harmonization.json with the fitted conversions.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .. import config as C

INTERVALS = ((0, 5), (5, 15), (15, 30), (30, 60), (60, 100), (100, 200))
MIN_COVERAGE = 0.5
# preferred family first; anything not listed is dropped (e.g. loss_on_ignition)
PRIORITY = {
    "soc": ("dry_combustion", "walkley_black", "unknown"),
    "ph": ("ph_h2o", "ph_cacl2"),
    "bd": ("bd_third_bar", "bd_fine_earth", "bd_oven_dry", "unknown"),
    "sand": ("pipette", "hydrometer", "laser_diffraction", "unknown"),
    "clay": ("pipette", "hydrometer", "laser_diffraction", "unknown"),
    "cec": ("cec_ph7", "cec_sum_cations_ph82", "ecec", "unknown"),
    "k": ("nh4oac_ph7", "mehlich3", "unknown"),
    "ca": ("nh4oac_ph7", "unknown"),       # Mehlich-3 Ca/Mg have no paired layers to fit a conversion: dropped
    "mg": ("nh4oac_ph7", "unknown"),
    "om": ("measured",),                   # derived OM (from harmonized SOC) is added after method choice
}
# (target, from_family) -> (to_family, kind); kind "linear" fits y = a + b x, "ratio" fits y = r x
CONVERTIBLE = {
    ("ph", "ph_cacl2"): ("ph_h2o", "linear"),
    ("soc", "walkley_black"): ("dry_combustion", "ratio"),
    ("bd", "bd_oven_dry"): ("bd_third_bar", "linear"),
    ("cec", "cec_sum_cations_ph82"): ("cec_ph7", "linear"),
    ("cec", "ecec"): ("cec_ph7", "linear"),
    ("k", "mehlich3"): ("nh4oac_ph7", "linear"),
}
# These families can diverge from the canonical one by soil type (ECEC vs CEC at pH 7 in acid soils, where
# aluminium dominates), so their conversion is used only when the fit explains most of the variance; otherwise
# the rows are dropped rather than mixed in unconverted.
GATED = {("cec", "cec_sum_cations_ph82"), ("cec", "ecec"), ("k", "mehlich3")}
MIN_CONVERSION_R2 = 0.8
SITE_COLS = ["site_id", "source", "lon", "lat", "coord_precision_m", "year", "license"]


def fit_conversions(labels: pd.DataFrame) -> dict:
    """Fit each conversion on layers that carry both families (KSSL in practice)."""
    out = {}
    for (target, src), (dst, kind) in CONVERTIBLE.items():
        p = labels[labels["target"] == target].pivot_table(
            index="layer_id", columns="method_family", values="value", aggfunc="first")
        if src not in p or dst not in p:
            continue
        p = p[[src, dst]].dropna()
        x, y = p[src].to_numpy(), p[dst].to_numpy()
        if kind == "ratio":
            ok = x > 0
            r = float(np.median(y[ok] / x[ok]))
            a, b = 0.0, r
        else:
            b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        out[f"{target}:{src}->{dst}"] = dict(target=target, src=src, dst=dst, kind=kind, a=float(a), b=float(b),
                                             n=int(len(p)), r2=float(1 - resid.var() / y.var()),
                                             resid_sd=float(resid.std()))
    return out


def to_intervals(labels: pd.DataFrame, intervals=INTERVALS, min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    """Thickness-weighted mean per (site, target, family, interval)."""
    parts = []
    for top, bot in intervals:
        ov = (np.minimum(labels["depth_bot"], bot) - np.maximum(labels["depth_top"], top)).clip(lower=0)
        m = ov > 0
        if not m.any():
            continue
        d = labels.loc[m, ["site_id", "target", "method_family", "value"]].assign(w=ov[m])
        d["wv"] = d["w"] * d["value"]
        g = d.groupby(["site_id", "target", "method_family"], sort=False).agg(wv=("wv", "sum"), w=("w", "sum"))
        g["value"] = g["wv"] / g["w"]
        g["coverage"] = (g["w"] / (bot - top)).clip(upper=1.0)
        g = g[g["coverage"] >= min_coverage].reset_index()
        g["depth_top"], g["depth_bot"] = float(top), float(bot)
        parts.append(g.drop(columns=["wv", "w"]))
    return pd.concat(parts, ignore_index=True)


def choose_method(iv: pd.DataFrame, conversions: dict, priority=PRIORITY) -> pd.DataFrame:
    """Keep the preferred family per (site, target, interval), converting where fitted."""
    rank = {(t, f): i for t, fams in priority.items() for i, f in enumerate(fams)}
    good = {(c["target"], c["src"]) for c in conversions.values() if c.get("r2", 1.0) >= MIN_CONVERSION_R2}
    for t, f in GATED - good:                       # no fit, or a poor one: drop instead of mixing methods
        rank.pop((t, f), None)
    conversions = {k: c for k, c in conversions.items() if (c["target"], c["src"]) not in GATED or
                   (c["target"], c["src"]) in good}
    iv = iv.copy()
    iv["rank"] = [rank.get(k, np.nan) for k in zip(iv["target"], iv["method_family"])]
    iv = iv.dropna(subset=["rank"]).sort_values("rank")
    best = iv.drop_duplicates(["site_id", "target", "depth_top"]).copy()
    best["converted"] = False
    for conv in conversions.values():
        m = (best["target"] == conv["target"]) & (best["method_family"] == conv["src"])
        best.loc[m, "value"] = conv["a"] + conv["b"] * best.loc[m, "value"]
        best.loc[m, "converted"] = True
    return best.drop(columns="rank")


def add_derived_om(best: pd.DataFrame, factor: float = C.OM_PER_SOC) -> pd.DataFrame:
    """Organic matter from harmonized SOC (x factor, van Bemmelen), so OM and SOC never disagree.

    A measured OM value is kept only where the site and interval have no SOC.
    """
    soc = best[best["target"] == "soc"].copy()
    soc["target"], soc["value"], soc["method_family"] = "om", soc["value"] * factor, "converted_from_soc"
    soc["converted"] = True
    measured = best[best["target"] == "om"]
    have = set(zip(soc["site_id"], soc["depth_top"]))
    measured = measured[[k not in have for k in zip(measured["site_id"], measured["depth_top"])]]
    return pd.concat([best[best["target"] != "om"], soc, measured], ignore_index=True)


def harmonize(c1: pd.DataFrame, conversions: dict | None = None) -> tuple[pd.DataFrame, dict]:
    if conversions is None:
        conversions = fit_conversions(c1[c1["source"] == "kssl_ldm"])
    best = add_derived_om(choose_method(to_intervals(c1), conversions))
    site_cols = [c for c in SITE_COLS + ["role", "near_holdout"] if c in c1]
    sites = c1[site_cols].drop_duplicates("site_id")
    out = best.merge(sites, on="site_id", how="left")
    out["interval"] = out["depth_top"].astype(int).astype(str) + "-" + out["depth_bot"].astype(int).astype(str)
    out["unit"] = out["target"].map(C.TARGET_UNITS)
    cols = site_cols + ["target", "interval", "depth_top", "depth_bot", "value", "unit", "method_family",
                        "converted", "coverage"]
    return out[cols].sort_values(["site_id", "target", "depth_top"]).reset_index(drop=True), conversions


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    c1 = pd.read_parquet(C.DATA_PROC / "labels_c1.parquet")
    out, conv = harmonize(c1)
    out.to_parquet(C.DATA_PROC / "labels_harmonized.parquet", index=False)
    summary = {
        "intervals_cm": [list(i) for i in INTERVALS], "min_coverage": MIN_COVERAGE, "conversions": conv,
        "rows": int(len(out)),
        "sites_by_target_interval": out.groupby(["target", "interval"])["site_id"].nunique().unstack().fillna(0)
        .astype(int).to_dict(),
        "converted_share_by_target": out.groupby("target")["converted"].mean().round(4).to_dict(),
    }
    (C.RESULTS / "harmonization.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("conversions", "rows", "converted_share_by_target")}, indent=2))


if __name__ == "__main__":
    main()
