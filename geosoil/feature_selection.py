"""Feature-selection report: which feature groups matter, per target, under spatial CV (step 2.3).

Groups, not columns: a group is a modality (terrain, AlphaEarth, climate annual, climate monthly,
radiometrics, location, depth). Column-level importance on correlated covariates is unstable,
and random-CV importance rewards features that just proxy location (Ploton et al. 2020), so
everything here is computed on held-out spatial folds.

Three measures per (target, group):
  perm_drop   grouped permutation importance: R2 lost when the group's columns are shuffled
              together on the held-out fold (other groups intact). Understates groups that a
              correlated group can stand in for.
  shap_share  share of mean |SHAP| (LightGBM pred_contrib) on the held-out fold.
  drop_group  R2 lost when the model is retrained without the group (--drop-group; slower).
  top_columns the 25 individual columns with the largest mean |SHAP| share (which AlphaEarth
              dimensions, climate months, lithology classes ... drive each target).
              This is the decision measure: keep a group if dropping it costs more than the
              fold-to-fold noise for at least one target (CORPUS-PLAN.md section D).

    python -m geosoil.feature_selection [--drop-group] [--max-rows 150000]
    -> geosoil/results/runs/feature_selection_<time>.json and .md
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import json

import numpy as np
import pandas as pd

from . import config as C
from .features import terrain
from .metrics import metrics

TOPSOIL = ("0-5", "5-15", "15-30")
TOP_COLUMNS = 25


def feature_groups(columns) -> dict[str, list[str]]:
    from .features.groups import feature_groups as fg
    return fg(columns)


def _model(seed: int = 0):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)


def _r2(y, p) -> float:
    return metrics(y, p)["r2"]


def importance(d: pd.DataFrame, y: np.ndarray, groups: dict[str, list[str]], drop_group: bool,
               seed: int = 0) -> dict:
    cols = [c for g in groups.values() for c in g]
    X = d[cols].to_numpy(np.float32)
    idx = {g: [cols.index(c) for c in v] for g, v in groups.items()}
    rng = np.random.default_rng(seed)
    folds = sorted(d["fold"].unique())
    base, perm, shap, drop = [], {g: [] for g in groups}, {g: [] for g in groups}, {g: [] for g in groups}
    col_shap = np.zeros(len(cols))
    for k in folds:
        tr, va = (d["fold"] != k).to_numpy(), (d["fold"] == k).to_numpy()
        m = _model(seed).fit(X[tr], y[tr])
        r0 = _r2(y[va], m.predict(X[va]))
        base.append(r0)
        contrib = np.abs(m.predict(X[va], pred_contrib=True)[:, :-1]).mean(axis=0)
        total = contrib.sum() or 1.0
        col_shap += contrib / total / len(folds)
        for g, ii in idx.items():
            Xp = X[va].copy()
            Xp[:, ii] = Xp[rng.permutation(len(Xp))][:, ii]          # shuffle the group's rows jointly
            perm[g].append(r0 - _r2(y[va], m.predict(Xp)))
            shap[g].append(float(contrib[ii].sum() / total))
            if drop_group and len(groups) > 1:
                keep = [j for j in range(X.shape[1]) if j not in ii]
                md = _model(seed).fit(X[tr][:, keep], y[tr])
                drop[g].append(r0 - _r2(y[va], md.predict(X[va][:, keep])))
    out = {"r2_all": float(np.mean(base)), "r2_all_sd": float(np.std(base)), "groups": {}}
    top = np.argsort(col_shap)[::-1][:TOP_COLUMNS]
    group_of = {c: g for g, v in groups.items() for c in v}
    out["top_columns"] = [{"column": cols[i], "group": group_of[cols[i]], "shap_share": float(col_shap[i])} for i in top]
    for g in groups:
        out["groups"][g] = {"n_cols": len(groups[g]),
                            "perm_drop": float(np.mean(perm[g])), "perm_drop_sd": float(np.std(perm[g])),
                            "shap_share": float(np.mean(shap[g]))}
        if drop[g]:
            out["groups"][g].update(drop_group=float(np.mean(drop[g])), drop_group_sd=float(np.std(drop[g])))
    return out


def markdown(res: dict) -> str:
    lines = ["# Feature-group importance (spatial CV, lab labels, topsoil)", ""]
    for t, r in res.items():
        lines += [f"## {t}: R² all groups {r['r2_all']:.3f} ± {r['r2_all_sd']:.3f}  (ceiling {r.get('r2_ceiling')})", "",
                  "| group | cols | permutation ΔR² | drop-group ΔR² | SHAP share |", "|---|---|---|---|---|"]
        for g, v in sorted(r["groups"].items(), key=lambda kv: -kv[1].get("drop_group", kv[1]["perm_drop"])):
            dg = f"{v['drop_group']:.3f} ± {v['drop_group_sd']:.3f}" if "drop_group" in v else "–"
            lines.append(f"| {g} | {v['n_cols']} | {v['perm_drop']:.3f} ± {v['perm_drop_sd']:.3f} | {dg} | {v['shap_share']:.2f} |")
        if r.get("top_columns"):
            lines += ["", "Top columns by SHAP share: " + ", ".join(
                f"{c['column']} ({c['shap_share']:.3f})" for c in r["top_columns"][:12])]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--drop-group", action="store_true")
    ap.add_argument("--max-rows", type=int, default=150_000, help="rows per target (0 = all)")
    a = ap.parse_args(argv)
    from .baseline_c1 import load_topsoil
    df = load_topsoil(train_only=True)
    df["depth_mid"] = (df["depth_top"] + df["depth_bot"]) / 2
    floor = json.loads((C.RESULTS / "noise_floor.json").read_text())["targets"]
    groups = feature_groups(df.columns)
    res = {}
    for t in C.TARGETS:
        d = df[df["target"] == t].dropna(subset=["value"])
        core = groups.get("alphaearth", []) + groups.get("terrain", [])
        d = d[d[core].notna().any(axis=1)] if core else d
        if a.max_rows and len(d) > a.max_rows:
            d = d.sample(a.max_rows, random_state=0)
        y = np.log1p(d["value"].to_numpy()) if t in C.LOG_TARGETS else d["value"].to_numpy()
        res[t] = importance(d, y, groups, a.drop_group)
        res[t]["n"], res[t]["sites"] = int(len(d)), int(d["site_id"].nunique())
        res[t]["r2_ceiling"] = floor.get(f"{t}/topsoil_0_30/{'log1p' if t in C.LOG_TARGETS else 'raw'}", {}).get("r2_ceiling")
        print(t, json.dumps({g: round(v.get("drop_group", v["perm_drop"]), 3) for g, v in res[t]["groups"].items()}))
    runs = C.RESULTS / "runs"
    runs.mkdir(exist_ok=True)
    stamp = f"{pd.Timestamp.now():%Y%m%dT%H%M}"
    (runs / f"feature_selection_{stamp}.json").write_text(json.dumps(res, indent=2))
    (runs / f"feature_selection_{stamp}.md").write_text(markdown(res))
    print(markdown(res))


if __name__ == "__main__":
    main()
