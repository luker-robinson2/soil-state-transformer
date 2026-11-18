"""Use a regional model only where it helps: nested selection of the serving tier per cell (LOCAL-MODELS.md).

local_eval.py --parts regional records, per target, radius, fold and validation cell, the squared error of the global
model, the regional blend, and the random-subset control. A rule picked by looking at the same cells it is scored on
would overstate the gain (selection bias), so selection is nested: for each held-out fold k, choose on the OTHER
folds' cells a density threshold N (use the blend in a cell only if at least N training labels lie within the
radius), apply it to fold k's cells, and score. Reported per target and radius:

  mse_global, mse_blend_all, mse_selected     row-weighted MSE over all cells, selection applied out-of-fold
  gain_selected                               (mse_global - mse_selected) / mse_global, with a block-bootstrap CI
                                              over cells and a sign-flip p-value (cells as the exchangeable unit)
  share_blended                               share of validation rows served by the blend under the selected rule
  locality_margin                             blend gain minus random-subset-control gain on the blended cells:
                                              > 0 means the gain comes from locality, not from a second model

    python -m geosoil.tiers results/runs/local_eval_<stamp>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS = (0, 25, 50, 100, 200, 400, 800, 1600, 3200, np.inf)   # inf = never blend (global only)


def _sse(c: pd.DataFrame, col: str) -> np.ndarray:
    return c[col].to_numpy() * c["n_val"].to_numpy()


def nested_selection(cells: pd.DataFrame) -> pd.DataFrame:
    """Per row (cell) of one target x radius: was it blended under the rule chosen on the other folds?"""
    out = []
    for k, held in cells.groupby("fold"):
        train = cells[cells["fold"] != k]
        best, best_sse = np.inf, np.inf
        for t in THRESHOLDS:
            use = train["n_local_train"] >= t
            sse = np.where(use, _sse(train, "mse_blend"), _sse(train, "mse_global")).sum()
            if sse < best_sse - 1e-12:
                best, best_sse = t, sse
        h = held.copy()
        h["threshold"] = best
        h["blended"] = h["n_local_train"] >= best
        out.append(h)
    return pd.concat(out, ignore_index=True)


def summarise(sel: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    sg, sb, sc = _sse(sel, "mse_global"), _sse(sel, "mse_blend"), _sse(sel, "mse_control")
    ss = np.where(sel["blended"], sb, sg)
    n = sel["n_val"].to_numpy()
    d = sg - ss                                     # per-cell SSE saved by selection
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    gains = d[idx].sum(1) / sg[idx].sum(1)
    flips = rng.choice([-1.0, 1.0], (n_boot, len(d)))
    null = (flips * d).sum(1)
    b = sel["blended"].to_numpy()
    margin = ((sg[b] - sb[b]).sum() - (sg[b] - sc[b]).sum()) / sg[b].sum() if b.any() else float("nan")
    return {"cells": int(len(sel)), "rows": int(n.sum()),
            "mse_global": float(sg.sum() / n.sum()), "mse_blend_all": float(sb.sum() / n.sum()),
            "mse_selected": float(ss.sum() / n.sum()),
            "gain_blend_all": float((sg.sum() - sb.sum()) / sg.sum()),
            "gain_selected": float(d.sum() / sg.sum()),
            "gain_lo": float(np.quantile(gains, 0.025)), "gain_hi": float(np.quantile(gains, 0.975)),
            "p_signflip": float((np.sum(np.abs(null) >= abs(d.sum())) + 1) / (n_boot + 1)),
            "share_blended": float(n[b].sum() / n.sum()), "locality_margin": float(margin),
            "thresholds": sorted({float(x) for x in sel["threshold"]})}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("result", type=Path)
    a = ap.parse_args(argv)
    res = json.loads(a.result.read_text())
    out = {}
    for t, by_scale in res.items():                  # local_eval layout: res[target][scale]["regional"]["cells"]
        if not isinstance(by_scale, dict):
            continue
        for split, r in by_scale.items():
            cells = (r.get("regional") or {}).get("cells") if isinstance(r, dict) else None
            if not cells:
                continue
            cdf = pd.DataFrame(cells)
            for R, c in cdf.groupby("radius_km"):
                s = summarise(nested_selection(c))
                out.setdefault(t, {}).setdefault(split, {})[f"{int(R)}km"] = s
                print(f"{t:5s} {split:7s} {int(R):4d} km  global {s['mse_global']:.4f}  blend-all {s['gain_blend_all']:+.1%}  "
                      f"selected {s['gain_selected']:+.1%} [{s['gain_lo']:+.1%}, {s['gain_hi']:+.1%}] p={s['p_signflip']:.3f}  "
                      f"blended {s['share_blended']:.0%}  locality {s['locality_margin']:+.1%}", flush=True)
    dest = a.result.with_name(a.result.stem + "_tiers.json")
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
