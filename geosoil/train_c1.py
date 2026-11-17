"""C1 neural trainer (roadmap I2, plan step 2.6): one multi-target network on the C1 feature table.

Same rows, features and spatial folds as the LightGBM baseline (baseline_c1.py), so the two are
directly comparable. One row per (site, depth interval); the five targets are predicted together and
the loss is masked where a target was not measured, so a site with only pH still trains the shared trunk.

Model: residual MLP (Gorishniy et al. 2021, "Revisiting deep learning models for tabular data", NeurIPS),
a strong tabular baseline. Its role is the bridge to the foundation model: the same heads later sit on
pretrained AlphaEarth sequence encoders instead of hand-built features.

Leakage guards: feature scaling and target standardisation are fit on training folds only; early
stopping uses an inner split of whole spatial blocks from the training folds; neighbour features
(--neighbours) are built per fold and per target with features/neighbours.py.

    python -m geosoil.train_c1 [--neighbours] [--seeds 0,1,2] [--epochs 60] [--width 512 --depth 4]
    # -> geosoil/results/runs/train_c1_<time>.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import config as C
from .metrics import metrics

TOPSOIL = ("0-5", "5-15", "15-30")


def wide_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Long label rows -> one row per (site, interval) with a value column per target."""
    from .baseline_c1 import groups
    df = df[df["interval"].isin(TOPSOIL) & (df["split"] == "train")].copy()
    df["depth_mid"] = (df["depth_top"] + df["depth_bot"]) / 2
    feats = ["depth_mid", "lon", "lat"] + groups(df.columns)["all"]
    keys = ["site_id", "interval"]
    y = df.pivot_table(index=keys, columns="target", values="value", aggfunc="mean")
    x = df.drop_duplicates(keys).set_index(keys)[feats + ["block", "fold"]]
    w = x.join(y, how="inner").reset_index()
    for t in C.TARGETS:
        if t not in w:
            w[t] = np.nan
        if t in C.LOG_TARGETS:
            w[t] = np.log1p(w[t])
    return w, feats


class Scaler:
    """Robust per-column scaling fit on training rows; NaN -> 0 after scaling, plus a missing flag per column."""

    def fit(self, X: np.ndarray):
        self.med = np.nanmedian(X, axis=0)
        q = np.nanpercentile(X, [25, 75], axis=0)
        self.iqr = np.where((q[1] - q[0]) > 0, q[1] - q[0], 1.0)
        self.med = np.where(np.isnan(self.med), 0.0, self.med)
        self.flag = np.isnan(X).any(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = np.clip((X - self.med) / self.iqr, -10, 10)
        miss = np.isnan(Z[:, self.flag]).astype(np.float32)
        return np.hstack([np.nan_to_num(Z, nan=0.0), miss]).astype(np.float32)


def build_model(n_in: int, n_out: int, width: int = 512, depth: int = 4, dropout: float = 0.15):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(2 * width, width), nn.Dropout(dropout))

        def forward(self, x):
            return x + self.f(x)

    return nn.Sequential(nn.Linear(n_in, width), *[Block() for _ in range(depth)], nn.LayerNorm(width),
                         nn.Linear(width, n_out))


def masked_mse(pred, y):
    import torch
    m = ~torch.isnan(y)
    return ((pred - torch.nan_to_num(y)) ** 2 * m).sum() / m.sum().clamp(min=1)


def fit_predict(Xtr, Ytr, Xva, blocks_tr, args, seed: int, device: str) -> np.ndarray:
    """Train on (Xtr, Ytr) with early stopping on held-out blocks; return predictions for Xva."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks_tr)
    inner = np.isin(blocks_tr, rng.choice(ub, max(1, len(ub) // 10), replace=False))
    ymu, ysd = np.nanmean(Ytr[~inner], axis=0), np.nanstd(Ytr[~inner], axis=0)
    ysd = np.where(ysd > 0, ysd, 1.0)
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    xa, ya = T(Xtr[~inner]), T((Ytr[~inner] - ymu) / ysd)
    xi, yi = T(Xtr[inner]), T((Ytr[inner] - ymu) / ysd)
    model = build_model(Xtr.shape[1], Ytr.shape[1], args.width, args.depth, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * max(1, int(np.ceil(len(xa) / args.batch)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    best, best_state, bad = np.inf, None, 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xa), device=device)
        for i in range(0, len(xa), args.batch):
            b = perm[i:i + args.batch]
            loss = masked_mse(model(xa[b]), ya[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            v = float(masked_mse(model(xi), yi))
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(T(Xva[i:i + 65536])) for i in range(0, len(Xva), 65536)]).cpu().numpy()
    return out * ysd + ymu


def neighbour_block(w: pd.DataFrame, tr: np.ndarray, rows: np.ndarray, exclude: bool) -> np.ndarray:
    from .features import neighbours as NB
    cols = []
    for t in C.TARGETS:
        cols.append(NB.fold_features(w[tr], w[rows], w.loc[tr, t].to_numpy(float), exclude_same_block=exclude)
                    .to_numpy(np.float32))
    return np.hstack(cols)


def cross_validate(w: pd.DataFrame, feats: list[str], args, seed: int, device: str) -> dict:
    X = w[feats].to_numpy(np.float64)
    Y = w[list(C.TARGETS)].to_numpy(np.float64)
    pred = np.full_like(Y, np.nan)
    for k in sorted(w["fold"].dropna().unique()):
        tr, va = (w["fold"] != k).to_numpy(), (w["fold"] == k).to_numpy()
        Xtr, Xva = X[tr], X[va]
        if args.neighbours:
            Xtr = np.hstack([Xtr, neighbour_block(w, tr, tr, True)])
            Xva = np.hstack([Xva, neighbour_block(w, tr, va, False)])
        sc = Scaler().fit(Xtr)
        pred[va] = fit_predict(sc.transform(Xtr), Y[tr], sc.transform(Xva), w.loc[tr, "block"].to_numpy(),
                               args, seed, device)
    out = {}
    for j, t in enumerate(C.TARGETS):
        m = ~np.isnan(Y[:, j])
        out[t] = {**metrics(Y[m, j], pred[m, j]), "rows": int(m.sum())}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--neighbours", action="store_true")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--table", default=str(C.DATA_PROC / "train_c1.parquet"))
    a = ap.parse_args(argv)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w, feats = wide_table(pd.read_parquet(a.table))
    if a.max_rows and len(w) > a.max_rows:
        w = w.sample(a.max_rows, random_state=0).reset_index(drop=True)
    n_params = sum(p.numel() for p in build_model(len(feats) * 2 + 25 * a.neighbours, 5, a.width, a.depth).parameters())
    print(f"rows {len(w)}  features {len(feats)}  params ~{n_params / 1e6:.1f}M  device {device}", flush=True)
    res = {"config": vars(a), "rows": len(w), "features": len(feats), "params": n_params, "seeds": {}}
    for s in map(int, a.seeds.split(",")):
        t0 = time.time()
        res["seeds"][s] = cross_validate(w, feats, a, s, device)
        print(f"seed {s} ({time.time() - t0:.0f}s): " +
              "  ".join(f"{t} {res['seeds'][s][t]['r2']:.3f}" for t in C.TARGETS), flush=True)
    res["mean_r2"] = {t: float(np.mean([v[t]["r2"] for v in res["seeds"].values()])) for t in C.TARGETS}
    runs = C.RESULTS / "runs"
    runs.mkdir(exist_ok=True)
    out = runs / f"train_c1_{pd.Timestamp.now():%Y%m%dT%H%M}{'_nb' if a.neighbours else ''}.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
