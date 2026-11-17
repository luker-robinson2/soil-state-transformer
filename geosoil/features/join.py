"""Join harmonized labels, fixed splits and feature shards into one training table (Phase 2).

One row per (site, target, interval). Static features (terrain) join on site_id. AlphaEarth
joins on site_id: for each site the embedding is averaged over the years extracted so far
(`ae_years`), excluding the temporal hold-out year, and renormalised to unit length.
Multi-year sequences come later.

    python -m geosoil.features.join                      # -> data/processed/train_c1.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..corpus.splits import TEMPORAL_HOLDOUT_YEAR
from . import terrain

FEAT = C.DATA_PROC / "features"


def read_shards(directory: Path, kind: str = "sites") -> pd.DataFrame:
    files = sorted(Path(directory).glob(f"{kind}-*-of-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def alphaearth_mean(feat: Path = FEAT / "alphaearth", holdout_year: int = TEMPORAL_HOLDOUT_YEAR) -> pd.DataFrame:
    """Per-site mean embedding over every extracted year except the hold-out year."""
    parts = [read_shards(d) for d in sorted(feat.glob("y*")) if d.is_dir()]
    ae = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    if ae.empty:
        return ae
    ae = ae[ae["year"] != holdout_year].drop_duplicates(["site_id", "year"])
    cols = list(C.AE_COLS)
    g = ae.groupby("site_id")
    m = g[cols].mean()
    norm = np.linalg.norm(m.to_numpy(), axis=1, keepdims=True)
    m[cols] = m.to_numpy() / np.where(norm > 0, norm, 1)
    m["ae_years"] = g["year"].nunique()
    return m.reset_index()


def build(labels: pd.DataFrame, splits: pd.DataFrame, terr: pd.DataFrame, ae: pd.DataFrame) -> pd.DataFrame:
    df = labels.merge(splits[["site_id", "block", "split", "fold"]], on="site_id", how="left")
    if not terr.empty:
        df = df.merge(terr.drop_duplicates("site_id"), on="site_id", how="left")
    if not ae.empty:
        df = df.merge(ae, on="site_id", how="left")
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / "train_c1.parquet")
    a = ap.parse_args(argv)
    labels = pd.read_parquet(C.DATA_PROC / "labels_harmonized.parquet")
    splits = pd.read_parquet(C.DATA_PROC / "splits.parquet")
    terr, ae = read_shards(FEAT / "terrain"), alphaearth_mean()
    df = build(labels, splits, terr, ae)
    df.to_parquet(a.out, index=False)
    print(json.dumps({"rows": len(df), "sites": int(df["site_id"].nunique()),
                      "with_terrain": float(df[list(terrain.FEATURES)].notna().all(axis=1).mean()) if not terr.empty else 0.0,
                      "with_alphaearth": float(df["A00"].notna().mean()) if "A00" in df else 0.0,
                      "ae_years": sorted(map(int, df["ae_years"].dropna().unique())) if "ae_years" in df else []},
                     indent=2))


if __name__ == "__main__":
    main()
