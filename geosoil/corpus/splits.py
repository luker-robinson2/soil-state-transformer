"""Fixed spatial splits for C1 (EXECUTION-PLAN step 1.10).

Every site belongs to a 1° x 1° block. A block's split comes from a hash of its
coordinates, so the assignment never changes when sites or sources are added:
  - ~10% of blocks -> "test" (never used for model selection)
  - the rest -> "train" with fold 0-4 for spatial-block cross-validation
RaCA sites keep split "holdout". `near_holdout` (training sites within 1 km of a RaCA
site) is carried through so RaCA-scored runs can exclude them.

The temporal hold-out for the satellite features is the latest year both embedding
sources cover (TEMPORAL_HOLDOUT_YEAR); it applies to inputs, not to these label splits.

Finer folds for the local and regional model tiers (docs/research/LOCAL-MODELS.md): inside the training
split, each site also gets a fold at 0.25° (~25 km) and 0.05° (~5 km) blocks, `fold_025` and `fold_005`. The
test blocks stay the 1° ones, so the held-out set is the same whichever scale a run cross-validates at.
Smaller blocks put validation sites closer to training sites, so their scores answer "how well does the model
do with labels nearby", not "how well does it extrapolate"; report them as such.

Output: data/processed/splits.parquet (site_id, block, split, fold, fold_025, fold_005, near_holdout).
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from .. import config as C

BLOCK_DEG = 1.0
N_FOLDS = 5
TEST_EVERY = 10          # 1 in 10 blocks is test
FINE_SCALES = {"fold_025": 0.25, "fold_005": 0.05}   # column -> block size in degrees
TEMPORAL_HOLDOUT_YEAR = 2024   # TESSERA covers 2024 only so far; AlphaEarth 2017-2025


def block_id(lon: pd.Series, lat: pd.Series, deg: float = BLOCK_DEG) -> pd.Series:
    bx = np.floor(lon / deg).astype(int)
    by = np.floor(lat / deg).astype(int)
    return bx.astype(str) + "_" + by.astype(str)


def _hash(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:7], "big")   # 7 bytes keeps it inside int64


def assign(sites: pd.DataFrame) -> pd.DataFrame:
    """sites: site_id, lon, lat [, role, near_holdout]."""
    out = sites[["site_id"]].copy()
    out["block"] = block_id(sites["lon"], sites["lat"]).values
    h = out["block"].map(_hash)
    out["split"] = np.where(h % TEST_EVERY == 0, "test", "train")
    out["fold"] = np.where(out["split"] == "train", (h // TEST_EVERY) % N_FOLDS, -1).astype(int)
    for col, deg in FINE_SCALES.items():
        hf = block_id(sites["lon"], sites["lat"], deg).map(lambda b, d=deg: _hash(f"{d}:{b}")).values
        out[col] = np.where(out["split"] == "train", hf % N_FOLDS, -1).astype(int)
    if "role" in sites:
        hold = (sites["role"] == "holdout").values
        out.loc[hold, "split"] = "holdout"
        out.loc[hold, ["fold", *FINE_SCALES]] = -1
    out["near_holdout"] = sites["near_holdout"].fillna(False).astype(bool).values if "near_holdout" in sites else False
    return out.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    c1 = pd.read_parquet(C.DATA_PROC / "labels_c1.parquet",
                         columns=["site_id", "lon", "lat", "role", "near_holdout"]).drop_duplicates("site_id")
    sp = assign(c1)
    sp.to_parquet(C.DATA_PROC / "splits.parquet", index=False)
    summary = {
        "block_deg": BLOCK_DEG, "n_folds": N_FOLDS, "test_every": TEST_EVERY,
        "temporal_holdout_year": TEMPORAL_HOLDOUT_YEAR,
        "sites_by_split": sp["split"].value_counts().to_dict(),
        "sites_by_fold": sp.loc[sp["split"] == "train", "fold"].value_counts().sort_index().to_dict(),
        "blocks_by_split": sp.drop_duplicates("block")["split"].value_counts().to_dict(),
    }
    (C.RESULTS / "splits_summary.json").write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps(summary, indent=2, default=int))


if __name__ == "__main__":
    main()
