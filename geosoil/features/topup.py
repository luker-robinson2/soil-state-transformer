"""Sites that have labels but no features yet (newly ingested sources) -> labels_new.parquet.

A site counts as featured when it appears in the terrain shards, which every site gets.

    python -m geosoil.features.topup
"""
from __future__ import annotations

import pandas as pd

from .. import config as C
from .join import read_shards

COLS = ["site_id", "lon", "lat", "coord_precision_m"]


def new_sites(labels: pd.DataFrame, featured: pd.DataFrame) -> pd.DataFrame:
    s = labels[COLS].drop_duplicates("site_id")
    done = set(featured["site_id"]) if "site_id" in featured else set()
    return s[~s["site_id"].isin(done)].reset_index(drop=True)


def main() -> None:
    labels = pd.read_parquet(C.DATA_PROC / "labels_c1.parquet", columns=COLS)
    out = new_sites(labels, read_shards(C.DATA_PROC / "features" / "terrain"))
    out.to_parquet(C.DATA_PROC / "labels_new.parquet", index=False)
    print(f"{len(out)} sites without features -> {C.DATA_PROC / 'labels_new.parquet'}")


if __name__ == "__main__":
    main()
