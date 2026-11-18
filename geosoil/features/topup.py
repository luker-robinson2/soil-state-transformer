"""Sites that have labels but no features yet (newly ingested sources) -> labels_new.parquet.

A site counts as featured when it appears in the terrain shards, which every site gets.

    python -m geosoil.features.topup
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from .. import config as C
from .join import read_shards

COLS = ["site_id", "lon", "lat", "coord_precision_m"]


def new_sites(labels: pd.DataFrame, featured: pd.DataFrame) -> pd.DataFrame:
    s = labels[COLS].drop_duplicates("site_id")
    done = set(featured["site_id"]) if "site_id" in featured else set()
    return s[~s["site_id"].isin(done)].reset_index(drop=True)


def free_ntasks(directory: Path, want: int) -> int:
    """The smallest task count >= want that no shard in `directory` (or its subdirectories) uses yet.

    Shards are named sites-<task>-of-<ntasks>; a top-up that reused an earlier top-up's ntasks would find those
    files and skip every task as already done (2026-11-18: the second top-up extracted nothing).
    """
    used = {int(m.group(1)) for f in Path(directory).rglob("sites-*-of-*.parquet")
            if (m := re.search(r"-of-(\d+)\.parquet$", f.name))}
    # a top-up still running has written few shards yet: it reserves its count with a marker at submit time
    used |= {int(m.group(1)) for f in Path(directory).glob(".reserved-of-*") if (m := re.search(r"-of-(\d+)$", f.name))}
    n = want
    while n in used:
        n += 1
    return n


def reserve(directory: Path, want: int) -> int:
    """free_ntasks plus a marker, so a concurrent top-up cannot pick the same count before shards exist."""
    n = free_ntasks(directory, want)
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / f".reserved-of-{n:04d}").touch()
    return n


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] in ("--free-ntasks", "--reserve"):   # used by scripts/slurm/run_topup.sh
        f = reserve if sys.argv[1] == "--reserve" else free_ntasks
        print(f(Path(sys.argv[2]), int(sys.argv[3])))
        return
    labels = pd.read_parquet(C.DATA_PROC / "labels_c1.parquet", columns=COLS)
    out = new_sites(labels, read_shards(C.DATA_PROC / "features" / "terrain"))
    out.to_parquet(C.DATA_PROC / "labels_new.parquet", index=False)
    print(f"{len(out)} sites without features -> {C.DATA_PROC / 'labels_new.parquet'}")


if __name__ == "__main__":
    main()
