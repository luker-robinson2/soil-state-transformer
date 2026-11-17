"""Run a static per-site feature source for every C1 site.

Raster sources declare `source()` and go through the shared sampler (rasters.py); sources with their
own logic (polygon lookups, tiled extraction) declare `extract(sites)`.

    python -m geosoil.features.static --source climate        # or radiometrics, thermal, regolith,
    python -m geosoil.features.static --source all            #    lithology, hand
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import pandas as pd

from .. import config as C
from . import rasters

SOURCES = ("climate", "radiometrics", "thermal", "regolith", "lithology", "hand")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="all", help=f"one of {SOURCES} or all; comma-separated list allowed")
    ap.add_argument("--labels", type=Path, default=C.DATA_PROC / "labels_c1.parquet")
    ap.add_argument("--limit-sites", type=int, default=0)
    a = ap.parse_args(argv)
    sites = pd.read_parquet(a.labels, columns=["site_id", "lon", "lat"]).drop_duplicates("site_id")
    if a.limit_sites:
        sites = sites.sample(a.limit_sites, random_state=0)
    names = SOURCES if a.source == "all" else tuple(a.source.split(","))
    unknown = set(names) - set(SOURCES)
    if unknown:
        raise SystemExit(f"unknown source(s): {sorted(unknown)}")
    for name in names:
        mod = importlib.import_module(f"geosoil.features.{name}")
        res = mod.extract(sites) if hasattr(mod, "extract") else rasters.extract(mod.source(), sites)
        print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
