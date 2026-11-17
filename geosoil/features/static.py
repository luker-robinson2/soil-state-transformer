"""Run a static-raster feature source (climate, radiometrics, ...) for every C1 site.

    python -m geosoil.features.static --source climate
    python -m geosoil.features.static --source radiometrics
    python -m geosoil.features.static --source all
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import pandas as pd

from .. import config as C
from . import rasters

SOURCES = ("climate", "radiometrics")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="all", choices=(*SOURCES, "all"))
    ap.add_argument("--labels", type=Path, default=C.DATA_PROC / "labels_c1.parquet")
    ap.add_argument("--limit-sites", type=int, default=0)
    a = ap.parse_args(argv)
    sites = pd.read_parquet(a.labels, columns=["site_id", "lon", "lat"]).drop_duplicates("site_id")
    if a.limit_sites:
        sites = sites.sample(a.limit_sites, random_state=0)
    for name in SOURCES if a.source == "all" else (a.source,):
        src = importlib.import_module(f"geosoil.features.{name}").source()
        print(json.dumps(rasters.extract(src, sites), indent=2))


if __name__ == "__main__":
    main()
