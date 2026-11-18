"""WoSIS "latest" (ISRIC's live WFS service): the datasets that are not in the 2023 December snapshot.

The 2023 snapshot stays the main WoSIS source (wosis.py). The live service has since dropped ~1.9k profiles and
added a few datasets; as of 2026-11-18 those are three CC-BY 4.0 IFDC sets of 0-20 cm topsoil from Burundi
(pH, CEC at pH 7), Rwanda (pH) and Uganda (pH), 2,620 profiles. This loader keeps only datasets absent from the
snapshot, so later additions are picked up automatically. Profile IDs are re-keyed between WoSIS versions, so
these sites get their own prefix and can never collide with snapshot sites.

Each property layer is fetched whole as CSV (the server returns a zip, ~40 s per layer); columns match the
snapshot TSVs, so rows go through wosis.property_rows unchanged.

    python -m geosoil.corpus.wosis_latest
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

from .. import config as C
from . import common, wosis

SOURCE = "wosis_latest"
WFS = ("https://maps.isric.org/mapserv?map=/map/wosis_latest.map&SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
       "&OUTPUTFORMAT=text/csv&TYPENAMES=ms:wosis_latest_{layer}")
RAW = common.RAW / "wosis_latest"


def _layer(layer: str) -> pd.DataFrame:
    path = common.fetch(WFS.format(layer=layer), RAW / f"{layer}.zip")
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        return pd.read_csv(io.BytesIO(z.read(name)), low_memory=False)


def snapshot_datasets(snapshot: Path = wosis.RAW) -> set[str]:
    """dataset_id values present in the 2023 snapshot (from its profiles table)."""
    z = zipfile.ZipFile(snapshot)
    name = next(n for n in z.namelist() if n.endswith("wosis_202312_profiles.tsv") and "MACOSX" not in n)
    return set(pd.read_csv(z.open(name), sep="\t", usecols=["dataset_code"])["dataset_code"].dropna().unique())   # dataset_id in the property tables


def build(tables: dict[str, pd.DataFrame], known: set[str]) -> tuple[pd.DataFrame, dict]:
    new = {p: t[~t["dataset_id"].isin(known)] for p, t in tables.items()}
    new = {p: t for p, t in new.items() if len(t)}
    if not new:
        return common.finalize(pd.DataFrame(), pd.DataFrame(), source=SOURCE, prefix="wosisl", license="see row")
    labels, drops = wosis.build(new, source=SOURCE, prefix="wosisl")
    drops["datasets"] = sorted({d for t in new.values() for d in t["dataset_id"].unique()})
    return labels, drops


def load() -> tuple[pd.DataFrame, dict]:
    known = snapshot_datasets(common.fetch(wosis.URL, wosis.RAW))
    tables = {}
    for prop in wosis.PROPERTIES:
        t = _layer(prop).rename(columns={"X": "longitude", "Y": "latitude"})   # the service names them X/Y
        tables[prop] = t[[c for c in wosis.COLUMNS if c in t.columns]]
    return build(tables, known)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
