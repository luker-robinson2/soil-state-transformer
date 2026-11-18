"""ISRIC WoSIS 2023 snapshot → canonical label table.

Reads one TSV per property straight from the snapshot zip. Each row already carries
coordinates, dataset_id, date, positional uncertainty and the per-profile licence.
Values come from `value_avg` (the `value` column is a Postgres array like "{36.8}").

US-NCSS rows are dropped here: they are the KSSL Lab Data Mart, ingested directly.
Units: orgc g/kg, sand/clay %, bulk density kg/dm3 (= g/cm3), phaq pH, CEC cmol(+)/kg, orgm g/kg.
WoSIS 2023 has no exchangeable-cation (K/Ca/Mg) property files.

    python -m geosoil.corpus.wosis
"""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd

from .. import config as C
from . import common

SOURCE = "wosis_2023"
URL = "https://files.isric.org/public/wosis_snapshot/WoSIS_2023_December.zip"
RAW = common.RAW / "wosis" / "WoSIS_2023_December.zip"
# dataset_id codes already covered by a directly ingested source
EXCLUDE_DATASETS = {"US-NCSS"}
# property -> (target, method family or None to classify from method_options)
PROPERTIES = {
    "orgc": ("soc", None),
    "phaq": ("ph", "ph_h2o"),
    "sand": ("sand", None),
    "clay": ("clay", None),
    "bdfiod": ("bd", "bd_oven_dry"),
    "bdfi33": ("bd", "bd_third_bar"),
    # CEC in cmol(+)/kg: at pH 7 (NH4OAc or another pH-7 index cation) is canonical; pH 8.2 and effective CEC
    # are separate families, converted in harmonize.py only where a good fit exists. orgm is measured OM, g/kg.
    "cecph7": ("cec", "cec_ph7"),
    "cecph8": ("cec", "cec_sum_cations_ph82"),
    "ecec": ("cec", "ecec"),
    "orgm": ("om", "measured"),
}
# positional_uncertainty is categorical; use each class's upper bound in metres
UNCERTAINTY_M = {"Circa 100 m": 100.0, "100 m - 1 km": 1_000.0, "1 km - 10 km": 10_000.0, "Over 10 km": 100_000.0}
COLUMNS = ["profile_id", "layer_id", "upper_depth", "lower_depth", "value_avg", "method_options",
           "dataset_id", "longitude", "latitude", "positional_uncertainty", "date", "licence"]


def _texture_family(text) -> str:
    t = "" if pd.isna(text) else str(text).lower()
    if "pipet" in t:
        return "pipette"
    if "hydrometer" in t:
        return "hydrometer"
    if "laser" in t:
        return "laser_diffraction"
    return "unknown"


def _ratio(text) -> str:
    m = re.search(r"ratio\s*=\s*\[\s*([0-9.:/ ]+?)\s*\]", "" if pd.isna(text) else str(text))
    return m.group(1).replace(" ", "") if m else "NA"


def property_rows(df: pd.DataFrame, prop: str) -> pd.DataFrame:
    target, family = PROPERTIES[prop]
    df = df[~df["dataset_id"].isin(EXCLUDE_DATASETS)]
    base = pd.DataFrame({"site_key": df["profile_id"].astype(str), "layer_key": df["layer_id"].astype(str),
                         "depth_top": common.num(df["upper_depth"]), "depth_bot": common.num(df["lower_depth"])},
                        index=df.index)
    opts = df["method_options"]
    if prop == "phaq":
        mid, fam = "wosis/phaq:" + opts.map(_ratio), family
    elif prop == "orgc":
        mid, fam = pd.Series("wosis/orgc", index=df.index), opts.map(common.carbon_family_from_text)
    elif prop in ("sand", "clay"):
        mid, fam = pd.Series(f"wosis/{prop}", index=df.index), opts.map(_texture_family)
    else:
        mid, fam = pd.Series(f"wosis/{prop}", index=df.index), family
    if isinstance(mid, str):
        mid = pd.Series(mid, index=df.index)
    vals = common.long_rows(base, target, common.num(df["value_avg"]), mid, fam)
    vals["license"] = df.loc[vals.index, "licence"].astype(str).values
    return vals


def build(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    vals = pd.concat([property_rows(df, p) for p, df in tables.items()], ignore_index=True)
    allrows = pd.concat([df[~df["dataset_id"].isin(EXCLUDE_DATASETS)] for df in tables.values()])
    sites = pd.DataFrame({
        "site_key": allrows["profile_id"].astype(str),
        "lon": common.num(allrows["longitude"]), "lat": common.num(allrows["latitude"]),
        "coord_source": "longitude/latitude",
        "coord_precision_m": allrows["positional_uncertainty"].map(UNCERTAINTY_M),
        "year": pd.to_datetime(allrows["date"], errors="coerce").dt.year,
    }).drop_duplicates("site_key")
    labels, drops = common.finalize(vals, sites, source=SOURCE, prefix="wosis", license="see row")
    drops["excluded_datasets"] = {d: int(sum((t["dataset_id"] == d).sum() for t in tables.values()))
                                  for d in EXCLUDE_DATASETS}
    return labels, drops


def read(path: Path) -> dict[str, pd.DataFrame]:
    z = zipfile.ZipFile(path)
    names = {Path(n).name: n for n in z.namelist() if not n.startswith("__MACOSX") and n.endswith(".tsv")}
    out = {}
    for prop in PROPERTIES:
        with z.open(names[f"wosis_202312_{prop}.tsv"]) as f:
            out[prop] = pd.read_csv(f, sep="\t", usecols=COLUMNS, low_memory=False)
    return out


def load() -> tuple[pd.DataFrame, dict]:
    return build(read(common.fetch(URL, RAW)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
