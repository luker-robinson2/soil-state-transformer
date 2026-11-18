"""Namibia Soil Profile Database NSPD2025 (Zenodo 17618737, CC-BY 4.0) → canonical labels.

An Excel workbook of Namibian soil profiles from national soil surveys (NSS by region), research projects
(BIOTA, SASSCAL, GIZ ...) and LandPKS. Sheets used: `Hor_lab` (one row per horizon, with coordinates) and
`Reg&Site` (year and coordinate reference system per profile; PRID has duplicates, so it is deduplicated
before mapping). Dataset == "LandPKS" is skipped (app entries without lab values).

Gotchas:
  * numeric columns can hold strings padded with non-breaking spaces ('-26.396908\\xa0'): \\xa0 is stripped
    before to_numeric. Column names contain newlines and trailing spaces, so columns are matched by regex.
  * depth: the numeric `Upper Depth cm ` / `Lower Depth cm ` columns, not the "[contains >]" ones.
  * coordinates: per profile either "Geographic, WGS84" or "Geographic, Schwarzeck" (EPSG:4293); the
    "WGS85".."WGS223" labels are spreadsheet auto-fill of WGS84. Schwarzeck coordinates are NOT datum-shifted:
    the WoSIS / OLM NAMSOTER copies of the same profiles hold the unshifted values (46% match within 100 m raw,
    8% after the ~97 m shift), and dedup must find them. They get coord_precision_m = 150.
Mapping and units:
  * clay `Clay % [< 2 µm]` %, family unknown.
  * sand `Sand % < 53 µm` %: despite the header the values are sand > 53 µm (q50 84, clay + silt + sand = 100).
    Silt is 2–53 µm, a non-USDA limit, so family unknown.
  * SOC `Organic Carbon g/kg WB` (or `Organic Carbon % WB` × 10 when the g/kg cell is empty), walkley_black.
    LOI carbon is not used.
  * pH `pH water (PHAQ)` → ph_h2o; `pH CaCl2` → ph_cacl2.
  * BD `Bulk Density kg/dm3` (= g/cm3), family unknown.
  * CEC `CEC Measured` cmol(+)/kg, family unknown. `CEC Effective` is exactly the sum of bases: skipped.
  * Ca/Mg/K: the cmol(+)/kg `Exchangeable ...` columns are inconsistent with the mg/kg columns in several
    datasets (TWO NSS Ca exceeds CEC 8×), so they are used only from BASES_DATASETS, family unknown.
Year: Reg&Site `Year` (1989–2023). Keys: site = PRID, layer = PRID/HONU/top-bottom.
NSPD republishes NAMSOTER / WoSIS profiles: dedup treats it as a republisher with gap fill.

    python -m geosoil.corpus.nspd
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import common

SOURCE = "nspd"
URL = "https://zenodo.org/api/records/17618737/files/Nam_Prof_Nov2025.xlsx/content"
RAW = common.RAW / "nspd" / "Nam_Prof_Nov2025.xlsx"
LICENSE = "CC-BY-4.0 (NSPD2025, Zenodo 17618737)"
SKIP_DATASETS = {"LandPKS"}
BASES_DATASETS = {"FEB", "DOR", "MAR", "JAN", "AO"}
SCHWARZECK_PRECISION_M = 150.0
COLS = {   # key -> regex on the Hor_lab header
    "top": r"^Upper Depth cm\s*$", "bot": r"^Lower Depth cm\s*$",
    "clay": r"^Clay %", "sand": r"^Sand %",
    "soc_gkg": r"^Organic Carbon g/kg WB", "soc_pct": r"^Organic Carbon % WB",
    "ph_h2o": r"^pH water \(PHAQ\)", "ph_cacl2": r"^pH CaCl2",
    "bd": r"^Bulk Density kg/dm3", "cec": r"^CEC Measured",
    "ca": r"^Exchangeable Calcium", "mg": r"^Exchangeable Magnesium", "k": r"^Exchangeable Potassium",
}


def col(df: pd.DataFrame, pattern: str) -> str:
    hits = [c for c in df.columns if re.search(pattern, str(c))]
    if len(hits) != 1:
        raise KeyError(f"expected one column matching {pattern!r}, found {hits}")
    return hits[0]


def nbsp_num(s: pd.Series) -> pd.Series:
    """Numbers stored as text padded with non-breaking spaces."""
    return common.num(s.astype(str).str.replace("\xa0", "", regex=False).str.strip())


def build(hor: pd.DataFrame, site: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n_all = len(hor)
    h = hor[~hor["Dataset"].astype(str).str.strip().isin(SKIP_DATASETS)].copy()
    v = {k: nbsp_num(h[col(h, p)]) for k, p in COLS.items()}
    site_key = h["PRID"].astype(str).str.strip()
    base = pd.DataFrame({"site_key": site_key,
                         "layer_key": site_key + "/" + h["HONU"].astype(str).str.strip() + "/"
                                      + v["top"].astype(str) + "-" + v["bot"].astype(str),
                         "depth_top": v["top"], "depth_bot": v["bot"]})
    bases_ok = h["Dataset"].astype(str).str.strip().isin(BASES_DATASETS)
    soc = v["soc_gkg"].fillna(v["soc_pct"] * 10.0)
    parts = [
        common.long_rows(base, "clay", v["clay"], "nspd/CLPC", "unknown"),
        common.long_rows(base, "sand", v["sand"], "nspd/SDTO_gt53um", "unknown"),
        common.long_rows(base, "soc", soc, "nspd/OC_WB", "walkley_black"),
        common.long_rows(base, "ph", v["ph_h2o"], "nspd/PHAQ", "ph_h2o"),
        common.long_rows(base, "ph", v["ph_cacl2"], "nspd/PHCA", "ph_cacl2"),
        common.long_rows(base, "bd", v["bd"], "nspd/BD", "unknown"),
        common.long_rows(base, "cec", v["cec"], "nspd/CEC_measured", "unknown"),
        *[common.long_rows(base, t, v[t].where(bases_ok), f"nspd/EX{t.upper()}", "unknown") for t in ("ca", "mg", "k")],
    ]
    vals = pd.concat(parts, ignore_index=True)

    s = site.assign(PRID=site["PRID"].astype(str).str.strip()).drop_duplicates("PRID").set_index("PRID")
    crs = site_key.map(s["Coordinate Reference System"].astype(str).str.strip())
    schwarzeck = crs.str.contains("Schwarzeck", case=False, na=False)
    sites = pd.DataFrame({
        "site_key": site_key,
        "lon": nbsp_num(h["Longitude"]),
        "lat": nbsp_num(h["Latitude"]),
        "coord_source": np.where(schwarzeck, "Latitude/Longitude (Schwarzeck, not shifted)",
                                 "Latitude/Longitude (WGS84)"),
        "coord_precision_m": np.where(schwarzeck, SCHWARZECK_PRECISION_M, np.nan),
        "year": site_key.map(nbsp_num(s["Year"])),
    })
    labels, drops = common.finalize(vals, sites, source=SOURCE, prefix="nspd", license=LICENSE)
    return labels, {"skipped_dataset_rows": n_all - len(h), **drops}


def load() -> tuple[pd.DataFrame, dict]:
    path = common.fetch(URL, RAW)
    hor = pd.read_excel(path, sheet_name="Hor_lab")
    site = pd.read_excel(path, sheet_name="Reg&Site", usecols=["PRID", "Year", "Coordinate Reference System"])
    return build(hor, site)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
