"""SISLAC v1.2, Latin America and Caribbean soil profiles (Díaz et al., Zenodo 7876731) → canonical labels.

SISLAC merges national surveys. Sources already in the corpus through WoSIS are skipped (the WoSIS 2016
rows, CHLSOC, Mexico INEGI Serie 1/2), as are rows flagged perfil_duplicado. Of the two Argentina
exports, only the newer SISINTA_v_01-12-2022 is used; the older one is in different SOC units. The
small DNOS-CORSAN set is skipped: it is in g/kg and is part of the Brazilian Soil Dataset.

Units, checked per source from value distributions (2026-11-17):
  organic_carbon %  → ×10 g/kg      clay, sand %       ph unspecified method      bulk_density g/cm3
Several small sources write 0 for "not measured": SOC = 0 is dropped, and texture is kept only when
clay + silt + sand is within 90–110 % (or silt is missing and clay + sand ≤ 100).
Dates of 1900 or 1905 are placeholders, so the year is left empty.
Licences are per row (CC-BY 4.0, PDDL, or CC-BY-NC-ND); rows derived from NC-ND data are never
committed or republished.

The archive is a zip holding a RAR that holds the CSV; extraction uses bsdtar (libarchive).

    python -m geosoil.corpus.sislac
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import common

SOURCE = "sislac"
URL = ("https://zenodo.org/api/records/7876731/files/"
       "DiazSergio/Revised-SISLAC-Database-Version1.2-v1.2.zip/content")
RAW = common.RAW / "sislac" / "sislac_v1.2.zip"
SKIP_SOURCES = {"WoSIS July 2016 Snapshot", "CHLSOC: the Chilean Soil Organic Carbon database",
                "México Serie-1", "México Serie-2", "sisINTA (sisinta.inta.gob.ar)",
                "reservatório do DNOS-CORSAN"}
PLACEHOLDER_YEARS = {1900, 1905}
BAD_CEC = {"ECOSUR-VT-2016", "SIAP"}


def texture_ok(clay: pd.Series, silt: pd.Series, sand: pd.Series) -> pd.Series:
    total = clay + silt + sand
    with_silt = total.between(90, 110)
    no_silt = silt.isna() & (clay + sand).between(1, 100)
    return with_silt | no_silt


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df[~df["source"].isin(SKIP_SOURCES) & (df["perfil_duplicado"].astype(str) != "t")].copy()
    site_key = df["profile_id"].astype(str)
    layer_key = site_key + "/" + df["layer_id"].astype(str)
    base = pd.DataFrame({"site_key": site_key, "layer_key": layer_key,
                         "depth_top": common.num(df["top"]), "depth_bot": common.num(df["bottom"])})
    clay, silt, sand = common.num(df["clay"]), common.num(df["silt"]), common.num(df["sand"])
    tex = texture_ok(clay, silt, sand)
    soc = common.num(df["organic_carbon"])
    col = lambda c: common.num(df[c]) if c in df else pd.Series(np.nan, index=df.index)
    parts = [
        common.long_rows(base, "soc", (soc * 10.0).where(soc > 0), "sislac/organic_carbon", "unknown"),
        common.long_rows(base, "ph", common.num(df["ph"]), "sislac/ph", "unknown"),
        common.long_rows(base, "sand", sand.where(tex), "sislac/sand", "unknown"),
        common.long_rows(base, "clay", clay.where(tex), "sislac/clay", "unknown"),
        common.long_rows(base, "bd", common.num(df["bulk_density"]), "sislac/bulk_density", "unknown"),
        # the "ecec" column mixes CEC methods across national sources (Argentina's medians look like pH-7 CEC), so
        # it goes in as unknown-method CEC; ECOSUR and SIAP report zeros or ~0.04 and are skipped
        common.long_rows(base, "cec", col("ecec").where(~df["source"].isin(BAD_CEC) & (col("ecec") > 0)),
                         "sislac/ecec", "unknown"),
        *[common.long_rows(base, el, col(el).where(col(el) > 0), f"sislac/{el}", "unknown") for el in ("k", "ca", "mg")],
    ]
    vals = pd.concat(parts, ignore_index=True)
    year = pd.to_numeric(df["fecha"].astype(str).str[-4:], errors="coerce")
    sites = pd.DataFrame({
        "site_key": site_key,
        "lon": common.num(df["longitude"]),
        "lat": common.num(df["latitude"]),
        "coord_source": "latitude/longitude",
        "coord_precision_m": np.nan,
        "year": year.where(~year.isin(PLACEHOLDER_YEARS)),
        "license": df["license"].astype(str),
    })
    return common.finalize(vals, sites, source=SOURCE, prefix="sl", license="see row")


def _csv(zpath: Path) -> Path:
    out = zpath.parent / "sislac_v2.csv"
    if out.exists():
        return out
    work = zpath.parent / "x"
    work.mkdir(exist_ok=True)
    subprocess.run(["bsdtar", "-xf", str(zpath), "-C", str(work)], check=True)
    rar = next(work.rglob("*.rar"))
    subprocess.run(["bsdtar", "-xf", str(rar), "-C", str(zpath.parent)], check=True)
    if not out.exists():
        raise FileNotFoundError(f"sislac_v2.csv not found after extracting {rar}")
    return out


def load() -> tuple[pd.DataFrame, dict]:
    return build(pd.read_csv(_csv(common.fetch(URL, RAW)), low_memory=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
