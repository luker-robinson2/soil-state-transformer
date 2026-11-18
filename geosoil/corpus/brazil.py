"""Brazilian Soil Dataset 2024 (SoilData / MapBiomas) → canonical label table.

Units in the source: carbono g/kg, areia/argila g/kg (÷10 → %), ph (water), dsi g/cm3, ctc cmolc (per dm3).
No per-value method codes are published, so methods are recorded as unspecified.
Licence is per dataset (CC BY 4.0 or CC BY-NC 4.0) and kept per row.

    python -m geosoil.corpus.brazil
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .. import config as C
from . import common

SOURCE = "brazil_soildata"
URL = "https://repositorio.soildata.mapbiomas.org/api/access/datafile/832"
RAW = common.RAW / "brazil" / "brazil_soil_dataset_2024.tsv"


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    site_key = df["dataset_id"].astype(str) + "/" + df["observacao_id"].astype(str)
    layer_key = site_key + "/" + df["camada_id"].astype(str) + "/" + df["amostra_id"].astype(str)
    base = pd.DataFrame({"site_key": site_key, "layer_key": layer_key,
                         "depth_top": common.num(df["profund_sup"]), "depth_bot": common.num(df["profund_inf"])})
    parts = [
        common.long_rows(base, "soc", common.num(df["carbono"]), "brazil/carbono", "unknown"),
        common.long_rows(base, "ph", common.num(df["ph"]), "brazil/ph_h2o", "ph_h2o"),
        common.long_rows(base, "sand", common.num(df["areia"]) / 10.0, "brazil/areia", "unknown"),
        common.long_rows(base, "clay", common.num(df["argila"]) / 10.0, "brazil/argila", "unknown"),
        common.long_rows(base, "bd", common.num(df["dsi"]), "brazil/dsi", "unknown"),
        # CTC (cmolc, usually per dm3 of soil, not per kg): kept as an unknown-method CEC
        *([common.long_rows(base, "cec", common.num(df["ctc"]), "brazil/ctc", "unknown")] if "ctc" in df else []),
    ]
    vals = pd.concat(parts, ignore_index=True)
    sites = pd.DataFrame({
        "site_key": site_key,
        "lon": common.num(df["coord_x"]),
        "lat": common.num(df["coord_y"]),
        "coord_source": "coord_x/coord_y",
        "coord_precision_m": common.num(df["coord_precisao"]),
        "year": common.num(df["data_ano"]),
        "license": df["dataset_licenca"].astype(str),
    })
    return common.finalize(vals, sites, source=SOURCE, prefix="br", license="see row")


def load() -> tuple[pd.DataFrame, dict]:
    return build(pd.read_csv(common.fetch(URL, RAW), sep="\t", low_memory=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
