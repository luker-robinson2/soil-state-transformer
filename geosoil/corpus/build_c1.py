"""Build the C1 label set: every open source → per-source tables → dedup → labels_c1.parquet.

    python -m geosoil.corpus.build_c1            # all sources
    python -m geosoil.corpus.build_c1 --skip wosis

Writes data/processed/labels_<source>.parquet, labels_c1.parquet, dedup_log.parquet, then
runs harmonization, splits and the noise floor (labels_harmonized.parquet, splits.parquet).
Small committed summaries land in geosoil/results/ (c1_summary, harmonization, splits_summary,
noise_floor).
"""
from __future__ import annotations

import argparse
import importlib
import json

import pandas as pd

from .. import config as C
from . import common, dedup, harmonize, noise_floor, splits

MODULES = ("kssl", "wosis", "npdb", "brazil", "sislac", "olm_compendium", "ossl", "napeshm", "usgs_ds801",
           "fia", "nwca", "sodes", "bze_lw", "chspd", "icraf_afsis", "qld_sali", "nspd", "wosis_latest", "rmqs", "lucas_bd2018", "icrisat_karnataka", "raca")


def summarize(c1: pd.DataFrame, log: pd.DataFrame, drops: dict) -> dict:
    sites = c1.drop_duplicates("site_id")
    conus = sites["lon"].between(-125, -66) & sites["lat"].between(24, 50)
    top = c1[c1["depth_top"] < 30]
    train = c1[c1["role"] == "train"]
    return {
        "sites": {"train": int((sites["role"] == "train").sum()), "holdout": int((sites["role"] == "holdout").sum()),
                  "train_conus": int(((sites["role"] == "train") & conus).sum()),
                  "near_holdout": int(sites.loc[sites["role"] == "train", "near_holdout"].sum())},
        "values": int(len(c1)),
        "train_sites_by_source": train.drop_duplicates("site_id")["source"].value_counts().to_dict(),
        "train_topsoil_sites_by_target": top[top["role"] == "train"].groupby("target")["site_id"].nunique().to_dict(),
        "license_values": train["license"].str.contains("NonCommercial|NC", regex=True).map(
            {True: "noncommercial", False: "open"}).value_counts().to_dict(),
        "dedup": log.groupby(["source", "action"]).size().rename("sites").reset_index().to_dict("records"),
        "dropped_cross_source_by_keeper": log[log["action"] == "dropped"].groupby(
            ["source", "keeper_source"]).size().rename("sites").reset_index().to_dict("records"),
        "per_source_qa_drops": drops,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    tables, drops = [], {}
    for name in MODULES:
        if name in a.skip:
            continue
        mod = importlib.import_module(f"geosoil.corpus.{name}")
        dest = C.DATA_PROC / f"labels_{mod.SOURCE}.parquet"
        try:
            labels, d = mod.load()
        except OSError as e:                      # a host is down: reuse the last build of this source, else skip
            if not dest.exists():
                print(f"SKIP {name}: {e} (no earlier build)", flush=True)
                drops[mod.SOURCE] = {"skipped": str(e)[:200]}
                continue
            print(f"REUSE {name}: {e}; using {dest.name}", flush=True)
            labels, d = pd.read_parquet(dest), {"reused_after": str(e)[:200]}
            tables.append(labels)
            drops[mod.SOURCE] = d
            continue
        common.write(labels, d, dest)
        tables.append(labels)
        drops[mod.SOURCE] = d
    c1, log = dedup.build_c1(tables)
    c1.to_parquet(C.DATA_PROC / "labels_c1.parquet", index=False)
    log.to_parquet(C.DATA_PROC / "dedup_log.parquet", index=False)
    summary = summarize(c1, log, drops)
    (C.RESULTS / "c1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("sites", "values", "train_sites_by_source",
                                              "train_topsoil_sites_by_target", "license_values")}, indent=2))
    for step in (harmonize, splits, noise_floor):
        step.main([])


if __name__ == "__main__":
    main()
