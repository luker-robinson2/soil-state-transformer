"""baseline_c1.main end to end on a tiny table: every CLI flag parses and OOF predictions are written."""
import json
import shutil

import numpy as np
import pandas as pd

from geosoil import baseline_c1 as B
from geosoil import config as C


def _table(n=600, seed=0):
    rng = np.random.default_rng(seed)
    ae = {c: rng.normal(size=n).astype("float32") for c in C.AE_COLS}
    df = pd.DataFrame(ae)
    blocks = rng.integers(0, 60, n)
    df["site_id"] = [f"s{i}" for i in range(n)]
    df["lon"], df["lat"] = rng.uniform(-100, -90, n), rng.uniform(35, 45, n)
    df["block"], df["fold"], df["split"] = blocks.astype(str), blocks % 5, "train"
    df["interval"], df["depth_top"], df["depth_bot"] = "0-5", 0.0, 5.0
    df["source"] = "kssl_ldm"
    df["target"] = np.where(np.arange(n) % 2 == 0, "cec", "ph")
    df["method_family"] = np.where(np.arange(n) % 4 == 0, "cec_ph7", np.where(df["target"] == "cec", "ecec", "ph_h2o"))
    df["value"] = np.abs(df[C.AE_COLS[0]] * 3 + 10 + rng.normal(size=n))
    return df


def test_main_parses_all_flags_and_writes_oof(tmp_path, monkeypatch):
    res = tmp_path / "results"
    res.mkdir()
    shutil.copy(C.RESULTS / "noise_floor.json", res / "noise_floor.json")
    monkeypatch.setattr(C, "RESULTS", res)
    table = tmp_path / "train.parquet"
    _table().to_parquet(table)
    B.main(["--table", str(table), "--groups", "all", "--targets", "cec", "--canonical-only", "cec"])
    oof = list((res / "runs").glob("oof_*/cec.parquet"))
    assert len(oof) == 1
    o = pd.read_parquet(oof[0])
    assert set(o["method_family"]) == {"cec_ph7"} and {"lon", "lat", "y", "pred", "block"} <= set(o.columns)
    out = json.loads(next((res / "runs").glob("baseline_c1_*.json")).read_text())
    assert list(out) == ["cec"] and "all" in out["cec"]
