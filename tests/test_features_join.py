import numpy as np
import pandas as pd
import pytest

from geosoil import config as C
from geosoil.features import join


def _ae(site, year, v):
    row = {c: 0.0 for c in C.AE_COLS}
    row["A00"], row["A01"] = v
    return {"site_id": site, "year": year, **row}


def test_alphaearth_mean_excludes_holdout_year_and_renormalises(tmp_path):
    for y, rows in ((2020, [_ae("s", 2020, (1.0, 0.0))]), (2021, [_ae("s", 2021, (0.0, 1.0))]),
                    (2024, [_ae("s", 2024, (-1.0, 0.0))])):
        d = tmp_path / f"y{y}"
        d.mkdir()
        pd.DataFrame(rows).to_parquet(d / "sites-0000-of-0001.parquet")
    m = join.alphaearth_mean(tmp_path, holdout_year=2024).set_index("site_id")
    assert m.loc["s", "ae_years"] == 2
    assert m.loc["s", "A00"] == pytest.approx(1 / np.sqrt(2)) and m.loc["s", "A01"] == pytest.approx(1 / np.sqrt(2))


def test_build_joins_splits_terrain_and_alphaearth():
    labels = pd.DataFrame({"site_id": ["a", "b"], "target": "ph", "interval": "0-5", "value": [6.0, 7.0]})
    splits = pd.DataFrame({"site_id": ["a", "b"], "block": ["x", "y"], "split": ["train", "test"], "fold": [0, -1]})
    terr = pd.DataFrame({"site_id": ["a"], "elevation": [100.0]})
    ae = pd.DataFrame({"site_id": ["b"], "A00": [0.5], "ae_years": [1]})
    df = join.build(labels, splits, terr, ae).set_index("site_id")
    assert df.loc["a", "elevation"] == 100.0 and np.isnan(df.loc["b", "elevation"])
    assert df.loc["b", "A00"] == 0.5 and df.loc["b", "split"] == "test"
