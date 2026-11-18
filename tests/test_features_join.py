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


def test_baseline_groups_pick_up_new_sources_by_prefix():
    from geosoil import baseline_c1
    cols = ["elevation", "slope", "A00", "A01", "clim_map", "rad_k", "value"]
    g = baseline_c1.groups(cols)
    assert g["climate"] == ["clim_map"] and g["radiometrics"] == ["rad_k"]
    assert set(g["all"]) == {"elevation", "slope", "A00", "A01", "clim_map", "rad_k"}


def test_build_merges_any_static_source():
    labels = pd.DataFrame({"site_id": ["a"], "target": "ph", "interval": "0-5", "value": [6.0]})
    splits = pd.DataFrame({"site_id": ["a"], "block": ["x"], "split": ["train"], "fold": [0]})
    df = join.build(labels, splits, pd.DataFrame(), pd.DataFrame(),
                    {"climate": pd.DataFrame({"site_id": ["a"], "clim_map": [500.0]})})
    assert df.loc[0, "clim_map"] == 500.0


def test_topup_finds_only_unfeatured_sites():
    from geosoil.features import topup
    labels = pd.DataFrame({"site_id": ["a", "a", "b", "c"], "lon": [0, 0, 1, 2], "lat": [0, 0, 1, 2],
                           "coord_precision_m": [None] * 4})
    out = topup.new_sites(labels, pd.DataFrame({"site_id": ["a", "c"]}))
    assert out["site_id"].tolist() == ["b"]
    assert topup.new_sites(labels, pd.DataFrame())["site_id"].tolist() == ["a", "b", "c"]


def test_topup_picks_a_task_count_no_shard_uses(tmp_path):
    from geosoil.features import topup
    (tmp_path / "y2017-2025").mkdir()
    for n in (16, 17):
        (tmp_path / "y2017-2025" / f"sites-0000-of-{n:04d}.parquet").touch()
    assert topup.free_ntasks(tmp_path, 16) == 18
    assert topup.free_ntasks(tmp_path, 4) == 4


def test_reserved_task_counts_are_skipped_before_any_shard_exists(tmp_path):
    from geosoil.features import topup
    assert topup.reserve(tmp_path, 18) == 18
    assert topup.free_ntasks(tmp_path, 18) == 19          # a running top-up with no shards yet still blocks 18
    assert topup.reserve(tmp_path, 18) == 19
