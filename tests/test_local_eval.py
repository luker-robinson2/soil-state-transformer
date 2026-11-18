import numpy as np
import pandas as pd
import pytest


def test_fine_folds_nest_inside_the_fixed_test_split():
    from geosoil.corpus import splits
    rng = np.random.default_rng(0)
    s = pd.DataFrame({"site_id": [f"s{i}" for i in range(3000)], "lon": rng.uniform(-110, -90, 3000),
                      "lat": rng.uniform(30, 45, 3000)})
    a = splits.assign(s)
    test = a["split"] == "test"
    assert (a.loc[test, ["fold", "fold_025", "fold_005"]] == -1).all().all()
    for col, deg in splits.FINE_SCALES.items():
        a["b"] = splits.block_id(s["lon"], s["lat"], deg)
        assert (a[~test].groupby("b")[col].nunique() == 1).all()          # one fold per fine block
        assert set(a.loc[~test, col]) == set(range(splits.N_FOLDS))
    # finer blocks really are finer: more distinct blocks
    assert splits.block_id(s["lon"], s["lat"], 0.05).nunique() > splits.block_id(s["lon"], s["lat"], 0.25).nunique()


def test_distance_to_training_and_bins():
    from geosoil import local_eval as L
    d = pd.DataFrame({"site_id": ["a", "b", "c"], "lon": [0.0, 0.0, 0.0], "lat": [0.0, 0.009, 1.0],
                      "fold": [0, 1, 1]})
    dist = L.distance_to_training(d, "fold")
    assert dist[0] == pytest.approx(1.0, rel=0.02)                       # a's nearest other-fold site is b, 1 km
    assert dist[2] == pytest.approx(111.2, rel=0.02)                     # c's nearest other-fold site is a
    y = np.arange(120, dtype=float)
    pred = y.copy()
    pred[60:] = y[60:].mean()                                            # far rows: no skill
    b = L.by_distance(y, pred, np.r_[np.full(60, 0.5), np.full(60, 50.0)])
    assert b["0-1km"]["rows"] == 60 and b["0-1km"]["r2"] == pytest.approx(1.0)
    assert b["25-100km"]["rows"] == 60 and b["25-100km"]["r2"] == pytest.approx(0.0, abs=1e-9)
    assert b[">300km"] == {"rows": 0}


def test_regional_models_score_only_cells_with_enough_local_data(monkeypatch):
    pytest.importorskip("lightgbm")
    from geosoil import local_eval as L
    monkeypatch.setattr(L, "MIN_LOCAL_ROWS", 100)
    monkeypatch.setattr(L, "MIN_CELL_ROWS", 20)
    rng = np.random.default_rng(1)
    n = 1200
    d = pd.DataFrame({"site_id": [f"s{i}" for i in range(n)], "lon": rng.uniform(-100, -98, n),
                      "lat": rng.uniform(40, 42, n), "depth_mid": 2.5, "A00": rng.normal(size=n)})
    d["fold_025"] = rng.integers(0, 5, n)
    # the relation flips sign east vs west of -99: a regional model can learn what one global fit cannot
    y = np.where(d["lon"] < -99, 1.0, -1.0) * d["A00"].to_numpy() + rng.normal(0, 0.1, n)
    global_pred = np.zeros(n)
    r = L.regional(d, y, global_pred, ["A00"], "fold_025", radii=(60,))
    assert r["60km"]["models"] > 0 and r["60km"]["rows_scored"] > 0
    assert r["60km"]["regional"]["r2"] > r["60km"]["global"]["r2"]
