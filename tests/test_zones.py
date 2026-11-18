import numpy as np
import pandas as pd
import pytest


def test_zones_separate_distinct_covariate_regimes():
    from geosoil import zones as Z
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"A00": rng.normal(-3, 0.3, 300), "A01": rng.normal(0, 0.3, 300)})
    b = pd.DataFrame({"A00": rng.normal(3, 0.3, 300), "A01": rng.normal(0, 0.3, 300)})
    s = pd.concat([a, b], ignore_index=True)
    s.loc[5, "A01"] = np.nan                                   # missing values must not break it
    z = Z.assign_zones(s, ["A00", "A01"], k=2)
    assert len(set(z[:300])) == 1 and len(set(z[300:])) == 1 and z[0] != z[300]


def test_per_zone_models_win_when_zones_differ(monkeypatch):
    pytest.importorskip("lightgbm")
    from geosoil import zones as Z
    monkeypatch.setattr(Z, "MIN_ZONE_ROWS", 50)
    rng = np.random.default_rng(1)
    n = 1200
    d = pd.DataFrame({"depth_mid": 2.5, "lon": rng.uniform(0, 1, n), "lat": rng.uniform(0, 1, n),
                      "A00": rng.normal(size=n), "fold": rng.integers(0, 5, n)})
    zone = (rng.uniform(size=n) > 0.5).astype(int)
    y = np.where(zone == 1, 2.0, -2.0) * d["A00"].to_numpy() + rng.normal(0, 0.1, n)   # opposite slopes per zone
    global_pred = np.zeros(n)
    r = Z.zone_scores(d, y, global_pred, ["A00"], zone)
    assert r["per_zone"]["r2"] > 0.9 and r["zone_feature"]["r2"] > 0.9
    assert r["per_zone"]["r2"] > r["global"]["r2"]
    assert np.isfinite(r["leave_zone_out"]["r2"])
