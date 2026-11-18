import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
from geosoil import feature_selection as FS


def test_groups_split_climate_annual_and_monthly():
    g = FS.feature_groups(["lon", "lat", "depth_mid", "elevation", "A00", "clim_map", "clim_ppt_01", "rad_k"])
    assert g["climate_annual"] == ["clim_map"] and g["climate_monthly"] == ["clim_ppt_01"]
    assert g["location"] == ["lon", "lat"] and g["radiometrics"] == ["rad_k"]


def test_informative_group_outranks_noise_group():
    rng = np.random.default_rng(0)
    n = 3000
    d = pd.DataFrame({"elevation": rng.normal(size=n), "rad_k": rng.normal(size=n),
                      "fold": rng.integers(0, 5, n)})
    y = 2.0 * d["elevation"].to_numpy() + rng.normal(scale=0.3, size=n)   # only terrain matters
    groups = FS.feature_groups(d.columns)
    r = FS.importance(d, y, groups, drop_group=True)
    t, noise = r["groups"]["terrain"], r["groups"]["radiometrics"]
    assert r["r2_all"] > 0.9
    assert t["perm_drop"] > 0.5 and abs(noise["perm_drop"]) < 0.05
    assert t["drop_group"] > 0.5 and abs(noise["drop_group"]) < 0.05
    assert t["shap_share"] > 0.8
    assert r["top_columns"][0]["column"] == "elevation" and r["top_columns"][0]["group"] == "terrain"
    assert "| terrain |" in FS.markdown({"x": {**r, "r2_ceiling": 0.9}})


def test_class_agreement_and_conformal_coverage():
    import numpy as np
    from geosoil.metrics import class_agreement, conformal_coverage
    a = class_agreement("ph", np.array([5.0, 6.8, 8.0]), np.array([5.2, 6.9, 7.0]))
    assert a["class_exact"] == pytest.approx(1 / 3) and a["class_within_one"] == pytest.approx(2 / 3)  # 5.0 vs 5.2 straddle 5.05
    rng = np.random.default_rng(0)
    y = rng.normal(size=5000)
    pred = y + rng.normal(scale=0.5, size=5000)
    c = conformal_coverage(y, pred, rng.integers(0, 5, 5000), level=0.9)
    assert abs(c["coverage"] - 0.9) < 0.02 and c["half_width"] == pytest.approx(1.645 * 0.5, rel=0.08)
