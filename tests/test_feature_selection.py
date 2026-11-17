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
    assert "| terrain |" in FS.markdown({"x": {**r, "r2_ceiling": 0.9}})
