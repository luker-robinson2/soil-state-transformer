import numpy as np
import pandas as pd
import pytest

from geosoil import config as C
from geosoil.features import gee_context as G, groups


def test_bare_soil_columns_and_indices():
    raw = pd.DataFrame({b: [0.1, np.nan] for b in G.S2_BANDS}, index=["a", "b"])
    raw["B8"], raw["B4"], raw["B11"], raw["B12"] = [0.3, np.nan], [0.1, np.nan], [0.24, np.nan], [0.2, np.nan]
    raw["frac"] = [0.2, 0.0]
    d = G.derive_bare_soil(raw)
    assert d.loc["a", "bs_clay_idx"] == pytest.approx(1.2) and d.loc["a", "bs_ndvi"] == pytest.approx(0.5)
    assert d.loc["b", "bs_frac"] == 0.0 and np.isnan(d.loc["b", "bs_b4"])     # never bare: NaN, not 0


def test_context_similarity_and_frame_layout():
    site = {b: 1.0 for b in C.AE_COLS}
    row = {"site_id": "s1", **site}
    row.update({f"c500_{b}": 1.0 for b in C.AE_COLS})                          # identical: similarity 1
    row.update({f"c3k_{b}": (-1.0 if i % 2 else 1.0) for i, b in enumerate(C.AE_COLS)})   # orthogonal: 0
    out = G.to_frame("alphaearth_context", pd.DataFrame([row]))
    assert out.loc[0, "site_id"] == "s1"
    assert out.loc[0, "aectx_sim500"] == pytest.approx(1.0) and out.loc[0, "aectx_sim3k"] == pytest.approx(0.0)
    assert {"aectx500_A00", "aectx3k_A63"} <= set(out.columns) and len(out.columns) == 1 + 2 * 65
    g = groups.feature_groups(list(out.columns) + ["bs_b4"])
    assert len(g["alphaearth_context"]) == 130 and g["bare_soil"] == ["bs_b4"]
    assert "A00" not in g["alphaearth_context"]


def test_state_keys_migrate_and_plans_are_found(tmp_path):
    import json
    (tmp_path / "ee_tasks.json").write_text(json.dumps({"3": {"task": "T1", "asset": "a", "ntasks": 79},
                                                         "12:0": {"task": "T2", "asset": "b", "ntasks": 12}}))
    pd.DataFrame({"site_id": ["x"], "task": [0]}).to_parquet(tmp_path / "ee_plan_12.parquet")
    st = G._load_state(tmp_path)
    assert set(st) == {"79:3", "12:0"}
    assert list(G._plans(tmp_path)) == [12]
