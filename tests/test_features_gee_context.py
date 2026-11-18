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


def test_spatial_tasks_keep_neighbours_together():
    import numpy as np
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"lon": rng.uniform(-100.01, -100.0, 50), "lat": rng.uniform(40.0, 40.01, 50)})
    b = pd.DataFrame({"lon": rng.uniform(10.0, 10.01, 50), "lat": rng.uniform(-5.0, -4.99, 50)})
    t = G.spatial_tasks(pd.concat([a, b], ignore_index=True), 2)
    assert len(set(t[:50])) == 1 and len(set(t[50:])) == 1 and t[0] != t[50]
    assert sorted(np.bincount(t)) == [50, 50]


def test_gap_fill_sites_copy_their_keepers_features(tmp_path, monkeypatch):
    from geosoil.features import shards
    out = tmp_path / "bare_soil"
    shards.write(pd.DataFrame({"site_id": ["wosis:1"], "bs_b4": [0.21]}), out, 0, 3, {"failed_blocks": []})
    log = tmp_path / "log.parquet"
    pd.DataFrame({"site_id": ["olm:9"], "keeper_site_id": ["wosis:1"], "action": ["gap_fill"]}).to_parquet(log)
    r = G.fill_from_keepers("bare_soil", tmp_path, log)
    assert r["filled"] == 1
    got = pd.read_parquet(out / "sites-0000-of-0001.parquet")
    assert got.loc[0, "site_id"] == "olm:9" and got.loc[0, "bs_b4"] == 0.21
