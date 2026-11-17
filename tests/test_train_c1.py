import argparse

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def _long_table(n_sites=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sites):
        lon, lat = rng.uniform(-100, -90), rng.uniform(30, 40)
        a0, a1 = rng.normal(), rng.normal()
        block = f"{int(lon)}_{int(lat)}"
        fold = hash(block) % 5
        for iv, (top, bot) in {"0-5": (0, 5), "5-15": (5, 15)}.items():
            base = dict(site_id=f"s{i}", interval=iv, depth_top=top, depth_bot=bot, lon=lon, lat=lat,
                        block=block, fold=fold, split="train", A00=a0, A01=a1, clim_map=np.nan if i % 7 == 0 else a1)
            rows.append(dict(base, target="ph", value=6 + a0))                   # learnable from A00
            if i % 3:
                rows.append(dict(base, target="soc", value=np.expm1(1 + 0.5 * a1)))
    return pd.DataFrame(rows)


def test_wide_table_pivots_and_logs_soc():
    from geosoil import train_c1 as T
    w, feats = T.wide_table(_long_table(50))
    assert {"A00", "A01", "clim_map", "depth_mid"} <= set(feats)
    assert len(w) == 100 and w["sand"].isna().all()                          # unmeasured target: NaN column
    soc = w["soc"].dropna()
    assert soc.between(-2, 4).all()                                          # log1p applied


def test_scaler_is_fit_on_training_rows_only():
    from geosoil import train_c1 as T
    tr = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]])
    sc = T.Scaler().fit(tr)
    z = sc.transform(np.array([[100.0, np.nan]]))
    assert z.shape == (1, 3) and z[0, 0] == 10.0 and z[0, 1] == 0.0 and z[0, 2] == 1.0   # clipped, filled, flagged


def test_cross_validation_learns_a_learnable_target():
    from geosoil import train_c1 as T
    w, feats = T.wide_table(_long_table())
    args = argparse.Namespace(neighbours=False, epochs=25, patience=25, batch=128, lr=3e-3, weight_decay=1e-4,
                              width=64, depth=2, dropout=0.0)
    res = T.cross_validate(w, feats, args, seed=0, device="cpu")
    assert res["ph"]["r2"] > 0.8 and res["soc"]["r2"] > 0.5
    assert np.isnan(res["sand"]["r2"])                                       # never measured: no score, no crash
