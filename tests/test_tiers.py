import numpy as np
import pandas as pd


def _cells(rng, n=400):
    dens = rng.integers(0, 3000, n)
    mg = rng.uniform(0.8, 1.2, n)
    helps = dens >= 500                                           # blend helps only in dense cells
    mb = np.where(helps, mg * 0.85, mg * 1.10)
    return pd.DataFrame({"radius_km": 150, "fold": rng.integers(0, 5, n), "cell": [str(i) for i in range(n)],
                         "n_local_train": dens, "n_val": rng.integers(20, 200, n), "var_val": 1.0,
                         "mse_global": mg, "mse_blend": mb, "mse_control": mg * 1.02})


def test_selection_beats_both_global_and_blend_everywhere():
    from geosoil import tiers as T
    s = T.summarise(T.nested_selection(_cells(np.random.default_rng(0))))
    assert s["gain_selected"] > max(0.0, s["gain_blend_all"]) and s["gain_lo"] > 0
    assert 0.5 < s["share_blended"] < 0.95 and s["locality_margin"] > 0


def test_never_blends_when_blend_always_hurts():
    from geosoil import tiers as T
    c = _cells(np.random.default_rng(1))
    c["mse_blend"] = c["mse_global"] * 1.1
    s = T.summarise(T.nested_selection(c))
    assert s["share_blended"] == 0 and s["gain_selected"] == 0
