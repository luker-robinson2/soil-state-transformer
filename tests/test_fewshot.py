import numpy as np
import pandas as pd
import pytest


def test_variance_components_recover_tau_and_sigma():
    from geosoil.fewshot import variance_components
    rng = np.random.default_rng(0)
    cells = np.repeat(np.arange(400), 20)
    r = np.repeat(rng.normal(0, 0.5, 400), 20) + rng.normal(0, 1.0, len(cells))
    tau2, sigma2 = variance_components(cells, r)
    assert tau2 == pytest.approx(0.25, rel=0.2) and sigma2 == pytest.approx(1.0, rel=0.05)


def test_offset_helps_when_cells_share_an_offset_and_not_when_they_do_not():
    from geosoil.fewshot import run
    rng = np.random.default_rng(1)
    n_cells, per = 300, 15
    cx, cy = rng.uniform(-100, -90, n_cells), rng.uniform(35, 45, n_cells)
    lon = np.repeat(cx, per) + rng.uniform(0, 0.01, n_cells * per)
    lat = np.repeat(cy, per) + rng.uniform(0, 0.01, n_cells * per)
    truth = rng.normal(size=n_cells * per)

    def frame(offset_sd):
        y = truth + np.repeat(rng.normal(0, offset_sd, n_cells), per) + rng.normal(0, 0.3, n_cells * per)
        return pd.DataFrame({"site_id": np.arange(len(y)).astype(str), "lon": lon, "lat": lat,
                             "fold": np.repeat(np.arange(n_cells) % 5, per), "y": y, "pred": truth})
    shared = run(frame(0.8), 0.05, [5], draws=3)["k5"]
    none = run(frame(0.0), 0.05, [5], draws=3)["k5"]
    assert shared["rel_mse_offset_eb"] < 0.5 and shared["rel_mse_offset_eb_hi"] < 1
    assert none["rel_mse_offset_eb"] == pytest.approx(1.0, abs=0.05)     # shrinkage protects: no harm
