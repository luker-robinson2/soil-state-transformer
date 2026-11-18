import numpy as np
import pandas as pd
import pytest

from geosoil import metrics as M


def test_point_metrics_decomposition_and_bias_blind_r2():
    rng = np.random.default_rng(0)
    y = rng.normal(10, 2, 5000)
    p = y + 1.0                                                   # perfectly correlated, biased by +1
    m = M.point_metrics(y, p)
    assert m["r2_pearson"] == pytest.approx(1.0) and m["r2"] < 1.0 and m["me"] == pytest.approx(1.0)
    assert m["ccc"] < 1.0 and m["mse_bias"] == pytest.approx(1.0, rel=1e-6)
    assert m["mse_bias"] + m["mse_scale"] + m["mse_corr"] == pytest.approx(m["mse"], rel=1e-6)


def test_floor_skill_is_one_at_the_noise_floor():
    rng = np.random.default_rng(1)
    truth = rng.normal(0, 1, 20000)
    y = truth + rng.normal(0, 0.5, 20000)                         # lab noise var 0.25
    f = M.floor_skill(y, truth, nugget=0.25)
    assert f["floor_skill"] == pytest.approx(1.0, abs=0.03) and f["excess_rmse"] < 0.05


def test_paired_compare_detects_a_better_model_and_respects_blocks():
    rng = np.random.default_rng(2)
    n, blocks = 4000, np.repeat(np.arange(200), 20)
    y = rng.normal(size=n)
    good, bad = y + rng.normal(0, 0.5, n), y + rng.normal(0, 0.8, n)
    r = M.paired_compare(y, bad, good, blocks, n_boot=500)
    assert r["delta_mse"] > 0 and r["delta_lo"] > 0 and r["p_signflip"] < 0.01
    same = M.paired_compare(y, good, good + rng.normal(0, 1e-3, n), blocks, n_boot=500)
    assert same["p_signflip"] > 0.05


def test_interval_score_rewards_coverage_and_kappa_orders_misses():
    y = np.array([1.0, 2.0, 3.0])
    assert M.interval_score(y, y - 0.5, y + 0.5) == pytest.approx(1.0)
    assert M.interval_score(y, y + 1, y + 2) > M.interval_score(y, y - 0.5, y + 0.5)
    yy = np.array([4.0, 5.3, 5.8, 6.3, 6.9, 7.6, 8.1])
    assert M.weighted_kappa("ph", yy, yy) == pytest.approx(1.0)
    near, far = yy + 0.5, yy + 2.5
    assert M.weighted_kappa("ph", yy, near) > M.weighted_kappa("ph", yy, far)


def test_block_bootstrap_ci_widens_with_correlated_blocks():
    rng = np.random.default_rng(3)
    blk = np.repeat(np.arange(50), 40)
    x = np.repeat(rng.normal(size=50), 40) + rng.normal(0, 0.1, 2000)   # strong within-block correlation
    wide = M.block_bootstrap(blk, np.mean, x, n_boot=400)
    iid = M.block_bootstrap(np.arange(2000), np.mean, x, n_boot=400)
    assert (wide["hi"] - wide["lo"]) > 3 * (iid["hi"] - iid["lo"])


def test_scorecard_end_to_end(tmp_path):
    from geosoil import scorecard as S
    rng = np.random.default_rng(4)
    n = 3000
    pred = rng.normal(2.5, 0.6, n)
    y = pred + rng.normal(0, 0.3, n)                              # truth = prediction + residual, as smearing assumes
    o = pd.DataFrame({"site_id": [f"s{i}" for i in range(n)], "block": np.repeat(np.arange(150), 20).astype(str),
                      "fold": np.repeat(np.arange(150), 20) % 5, "interval": "0-5", "source": "a", "target": "soc",
                      "y": y, "pred": pred})
    r = S.score(o, n_boot=100)
    assert 0.85 <= r["interval"]["coverage"] <= 0.95 and r["cis"]["rmse"]["lo"] < r["point"]["rmse"] < r["cis"]["rmse"]["hi"]
    os_ = r["original_scale"]
    assert abs(os_["smeared_mean"]["me"]) < 0.2 < abs(os_["median"]["me"])   # the median back-transform is biased low
    better = o.assign(pred=y + rng.normal(0, 0.1, n))           # closer to the truth
    c = S.compare(o, better, n_boot=300)
    assert c["delta_mse"] > 0 and c["p_signflip"] < 0.01
