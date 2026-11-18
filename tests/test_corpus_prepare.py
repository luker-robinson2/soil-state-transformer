"""Harmonization (1.8), splits (1.10) and noise floor (1.11)."""
import numpy as np
import pandas as pd
import pytest

from geosoil.corpus import harmonize, noise_floor, splits


def _lab(site, target, top, bot, value, family, layer=None, source="kssl_ldm"):
    return dict(site_id=site, layer_id=layer or f"{site}/{top}", source=source, lon=-100.0, lat=40.0,
                coord_precision_m=np.nan, year=2000, license="x", target=target, depth_top=float(top),
                depth_bot=float(bot), value=float(value), method_family=family)


def test_interval_mean_is_thickness_weighted_and_needs_coverage():
    df = pd.DataFrame([_lab("s", "clay", 0, 10, 10, "pipette"), _lab("s", "clay", 10, 40, 40, "pipette")])
    iv = harmonize.to_intervals(df).set_index("depth_top")
    assert iv.loc[5.0, "value"] == pytest.approx(25.0)         # 5-15: half 10, half 40
    assert iv.loc[15.0, "value"] == pytest.approx(40.0)
    assert 30.0 not in iv.index                                # 30-60 only a third covered
    assert harmonize.to_intervals(df, min_coverage=0.3).set_index("depth_top").loc[30.0, "coverage"] == \
        pytest.approx(10 / 30)


def test_preferred_method_wins_and_others_are_converted():
    df = pd.DataFrame([
        _lab("a", "ph", 0, 5, 6.0, "ph_h2o"), _lab("a", "ph", 0, 5, 5.5, "ph_cacl2", layer="a/x"),
        _lab("b", "ph", 0, 5, 5.0, "ph_cacl2"),
        _lab("c", "soc", 0, 5, 7.0, "loss_on_ignition"),
    ])
    conv = {"ph:ph_cacl2->ph_h2o": dict(target="ph", src="ph_cacl2", dst="ph_h2o", kind="linear", a=1.0, b=1.0)}
    out, _ = harmonize.harmonize(df, conversions=conv)
    by = out.set_index("site_id")
    assert by.loc["a", "value"] == 6.0 and not by.loc["a", "converted"]
    assert by.loc["b", "value"] == 6.0 and by.loc["b", "converted"]
    assert "c" not in by.index                                  # LOI is not in the priority list


def test_conversions_are_fitted_on_paired_layers():
    rows = []
    for i, x in enumerate(np.linspace(4, 8, 50)):
        rows += [_lab(f"s{i}", "ph", 0, 5, x, "ph_cacl2", layer=f"L{i}"),
                 _lab(f"s{i}", "ph", 0, 5, 0.9 + 0.94 * x, "ph_h2o", layer=f"L{i}")]
    conv = harmonize.fit_conversions(pd.DataFrame(rows))["ph:ph_cacl2->ph_h2o"]
    assert conv["a"] == pytest.approx(0.9) and conv["b"] == pytest.approx(0.94) and conv["n"] == 50


def test_splits_are_by_block_stable_and_mark_holdout():
    rng = np.random.default_rng(0)
    sites = pd.DataFrame({"site_id": [f"s{i}" for i in range(5000)],
                          "lon": rng.uniform(-120, -70, 5000), "lat": rng.uniform(25, 49, 5000),
                          "role": "train", "near_holdout": False})
    sites.loc[0, "role"] = "holdout"
    a, b = splits.assign(sites), splits.assign(sites.sample(frac=1, random_state=1))
    merged = a.merge(b, on="site_id")
    assert (merged.split_x == merged.split_y).all() and (merged.fold_x == merged.fold_y).all()
    per_block = a[a.split != "holdout"].groupby("block")[["split", "fold"]].nunique()
    assert (per_block <= 1).all().all()                         # one split/fold per block
    assert a.loc[a.site_id == "s0", "split"].item() == "holdout"
    assert 0.03 < (a.split == "test").mean() < 0.2
    assert set(a.loc[a.split == "train", "fold"]) == set(range(5))


def test_noise_floor_recovers_known_nugget():
    rng = np.random.default_rng(0)
    n = 400
    truth = rng.normal(30, 10, n)                               # total variance ~100 + nugget
    rows, sites = [], []
    for i in range(n):
        for k in range(2):
            sid = f"p{i}_{k}"
            lat = 30 + i * 0.01 + k * 1e-4                     # pair ~11 m apart
            sites.append(dict(site_id=sid, lon=-100.0, lat=lat))
            rows.append(dict(site_id=sid, target="clay", method_family="pipette", depth_top=0.0,
                             value=truth[i] + rng.normal(0, 2)))   # nugget variance 4
    pairs = noise_floor.colocated_pairs(pd.DataFrame(sites))
    assert len(pairs) == n
    res = noise_floor.floor_table(pd.DataFrame(rows), pairs)["clay/topsoil_0_30/raw"]
    assert res["nugget"] == pytest.approx(4.0, rel=0.2)
    assert res["r2_ceiling"] == pytest.approx(1 - 4 / 104, abs=0.02)


def test_crowded_placeholder_coordinates_are_skipped():
    sites = pd.DataFrame({"site_id": list("abcde"), "lon": [-100.0] * 5, "lat": [40.0] * 5})
    assert noise_floor.colocated_pairs(sites, max_sites_per_point=3).empty


def test_new_targets_derived_om_and_gated_conversions():
    import numpy as np
    from geosoil.corpus import harmonize as H
    iv = pd.DataFrame({
        "site_id": ["a", "a", "b", "b", "c"], "target": ["soc", "cec", "cec", "om", "om"],
        "method_family": ["dry_combustion", "ecec", "cec_ph7", "measured", "measured"],
        "value": [10.0, 5.0, 20.0, 99.0, 40.0], "coverage": 1.0, "depth_top": 0.0, "depth_bot": 5.0})
    good = {"cec:ecec->cec_ph7": dict(target="cec", src="ecec", dst="cec_ph7", kind="linear", a=1.0, b=2.0, r2=0.9)}
    out = H.add_derived_om(H.choose_method(iv, good))
    om = out[out["target"] == "om"].set_index("site_id")["value"]
    assert om["a"] == pytest.approx(17.24) and om["c"] == 40.0 and om["b"] == 99.0   # measured OM only where no SOC
    cec = out[out["target"] == "cec"].set_index("site_id")["value"]
    assert cec["a"] == pytest.approx(11.0) and cec["b"] == 20.0                      # ecec converted (r2 0.9)
    bad = {"cec:ecec->cec_ph7": dict(good["cec:ecec->cec_ph7"], r2=0.5)}
    out2 = H.choose_method(iv, bad)
    assert "a" not in set(out2.loc[out2["target"] == "cec", "site_id"])             # poor fit: dropped, not mixed
