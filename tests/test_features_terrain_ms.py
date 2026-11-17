import numpy as np
import pytest

from geosoil.features import terrain_ms as TM


def test_box_sums_match_brute_force():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 50))
    I = TM.integral(a)
    r, c = np.array([0, 10, 39]), np.array([0, 25, 49])
    got = TM.box(I, r, c, 3, 4)
    want = [a[max(0, ri - 3):ri + 4, max(0, ci - 4):ci + 5].sum() for ri, ci in zip(r, c)]
    assert np.allclose(got, want)


def test_stats_ignore_nans():
    z = np.arange(25, dtype=float).reshape(5, 5)
    z[0, 0] = np.nan
    m, sd, cover = TM.Stats(z).mean_std(np.array([1]), np.array([1]), 1, 1)
    block = z[0:3, 0:3]
    assert m[0] == pytest.approx(np.nanmean(block)) and sd[0] == pytest.approx(np.nanstd(block))
    assert cover[0] == pytest.approx(8 / 9)


def test_multiscale_on_a_valley_and_a_plane():
    n = 801                                            # 90 m pixels, ~72 km across
    rr, cc = np.mgrid[0:n, 0:n].astype(float)
    centre = n // 2
    valley = 0.002 * ((cc - centre) * 90.0) ** 2 / 90.0  # parabola across columns, low at centre
    f = TM.multiscale(valley, np.array([centre]), np.array([centre]), 90.0, 90.0).iloc[0]
    assert f["tms_tpi_5km"] < 0 and f["tms_dev_20km"] < 0          # valley bottom sits below its surroundings
    assert f["tms_relief_20km"] > f["tms_relief_1km"] > 0
    assert f["tms_dev_max"] < 0 and f["tms_dev_scale"] in TM.SCALES_KM

    plane = 100 + 0.05 * (n - 1 - rr) * 90.0             # rises northwards at 5 %
    g = TM.multiscale(plane, np.array([centre]), np.array([centre]), 90.0, 90.0).iloc[0]
    assert g["tms_slope_2km"] == pytest.approx(np.degrees(np.arctan(0.05)), rel=1e-3)
    assert g["tms_tpi_5km"] == pytest.approx(0.0, abs=1e-6)


def test_columns_and_tile_names():
    assert len(TM.columns()) == 4 * len(TM.SCALES_KM) + 2 and "tms_dev_0p5km" in TM.columns()
    assert TM.tile_name(40, -106) == "Copernicus_DSM_COG_30_N40_00_W106_00_DEM"
    assert TM.tile_name(-4, 39) == "Copernicus_DSM_COG_30_S04_00_E039_00_DEM"


def test_low_coverage_windows_are_nan():
    z = np.full((101, 101), np.nan)
    z[50, 50] = 10.0
    f = TM.multiscale(z, np.array([50]), np.array([50]), 90.0, 90.0).iloc[0]
    assert np.isnan(f["tms_tpi_1km"]) and np.isnan(f["tms_dev_max"])
