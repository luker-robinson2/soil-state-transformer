import math

import numpy as np
import pytest

from geosoil.features import terrain as T

N = 2 * T.R + 1
DX = DY = 30.0


def _plane(east=0.0, north=0.0):
    """Elevation window rising `east` m/m eastwards and `north` m/m northwards (rows run north->south)."""
    r, c = np.mgrid[0:N, 0:N].astype(float)
    return 100 + east * c * DX + north * (N - 1 - r) * DY


def test_tile_names():
    assert T.tile_name(-105.5, 40.2) == "Copernicus_DSM_COG_10_N40_00_W106_00_DEM"
    assert T.tile_name(-60.5, -3.5) == "Copernicus_DSM_COG_10_S04_00_W061_00_DEM"
    assert T.tile_name(6.2, 0.1) == "Copernicus_DSM_COG_10_N00_00_E006_00_DEM"


def test_slope_and_aspect_of_an_east_rising_plane():
    f = T.features(_plane(east=0.1), DX, DY)
    assert f["slope"] == pytest.approx(math.degrees(math.atan(0.1)))
    assert f["aspect_sin"] == pytest.approx(-1.0) and f["aspect_cos"] == pytest.approx(0.0, abs=1e-9)  # faces west


def test_north_rising_plane_faces_south_and_flat_tpi():
    f = T.features(_plane(north=0.05), DX, DY)
    assert f["aspect_cos"] == pytest.approx(-1.0)
    assert f["tpi_100"] == pytest.approx(0.0, abs=1e-6) and f["curvature"] == pytest.approx(0.0, abs=1e-9)
    assert f["relief_500"] == pytest.approx(0.05 * (N - 1) * DY)


def test_a_peak_has_positive_tpi_and_negative_curvature():
    r, c = np.mgrid[0:N, 0:N].astype(float)
    peak = 200 - 0.5 * ((r - T.R) ** 2 + (c - T.R) ** 2)
    f = T.features(peak, DX, DY)
    assert f["tpi_100"] > 0 and f["tpi_500"] > f["tpi_100"] and f["curvature"] < 0
    assert f["slope"] == pytest.approx(0.0, abs=1e-9) and f["elevation"] == 200


def test_missing_centre_gives_all_nan_and_edges_degrade_gracefully():
    w = _plane(east=0.1)
    w[T.R, T.R] = np.nan
    assert all(np.isnan(v) for v in T.features(w, DX, DY).values())
    w = _plane(east=0.1)
    w[:, : T.R - 5] = np.nan                       # site near a tile edge
    f = T.features(w, DX, DY)
    assert np.isfinite(f["slope"]) and np.isfinite(f["tpi_100"])
