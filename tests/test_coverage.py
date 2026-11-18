import numpy as np
import pandas as pd


def test_dissimilarity_flags_points_far_from_training():
    from geosoil import coverage as Cv
    rng = np.random.default_rng(0)
    cols = ["clim_a", "clim_b", "lst_c"]
    train = pd.DataFrame(rng.normal(0, 1, (600, 3)), columns=cols)
    folds = rng.integers(0, 5, 600)
    near = pd.DataFrame(rng.normal(0, 1, (50, 3)), columns=cols)
    far = pd.DataFrame(rng.normal(8, 1, (50, 3)), columns=cols)
    r = Cv.dissimilarity(train, pd.concat([near, far], ignore_index=True), cols, folds=folds)
    inside = r["di"] <= r["threshold"]
    assert inside[:50].mean() > 0.9 and inside[50:].mean() < 0.05


def test_sample_points_are_equal_area_and_in_range():
    from geosoil import coverage as Cv
    p = Cv.sample_points(20000)
    assert p["lat"].between(-56, 78).all() and p["lon"].between(-180, 180).all()
    # equal-area: more points near the equator than near 70°N per degree of latitude
    assert (p["lat"].abs() < 5).sum() > 2 * (p["lat"].between(65, 75)).sum()
