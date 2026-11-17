import numpy as np
import pandas as pd
import pytest

from geosoil.features import neighbours as NB


def _sites():
    # block A: two sites 1 km apart near (0, 0); block B: one site ~11 km east; block C: one ~111 km north
    return pd.DataFrame({"lon": [0.0, 0.009, 0.1, 0.0], "lat": [0.0, 0.0, 0.0, 1.0],
                         "block": ["A", "A", "B", "C"], "interval": ["0-5"] * 4})


def test_training_rows_skip_their_own_block():
    t = _sites()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    f = NB.fold_features(t, t, y, exclude_same_block=True, k=1)
    # site 0's nearest neighbour is site 1 (1 km, same block), so it must get site 2 (block B) instead
    assert f.loc[0, "nb_near"] == 3.0 and f.loc[0, "nb_d1_km"] == pytest.approx(11.1, rel=0.02)
    assert f.loc[2, "nb_near"] in (1.0, 2.0)                       # block B sees block A
    g = NB.fold_features(t, t, y, exclude_same_block=False, k=1)
    assert g.loc[0, "nb_near"] == 1.0 and g.loc[0, "nb_d1_km"] == 0.0   # no exclusion: itself


def test_validation_rows_use_only_training_rows_of_the_same_interval():
    t = _sites()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    q = pd.DataFrame({"lon": [0.0, 0.0], "lat": [0.5, 0.5], "block": ["D", "D"], "interval": ["0-5", "5-15"]})
    f = NB.fold_features(t, q, y, exclude_same_block=False, k=2)
    assert not np.isnan(f.loc[0, "nb_idw"])
    assert f.loc[1].isna().all()                                   # no 5-15 training rows
    assert f.loc[0, "nb_dmean_km"] >= f.loc[0, "nb_d1_km"]


def test_missing_labels_are_ignored_and_short_lists_padded():
    t = _sites()
    y = np.array([np.nan, 2.0, np.nan, np.nan])
    q = pd.DataFrame({"lon": [5.0], "lat": [5.0], "block": ["Z"], "interval": ["0-5"]})
    f = NB.fold_features(t, q, y, exclude_same_block=False, k=3)
    assert f.loc[0, "nb_near"] == 2.0 and f.loc[0, "nb_idw"] == pytest.approx(2.0)


def test_dense_block_exclusion_needs_wide_search():
    rng = np.random.default_rng(0)
    n = 300                                                        # 299 sites in one block, 1 far outside
    t = pd.DataFrame({"lon": np.r_[rng.uniform(0, 0.5, n - 1), 3.0], "lat": np.r_[rng.uniform(0, 0.5, n - 1), 3.0],
                      "block": ["A"] * (n - 1) + ["B"], "interval": ["0-5"] * n})
    y = np.r_[np.zeros(n - 1), 9.0]
    f = NB.fold_features(t, t.iloc[:5], y, exclude_same_block=True, k=1)
    assert (f["nb_near"] == 9.0).all()                             # found past 299 same-block sites
