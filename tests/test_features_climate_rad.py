import numpy as np
import pandas as pd
import pytest

from geosoil.features import climate, radiometrics


def _clim_row(ppt, tmax, tmin, pet):
    row = {"site_id": "s"}
    for v in climate.VARS:
        for m in range(1, 13):
            row[f"clim_{v}_{m:02d}"] = 1.0
    for m in range(1, 13):
        row[f"clim_ppt_{m:02d}"] = ppt[m - 1]
        row[f"clim_tmax_{m:02d}"] = tmax[m - 1]
        row[f"clim_tmin_{m:02d}"] = tmin[m - 1]
        row[f"clim_pet_{m:02d}"] = pet
    return pd.DataFrame([row])


def test_climate_annual_summaries():
    ppt = [10.0] * 6 + [90.0] * 6
    tmax = list(np.linspace(5, 27, 12))
    tmin = [t - 10 for t in tmax]
    d = climate.derive(_clim_row(ppt, tmax, tmin, pet=50.0)).iloc[0]
    assert d["clim_map"] == pytest.approx(600.0)
    assert d["clim_pet"] == pytest.approx(600.0) and d["clim_ai"] == pytest.approx(1.0)
    assert d["clim_mat"] == pytest.approx(np.mean([(a + b) / 2 for a, b in zip(tmax, tmin)]))
    assert d["clim_trange"] == pytest.approx(22.0)
    assert d["clim_ppt_cv"] == pytest.approx(np.std(ppt, ddof=1) / np.mean(ppt))


def test_climate_layers_cover_every_variable_and_month():
    names = [l.name for l in climate.source().layers]
    assert len(names) == len(climate.VARS) * 12 and "clim_tmin_07" in names
    assert all(l.subdataset in climate.VARS for l in climate.source().layers)


def test_radiometrics_coalesces_regions_and_flags_source():
    df = pd.DataFrame({"site_id": ["us", "au", "none"],
                       "nure_k": [1.7, np.nan, np.nan], "nure_u": [2.5, np.nan, np.nan], "nure_th": [10.0, np.nan, np.nan],
                       "radmap_k": [np.nan, 0.8, np.nan], "radmap_u": [np.nan, 1.5, np.nan], "radmap_th": [np.nan, 8.0, np.nan]})
    d = radiometrics.derive(df).set_index("site_id")
    assert d.loc["us", "rad_k"] == 1.7 and d.loc["us", "rad_src"] == 1
    assert d.loc["au", "rad_th"] == 8.0 and d.loc["au", "rad_src"] == 2
    assert d.loc["us", "rad_th_k"] == pytest.approx(10 / 1.7)
    assert np.isnan(d.loc["none", "rad_k"]) and np.isnan(d.loc["none", "rad_src"])
    assert not any(c.startswith(("nure_", "radmap_")) for c in d.columns)


def test_radiometrics_uses_unique_local_filenames():
    files = [l.filename for l in radiometrics.source().layers if l.name.startswith("nure_")]
    assert len(set(files)) == 3 and all(f.endswith(".tif") for f in files)
