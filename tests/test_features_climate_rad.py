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
    assert d["clim_p_minus_pet"] == pytest.approx(0.0)
    assert d["clim_wet_months"] == 6                        # 90 mm months exceed 50 mm PET
    tmean = np.array([(a + b) / 2 for a, b in zip(tmax, tmin)])
    assert d["clim_gdd5"] == pytest.approx(np.sum(np.clip(tmean - 5, 0, None) * climate.DAYS))
    assert d["clim_frost_months"] == int((np.array(tmin) < 0).sum())


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


def test_radiometrics_falls_back_to_north_american_grid():
    df = pd.DataFrame({"site_id": ["us", "ca"], "nure_k": [1.7, np.nan], "nure_u": [2.0, np.nan],
                       "nure_th": [10.0, np.nan], "namrad_k": [9.0, 2.0], "namrad_u": [9.0, 1.0],
                       "namrad_th": [9.0, 6.0]})
    d = radiometrics.derive(df).set_index("site_id")
    assert d.loc["us", "rad_k"] == 1.7 and d.loc["us", "rad_src"] == 1      # NURE wins where present
    assert d.loc["ca", "rad_th"] == 6.0 and d.loc["ca", "rad_src"] == 3
    lay = {l.name: l for l in radiometrics.source().layers}
    assert lay["namrad_k"].member == "NAMrad_K.flt" and "+R=6370997" in lay["namrad_k"].crs


def test_sample_layer_reads_zip_member_with_crs_override(tmp_path):
    import zipfile
    from geosoil.features import rasters
    # 4x4 EHdr float grid in spherical TM, no .prj: the Layer supplies the CRS
    (tmp_path / "g.hdr").write_text("ncols 4\nnrows 4\nxllcorner -4000\nyllcorner 4400000\ncellsize 2000\n"
                                    "nodata_value -9999\nbyteorder lsbfirst\n")
    a = np.arange(16, dtype="<f4").reshape(4, 4)
    a[0, 0] = -9999
    a.tofile(tmp_path / "g.flt")
    with zipfile.ZipFile(tmp_path / "g.zip", "w") as z:
        z.write(tmp_path / "g.flt", "g.flt")
        z.write(tmp_path / "g.hdr", "g.hdr")
    lay = rasters.Layer("x", str(tmp_path / "g.zip"), member="g.flt", crs=radiometrics.NAMRAD_CRS, scale=1.0, offset=0.0)
    path = rasters._gdal_path(lay, "t")
    assert path == f"/vsizip/{tmp_path}/g.zip/g.flt" and rasters._local_file(path) == tmp_path / "g.zip"
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(lay.crs, "EPSG:4326", always_xy=True).transform(
        np.array([-3000.0, 3000.0, 1000.0]), np.array([4407000.0, 4401000.0, 4500000.0]))
    v = rasters.sample_layer(path, lay, lon, lat)
    assert np.isnan(v[0]) and v[1] == 15.0 and np.isnan(v[2])               # nodata, last cell, off-grid


def test_extract_reads_ehdr_sidecars_in_zips(tmp_path):
    """extract() must not hide the .hdr sidecar (GDAL_DISABLE_READDIR_ON_OPEN applies to remote reads only)."""
    import zipfile
    from geosoil.features import rasters
    (tmp_path / "g.hdr").write_text("ncols 2\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 1\n"
                                    "nodata_value -9999\nbyteorder lsbfirst\n")
    np.array([[1, 2], [3, 4]], dtype="<f4").tofile(tmp_path / "g.flt")
    with zipfile.ZipFile(tmp_path / "g.zip", "w") as z:
        z.write(tmp_path / "g.flt", "g.flt")
        z.write(tmp_path / "g.hdr", "g.hdr")
    src = rasters.Source("t", [rasters.Layer("v", str(tmp_path / "g.zip"), member="g.flt", crs="EPSG:4326",
                                             scale=1.0, offset=0.0)], licence="test")
    res = rasters.extract(src, pd.DataFrame({"site_id": ["a"], "lon": [1.5], "lat": [0.5]}), out_root=tmp_path / "out")
    assert res["failed"] == 0
