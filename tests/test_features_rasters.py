import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from affine import Affine

from geosoil.features import rasters as RS


def _write(path, data, transform, crs, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[-1], height=data.shape[-2],
                       count=data.shape[0], dtype=data.dtype, crs=crs, transform=transform, nodata=nodata) as ds:
        ds.write(data)


def test_samples_containing_cell_with_scale_nodata_and_outside(tmp_path):
    data = np.arange(100, dtype=np.int16).reshape(1, 10, 10)
    data[0, 0, 0] = -9999
    p = tmp_path / "g.tif"
    _write(p, data, Affine(1, 0, -105, 0, -1, 45), "EPSG:4326", nodata=-9999)   # 1-degree cells
    lyr = RS.Layer("v", str(p), scale=0.5, offset=1.0)
    vals = RS.sample_layer(str(p), lyr, np.array([-104.5, -100.2, -104.9, -120.0]), np.array([44.5, 40.1, 44.9, 40.0]))
    assert vals[0] == pytest.approx(np.nan, nan_ok=True)             # nodata cell (row 0, col 0)
    assert vals[1] == pytest.approx(data[0, 4, 4] * 0.5 + 1.0)       # row 4, col 4
    assert np.isnan(vals[3])                                         # outside the grid


def test_projected_grid_and_valid_range(tmp_path):
    from pyproj import Transformer
    data = np.full((1, 20, 20), 3.0, dtype=np.float32)
    data[0, 5, 7] = 250.0
    x0, y0 = 500_000.0, 4_500_000.0
    p = tmp_path / "u.tif"
    _write(p, data, Affine(100, 0, x0, 0, -100, y0), "EPSG:32613")
    lon, lat = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True).transform(
        [x0 + 750, x0 + 50], [y0 - 550, y0 - 50])
    vals = RS.sample_layer(str(p), RS.Layer("k", str(p), valid_range=(0, 100)), np.array(lon), np.array(lat))
    assert np.isnan(vals[0]) and vals[1] == 3.0                     # 250 is outside the valid range


def test_extract_writes_shard_with_manifest_and_derived_columns(tmp_path):
    data = np.ones((2, 4, 4), dtype=np.float32)
    data[1] *= 3
    p = tmp_path / "m.tif"
    _write(p, data, Affine(1, 0, 0, 0, -1, 4), "EPSG:4326")
    src = RS.Source("demo", [RS.Layer("a", str(p), band=1), RS.Layer("b", str(p), band=2),
                             RS.Layer("bad", str(tmp_path / "missing.tif"))],
                    licence="CC0", derive=lambda d: d.assign(ratio=d["b"] / d["a"]))
    sites = pd.DataFrame({"site_id": ["s1", "s2"], "lon": [0.5, 9.0], "lat": [3.5, 3.5]})
    meta = RS.extract(src, sites, out_root=tmp_path / "out")
    df = pd.read_parquet(tmp_path / "out" / "demo" / "sites-0000-of-0001.parquet").set_index("site_id")
    assert df.loc["s1", "ratio"] == 3.0 and np.isnan(df.loc["s2", "a"])
    assert meta["failed"] == 1                                      # the missing file is logged, not fatal


def test_file_declared_scale_and_offset_are_applied(tmp_path):
    data = np.full((1, 4, 4), 811, dtype=np.int16)                  # TerraClimate-style tmax
    p = tmp_path / "s.tif"
    with rasterio.open(p, "w", driver="GTiff", width=4, height=4, count=1, dtype="int16", crs="EPSG:4326",
                       transform=Affine(1, 0, 0, 0, -1, 4), nodata=-32768) as ds:
        ds.write(data)
        ds.scales, ds.offsets = (0.1,), (-73.0,)
    v = RS.sample_layer(str(p), RS.Layer("tmax", str(p)), np.array([1.5]), np.array([2.5]))
    assert v[0] == pytest.approx(8.1)
