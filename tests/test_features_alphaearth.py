"""Block planner + extractor on a small synthetic bottom-up COG (the AlphaEarth layout)."""
import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from affine import Affine
from pyproj import Transformer

from geosoil.features import alphaearth as A

CRS = "EPSG:32613"
X0, Y0 = 500_000.0, 4_400_000.0          # tile south-west corner (UTM 13N)
SIZE, BLOCK = 32, 16


@pytest.fixture
def tile(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "BLOCK", BLOCK)
    path = tmp_path / "t.tiff"
    # band 0 encodes the column, band 1 the row (counted from the southern edge), band 2 is a constant
    data = np.zeros((A.N_BANDS, SIZE, SIZE), dtype=np.int8)
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    data[0], data[1], data[2] = cols + 1, rows + 1, 50
    data[:, 0, 0] = -128                                   # one no-data pixel
    with rasterio.open(path, "w", driver="GTiff", width=SIZE, height=SIZE, count=A.N_BANDS, dtype="int8",
                       crs=CRS, transform=Affine(10, 0, X0, 0, 10, Y0), tiled=True, blockxsize=BLOCK,
                       blockysize=BLOCK, nodata=-128) as ds:   # positive y-res = bottom-up, like AlphaEarth
        ds.write(data)
    monkeypatch.setattr(A, "tile_url", lambda p: str(path))
    index = pd.DataFrame([dict(path="s3://b/tge-labs/aef/v1/annual/2020/13N/t.tiff", year=2020, crs=CRS,
                               utm_west=X0, utm_south=Y0, utm_east=X0 + SIZE * 10, utm_north=Y0 + SIZE * 10,
                               wgs84_west=-106.0, wgs84_east=-104.0, wgs84_south=39.0, wgs84_north=41.0)])
    return index


def _site(site_id, col, row):
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(X0 + (col + 0.5) * 10,
                                                                                Y0 + (row + 0.5) * 10)
    return dict(site_id=site_id, lon=lon, lat=lat)


def test_plan_maps_sites_to_block_and_pixel(tile):
    sites = pd.DataFrame([_site("a", 3, 5), _site("b", 20, 30), _site("far", 0, 0)])
    sites.loc[2, "lon"] = -110.0                           # outside the tile
    p = A.plan(sites, tile, [2020]).set_index("site_id")
    assert "far" not in p.index
    assert (p.loc["a", ["bx", "by", "px", "py"]] == [0, 0, 3, 5]).all()
    assert (p.loc["b", ["bx", "by", "px", "py"]] == [1, 1, 4, 14]).all()


def test_extract_reads_the_right_pixel_once_per_block(tile):
    sites = pd.DataFrame([_site("a", 3, 5), _site("c", 7, 9), _site("b", 20, 30)])
    p = A.plan(sites, tile, [2020])
    p = p.assign(key=p["path"] + "#" + p["bx"].astype(str) + "_" + p["by"].astype(str))
    out = pd.concat([A.extract_group(g, neighbors=0)[0] for _, g in p.groupby("key")]).set_index("site_id")
    for sid, col, row in (("a", 3, 5), ("c", 7, 9), ("b", 20, 30)):
        assert out.loc[sid, "A00"] == pytest.approx(A.dequantize(np.int8([col + 1]))[0])
        assert out.loc[sid, "A01"] == pytest.approx(A.dequantize(np.int8([row + 1]))[0])
    assert p["key"].nunique() == 2                         # a and c share block (0, 0)


def test_unlabelled_sample_is_deterministic_valid_and_georeferenced(tile):
    p = A.plan(pd.DataFrame([_site("a", 3, 5)]), tile, [2020])
    _, u1 = A.extract_group(p, neighbors=20)
    _, u2 = A.extract_group(p, neighbors=20)
    assert len(u1) == 20 and u1.equals(u2)
    assert (u1["A02"] == 50).all()                         # no-data pixel never sampled
    col = (u1["utm_x"] - X0) / 10 - 0.5
    row = (u1["utm_y"] - Y0) / 10 - 0.5
    assert np.allclose(u1["A00"], col + 1) and np.allclose(u1["A01"], row + 1)


def test_dequantize_matches_readme():
    raw = np.int8([-127, -1, 0, 1, 127, -128])
    d = A.dequantize(raw)
    assert d[2] == 0 and np.isnan(d[5])
    assert d[4] == pytest.approx((127 / 127.5) ** 2) and d[0] == pytest.approx(-(127 / 127.5) ** 2)


def test_status_reports_missing_tasks(tmp_path):
    (tmp_path / "sites-0000-of-0002.parquet").write_bytes(b"x")
    (tmp_path / "sites-0000-of-0002.json").write_text('{"failed_blocks": [{"block": "k"}]}')
    s = A.status(2, tmp_path)
    assert s["done"] == 1 and s["missing_tasks"] == [1] and s["failed_blocks"] == 1


def test_transient_failures_are_retried_before_being_recorded(tile, tmp_path, monkeypatch):
    p = A.plan(pd.DataFrame([_site("a", 3, 5), _site("b", 20, 30)]), tile, [2020])
    calls = {}
    real = A.extract_group

    def flaky(g, neighbors, seed=0):
        key = g.iloc[0].bx
        calls[key] = calls.get(key, 0) + 1
        if key == 1 and calls[key] < 3:
            raise OSError("Could not resolve host")
        return real(g, neighbors, seed)

    monkeypatch.setattr(A, "extract_group", flaky)
    monkeypatch.setattr(A, "RETRY_BACKOFF_S", 0)
    meta = A.extract(p, 0, 1, threads=2, neighbors=0, out=tmp_path)
    assert meta["failed_blocks"] == [] and meta["site_years"] == 2 and calls[1] == 3
