import numpy as np
import pandas as pd
import pytest

from geosoil.features import groups, hand, regolith, thermal


def test_feature_groups_cover_every_prefix():
    cols = ["lon", "lat", "depth_mid", "elevation", "A00", "clim_map", "clim_ppt_01", "rad_k", "tms_dev_1km",
            "litho_glim_su", "lst_day_mean", "regolith_bdticm_cm", "hydro_hand"]
    g = groups.feature_groups(cols)
    assert g["terrain_multiscale"] == ["tms_dev_1km"] and g["lithology"] == ["litho_glim_su"]
    assert g["thermal"] == ["lst_day_mean"] and g["hydrology"] == ["hydro_hand"]
    assert g["regolith"] == ["regolith_bdticm_cm"] and "location" in g
    assert "location" not in groups.feature_groups(cols, include_location=False)


def test_baseline_groups_include_new_sources_and_all():
    from geosoil import baseline_c1
    g = baseline_c1.groups(["elevation", "A00", "clim_map", "litho_sgmc_water", "hydro_hand"])
    assert set(g) >= {"terrain", "alphaearth", "climate", "lithology", "hydrology", "terrain+alphaearth", "all"}
    assert len(g["all"]) == 5


def test_lithology_point_in_polygon_and_one_hot():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box
    from geosoil.features import lithology as L
    polys = gpd.GeoDataFrame({"xx": ["su", "va"]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:4326")
    sites = pd.DataFrame({"site_id": ["a", "b", "c"], "lon": [0.5, 1.5, 5.0], "lat": [0.5, 0.5, 5.0]})
    v = L.point_in_polygon(sites, polys, "xx")
    assert list(v[:2]) == ["su", "va"] and pd.isna(v.iloc[2])
    oh = L.one_hot(v, "litho_glim_", list(L.GLIM_CLASSES))
    assert oh.loc[0, "litho_glim_su"] == 1.0 and oh.loc[0, "litho_glim_va"] == 0.0
    assert oh.loc[2].isna().all() and len(oh.columns) == 16
    assert L.slug("Sedimentary, clastic") == "sedimentary_clastic"


def test_thermal_summaries():
    row = {}
    for m in range(1, 13):
        row[f"lst_day_{m:02d}"] = 10.0 + m
        row[f"lst_night_{m:02d}"] = 0.0 + m / 2
    d = thermal.derive(pd.DataFrame([row])).iloc[0]
    assert d["lst_day_amp"] == pytest.approx(11.0)
    assert d["lst_diurnal_mean"] == pytest.approx(np.mean([10 + m - m / 2 for m in range(1, 13)]))
    assert d["lst_diurnal_max"] == pytest.approx(10 + 12 - 6)
    lay = thermal.source().layers
    assert len(lay) == 24 and lay[0].scale == 0.02 and lay[0].offset == -273.15


def test_regolith_log_and_hand_window():
    d = regolith.derive(pd.DataFrame({"regolith_bdticm_cm": [0.0, 999.0]}))
    assert d["regolith_log_bdticm"].tolist() == pytest.approx([0.0, np.log1p(999.0)])
    w = np.arange(81, dtype=float).reshape(9, 9)
    assert hand.window_features(w) == [40.0, 40.0, 0.0]
    w[4, 4] = np.nan
    assert all(np.isnan(hand.window_features(w)))
    assert hand.tile_name(-105.5, 40.2) == "Copernicus_DSM_COG_10_N40_00_W106_00_HAND"


def test_static_rejects_unknown_sources(tmp_path):
    from geosoil.features import static
    pd.DataFrame({"site_id": ["a"], "lon": [0.0], "lat": [0.0]}).to_parquet(tmp_path / "l.parquet")
    with pytest.raises(SystemExit):
        static.main(["--source", "nope", "--labels", str(tmp_path / "l.parquet")])


def test_landuse_cell_area_and_history():
    from geosoil.features import landuse as LU
    a = LU.cell_area_km2(np.array([0.01, 60.01]))
    assert a[0] == pytest.approx(85.5, rel=0.01) and a[1] == pytest.approx(a[0] * np.cos(np.radians(60.04)), rel=0.01)
    years = LU.DECADES
    crop = {y: np.array([0.0 if y < 1900 else 0.5, 0.0, np.nan]) for y in years}   # site 0 farmed from 1900
    graz = {y: np.array([0.1, 0.3, np.nan]) for y in LU.GRAZ_YEARS}
    d = LU.derive(crop, graz)
    assert d.loc[0, "hist_years_since_conv"] == 2017 - 1900
    assert d.loc[0, "hist_crop_max"] == 0.5 and d.loc[0, "hist_crop_1850"] == 0.0
    assert d.loc[0, "hist_crop_years"] == pytest.approx(2025 - 1900)
    assert np.isnan(d.loc[1, "hist_years_since_conv"]) and d.loc[1, "hist_crop_years"] == 0
    assert d.loc[2].isna().all()
    assert LU.member("cropland", 1950) == "baseline/asc/1950AD_lu/cropland1950AD.asc"


def test_surficial_parses_origin_grain_and_thickness():
    from geosoil.features import surficial as S
    assert S.origin(11) == "alluvial" and S.origin(311) == "eolian_loess" and S.origin(331) == "eolian_sand"
    assert S.origin(451) == "glaciofluvial" and S.origin(422) == "glacial_till" and S.origin(985) == "volcanic"
    assert S.origin(999) == "water" and S.origin(930) == "residual" and S.origin(702) == "organic"
    assert S.grain("Glacial till sediments, mostly clayey, thick") == 0.0
    assert S.grain("Eolian sediments, mostly loess, thin") == 1.0
    assert S.grain("Proglacial sediments, mostly coarse-grained, discontinuous") == 2.0
    assert np.isnan(S.grain("Residual materials developed in bedrock, thin"))
    assert S.thickness("Alluvial sediments, thick") == 2.0 and S.thickness("Colluvial sediments, discontinuous") == 0.0
    e = S.encode(pd.Series([422, np.nan]), pd.Series(["Glacial till sediments, mostly silty, thick", np.nan]))
    assert e.loc[0, "surf_glacial_till"] == 1.0 and e.loc[0, "surf_alluvial"] == 0.0
    assert e.loc[0, "surf_grain"] == 1.0 and e.loc[0, "surf_thick"] == 2.0 and e.loc[1].isna().all()
    assert groups.feature_groups(["surf_grain", "litho_glim_su"])["surficial"] == ["surf_grain"]
