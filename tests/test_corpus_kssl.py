import numpy as np
import pandas as pd
import pytest

from geosoil.corpus import kssl, schema


def _sites(**over):
    base = dict(
        pedon_key=[1, 2, 3],
        site_key=[10, 20, 30],
        pedlabsampnum=["a", "b", "c"],
        country_key=[244, 244, 244],
        site_obsdate=["1990-06-01T00:00:00.0Z", "9863-01-01T00:00:00.0Z", None],
        latitude_std_decimal_degrees=[40.0, np.nan, np.nan],
        longitude_std_decimal_degrees=[-105.0, np.nan, np.nan],
        latitude_decimal_degrees=[40.1, 38.5, np.nan],
        longitude_decimal_degrees=[-105.1, -100.5, np.nan],
    )
    base.update(over)
    return pd.DataFrame(base)


def _layer(**over):
    row = dict(
        pedon_key=1, layer_key=100, labsampnum="x", hzn_top=0.0, hzn_bot=10.0, prep_code="S",
        sand_total=np.nan, silt_total=np.nan, clay_total=np.nan, particle_size_method=np.nan,
        total_carbon_ncs=np.nan, total_carbon_ncs_method=np.nan, caco3_lt_2_mm=np.nan,
        organic_carbon_walkley_black=np.nan, oc_walkley_black_method=np.nan,
        ph_h2o=np.nan, ph_h2o_method=np.nan, ph_cacl2=np.nan, ph_cacl2_method=np.nan,
        bulk_density_third_bar=np.nan, bulk_density_third_bar_method=np.nan,
        bulk_density_oven_dry=np.nan, bulk_density_oven_dry_method=np.nan,
    )
    row.update(over)
    return row


def _build(rows, sites=None):
    return kssl.build(_sites() if sites is None else sites, pd.DataFrame(rows))


def test_soc_subtracts_carbonate_carbon_and_converts_to_g_per_kg():
    labels, _ = _build([_layer(total_carbon_ncs=3.0, total_carbon_ncs_method="4H2a", caco3_lt_2_mm=10.0, ph_h2o=8.0)])
    soc = labels[(labels.target == "soc") & (labels.method_family == "dry_combustion")]
    assert soc.value.item() == pytest.approx((3.0 - 0.12 * 10.0) * 10)


def test_total_carbon_without_carbonate_is_dropped_on_alkaline_soil():
    labels, _ = _build([_layer(total_carbon_ncs=3.0, total_carbon_ncs_method="4H2a", ph_h2o=8.2)])
    assert labels[labels.target == "soc"].empty


def test_total_carbon_without_carbonate_is_kept_on_acid_soil():
    labels, _ = _build([_layer(total_carbon_ncs=2.0, total_carbon_ncs_method="4H2a", ph_h2o=5.5)])
    assert labels[labels.target == "soc"].value.item() == pytest.approx(20.0)


def test_carbon_method_comes_from_code_not_column_name():
    labels, _ = _build([
        _layer(layer_key=1, organic_carbon_walkley_black=1.0, oc_walkley_black_method="6A1c"),
        _layer(layer_key=2, organic_carbon_walkley_black=1.0, oc_walkley_black_method="6A2d"),
    ])
    fam = labels.set_index("layer_id").method_family
    assert fam["kssl:1"] == "walkley_black"
    assert fam["kssl:2"] == "dry_combustion"


def test_ph_keeps_both_methods_as_separate_rows():
    labels, _ = _build([_layer(ph_h2o=6.5, ph_h2o_method="4C1a2a", ph_cacl2=6.0, ph_cacl2_method="4C1a2a")])
    ph = labels[labels.target == "ph"].set_index("method_family").value
    assert ph["ph_h2o_1to1"] == 6.5 and ph["ph_cacl2_0.01M"] == 6.0
    assert labels[labels.target == "ph"].method_id.nunique() == 2


def test_coordinates_fall_back_and_unlocated_sites_are_dropped():
    labels, drops = _build([
        _layer(pedon_key=1, layer_key=1, ph_h2o=6.0),
        _layer(pedon_key=2, layer_key=2, ph_h2o=6.0),
        _layer(pedon_key=3, layer_key=3, ph_h2o=6.0),
    ])
    by = labels.set_index("site_id")
    assert by.loc["kssl:1", "lat"] == 40.0          # standardized coords preferred
    assert by.loc["kssl:2", "lat"] == 38.5          # falls back to raw
    assert "kssl:3" not in by.index and drops["no_coordinates"] == 1


def test_typo_years_become_missing():
    labels, _ = _build([_layer(pedon_key=1, layer_key=1, ph_h2o=6.0), _layer(pedon_key=2, layer_key=2, ph_h2o=6.0)])
    years = labels.set_index("site_id").year
    assert years["kssl:1"] == 1990 and pd.isna(years["kssl:2"])


def test_out_of_range_bad_depth_and_unsieved_rows_are_dropped():
    labels, drops = _build([
        _layer(layer_key=1, clay_total=-62.0, particle_size_method="3A1a1a"),
        _layer(layer_key=2, ph_h2o=6.0, hzn_top=10.0, hzn_bot=10.0),
        _layer(layer_key=3, ph_h2o=6.0, prep_code="GP"),
        _layer(layer_key=4, sand_total=40.0, particle_size_method="hyd1"),
    ])
    assert list(labels.layer_id) == ["kssl:4"]
    assert labels.method_family.item() == "hydrometer"
    assert drops["out_of_range"] == 1 and drops["bad_depth"] == 1


def test_output_matches_schema():
    labels, _ = _build([_layer(sand_total=40.0, clay_total=20.0, particle_size_method="3A1a1a",
                              bulk_density_third_bar=1.4, bulk_density_third_bar_method="4A1d")])
    assert list(labels.columns) == list(schema.COLUMNS)
    schema.validate(labels)
    assert set(labels.unit) == {"%", "g/cm3"}
