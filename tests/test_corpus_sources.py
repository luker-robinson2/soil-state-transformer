"""Small synthetic fixtures using each source's real column names."""
import numpy as np
import pandas as pd
import pytest

from geosoil.corpus import brazil, napeshm, npdb, ossl, raca, schema, usgs_ds801


def _check(labels):
    assert list(labels.columns) == list(schema.COLUMNS)
    schema.validate(labels)


def _val(labels, target, family=None):
    m = labels.target == target
    if family:
        m &= labels.method_family == family
    return labels[m].value


def test_brazil_converts_g_per_kg_texture_and_keeps_row_licence():
    df = pd.DataFrame(dict(
        dataset_id=["ctb1", "ctb2"], observacao_id=["p1", "p2"], camada_id=[1, 1], amostra_id=[1, 1],
        profund_sup=[0, 0], profund_inf=[20, 20], coord_x=[-50.0, np.nan], coord_y=[-15.0, np.nan],
        coord_precisao=[30.0, np.nan], data_ano=[1999, 2001], dataset_licenca=["CC BY 4.0", "CC BY-NC 4.0"],
        carbono=[12.0, 5.0], ph=[5.1, 5.5], areia=[430.0, 100.0], argila=[342.0, 200.0], dsi=[1.3, np.nan]))
    labels, drops = brazil.build(df)
    _check(labels)
    assert _val(labels, "sand").item() == pytest.approx(43.0)
    assert _val(labels, "clay").item() == pytest.approx(34.2)
    assert _val(labels, "soc").item() == 12.0                       # already g/kg
    assert set(labels.license) == {"CC BY 4.0"} and drops["no_coordinates"] > 0


def test_npdb_shifts_negative_organic_depths_and_converts_carbon():
    chem = pd.DataFrame(dict(PEDON_ID=[1, 1], LAYER_ID=[1, 2], U_DEPTH=["-10", "0"], L_DEPTH=["0", "20"],
                             CARB_ORG=["40.0", "2.5"], PH_H2O=[" ", "6.1"], PH_CACL2=["5.0", " "]))
    phys = pd.DataFrame(dict(PEDON_ID=[1], LAYER_ID=[2], U_DEPTH=["0"], L_DEPTH=["20"],
                             T_SAND=["40"], T_CLAY=["20"], BULK_DEN=[" "]))
    info = pd.DataFrame(dict(PEDON_ID=[1], DD_LAT_N=[52.0], DD_LONG_N=[-106.0], CAL_YEAR=[1985], CONF_METRS=["150"]))
    labels, _ = npdb.build(chem, phys, info)
    _check(labels)
    soc = labels[labels.target == "soc"].sort_values("depth_top")
    assert soc.depth_top.tolist() == [0.0, 10.0] and soc.value.tolist() == [400.0, 25.0]
    assert set(labels[labels.target == "ph"].method_family) == {"ph_h2o", "ph_cacl2"}
    assert labels.coord_precision_m.iloc[0] == 150.0


def test_ossl_drops_kssl_and_lucas_rows():
    site = pd.DataFrame({
        "dataset.code_ascii_txt": ["KSSL.SSL", "LUCAS.WOODWELL.SSL", "ICRAF.ISRIC"],
        "id.layer_uuid_txt": ["a", "b", "c"],
        "longitude.point_wgs84_dd": [-100.0, 10.0, 36.8], "latitude.point_wgs84_dd": [40.0, 50.0, -1.3],
        "layer.upper.depth_usda_cm": [0, 0, 0], "layer.lower.depth_usda_cm": [20, 20, 20],
        "observation.date.begin_iso.8601_yyyy.mm.dd": ["2011-01-01"] * 3,
        "location.point.error_any_m": [30, 30, 30],
        "dataset.license.title_ascii_txt": ["CC-BY", "JRC License Agreement", "CC-BY"]})
    lab = pd.DataFrame({"id.layer_uuid_txt": ["a", "b", "c"], "oc_usda.c729_w.pct": [1.0, 1.0, 0.59],
                        "ph.h2o_usda.a268_index": [6, 6, 5.8], "ph.cacl2_usda.a481_index": [np.nan] * 3,
                        "sand.tot_usda.c60_w.pct": [30, 30, 35], "clay.tot_usda.a334_w.pct": [20, 20, 31],
                        "bd_usda.a4_g.cm3": [np.nan] * 3})
    labels, _ = ossl.build(site, lab)
    _check(labels)
    assert labels.site_id.str.startswith("ossl:ICRAF.ISRIC").all()
    assert _val(labels, "soc").item() == pytest.approx(5.9)


def test_napeshm_uses_site_coordinates_and_dictionary_units():
    soil = pd.DataFrame(dict(siteid=[7, 7], euid=[1, 2], b_soc=[1.5, 2.0], b_ph=[6.1, 6.3],
                             b_sand=[30.0, 31.0], b_clay=[21.0, 22.0], bd_2mm=[1.2, np.nan]))
    sites = pd.DataFrame(dict(siteid=[7], site_latitude=[38.0], site_longitude=[-92.0], start_year=[1990]))
    labels, _ = napeshm.build(soil, sites)
    _check(labels)
    assert sorted(_val(labels, "soc")) == [15.0, 20.0]
    assert labels.site_id.nunique() == 2 and labels.lat.nunique() == 1
    assert (labels.depth_bot == 15.0).all()


def test_raca_subtracts_carbonate_and_uses_only_measured_bulk_density():
    samples = pd.DataFrame(dict(
        samp=["s1", "s2", "s3"], rcasiteid=["R1", "R1", "R1"], TOP=[0, 5, 20], BOT=[5, 20, 40],
        c_tot_ncs=[3.0, 2.0, 1.0], caco3=[10.0, np.nan, np.nan], efferv_1nhcl=["strong", "none", "slight"],
        Bulkdensity=[1.1, 1.2, 1.3], BDmeasured=["measured", "modeled", "measured"]))
    locations = pd.DataFrame(dict(RaCA_Id=["R1"], Gen_lat=[40.12], Gen_long=[-100.34]))
    labels, _ = raca.build(samples, locations)
    _check(labels)
    soc = labels[labels.target == "soc"].set_index("layer_id").value
    assert soc["raca:s1"] == pytest.approx((3.0 - 1.2) * 10)
    assert soc["raca:s2"] == pytest.approx(20.0)       # no carbonate, no effervescence
    assert "raca:s3" not in soc.index                  # effervescent, carbonate unknown
    assert sorted(_val(labels, "bd")) == [1.1, 1.3]
    assert (labels.coord_precision_m == 1000.0).all()


def test_usgs_parses_depth_strings_and_skips_units_row():
    df = pd.DataFrame(dict(A_LabID=[np.nan, "L1", "L2"], SiteID=[np.nan, "S1", "S2"],
                           Latitude=["Degrees", "31.3", "33.8"], Longitude=["Degrees", "-85.1", "-84.2"],
                           CollDate=[np.nan, "02/18/09", "02/26/09"], A_Depth=["cm", "2-30", "0-20"],
                           A_C_Org=["wt. %", "0.71", "<0.05"]))
    labels, _ = usgs_ds801.build(df)
    _check(labels)
    assert labels.layer_id.tolist() == ["usgs801:L1"]
    row = labels.iloc[0]
    assert (row.depth_top, row.depth_bot, row.value, row.year) == (2.0, 30.0, pytest.approx(7.1), 2009)


def _wosis(prop, values, datasets, opts, licences, upper=0, lower=20):
    n = len(values)
    return pd.DataFrame(dict(
        profile_id=list(range(1, n + 1)), layer_id=[10 * i for i in range(1, n + 1)],
        upper_depth=[upper] * n, lower_depth=[lower] * n, value_avg=values, method_options=opts,
        dataset_id=datasets, longitude=[5.0 + i for i in range(n)], latitude=[50.0] * n,
        positional_uncertainty=["Circa 100 m"] * n, date=["1995-06-01"] * n, licence=licences))


def test_wosis_drops_ncss_classifies_methods_and_keeps_licence():
    from geosoil.corpus import wosis
    tables = {
        "orgc": _wosis("orgc", [12.0, 30.0, 8.0], ["US-NCSS", "NL-X", "BR-Y"],
                       ["treatment = [ dry combustion ]", "treatment = [ wet oxidation, dichromate ]", None],
                       ["U.S. Public Domain", "CC BY 4.0", "Attribution-NonCommercial 3.0 (CC BY-NC 3.0)"]),
        "phaq": _wosis("phaq", [6.2], ["NL-X"], ["instrument = [ electrode ] ratio = [ 1:2.5 ]"], ["CC BY 4.0"]),
    }
    labels, drops = wosis.build(tables)
    _check(labels)
    assert drops["excluded_datasets"]["US-NCSS"] == 1
    soc = labels[labels.target == "soc"].set_index("site_id")
    assert "wosis:1" not in soc.index
    assert soc.loc["wosis:2", "method_family"] == "walkley_black"
    assert soc.loc["wosis:3", "method_family"] == "unknown"
    assert soc.loc["wosis:3", "license"].startswith("Attribution-NonCommercial")
    assert labels[labels.target == "ph"].method_id.item() == "wosis/phaq:1:2.5"
    assert (labels.coord_precision_m == 100.0).all()


def test_sislac_skips_republished_sources_and_fixes_units():
    from geosoil.corpus import sislac
    base = dict(country_code="ECU", perfil_duplicado="f", fecha="01/01/2014", license="CC-BY",
                ph=6.1, bulk_density=1.3, silt=30.0)
    rows = [
        dict(base, profile_id=1, layer_id=1, latitude=-1.0, longitude=-78.0, top=0, bottom=20,
             source="Harmonized soil database of Ecuador (HESD)", organic_carbon=2.0, clay=30.0, sand=40.0),
        dict(base, profile_id=2, layer_id=2, latitude=-2.0, longitude=-79.0, top=0, bottom=20,
             source="WoSIS July 2016 Snapshot", organic_carbon=2.0, clay=30.0, sand=40.0),
        dict(base, profile_id=3, layer_id=3, latitude=-3.0, longitude=-60.0, top=0, bottom=20,
             source="SISDELAV", organic_carbon=0.0, clay=0.0, sand=0.0, fecha="01/01/1900"),
        dict(base, profile_id=4, layer_id=4, latitude=-4.0, longitude=-61.0, top=0, bottom=20,
             source="ZONISIG", organic_carbon=1.0, clay=30.0, sand=40.0, perfil_duplicado="t"),
    ]
    labels, drops = sislac.build(pd.DataFrame(rows))
    assert labels["site_id"].nunique() == 2                       # WoSIS copy and duplicate flag skipped
    hesd = labels[labels["value"].notna() & labels["site_id"].str.endswith(":1")]
    assert hesd.loc[hesd["target"] == "soc", "value"].iloc[0] == pytest.approx(20.0)   # % -> g/kg
    zero = labels[labels["site_id"].str.endswith(":3")]
    assert set(zero["target"]) == {"ph", "bd"}                    # 0 = not measured for SOC and texture
