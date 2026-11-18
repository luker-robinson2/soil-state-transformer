"""FIA, NWCA, ChSPD and ICRAF AfSIS loaders: synthetic rows with the real column names."""
import numpy as np
import pandas as pd
import pytest

from geosoil.corpus import chspd, fia, icraf_afsis, nwca


def _vals(labels, target, method_id=None):
    t = labels[labels["target"] == target]
    if method_id is not None:
        t = t[t["method_id"] == method_id]
    return t


# --------------------------------------------------------------------------- FIA
def _fia_lab():
    rows = [
        # CN, PLT_CN, LAYER_TYPE, QASTATCD, BD, C_ORG_PCT, PH_H2O, PH_CACL2, K, MG, CA, ECEC, MEASYEAR, STATECD
        (11, 100, "MIN_1", 1.0, 1.2, 2.5, 5.5, 5.0, 391.0, 121.5, 2004.0, 12.0, 2010, 27),
        (12, 100, "MIN_2", 1.0, 1.4, 1.0, 5.8, 5.2, 195.5, 243.0, 400.8, 5.0, 2010, 27),
        (13, 100, "FF_TOTAL", 1.0, 0.3, 30.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2010, 27),
        (14, 100, "MIN_1", 7.0, 1.1, 2.0, 5.0, 4.5, 100.0, 100.0, 100.0, 1.0, 2010, 27),   # QA remeasure
        (15, 200, "MIN_1", 1.0, 8.8, np.nan, 6.0, 5.5, np.nan, np.nan, np.nan, np.nan, 2015, 27),  # BD out of range
        (16, 300, "MIN_1", 1.0, 1.0, 1.0, 6.0, 5.5, 10.0, 10.0, 10.0, 1.0, 2015, 27),     # no coordinates
    ]
    cols = ["CN", "PLT_CN", "LAYER_TYPE", "QASTATCD", "BULK_DENSITY", "C_ORG_PCT", "PH_H2O", "PH_CACL2",
            "EXCHNG_K", "EXCHNG_MG", "EXCHNG_CA", "ECEC", "MEASYEAR", "STATECD"]
    return pd.DataFrame(rows, columns=cols)


def _fia_plots():
    return pd.DataFrame({"CN": [100, 200], "LAT": [45.1, 46.2], "LON": [-93.1, -94.2]})


def test_fia_layers_units_and_qa():
    labels, drops = fia.build(_fia_lab(), _fia_plots())
    assert set(labels["layer_id"]) <= {"fia:11", "fia:12", "fia:15"}          # no forest floor, no QA rows
    k = _vals(labels, "k").set_index("layer_id")["value"]
    assert k["fia:11"] == pytest.approx(1.0) and k["fia:12"] == pytest.approx(0.5)
    assert _vals(labels, "ca").set_index("layer_id").loc["fia:11", "value"] == pytest.approx(10.0)
    assert _vals(labels, "mg").set_index("layer_id").loc["fia:12", "value"] == pytest.approx(2.0)
    assert set(_vals(labels, "k")["method_family"]) == {"unknown"}
    assert set(_vals(labels, "k")["method_id"]) == {"fia/nh4cl"}
    assert "cec" not in set(labels["target"])                                  # ECEC is a sum: skipped
    soc = _vals(labels, "soc").set_index("layer_id")["value"]
    assert soc["fia:11"] == pytest.approx(25.0)
    assert set(_vals(labels, "ph")["method_family"]) == {"ph_h2o", "ph_cacl2"}
    l2 = labels[labels["layer_id"] == "fia:12"].iloc[0]
    assert (l2["depth_top"], l2["depth_bot"]) == (pytest.approx(10.16), pytest.approx(20.32))
    assert (labels["coord_precision_m"] == 1600).all()
    assert drops["out_of_range"] == 1 and drops["no_coordinates"] > 0
    assert labels["year"].dropna().isin([2010, 2015]).all()


# --------------------------------------------------------------------------- NWCA
def _nwca11():
    chem = pd.DataFrame({
        "UID": [1, 1, 1], "PIT": ["A", "A", "A"], "LAYER": [1, 3, 4], "DEPTH": [10, 40, 60],
        "DATE_COL": ["5/3/2011"] * 3,
        "CLAY": [20.0, 25.0, np.nan], "CLAY_FLAG": [None, None, "Analysis could not be done or was not relevant to the sample"],
        "SAND": [50.0, 45.0, np.nan], "SAND_FLAG": [None] * 3,
        "TOT_CARBON": [2.0, 1.0, 0.5], "TOT_CARBON_FLAG": [None, None, "< PQL 0.2 %"],
        "CACO3_2MM": [np.nan, 5.0, np.nan], "CACO3_2MM_FLAG": [None] * 3,
        "PH_H2O": [5.0, 6.0, 7.0], "PH_H2O_FLAG": [None] * 3,
        "PH_CACL2": [4.5, 5.5, 6.5], "PH_CACL2_FLAG": [None] * 3,
        "CEC": [10.0, 8.0, 6.0], "CEC_FLAG": [None] * 3,
        "CA": [5.0, 4.0, 3.0], "CA_FLAG": [None] * 3,
        "K": [0.3, 0.0, 0.2], "K_FLAG": [None, "< MDL 0.06 cmol(+) / kg", None],
        "MG": [1.0, 1.0, 1.0], "MG_FLAG": [None] * 3,
        "BULK_DEN_DBF": [1.1, 1.3, 1.5], "BULK_DEN_DBF_FLAG": [None, None, "Db inconsistent with layer position, g/cc"],
    })
    hor = pd.DataFrame({"UID": [1] * 4, "PIT": ["A"] * 4, "HORIZON": [1, 2, 3, 4], "DEPTH": [10, 25, 40, 60]})
    site = pd.DataFrame({"UID": [1], "LAT_DD83": [30.0], "LON_DD83": [-88.0]})
    return chem, hor, site


def _nwca16():
    chem = pd.DataFrame({
        "UID": [9, 9], "HORIZON": [1, 2], "HORIZON_DEPTH": [5, 30], "DATE_COL": ["4/17/2016"] * 2,
        "CLAY_PSDA": [30.0, np.nan], "CLAY_PSDA_NARS_FLAG": [None, "NA1"],
        "SAND_PSDA": [40.0, np.nan], "SAND_PSDA_NARS_FLAG": [None, "NA1"],
        "C_TOT_ELEM_ANLYS": [3.0, 25.0], "C_TOT_ELEM_ANLYS_NARS_FLAG": [None, None],
        "CACO3_WETOX_2MM": [np.nan, np.nan], "CACO3_WETOX_2MM_NARS_FLAG": ["NA2", "NA2"],
        "PH_H2O_SOLN": [6.0, np.nan], "PH_H2O_SOLN_NARS_FLAG": [None, "N"],
        "PH_CACL2_SOLN": [5.5, np.nan], "PH_CACL2_SOLN_NARS_FLAG": [None, "N"],
        "CEC_NH4OAC_EXTR": [12.0, 90.0], "CEC_NH4OAC_EXTR_NARS_FLAG": [None, None],
        "CA_NH4OAC_EXTR": [6.0, 20.0], "CA_NH4OAC_EXTR_NARS_FLAG": [None, None],
        "K_NH4OAC_EXTR": [0.1, np.nan], "K_NH4OAC_EXTR_NARS_FLAG": ["L", "ND"],
        "MG_NH4OAC_EXTR": [2.0, 5.0], "MG_NH4OAC_EXTR_NARS_FLAG": [None, None],
    })
    hor = pd.DataFrame({"UID": [9, 9], "HORIZON": [1.0, 2.0], "HORIZON_DEPTH": [5, 30]})
    site = pd.DataFrame({"UID": [9], "LAT_DD83": [40.0], "LON_DD83": [-100.0]})
    return chem, hor, site


def test_nwca_depths_from_lower_boundaries_and_conversions():
    c11, h11, s11 = _nwca11()
    c16, h16, s16 = _nwca16()
    labels, _ = nwca.build(c11, h11, s11, c16, h16, s16)
    layers = labels.drop_duplicates("layer_id").set_index("layer_id")
    assert tuple(layers.loc["nwca:2011/1/A/1", ["depth_top", "depth_bot"]]) == (0, 10)
    # layer 3 sits below the unsampled layer 2: its top comes from the horizon file, not the chemistry rows
    assert tuple(layers.loc["nwca:2011/1/A/3", ["depth_top", "depth_bot"]]) == (25, 40)
    assert tuple(layers.loc["nwca:2016/9/2", ["depth_top", "depth_bot"]]) == (5, 30)
    soc = _vals(labels, "soc").set_index("layer_id")["value"]
    assert soc["nwca:2011/1/A/1"] == pytest.approx(20.0)
    assert soc["nwca:2011/1/A/3"] == pytest.approx(10 * (1.0 - 0.12 * 5.0))      # TC − 0.12 CaCO3
    assert soc["nwca:2016/9/2"] == pytest.approx(250.0)
    assert set(_vals(labels, "soc")["method_id"]) == {"nwca/tc_minus_caco3"}
    k = _vals(labels, "k").set_index("layer_id")["value"]
    assert "nwca:2011/1/A/3" not in k.index                                     # < MDL written as 0: dropped
    assert k["nwca:2016/9/1"] == pytest.approx(0.1)                             # "L" kept
    assert "nwca:2011/1/A/4" not in set(_vals(labels, "clay")["layer_id"])       # could not be done
    assert "nwca:2011/1/A/4" not in set(_vals(labels, "bd")["layer_id"])         # inconsistent BD flag
    assert set(_vals(labels, "cec")["method_family"]) == {"cec_ph7"}
    assert set(_vals(labels, "ca")["method_family"]) == {"nh4oac_ph7"}
    assert set(_vals(labels, "bd")["method_family"]) == {"unknown"}
    assert set(labels.loc[labels["site_id"].str.startswith("nwca:2016"), "year"]) == {2016}


# --------------------------------------------------------------------------- ChSPD
def _chspd():
    return pd.DataFrame({
        "ID": [1, 1, 2, 3, 4], "source": ["CIREN", "CIREN", "CHLSOC", "UChile", "WoSIS"],
        "x": [-71.12345, -71.12345, -71.5, -70.12, -72.0], "y": [-33.54321, -33.54321, -33.7, -34.56, -35.0],
        "top": [0.0, 20.0, 0.0, 0.0, 0.0], "bottom": [20.0, 50.0, 10.0, 15.0, 10.0],
        "clay": [30.0, 35.0, 20.0, 40.0, 10.0], "silt": [30.0, 30.0, 30.0, 10.0, 30.0],
        "sand": [40.0, 35.0, 50.0, 10.0, 60.0],              # UChile row sums to 60: texture dropped
        "bulk_density": [1.3, 1.4, 1.2, 1.1, 1.0], "bd_method": ["Clod", "Clod", None, "Cylinder", None],
        "organic_carbon": [20.0, 10.0, 2.0, 30.0, 1.0], "pH": [6.0, 6.5, 7.0, 5.0, 6.0],
        "year": [np.nan, np.nan, np.nan, 2010.0, np.nan],
    })


def test_chspd_skips_wosis_sources_soc_ph_and_maps_bd():
    labels, _ = chspd.build(_chspd())
    assert set(labels["site_id"]) == {"chspd:1", "chspd:3"}
    assert set(labels["target"]) == {"clay", "sand", "bd"}
    bd = _vals(labels, "bd").set_index("site_id")
    assert set(bd.loc["chspd:1", "method_family"]) == {"unknown"}
    assert bd.loc["chspd:3", "method_family"] == "bd_oven_dry"
    assert "chspd:3" not in set(_vals(labels, "clay")["site_id"])
    prec = labels.drop_duplicates("site_id").set_index("site_id")["coord_precision_m"]
    assert prec["chspd:1"] == pytest.approx(1.1132) and prec["chspd:3"] == pytest.approx(1113.2)
    assert labels["license"].str.startswith("CC-BY-4.0").all()


# --------------------------------------------------------------------------- ICRAF AfSIS
def _icraf():
    return pd.DataFrame({
        "SSN": ["a1", "a2", "b1"], "Country": ["Tanzania"] * 2 + ["Mali"], "Site": ["S"] * 3,
        "Cluster": [1, 1, 2], "Plot": [4, 4, 1], "Latitude": [-6.9, -6.9, np.nan], "Longitude": [36.1, 36.1, np.nan],
        "Depth": ["Topsoil", "Subsoil", "Topsoil"], "pH": [7.3, 7.0, 6.0],
        "Percent_Org_C": [0.66, 0.4, 1.0], "Caex": [5.58, 4.0, 1.0], "Kex": [0.39, 0.2, 0.1],
        "Mgex": [1.7, 1.5, 0.5], "eCEC": [2.8, -1.0, 2.0],
    })


def test_icraf_cohex_families_depths_and_no_ph():
    labels, drops = icraf_afsis.build(_icraf())
    assert set(labels["site_id"]) == {"icraf:Tanzania/S/1/4"}
    assert "ph" not in set(labels["target"])
    layers = labels.drop_duplicates("layer_id").set_index("layer_id")
    assert tuple(layers.loc["icraf:a2", ["depth_top", "depth_bot"]]) == (20, 50)
    assert set(_vals(labels, "k")["method_id"]) == {"icraf/cohex"}
    assert set(labels.loc[labels["target"].isin(["k", "ca", "mg"]), "method_family"]) == {"unknown"}
    cec = _vals(labels, "cec")
    assert list(cec["layer_id"]) == ["icraf:a1"] and set(cec["method_family"]) == {"ecec"}
    assert _vals(labels, "soc").set_index("layer_id").loc["icraf:a1", "value"] == pytest.approx(6.6)
    assert drops["no_coordinates"] > 0
