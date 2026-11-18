"""Synthetic-fixture tests for the qld_sali, sodes, bze_lw and nspd label loaders (real column names)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geosoil.corpus import bze_lw, nspd, qld_sali, sodes


def _vals(labels, target, method_id=None):
    d = labels[labels["target"] == target]
    if method_id is not None:
        d = d[d["method_id"] == method_id]
    return d


# ---------------------------------------------------------------- Queensland SALI
def _sali_samples():
    return pd.DataFrame({
        "projectCode": ["ABC", "ABC", "ABC", "XYZ"], "siteId": [1, 1, 1, 7], "observationNumber": [1, 1, 1, 1],
        "sampleNumber": [31, 33, 40, 2], "upperDepth": [0.0, 0.2, 3.2, 0.0], "lowerDepth": [0.1, 0.3, 3.5, 0.1],
        "longitude": [151.88, 151.88, 151.88, 145.0], "latitude": [-25.63, -25.63, -25.63, -20.0],
    })


def _r(sample, code, value, valid="Y", status="A", site=1, proj="ABC", date="1980-01-01T00:00:00"):
    return dict(siteId=site, projectCode=proj, observationNumber=1, sampleNumber=sample, labMethodCode=code,
                numericValue=str(value), validFlag=valid, resultStatus=status, analysisDate=date)


def test_sali_units_methods_and_filters():
    res = pd.DataFrame([
        _r(31, "2Z2_Clay", 40), _r(31, "2Z2_CS", 10), _r(31, "2Z2_FS", 25),
        _r(31, "6A1", 1.5), _r(31, "6B2", 1.7), _r(31, "4A1", 6.5), _r(31, "4B1", 5.8),
        _r(31, "15C1_CEC", 24), _r(31, "15C1_K", 0.3), _r(31, "15A1_K", 0.25), _r(31, "503.01a", 1.5),
        _r(33, "2Z2_Clay", 45, valid="N"),            # invalid flag -> dropped
        _r(33, "6A1", 0.1, status="BQ"),             # below quantification -> dropped
        _r(33, "2Z2_CS", 12),                         # FS missing -> no sand
        _r(40, "2Z2_Clay", 50),                       # top 320 cm -> dropped
        _r(2, "4A1", 7.0, site=7, proj="XYZ", date="0074-11-01T00:00:00"),
    ])
    lab, drops = qld_sali.build(_sali_samples(), res)
    assert drops["invalid_flag"] == 1 and drops["below_quantification"] == 1 and drops["deeper_than_300cm"] == 1
    l31 = lab[lab["layer_id"] == "qld:ABC/1/1/31"]
    assert l31[["depth_top", "depth_bot"]].iloc[0].tolist() == [0.0, 10.0]          # metres -> cm
    soc = _vals(l31, "soc").set_index("method_id")
    assert soc.loc["sali/6A1", "value"] == pytest.approx(15.0) and soc.loc["sali/6A1", "method_family"] == "walkley_black"
    assert soc.loc["sali/6B2", "method_family"] == "dry_combustion"
    sand = _vals(lab, "sand")
    assert len(sand) == 1 and sand["value"].iloc[0] == 35 and sand["method_id"].iloc[0] == "sali/2Z2_CS+FS_isss"
    assert sand["method_family"].iloc[0] == "sand_isss"            # ISSS limits: kept in the corpus, dropped by harmonize
    fam = dict(zip(l31["method_id"], l31["method_family"]))
    assert fam["sali/4A1"] == "ph_h2o" and fam["sali/4B1"] == "ph_cacl2" and fam["sali/15C1_CEC"] == "cec_ph7"
    assert fam["sali/15C1_K"] == "nh4oac_ph7" and fam["sali/15A1_K"] == "unknown" and fam["sali/503.01a"] == "bd_oven_dry"
    assert len(_vals(lab, "clay")) == 1                                            # invalid + deep clay dropped
    assert lab.loc[lab["site_id"] == "qld:XYZ/7/1", "year"].iloc[0] == 1974       # 0074 -> 1974
    assert (lab["license"].str.startswith("CC-BY-4.0")).all()


# ---------------------------------------------------------------- SODES
def test_sodes_crs_sand_soc_cec_and_open_horizons():
    prof = pd.DataFrame({"PROF_CODE": ["A001", "C010"], "X_UTM_ETRS": [747230.139, 2747.6],
                         "Y_UTM_ETRS": [4303855.596, 4776000.0]})
    hor = pd.DataFrame({
        "PROF_CODE": ["A001", "A001", "C010"], "HOR_CODE": ["A00101", "A00102", "C01001"],
        "M_DEPTH_UPP": [0.0, 30.0, 0.0], "M_THICKNESS": [30.0, 0.0, 20.0],
        "A_PH": [7.8, 7.9, 5.5], "A_CLAY": [18.5, 3.0, 30.0], "A_SAND_TOT": [38.2, 49.2, 90.0],
        "A_COARSE_SAND": ["20.4", "", "10"], "A_FINE_SAND": ["44.9", "", "15"],
        "A_OM": [1.724, 1.2, 3.448], "A_CEC": [30.9, 7.1, 15.0], "A_SUM_BASES_S": [30.9, 7.1, 9.0],
        "A_EXCH_CA": [28.6, 6.4, 6.0], "A_EXCH_MG": [1.6, 0.4, 2.0], "A_EXCH_K": [0.5, 0.1, 0.3],
        "A_BDENSITY": [1.2, 1.5, 1.3],
    })
    lab, drops = sodes.build(prof, hor)
    assert drops["bad_depth"] > 0 and not (lab["layer_id"] == "sodes:A00102").any()   # thickness 0 dropped
    a = lab[lab["site_id"] == "sodes:A001"].iloc[0]
    assert a["lon"] == pytest.approx(-0.151, abs=0.005) and a["lat"] == pytest.approx(38.849, abs=0.005)  # Adsubia, Alicante
    c = lab[lab["site_id"] == "sodes:C010"].iloc[0]
    assert c["lon"] < -8.5                                                            # Galicia via extended zone 30
    assert _vals(lab, "sand", "sodes/A_COARSE_SAND+A_FINE_SAND")["value"].sort_values().tolist() == pytest.approx([25.0, 65.3])
    assert _vals(lab, "soc")["value"].sort_values().tolist() == pytest.approx([10.0, 20.0])
    assert set(_vals(lab, "ph")["method_family"]) == {"ph_h2o"}
    cec = dict(zip(_vals(lab, "cec")["layer_id"], _vals(lab, "cec")["method_id"]))
    assert cec == {"sodes:A00101": "sodes/A_CEC_eq_sum_bases", "sodes:C01001": "sodes/A_CEC"}
    assert "bd" not in set(lab["target"])
    assert (lab["coord_precision_m"] == 100).all()


# ---------------------------------------------------------------- BZE-LW (GSBDDB)
def test_bze_lw_filters_source_and_reads_latin1(tmp_path):
    df = pd.DataFrame({
        "ProfileID": [2, 2, 99], "Source": ["BZE_LW", "BZE_LW", "FIADB"], "SiteName": ["Lüneburg", "Lüneburg", "x"],
        "SamplingYear": ["2015/11/1", "2015/11/1", "2001"], "Longitude": [8.41, 8.41, -100.0],
        "Latitude": [54.86, 54.86, 40.0], "UpperDepth(cm)": [0, 10, 0], "LowerDepth(cm)": [10, 30, 20],
        "Clay(%)": [7.6, np.nan, 20], "Silt(%)": [15.18, np.nan, 40], "Sand(%)": [77.22, np.nan, 40],
        "SOC(g/kg)": [21.83, 11.5, 5], "BD(g/cm3)": [1.16, 1.45, 1.3],
    })
    p = tmp_path / "g.csv"
    df.to_csv(p, index=False, encoding="latin-1")
    lab, drops = bze_lw.build(pd.read_csv(p, encoding="latin-1"))
    assert set(lab["site_id"]) == {"bze:2"} and drops["other_gsbddb_rows"] == 1
    assert _vals(lab, "soc")["method_family"].eq("dry_combustion").all()
    assert _vals(lab, "soc")["value"].max() == pytest.approx(21.83)                  # already g/kg
    assert len(_vals(lab, "clay")) == 1 and set(lab["year"]) == {2015}
    assert lab["coord_precision_m"].isna().all()


# ---------------------------------------------------------------- NSPD2025
def _nspd_hor():
    return pd.DataFrame({
        "PRID": ["N1", "N1", "S2", "L3", "T4"], "HONU": [1, 2, 1, 1, 1],
        "Dataset": ["FEB", "FEB", "OKA NSS", "LandPKS", "TWO NSS"],
        "Latitude": ["-22.5\xa0", "-22.5\xa0", "-19.0", "-20.0", "-21.0"],
        "Longitude": ["17.1", "17.1", "18.2\xa0", "16.0", "15.0"],
        "Upper Depth cm [contains >]": ["0", "20", "0", "0", "0"], "Lower Depth cm [contains >]": ["20", ">60", "15", "10", "10"],
        "Upper Depth cm ": ["0", "20", "0", "0", "0"], "Lower Depth cm ": ["20", "60\xa0", "15", "10", "10"],
        "Sand %\n< 53 µm \n(SDTO)": ["84", "80", "90", "50", "70"],
        "Silt % \n[2-53µm]\n(STPC)": ["6", "8", np.nan, "20", "10"],
        "Clay % \n[< 2 µm] \n(CLPC)": ["10", "12", np.nan, "30", "20"],
        "Bulk Density kg/dm3": ["1.5", np.nan, np.nan, np.nan, np.nan],
        "pH CaCl2 (PHCA) [2 soil:5 CaCl2 suspension]": ["5.8", np.nan, np.nan, np.nan, np.nan],
        "pH water (PHAQ)": ["7.0", "7.2", "6.0", "6.0", "8.0"],
        "Organic Carbon % WB": [0.23, 0.1, 0.5, 0.5, 0.2],
        "Organic Carbon g/kg WB": ["2.3", np.nan, "5.0", "5.0", "2.0"],
        "Organic Carbon % LOI": [9.9, 9.9, 9.9, 9.9, 9.9],
        "Exchangeable Calcium\\\ncmol(+)/kg \n(EXCA)": ["1.7", "1.0", "2.0", "1.0", "40.0"],
        "Exchangeable Magnesium\ncmol(+)/kg \n(EXMG)": ["0.6", "0.5", "0.5", "0.5", "5.0"],
        "Exchangeable Potassium\ncmol(+)/kg \n(EXCK)": ["0.21", "0.2", "0.3", "0.2", "1.0"],
        "CEC Effective \ncmol(+)/kg\n[Sum of Exchangeable Bases]": ["2.5", "1.7", "2.8", "1.7", "46"],
        "CEC Measured \ncmol(+)/kg \n": ["3.0", np.nan, "4.0", np.nan, "5.0"],
    })


def _nspd_site():
    return pd.DataFrame({"PRID": ["N1", "S2", "S2", "L3", "T4"], "Year": [2000, 1995, 1995, 2020, np.nan],
                         "Coordinate Reference System": ["Geographic, WGS87", "Geographic, Schwarzeck",
                                                         "Geographic, Schwarzeck", "Geographic, WGS84",
                                                         "Geographic, WGS84"]})


def test_nspd_nbsp_schwarzeck_raw_and_column_choices():
    lab, drops = nspd.build(_nspd_hor(), _nspd_site())
    assert drops["skipped_dataset_rows"] == 1 and not lab["site_id"].eq("nspd:L3").any()
    s2 = lab[lab["site_id"] == "nspd:S2"].iloc[0]
    assert s2["lon"] == pytest.approx(18.2) and s2["lat"] == pytest.approx(-19.0)     # NBSP stripped, not shifted
    assert s2["coord_precision_m"] == 150 and "Schwarzeck" in s2["coord_source"] and s2["year"] == 1995
    n1 = lab[lab["site_id"] == "nspd:N1"]
    assert np.isnan(n1["coord_precision_m"].iloc[0]) and n1["lat"].iloc[0] == pytest.approx(-22.5)
    assert n1["depth_bot"].max() == 60                                                # NBSP depth parsed
    soc = _vals(n1, "soc").set_index("layer_id")["value"]
    assert soc.tolist() == pytest.approx([2.3, 1.0])                                  # g/kg, else % x 10; not LOI
    assert set(_vals(lab, "soc")["method_family"]) == {"walkley_black"}
    cec = _vals(lab, "cec")
    assert cec["value"].sort_values().tolist() == [3.0, 4.0, 5.0]                     # Measured only, not Effective
    assert set(_vals(lab, "ca")["site_id"]) == {"nspd:N1"}                            # bases only from trusted datasets
    assert set(_vals(lab, "sand")["method_id"]) == {"nspd/SDTO_gt53um"}
    ph = _vals(lab, "ph")
    assert set(ph["method_family"]) == {"ph_h2o", "ph_cacl2"}


def test_nspd_column_regex_is_unique():
    with pytest.raises(KeyError):
        nspd.col(pd.DataFrame(columns=["Clay %", "Clay % 2"]), r"^Clay %")
