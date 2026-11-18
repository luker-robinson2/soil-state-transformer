"""LUCAS 2018 BD and RMQS1 loaders: synthetic rows with the real column names."""
import numpy as np
import pandas as pd
import pytest

from geosoil.corpus import lucas_bd2018, rmqs


def _vals(labels, target):
    return labels[labels["target"] == target].set_index("layer_id")


# --------------------------------------------------------------------------- LUCAS 2018 BD
def _lucas():
    return pd.DataFrame({
        "POINTID": [35582326, 46764408, 29901692, 51464814, 11111111],
        "Bdfine (g cm-3)": [1.19, 0.96, 1.17, 2.90, 0.80],
        "SOCS (kg cm-2)": [5.24, 4.35, 2.06, 6.33, 1.0],
        "coarse_vol": [0.14, 0.05, 0.08, 0.01, 0.0],
        "GPS_LAT": [43.609, 62.617, 37.127, 65.473, np.nan],
        "GPS_LONG": [0.544, 16.897, -4.963, 27.910, np.nan],
        # the real header has a double space and a trailing space
        "BDfine  method ": ["Measurement", "Local-Prediction", "PTF-4-Prediction", "Measurement", "Measurement"],
    })


def test_lucas_keeps_only_measured_bd_fine_earth_0_20():
    labels, drops = lucas_bd2018.build(_lucas())
    assert set(labels["site_id"]) == {"lucas:35582326"}           # predictions, out-of-range, no-coords dropped
    assert drops["ptf_predicted_rows"] == 2
    assert drops["out_of_range"] == 1 and drops["no_coordinates"] == 1
    row = labels.iloc[0]
    assert row["target"] == "bd" and row["unit"] == "g/cm3" and row["value"] == pytest.approx(1.19)
    assert row["method_family"] == "bd_fine_earth"
    assert row["method_id"] == "lucas2018/Bdfine_measured"
    assert (row["depth_top"], row["depth_bot"]) == (0, 20)
    assert row["year"] == 2018
    assert row["license"].startswith("CC-BY-4.0")
    assert np.isnan(row["coord_precision_m"])
    assert set(labels["target"]) == {"bd"}                        # SOCS is not a target


# --------------------------------------------------------------------------- RMQS1
RMQS_COLS = ("no_campagne;id_site;date_complete;code_dept;site_officiel;x_theo;y_theo;type_profil_rmqs;no_couche;"
             "profondeur_hz_sup;profondeur_hz_inf;teneur_eau_res_166_1;argile;limon_fin;limon_grossier;sable_fin;"
             "sable_grossier;cec_40_1;ca_ech_40_3;k_ech_40_3;mg_ech_40_3;na_ech_40_3;ph_eau_6_1;carbone_16_5_1;"
             "mat_org_0").split(";")


def _rmqs():
    rows = [
        # site 1820, layers 1 and 2; layer 9 is holorganic (negative depth) and is skipped
        ["1", "1820", "12/12/2006", "40", "True", "374439.21", "6368041.53", "C", "1", "0", "30", "3.52",
         "2", "1", "12", "91", "894", "1.85", "0.45", "0.062", "0.19", "0.057", "4.4", "18.00", "31.2"],
        ["1", "1820", "12/12/2006", "40", "True", "374439.21", "6368041.53", "C", "2", "30", "50", "0.5",
         "8", "3", "6", "68", "915", "0.500", "0.05", "0.030", "0.02", "0.021", "5.0", "2.1", "ND"],
        ["1", "1820", "12/12/2006", "40", "True", "374439.21", "6368041.53", "C", "9", "-3", "0", "ND",
         "ND", "ND", "ND", "ND", "ND", "40.0", "20.0", "1.0", "3.0", "ND", "5.5", "300", "ND"],
        # site 77: decimal commas, ND for pH and CEC
        ["1", "77", "05/06/2002", "51", "False", "780000,5", "6880000,25", "C", "1", "0", "25", "2,0",
         "300", "250", "150", "100", "200", "ND", "15,5", "0,41", "1,2", "0,1", "ND", "12,5", "21,5"],
    ]
    return pd.DataFrame(rows, columns=RMQS_COLS)


def test_rmqs_units_families_depths_and_coordinates():
    labels, drops = rmqs.build(_rmqs())
    assert drops["holorganic_layer_rows"] == 1
    assert set(labels["layer_id"]) == {"rmqs:1820/1", "rmqs:1820/2", "rmqs:77/1"}
    layers = labels.drop_duplicates("layer_id").set_index("layer_id")
    assert tuple(layers.loc["rmqs:1820/2", ["depth_top", "depth_bot"]]) == (30, 50)
    assert tuple(layers.loc["rmqs:77/1", ["depth_top", "depth_bot"]]) == (0, 25)

    clay, sand = _vals(labels, "clay"), _vals(labels, "sand")
    assert clay.loc["rmqs:1820/1", "value"] == pytest.approx(0.2)            # g/kg → %
    assert sand.loc["rmqs:1820/1", "value"] == pytest.approx(98.5)           # (91 + 894) / 10: 50–2000 µm
    assert sand.loc["rmqs:77/1", "value"] == pytest.approx(30.0)
    assert set(clay["method_family"]) == set(sand["method_family"]) == {"pipette"}

    soc = _vals(labels, "soc")
    assert soc.loc["rmqs:77/1", "value"] == pytest.approx(12.5)              # g/kg already, decimal comma
    assert set(soc["method_family"]) == {"dry_combustion"}
    assert "om" not in set(labels["target"])                                  # derived OM not used

    ph = _vals(labels, "ph")
    assert list(ph.index) == ["rmqs:1820/1", "rmqs:1820/2"] and set(ph["method_family"]) == {"ph_h2o"}

    cec = _vals(labels, "cec")
    assert "rmqs:77/1" not in cec.index                                       # ND
    assert cec.loc["rmqs:1820/1", "value"] == pytest.approx(1.85)
    assert set(cec["method_family"]) == {"cec_ph7"}
    for t, v in (("ca", 15.5), ("k", 0.41), ("mg", 1.2)):
        b = _vals(labels, t)
        assert b.loc["rmqs:77/1", "value"] == pytest.approx(v)
        assert set(b["method_family"]) == {"nh4oac_ph7"}
        assert set(b["unit"]) == {"cmol(+)/kg"}
    assert "bd" not in set(labels["target"])

    sites = labels.drop_duplicates("site_id").set_index("site_id")
    assert sites.loc["rmqs:1820", "lon"] == pytest.approx(-1.085, abs=0.01)   # Landes, Lambert-93 → WGS84
    assert sites.loc["rmqs:1820", "lat"] == pytest.approx(44.336, abs=0.01)
    assert (labels["coord_precision_m"] == rmqs.COORD_PRECISION_M).all()
    assert sites.loc["rmqs:1820", "year"] == 2006 and sites.loc["rmqs:77", "year"] == 2002
    assert labels["license"].str.startswith("Etalab-2.0").all()


@pytest.mark.parametrize("sep", [";", ",", "\t"])
def test_rmqs_read_raw_sniffs_separator_and_encoding(tmp_path, sep):
    df = _rmqs().iloc[:2].copy()
    df["code_dept"] = "Hérault"                                               # non-ASCII in cp1252
    p = tmp_path / "rmqs.csv"
    df.to_csv(p, sep=sep, index=False, encoding="cp1252")
    back = rmqs.read_raw(p)
    assert list(back.columns) == RMQS_COLS
    assert back["code_dept"].iloc[0] == "Hérault"
    assert back["mat_org_0"].isna().iloc[1]                                   # "ND" read as missing
    labels, _ = rmqs.build(back)
    assert set(labels["layer_id"]) == {"rmqs:1820/1", "rmqs:1820/2"}
