"""Karnataka soil health loader: synthetic rows with the real column names and a toy village gazetteer."""
import numpy as np
import pandas as pd
import pytest

from geosoil.corpus import icrisat_karnataka as ik

COLS = ["card_no", "farmer_number", "sau", "state", "district", "taluk", "village", "farmer_name",
        "survey_number", "soil_type", "authority", "ph", "ec", "oc", "av_p", "av_k", "av_s", "av_zn", "av_b",
        "av_fe", "av_cu", "av_mn"]


def _square(lon, lat, half_deg):
    return {"type": "Polygon", "coordinates": [[[lon - half_deg, lat - half_deg], [lon + half_deg, lat - half_deg],
                                                [lon + half_deg, lat + half_deg], [lon - half_deg, lat + half_deg],
                                                [lon - half_deg, lat - half_deg]]]}


def _gazetteer():
    feats = [
        ("1", "Aladakatti", "Badami", "Bagalkot", 75.70, 16.00, 0.01),
        ("2", "Kerakalamatti", "Badami", "Bagalkot", 75.80, 16.05, 0.01),
        ("3", "Kalalghatta", "Nelamagala", "Bangalore Rural", 77.40, 13.10, 0.005),
        ("4", "Hosur", "Bellary", "Bellary", 76.90, 15.10, 0.01),      # two Hosur in one taluk: ambiguous
        ("5", "Hosur", "Bellary", "Bellary", 76.95, 15.20, 0.01),
        ("6", "Big Village", "Bidar", "Bidar", 77.50, 17.90, 0.10),    # large polygon: precision above floor
    ]
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": _square(lon, lat, h),
         "properties": {"NAME": n, "LOC_CODE": code, "TALUK": t, "DISTRICT": d}}
        for code, n, t, d, lon, lat, h in feats]}


def _row(card, district, taluk, village, authority, ph, oc, av_k, state="Karnataka"):
    r = dict.fromkeys(COLS, "")
    r.update(card_no=card, state=state, district=district, taluk=taluk, village=village,
             farmer_name="SHOULD NOT BE USED", authority=authority, ph=ph, oc=oc, av_k=av_k)
    return r


def _samples():
    rows = [
        # exact match, central lab: three samples, medians pH 8.0, OC 0.5 % → 5 g/kg, K 39.1 mg/kg → 0.1
        _row("K1", "Bagalkot", "Badami", "Aladakatti", "ICR", "8.2", "0.59", "41"),
        _row("K2", "Bagalkot", "Badami", "Aladakatti", "ICR", "8.0", "0.50", "39.1"),
        _row("K3", "Bagalkot", "Badami", "Aladakatti", "ICR", "7.0", "0.44", "20"),
        # 10-char truncated name, fuzzy against "Kerakalamatti"; state lab: K skipped
        _row("K4", "Bagalkot", "Badami", "Kerkalamat", "DOA", "6.8", "0.62", "31"),
        # new district name + renamed taluk alias; truncated name is a prefix of the gazetteer key
        _row("K5", "Bengalur", "Nelama", "Kalalghatt", "DOA", "5.0", "1.10", "80"),
        # ambiguous village, unknown village, malformed row
        _row("K6", "Ballari", "Ballar", "Hosur", "DOA", "7.0", "0.4", "50"),
        _row("K7", "Bidar", "Bidar", "Nowhereville", "ICR", "7.0", "0.4", "50"),
        _row("K8", "0.22", "0.79", "30.3", "0.25", "", "2.54", "", state="7.5"),
        # big polygon; out-of-range pH sample removed before the median
        _row("K9", "Bidar", "Bidar", "Big Village", "ICR", "12.5", "0.3", "117.3"),
        _row("K10", "Bidar", "Bidar", "Big Village", "ICR", "6.0", "0.3", "117.3"),
    ]
    return pd.DataFrame(rows, columns=COLS)


@pytest.fixture(scope="module")
def built():
    gaz = ik.villages(_gazetteer())
    return ik.build_with_villages(_samples(), gaz)


def test_name_key_and_polygon_centroid():
    assert ik.name_key("Kalalghatta") == ik.name_key("Kalalgatta")
    assert ik.name_key("Hadnoor") == ik.name_key("Hadnur")
    lon, lat, rmax = ik.polygon_centroid(_square(75.7, 16.0, 0.01))
    assert lon == pytest.approx(75.7) and lat == pytest.approx(16.0)
    assert rmax == pytest.approx(np.hypot(0.01 * 111_320 * np.cos(np.radians(16.0)), 0.01 * 110_574), rel=1e-3)


def test_matching_kinds(built):
    _, drops, villages = built
    assert drops["malformed_rows"] == 1
    assert drops["villages_exact"] == 3            # Aladakatti, Kalalghatt (prefix), Big Village
    assert drops["villages_fuzzy"] == 1            # Kerkalamat
    assert drops["villages_ambiguous"] == 1 and drops["villages_none"] == 1
    assert drops["samples_unplaced"] == 2
    assert set(villages["site_key"]) == {"1", "2", "3", "6"}
    assert villages.set_index("site_key").loc["1", "n_samples"] == 3


def test_village_medians_units_and_families(built):
    labels, _, _ = built
    v = labels.set_index(["site_id", "target"])
    a = "icrisat_ka:1"
    assert v.loc[(a, "ph"), "value"] == pytest.approx(8.0)
    assert v.loc[(a, "soc"), "value"] == pytest.approx(5.0)                 # % → g/kg
    assert v.loc[(a, "k"), "value"] == pytest.approx(39.1 / 391.0)         # mg/kg → cmol(+)/kg
    assert v.loc[(a, "soc"), "method_family"] == "walkley_black"
    assert v.loc[(a, "ph"), "method_family"] == "ph_h2o"
    assert v.loc[(a, "k"), "method_family"] == "unknown"            # unit unconfirmed: not a standard-method label
    assert v.loc[(a, "k"), "method_id"] == "icrisat_karnataka/avK_NH4OAc_ICR"
    # state-lab K is never converted
    assert not ((labels["target"] == "k") & labels["method_id"].str.endswith("_DOA")).any()
    assert set(labels.loc[labels["site_id"] == "icrisat_ka:2", "target"]) == {"ph", "soc"}
    # out-of-range pH sample (12.5) dropped before the median
    assert v.loc[("icrisat_ka:6", "ph"), "value"] == pytest.approx(6.0)
    assert v.loc[("icrisat_ka:6", "k"), "value"] == pytest.approx(0.3)


def test_depth_coords_precision_licence(built):
    labels, _, _ = built
    assert (labels["depth_top"] == 0).all() and (labels["depth_bot"] == 15).all()
    assert labels["source"].eq("icrisat_karnataka").all()
    assert labels["license"].str.startswith("CC0-1.0").all() and labels["license"].str.contains("ODbL").all()
    assert labels["year"].isna().all()
    s = labels.drop_duplicates("site_id").set_index("site_id")
    assert s.loc["icrisat_ka:1", "coord_precision_m"] == 2000                # exact floor
    assert s.loc["icrisat_ka:2", "coord_precision_m"] == 5000                # fuzzy floor
    assert s.loc["icrisat_ka:6", "coord_precision_m"] > 10_000               # farthest vertex of a big village
    assert s.loc["icrisat_ka:1", "lon"] == pytest.approx(75.70)
    assert "fuzzy" in s.loc["icrisat_ka:2", "coord_source"]
    assert not labels.astype(str).apply(lambda c: c.str.contains("SHOULD NOT")).any().any()
