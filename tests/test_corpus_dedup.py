import numpy as np
import pandas as pd

from geosoil.corpus import dedup

M_PER_DEG_LAT = 111_195.0


def _rows(site_id, source, lat, lon=-100.0, top=0.0, bot=20.0):
    return dict(site_id=site_id, source=source, lat=lat, lon=lon, depth_top=top, depth_bot=bot,
                layer_id=site_id + "/1", target="ph", value=6.0, license="x")


def _df(*rows):
    return pd.DataFrame(list(rows))


def test_higher_priority_source_is_kept_within_radius():
    labels = _df(_rows("w:1", "wosis_2023", 40.0), _rows("k:1", "kssl_ldm", 40.0 + 50 / M_PER_DEG_LAT))
    kept, log = dedup.drop_cross_source(labels)
    assert kept.site_id.tolist() == ["k:1"]
    assert log.iloc[0].keeper_site_id == "k:1" and 45 < log.iloc[0].distance_m < 55


def test_radius_boundary():
    near = _df(_rows("k:1", "kssl_ldm", 40.0), _rows("w:1", "wosis_2023", 40.0 + 99 / M_PER_DEG_LAT))
    far = _df(_rows("k:1", "kssl_ldm", 40.0), _rows("w:1", "wosis_2023", 40.0 + 101 / M_PER_DEG_LAT))
    assert len(dedup.drop_cross_source(near)[1]) == 1
    assert len(dedup.drop_cross_source(far)[1]) == 0


def test_no_depth_overlap_means_not_a_duplicate():
    labels = _df(_rows("k:1", "kssl_ldm", 40.0, top=0, bot=20), _rows("w:1", "wosis_2023", 40.0, top=50, bot=80))
    assert len(dedup.drop_cross_source(labels)[1]) == 0


def test_nearby_sites_in_the_same_source_are_both_kept():
    labels = _df(_rows("k:1", "kssl_ldm", 40.0), _rows("k:2", "kssl_ldm", 40.0 + 10 / M_PER_DEG_LAT))
    assert len(dedup.drop_cross_source(labels)[0]) == 2


def test_same_coordinates_within_source_merge_but_keep_layers():
    a, b = _rows("n:1", "napeshm", 38.0), _rows("n:2", "napeshm", 38.0)
    b["layer_id"] = "n:2/1"
    merged, log = dedup.merge_same_coordinates(_df(a, b))
    assert merged.site_id.nunique() == 1 and len(merged) == 2
    assert log.iloc[0].action == "merged"


def test_holdout_is_never_deduplicated_but_flags_neighbours():
    tables = [_df(_rows("k:1", "kssl_ldm", 40.0), _rows("k:2", "kssl_ldm", 45.0)),
              _df(_rows("raca:1", "raca", 40.0 + 500 / M_PER_DEG_LAT))]
    c1, log = dedup.build_c1(tables)
    assert set(c1.site_id) == {"k:1", "k:2", "raca:1"} and log.empty
    flags = c1.set_index("site_id")
    assert bool(flags.loc["k:1", "near_holdout"]) and not bool(flags.loc["k:2", "near_holdout"])
    assert flags.loc["raca:1", "role"] == "holdout"


def test_every_removed_site_is_logged():
    labels = _df(_rows("k:1", "kssl_ldm", 40.0), _rows("w:1", "wosis_2023", 40.0),
                 _rows("o:1", "ossl_v1_2", 40.0), _rows("w:2", "wosis_2023", 41.0))
    kept, log = dedup.drop_cross_source(labels)
    assert set(kept.site_id) | set(log.site_id) == set(labels.site_id)
    assert set(kept.site_id) == {"k:1", "w:2"} and set(log.keeper_site_id) == {"k:1"}


def test_independent_surveys_are_not_dropped_near_other_sources():
    labels = _df(_rows("k:1", "kssl_ldm", 40.0), _rows("u:1", "usgs_ds801", 40.0),
                 _rows("n:1", "napeshm", 40.0))
    kept, log = dedup.drop_cross_source(labels)
    assert len(kept) == 3 and log.empty


def test_native_source_outranks_the_compilation_copy():
    labels = _df(_rows("w:1", "wosis_2023", 52.0), _rows("npdb:1", "canada_npdb", 52.0))
    kept, log = dedup.drop_cross_source(labels)
    assert kept.site_id.tolist() == ["npdb:1"] and log.iloc[0].site_id == "w:1"


def test_gap_fill_sources_keep_only_targets_the_keeper_lacks():
    from geosoil.corpus import dedup
    rows = []
    for t, v in (("soc", 10.0), ("ph", 6.0)):                      # WoSIS copy: SOC and pH, no bases
        rows.append(dict(site_id="wosis:1", source="wosis_2023", lon=10.0, lat=10.0, depth_top=0.0, depth_bot=20.0,
                         target=t, value=v))
    for t, v in (("soc", 11.0), ("k", 0.4), ("ca", 5.0)):          # compendium copy 30 m away: SOC plus bases
        rows.append(dict(site_id="olm:1", source="olm_compendium", lon=10.00027, lat=10.0, depth_top=0.0,
                         depth_bot=20.0, target=t, value=v))
    out, log = dedup.drop_cross_source(pd.DataFrame(rows).astype({"site_id": "string", "target": "string", "source": "string"}))
    kept = set(zip(out["site_id"], out["target"]))
    assert ("olm:1", "soc") not in kept and {("olm:1", "k"), ("olm:1", "ca")} <= kept
    assert log.loc[log["site_id"] == "olm:1", "action"].item() == "gap_fill"
    # a non-gap-fill republisher is still dropped whole
    out2, _ = dedup.drop_cross_source(pd.DataFrame(rows).replace({"olm_compendium": "ossl_v1_2"}))
    assert "olm:1" not in set(out2["site_id"])
