"""US EPA National Wetland Condition Assessment (NWCA) 2011 + 2016 soil chemistry → canonical labels.

Source: EPA National Aquatic Resource Surveys data pages (CSV). Licence: US Government work, public domain.
Soil samples were analysed by the NRCS Kellogg Soil Survey Laboratory (KSSL), so a few locations may also be
in kssl_ldm (dedup.REPUBLISHES lists kssl_ldm only).

Depth: DEPTH (2011) and HORIZON_DEPTH (2016) are the **lower** boundary of a horizon in cm ("thickness = depth of
lower boundary − depth of lower boundary of horizon above"). The top is the previous horizon's lower depth
within the same UID + PIT (2011) or UID (2016), taken from the horizon-description files because the chemistry
tables skip unsampled horizons. Horizon 1 starts at 0. A horizon with no description and no chemistry row for
the horizon above has no known top and is dropped.

Units and mapping (all % or cmol(+)/kg in the source):
  2011: CLAY/SAND %; TOT_CARBON %; PH_H2O, PH_CACL2; CEC (NH4OAc pH 7); CA/K/MG (NH4OAc pH 7); BULK_DEN_DBF.
  2016: CLAY_PSDA/SAND_PSDA %; C_TOT_ELEM_ANLYS %; PH_H2O_SOLN, PH_CACL2_SOLN; CEC_NH4OAC_EXTR;
        CA/K/MG_NH4OAC_EXTR. There is no bulk density in 2016.
  SOC g/kg = 10 × (TC − 0.12 × CaCO3) when CaCO3 (CACO3_2MM / CACO3_WETOX_2MM) is reported, else 10 × TC;
  family dry_combustion, method_id nwca/tc_minus_caco3. Texture family unknown (KSSL pipette). The 2011 bulk
  density is field-moist Dbf, so family unknown; rows flagged as inconsistent or outside the valid range are
  dropped.
Flags: 2011 values flagged "Analysis could not be done…" are dropped, and so are "< MDL" values, which are
written as 0 (the 2016 table leaves the same not-detected values empty). "< PQL" values are real low readings
and are kept, as are the 2016 "L" (below reporting limit) values. 2016 "N"/"ND" rows carry no value.
Year = DATE_COL. Coordinates: LAT_DD83 / LON_DD83 (NAD83) from the site-information files, keyed by UID.
249 locations were revisited between 2011 and 2016; both visits are kept (different years).

    python -m geosoil.corpus.nwca
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import common

SOURCE = "nwca"
URLS = {
    "chem11": "https://www.epa.gov/sites/default/files/2016-10/nwca2011_soilchem.csv",
    "hor11": "https://www.epa.gov/sites/default/files/2016-10/nwca2011_soilprofhorizons.csv",
    "site11": "https://www.epa.gov/system/files/other-files/2023-04/nwca11_siteinfo.csv",
    "chem16": "https://www.epa.gov/sites/default/files/2021-04/nwca_2016_soil_horizon_chemistry_-_data_csv.csv",
    "hor16": "https://www.epa.gov/sites/default/files/2021-04/nwca_2016_soil_horizon_description_-_data_csv.csv",
    "site16": "https://www.epa.gov/system/files/other-files/2022-04/nwca-2016-site-information-data_0.csv",
}
RAW = common.RAW / "nwca"
LICENSE = "US Government work, public domain (EPA NWCA)"
BD_BAD_FLAG = r"inconsistent|outside of valid range|wrong core volume"


def _tops(hor: pd.DataFrame, keys: list[str], hz: str, depth: str) -> pd.DataFrame:
    """Top depth of each horizon = lower depth of the horizon above (0 for horizon 1)."""
    h = hor[keys + [hz, depth]].copy()
    h[hz] = common.num(h[hz])
    h[depth] = common.num(h[depth])
    h = h.dropna(subset=[hz]).drop_duplicates(keys + [hz]).sort_values(keys + [hz])
    prev = h.groupby(keys)[depth].shift(1)
    prev_hz = h.groupby(keys)[hz].shift(1)
    h["top"] = np.where(h[hz] == 1, 0.0, np.where(prev_hz == h[hz] - 1, prev, np.nan))
    return h[keys + [hz, "top"]]


def _with_top(chem: pd.DataFrame, hor: pd.DataFrame, keys: list[str], hz: str, depth: str,
              hor_hz: str) -> pd.Series:
    """Top depth for each chemistry row: from the description file, else from the chemistry row above."""
    c = chem[keys + [hz, depth]].copy()
    c["_hz"] = common.num(c[hz])
    t = _tops(hor.rename(columns={hor_hz: "_hz"}), keys, "_hz", depth)
    top = c.merge(t, on=keys + ["_hz"], how="left")["top"].to_numpy()
    own = _tops(c, keys, "_hz", depth)                # chemistry-only fallback
    top2 = c.merge(own, on=keys + ["_hz"], how="left")["top"].to_numpy()
    return pd.Series(np.where(np.isnan(top), top2, top), index=chem.index)


def _soc(tc: pd.Series, caco3: pd.Series) -> pd.Series:
    soc = 10.0 * (tc - common.CACO3_C_FRACTION * caco3.fillna(0.0))
    return soc.where(soc > 0)


def _ok(df: pd.DataFrame, col: str) -> pd.Series:
    """2011 values minus 'could not be done' and '< MDL' (written as 0)."""
    flag = df.get(col + "_FLAG", pd.Series("", index=df.index)).fillna("").astype(str)
    v = common.num(df[col])
    return v.where(~flag.str.contains(r"could not be done|< MDL", regex=True))


def _ok16(df: pd.DataFrame, col: str) -> pd.Series:
    flag = df.get(col + "_NARS_FLAG", pd.Series("", index=df.index)).fillna("").astype(str)
    return common.num(df[col]).where(~flag.isin(["N", "ND"]))


def build_2011(chem: pd.DataFrame, hor: pd.DataFrame) -> pd.DataFrame:
    keys = ["UID", "PIT"]
    site_key = "2011/" + chem["UID"].astype(str) + "/" + chem["PIT"].astype(str)
    base = pd.DataFrame({"site_key": site_key, "layer_key": site_key + "/" + chem["LAYER"].astype(str),
                         "depth_top": _with_top(chem, hor, keys, "LAYER", "DEPTH", "HORIZON"),
                         "depth_bot": common.num(chem["DEPTH"])})
    bd_flag = chem["BULK_DEN_DBF_FLAG"].fillna("").astype(str)
    bd = _ok(chem, "BULK_DEN_DBF").where(~bd_flag.str.contains(BD_BAD_FLAG, case=False, regex=True))
    return pd.concat([
        common.long_rows(base, "soc", _soc(_ok(chem, "TOT_CARBON"), _ok(chem, "CACO3_2MM")),
                         "nwca/tc_minus_caco3", "dry_combustion"),
        common.long_rows(base, "ph", _ok(chem, "PH_H2O"), "nwca11/PH_H2O", "ph_h2o"),
        common.long_rows(base, "ph", _ok(chem, "PH_CACL2"), "nwca11/PH_CACL2", "ph_cacl2"),
        common.long_rows(base, "sand", _ok(chem, "SAND"), "nwca11/SAND", "unknown"),
        common.long_rows(base, "clay", _ok(chem, "CLAY"), "nwca11/CLAY", "unknown"),
        common.long_rows(base, "bd", bd, "nwca11/BULK_DEN_DBF", "unknown"),
        common.long_rows(base, "cec", _ok(chem, "CEC"), "nwca11/CEC_NH4OAc", "cec_ph7"),
        *[common.long_rows(base, t, _ok(chem, c), f"nwca11/{c}_NH4OAc", "nh4oac_ph7")
          for t, c in (("k", "K"), ("ca", "CA"), ("mg", "MG"))],
    ], ignore_index=True)


def build_2016(chem: pd.DataFrame, hor: pd.DataFrame) -> pd.DataFrame:
    keys = ["UID"]
    site_key = "2016/" + chem["UID"].astype(str)
    hz = common.num(chem["HORIZON"])
    base = pd.DataFrame({"site_key": site_key,
                         "layer_key": site_key + "/" + hz.astype("Int64").astype(str),
                         "depth_top": _with_top(chem, hor, keys, "HORIZON", "HORIZON_DEPTH", "HORIZON"),
                         "depth_bot": common.num(chem["HORIZON_DEPTH"])})
    return pd.concat([
        common.long_rows(base, "soc", _soc(_ok16(chem, "C_TOT_ELEM_ANLYS"), _ok16(chem, "CACO3_WETOX_2MM")),
                         "nwca/tc_minus_caco3", "dry_combustion"),
        common.long_rows(base, "ph", _ok16(chem, "PH_H2O_SOLN"), "nwca16/PH_H2O_SOLN", "ph_h2o"),
        common.long_rows(base, "ph", _ok16(chem, "PH_CACL2_SOLN"), "nwca16/PH_CACL2_SOLN", "ph_cacl2"),
        common.long_rows(base, "sand", _ok16(chem, "SAND_PSDA"), "nwca16/SAND_PSDA", "unknown"),
        common.long_rows(base, "clay", _ok16(chem, "CLAY_PSDA"), "nwca16/CLAY_PSDA", "unknown"),
        common.long_rows(base, "cec", _ok16(chem, "CEC_NH4OAC_EXTR"), "nwca16/CEC_NH4OAC_EXTR", "cec_ph7"),
        *[common.long_rows(base, t, _ok16(chem, f"{c}_NH4OAC_EXTR"), f"nwca16/{c}_NH4OAC_EXTR", "nh4oac_ph7")
          for t, c in (("k", "K"), ("ca", "CA"), ("mg", "MG"))],
    ], ignore_index=True)


def _sites(chem: pd.DataFrame, site: pd.DataFrame, site_key: pd.Series) -> pd.DataFrame:
    s = site.drop_duplicates("UID")[["UID", "LAT_DD83", "LON_DD83"]]
    out = pd.DataFrame({"site_key": site_key, "UID": chem["UID"],
                        "year": pd.to_datetime(chem["DATE_COL"], format="%m/%d/%Y", errors="coerce").dt.year})
    out = out.drop_duplicates("site_key").merge(s, on="UID", how="left")
    return pd.DataFrame({"site_key": out["site_key"], "lon": common.num(out["LON_DD83"]),
                         "lat": common.num(out["LAT_DD83"]), "coord_source": "LAT_DD83/LON_DD83 (NAD83)",
                         "coord_precision_m": np.nan, "year": out["year"]})


def build(chem11: pd.DataFrame, hor11: pd.DataFrame, site11: pd.DataFrame,
          chem16: pd.DataFrame, hor16: pd.DataFrame, site16: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    vals = pd.concat([build_2011(chem11, hor11), build_2016(chem16, hor16)], ignore_index=True)
    sk11 = "2011/" + chem11["UID"].astype(str) + "/" + chem11["PIT"].astype(str)
    sk16 = "2016/" + chem16["UID"].astype(str)
    sites = pd.concat([_sites(chem11, site11, sk11), _sites(chem16, site16, sk16)], ignore_index=True)
    return common.finalize(vals, sites, source=SOURCE, prefix="nwca", license=LICENSE)


def load() -> tuple[pd.DataFrame, dict]:
    t = {k: pd.read_csv(common.fetch(u, RAW / u.rsplit("/", 1)[1]), encoding="utf-8-sig", low_memory=False)
         for k, u in URLS.items()}
    return build(t["chem11"], t["hor11"], t["site11"], t["chem16"], t["hor16"], t["site16"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
