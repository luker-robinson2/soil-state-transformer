"""Queensland SALI Soil Chemistry API (Queensland Government, CC-BY 4.0) → canonical labels.

Source: the OData v4 service https://soil-chem.information.qld.gov.au/odata/ (anonymous). `Samples` holds one
row per sample with depths and coordinates; `SiteLabMethodResults` holds one row per lab result, filtered here
by Rayment & Lyons method code. Licence: CC-BY 4.0, one value for every row.

API gotchas (data scout pass 3, 2026-11-18):
  * `$filter`/`$orderby` take PascalCase property names (`LabMethodCode`); the JSON keys come back camelCase.
  * `$top` is capped at 1000 and `$expand` is broken (returns a bare "["), so samples and results are pulled
    separately, page by page, and joined here. Pages are cached under data/raw/qld_sali/ and the pull resumes.
  * `NumericValue` is a string.
Units and conversions:
  * depths are in metres → ×100 cm; layers starting at 300 cm or deeper are dropped.
  * texture 2Z2 is on ISSS limits (fine sand 20–200 µm, coarse sand 200–2000 µm, silt 2–20 µm), so
    sand = 2Z2_CS + 2Z2_FS (same sample) is sand > 20 µm, not USDA > 50 µm: it gets its own method id
    `sali/2Z2_CS+FS_isss`, family unknown. Clay (< 2 µm) is comparable.
  * organic carbon 6A1 (Walkley-Black) and 6B2/6B3 (dry combustion) are % → ×10 g/kg.
  * pH 4A1 is 1:5 water, 4B1 is 1:5 CaCl2. CEC and exchangeable bases are cmol(+)/kg already.
  * 15C1 is NH4OAc pH 7 (prewash) → cec_ph7 / nh4oac_ph7. 15B1 CEC (NH4Cl pH 7), 15A2 and 15D1 CEC
    (alcoholic, calcareous soils) are pH-7 index-cation CECs → cec_ph7. Bases by 15A1 (1 M NH4Cl), 15A2 and
    15D3 are not NH4OAc → unknown.
  * BD 503.01a (small core) g/cm3 → bd_oven_dry.
Filters: only ValidFlag == "Y" (about 25% of clay/CEC rows are "N", meaning undocumented) and ResultStatus
!= "BQ" (below quantification). Year: only an analysis date exists; it is used as an upper bound on the
sampling year (earliest per site). Dates written like 0074-11-01 get +1900.
Keys: site = ProjectCode/SiteId/ObservationNumber, layer = site/SampleNumber, method_id = sali/<code>.
SALI republishes the CSIRO NatSoil profiles that are in WoSIS: dedup treats it as a WoSIS republisher with
gap fill (WoSIS has no exchangeable bases).

    python -m geosoil.corpus.qld_sali
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import common

SOURCE = "qld_sali"
URL = "https://soil-chem.information.qld.gov.au/odata"
RAW = common.RAW / "qld_sali"
LICENSE = "CC-BY-4.0 (Queensland Government, SALI)"
PAGE = 1000
ORDER = "SiteId,ProjectCode,ObservationNumber,SampleNumber"
KEYS = ["projectCode", "siteId", "observationNumber", "sampleNumber"]
MAX_TOP_CM = 300.0

# method code -> (target, method_family); sand (2Z2_CS + 2Z2_FS) is built separately
METHODS = {
    "2Z2_Clay": ("clay", "unknown"),
    "6A1": ("soc", "walkley_black"), "6B2": ("soc", "dry_combustion"), "6B3": ("soc", "dry_combustion"),
    "4A1": ("ph", "ph_h2o"), "4B1": ("ph", "ph_cacl2"),
    "15C1_CEC": ("cec", "cec_ph7"), "15B1_CEC": ("cec", "cec_ph7"),
    "15A2_CEC": ("cec", "cec_ph7"), "15D1_CEC": ("cec", "cec_ph7"),
    "503.01a": ("bd", "bd_oven_dry"),
    **{f"15C1_{e}": (e.lower(), "nh4oac_ph7") for e in ("Ca", "Mg", "K")},
    **{f"{m}_{e}": (e.lower(), "unknown") for m in ("15A1", "15A2", "15D3") for e in ("Ca", "Mg", "K")},
}
SAND_CODES = ("2Z2_CS", "2Z2_FS")
SAND_METHOD = "sali/2Z2_CS+FS_isss"
SCALE = {"soc": 10.0}            # % -> g/kg
CODES = (*METHODS, *SAND_CODES)


# ---------------------------------------------------------------- paged OData pull (cached, resumable)
def _get(url: str, tries: int = 5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "geosoil-corpus/0.1"}),
                                        timeout=300) as r:
                return json.loads(r.read())
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def _query(entity: str, params: dict) -> str:
    return f"{URL}/{entity}?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def _page_ok(path: Path, expected: int) -> bool:
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    rows = rows.get("value", rows) if isinstance(rows, dict) else rows
    return isinstance(rows, list) and len(rows) == expected


def fetch_pages(name: str, entity: str, filt: str | None = None, workers: int = 8) -> pd.DataFrame:
    """All rows of `entity` (optionally filtered), one cached JSON file per 1000-row page."""
    RAW.mkdir(parents=True, exist_ok=True)
    base = {"$filter": filt} if filt else {}
    n = int(_get(f"{URL}/{entity}/$count?" + urllib.parse.urlencode(base, quote_via=urllib.parse.quote)))

    def one(skip: int) -> Path:
        path = RAW / f"{name}.{skip}.json"
        expected = min(PAGE, n - skip)
        if not _page_ok(path, expected):
            rows = _get(_query(entity, {**base, "$orderby": ORDER, "$skip": skip, "$top": PAGE}))
            tmp = path.with_suffix(".part")
            tmp.write_text(json.dumps(rows))
            tmp.rename(path)
        return path

    with ThreadPoolExecutor(workers) as ex:
        paths = list(ex.map(one, range(0, n, PAGE)))
    frames = []
    for p in paths:
        rows = json.loads(p.read_text())
        frames.append(pd.DataFrame(rows.get("value", rows) if isinstance(rows, dict) else rows))
    (RAW / f"{name}.manifest.json").write_text(json.dumps(
        dict(url=_query(entity, {**base, "$orderby": ORDER}), rows=n, pages=len(paths),
             fetched=pd.Timestamp.now("UTC").isoformat(timespec="seconds")), indent=2))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------- pure transform
def _year(dates: pd.Series) -> pd.Series:
    y = pd.to_numeric(dates.astype(str).str[:4], errors="coerce")
    return y.where(y >= 100, y + 1900)


def build(samples: pd.DataFrame, results: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """samples: Samples rows (camelCase keys); results: SiteLabMethodResults rows for CODES."""
    pre = {"input_results": len(results)}
    r = results[results["labMethodCode"].isin(CODES)].copy()
    valid = r["validFlag"].astype(str).str.upper() == "Y"
    pre["invalid_flag"] = int((~valid).sum())
    r = r[valid]
    bq = r["resultStatus"].astype(str).str.upper() == "BQ"
    pre["below_quantification"] = int(bq.sum())
    r = r[~bq]
    r["value"] = common.num(r["numericValue"])
    r = r[r["value"].notna()]
    r["year"] = _year(r["analysisDate"])

    s = samples.drop_duplicates(KEYS).copy()
    s["depth_top"] = common.num(s["upperDepth"]) * 100.0
    s["depth_bot"] = common.num(s["lowerDepth"]) * 100.0
    for df in (s, r):
        df["site_key"] = (df["projectCode"].astype(str) + "/" + df["siteId"].astype(str) + "/"
                          + df["observationNumber"].astype(str))
        df["layer_key"] = df["site_key"] + "/" + df["sampleNumber"].astype(str)
    r = r.merge(s[["layer_key", "depth_top", "depth_bot"]], on="layer_key", how="left")
    pre["no_sample"] = int(r["depth_top"].isna().sum())
    deep = r["depth_top"] >= MAX_TOP_CM
    pre["deeper_than_300cm"] = int(deep.sum())
    r = r[~deep & r["depth_top"].notna()]

    parts = []
    m = r[r["labMethodCode"].isin(METHODS)].reset_index(drop=True)
    tgt = m["labMethodCode"].map(lambda c: METHODS[c][0])
    for t, g in m.groupby(tgt):
        val = g["value"] * SCALE.get(t, 1.0)
        parts.append(common.long_rows(g, t, val, "sali/" + g["labMethodCode"].astype(str),
                                      g["labMethodCode"].map(lambda c: METHODS[c][1])))
    sand = r[r["labMethodCode"].isin(SAND_CODES)].drop_duplicates(["layer_key", "labMethodCode"])
    wide = sand.pivot(index="layer_key", columns="labMethodCode", values="value")
    if set(SAND_CODES) <= set(wide.columns):
        tot = (wide["2Z2_CS"] + wide["2Z2_FS"]).dropna().rename("value").reset_index()
        g = tot.merge(r.drop_duplicates("layer_key")[["layer_key", "site_key", "depth_top", "depth_bot"]],
                      on="layer_key")
        parts.append(common.long_rows(g, "sand", g["value"], SAND_METHOD, "unknown"))
    vals = pd.concat(parts, ignore_index=True)

    site_year = r.groupby("site_key")["year"].min()
    sites = s.drop_duplicates("site_key")[["site_key", "longitude", "latitude"]]
    sites = pd.DataFrame({
        "site_key": sites["site_key"],
        "lon": common.num(sites["longitude"]),
        "lat": common.num(sites["latitude"]),
        "coord_source": "Longitude/Latitude (GDA94)",
        "coord_precision_m": np.nan,
        "year": sites["site_key"].map(site_year),
    })
    labels, drops = common.finalize(vals, sites, source=SOURCE, prefix="qld", license=LICENSE)
    return labels, {**pre, **drops}


def load() -> tuple[pd.DataFrame, dict]:
    samples = fetch_pages("samples", "Samples")
    results = pd.concat([fetch_pages(f"r_{c}", "SiteLabMethodResults",
                                     f"LabMethodCode eq '{c}' and NumericValue ne null") for c in CODES],
                        ignore_index=True)
    return build(samples, results)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
