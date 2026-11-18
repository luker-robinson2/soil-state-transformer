"""Karnataka soil health data (Bhoochetana; doi:10.21421/D2/QYCEGR, CC0) → canonical village-level labels.

One tab-separated file of 92,832 farm-field topsoil samples (about 2009–2011) analysed for the Bhoochetana
soil-health mapping programme: pH, EC, OC, available P/K/S and micronutrients. There are NO coordinates: each
row names district / taluk / village (and a cadastral survey number), so samples are placed on village
centroids from the DataMeet Indian Village Boundaries layer for Karnataka (Census 2001 villages, ODbL-1.0).

Gotchas:
  * names are truncated in the file: district to 8 characters ("Chikball"), taluk to 6 ("Bagepa"), village to
    10 ("Kerkalamat"). District names are the post-2010 ones (Ballari, Kalaburagi, Yadgir, Ramanagara,
    Chikkaballapura), while the gazetteer uses 2001 districts, so districts map to 2001 parent districts
    (DISTRICTS) and renamed taluks have explicit aliases (TALUK_ALIASES).
  * villages match on a transliteration key (letters only, doubled letters collapsed, aspirates dropped:
    "Kalalugatt" ~ "Kalalghatta") as an exact (or, for 10-character truncated names, prefix) match inside the
    taluk; otherwise a fuzzy match (difflib ratio >= 0.85 and >= 0.05 ahead of the runner-up). Names that
    match several villages in the taluk, or nothing, are dropped (about a quarter of villages: hamlets and
    spellings too far from the Census form).
  * one malformed row (fields shifted by a stray quote; state == "7.5") is skipped; farmer_name is never read.
Aggregation: sampled fields inside a village are not located, so all samples of a village collapse to ONE site
at the village centroid, with the median per target and lab (`authority`). The sample count per village is in
the sidecar `icrisat_karnataka_villages.parquet` written by main(). Sample values outside schema.VALID_RANGE
are removed before the median.
Coordinates: polygon centroid of the matched village; coord_precision_m = distance from the centroid to the
farthest polygon vertex (every field of the village lies within it), floored at 2,000 m for exact name
matches and 5,000 m for fuzzy ones.
Depth 0–15 cm: not stated in the dataset metadata; the Bhoochetana sampling protocol (stratified farm-field
sampling, published by the programme) takes 0–15 cm topsoil cores.
Mapping and units (methods are not stated in the file; labs are the `authority` column: ICR = the
programme's central research lab, DOA = state Department of Agriculture labs, Dak = Dakshina Kannada lab):
  * ph → ph_h2o for every lab. The central lab uses 1:2 soil:water; the state labs' ratio is not recorded.
  * oc % (Walkley-Black wet oxidation, the Indian soil-health standard) × 10 → soc g/kg, walkley_black.
  * av_k: ammonium acetate (1 N, pH 7) extractable K in mg/kg (critical limit 50 mg/kg; ICR median 115,
    13% below 50, matching the programme's published deficiency rates; a kg/ha figure would be ~2.24× larger).
    Converted ÷ 391.0 → cmol(+)/kg, family "unknown" until the unit is confirmed (kept out of standard-method scoring), ONLY for ICR rows. DOA/Dak K is skipped: its unit and
    method are unconfirmed and it is internally inconsistent (DOA black soils read lower than red soils,
    the opposite of the ICR rows and of Vertisol mineralogy).
  * ec, av_p, av_s, micronutrients: not targets. soil_type is a farmer-reported class, not texture: skipped.
  * no Ca, Mg, CEC, BD or particle-size data in this file.
Year: not recorded per row (programme sampling 2009–2011), left empty.

    python -m geosoil.corpus.icrisat_karnataka
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import common, schema

SOURCE = "icrisat_karnataka"
PREFIX = "icrisat_ka"
URL = "https://dataverse.icrisat.org/api/access/datafile/1171"
RAW = common.RAW / SOURCE / "Karnataka_soil_profile.tab"
GAZ_URL = ("https://raw.githubusercontent.com/datameet/indian_village_boundaries/"
           "0c83ba6aec9069add004bdbbbaa7817fafb359bf/ka/ka.geojson")
GAZ_RAW = common.RAW / SOURCE / "datameet_ka_villages_2001.geojson"
LICENSE = ("CC0-1.0 (Karnataka soil health data, doi:10.21421/D2/QYCEGR); village centroids from DataMeet "
           "Indian Village Boundaries (ODbL-1.0)")
DEPTH = (0.0, 15.0)
K_MG_PER_CMOL = 391.0
K_LABS = {"ICR"}
FUZZY_MIN, FUZZY_MARGIN = 0.85, 0.05
PRECISION_FLOOR_M = {"exact": 2000.0, "fuzzy": 5000.0}
TRUNC_VILLAGE = 10

# file district (first 8 chars, letters only, lower) -> 2001 gazetteer district(s)
DISTRICTS = {
    "bagalkot": ["Bagalkot"], "ballari": ["Bellary"], "belagavi": ["Belgaum"],
    "bengalur": ["Bangalore", "Bangalore Rural"], "bidar": ["Bidar"], "chamaraj": ["Chamarajnagar"],
    "chikball": ["Kolar"], "chikkama": ["Chikmagalur"], "chitradu": ["Chitradurga"],
    "dakshina": ["Dakshina Kannada"], "davanage": ["Davangere"], "dharwad": ["Dharwad"], "gadag": ["Gadag"],
    "hassan": ["Hassan"], "haveri": ["Haveri"], "kalabura": ["Gulbarga"], "kodagu": ["Kodagu"],
    "kolar": ["Kolar"], "koppal": ["Koppal"], "mandya": ["Mandya"], "mysuru": ["Mysore"],
    "raichur": ["Raichur"], "ramanaga": ["Bangalore Rural"], "shivamog": ["Shimoga"], "tumakuru": ["Tumkur"],
    "udupi": ["Udupi"], "uttarak": ["Uttar Kannad"], "vijayapu": ["Bijapur"], "yadgir": ["Gulbarga"],
}
# file taluk (letters only, lower) -> 2001 taluk name(s), where the prefix does not match the 2001 spelling
TALUK_ALIASES = {
    "ballar": ["Bellary"], "hosape": ["Hospet"], "hoovin": ["Hadagalli"], "bailah": ["Bailhongal"],
    "belaga": ["Belgaum"], "hukker": ["Hukeri"], "devana": ["Devanhalli"], "doddab": ["Dod Ballapur"],
    "bengal": ["Bangalore North", "Bangalore South"], "chikka": ["Chikmagalur"], "hosadu": ["Hosdurga"],
    "bantwa": ["Bantval"], "davana": ["Davangere"], "hubbal": ["Hubli"], "byadgi": ["Bydagi"],
    "afzalp": ["Afsalpur"], "kalabu": ["Gulbarga"], "malava": ["Malvalli"], "mysuru": ["Mysore"],
    "tnars": ["Tirumakudal Narsipur"], "kanaka": ["Kankapura"], "shivam": ["Shimoga"],
    "thirth": ["Tirthahalli"], "korata": ["Korategere"], "tumaku": ["Tumkur"], "kundap": ["Coondapur"],
    "honnav": ["Honavar"], "joida": ["Supa"], "vijaya": ["Bijapur"],
}
_KEY_RULES = [(r"[^a-z]", ""), (r"ee", "i"), (r"oo", "u"), (r"(.)\1+", r"\1"), (r"([kgcjtdpb])h", r"\1"),
              (r"sh", "s"), (r"w", "v"), (r"z", "j"), (r"y", "i"), (r"q", "k"), (r"(.)\1+", r"\1")]


def letters(s) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def name_key(s) -> str:
    """Transliteration-tolerant key: Kalalugatta / Kalalghatta, Hadnur / Hadnoor share most of it."""
    k = str(s).lower()
    for pat, rep in _KEY_RULES:
        k = re.sub(pat, rep, k)
    return k


# --------------------------------------------------------------------------- gazetteer
def _ring_centroid_m(ring: np.ndarray, lat0: float) -> tuple[float, float, float]:
    """Signed area and centroid of one ring in local equirectangular metres (fine at village scale)."""
    mx = 111_320.0 * np.cos(np.radians(lat0))
    x, y = ring[:, 0] * mx, ring[:, 1] * 110_574.0
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    a = cross.sum() / 2.0
    if a == 0:
        return 0.0, float(x.mean()), float(y.mean())
    return a, float(((x[:-1] + x[1:]) * cross).sum() / (6 * a)), float(((y[:-1] + y[1:]) * cross).sum() / (6 * a))


def polygon_centroid(geom: dict) -> tuple[float, float, float]:
    """(lon, lat, farthest-vertex distance in m) of a GeoJSON Polygon / MultiPolygon (holes ignored)."""
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    rings = [np.asarray(p[0], float) for p in polys if p and len(p[0]) >= 3]
    rings = [np.vstack([r, r[:1]]) if not np.array_equal(r[0], r[-1]) else r for r in rings]
    lat0 = float(np.concatenate([r[:, 1] for r in rings]).mean())
    mx = 111_320.0 * np.cos(np.radians(lat0))
    parts = [_ring_centroid_m(r, lat0) for r in rings]
    w = np.array([abs(a) for a, _, _ in parts])
    if w.sum() == 0:
        w = np.ones(len(parts))
    cx = float(np.average([p[1] for p in parts], weights=w))
    cy = float(np.average([p[2] for p in parts], weights=w))
    pts = np.concatenate(rings)
    rmax = float(np.hypot(pts[:, 0] * mx - cx, pts[:, 1] * 110_574.0 - cy).max())
    return cx / mx, cy / 110_574.0, rmax


def villages(gj: dict) -> pd.DataFrame:
    """One row per gazetteer village: loc_code, name, taluk, district, lon, lat, rmax_m."""
    rows = []
    for f in gj["features"]:
        p, g = f["properties"], f.get("geometry")
        if not g or not p.get("NAME"):
            continue
        lon, lat, rmax = polygon_centroid(g)
        rows.append((str(p["LOC_CODE"]), p["NAME"], p["TALUK"], p["DISTRICT"], lon, lat, rmax))
    return pd.DataFrame(rows, columns=["loc_code", "name", "taluk", "district", "lon", "lat", "rmax_m"])


# --------------------------------------------------------------------------- matching
def taluk_candidates(gaz: pd.DataFrame, district: str, taluk: str) -> pd.DataFrame:
    g = gaz[gaz["district"].isin(DISTRICTS.get(letters(district)[:8], []))]
    t = letters(taluk)
    if t in TALUK_ALIASES:
        return g[g["taluk"].isin(TALUK_ALIASES[t])]
    return g[g["taluk"].map(letters).str.startswith(t)] if t else g.iloc[:0]


def match_village(cand: pd.DataFrame, village: str) -> tuple[str | None, str]:
    """(loc_code, kind) with kind in exact / fuzzy / ambiguous / none."""
    q = name_key(village)
    if not q or cand.empty:
        return None, "none"
    keys = cand["name"].map(name_key)
    trunc = len(str(village)) >= TRUNC_VILLAGE
    hit = cand[keys.str.startswith(q)] if trunc else cand[keys == q]
    if len(hit) == 1:
        return hit["loc_code"].iloc[0], "exact"
    if len(hit) > 1:
        return None, "ambiguous"
    score = keys.map(lambda k: difflib.SequenceMatcher(None, q, k[:len(q)] if trunc else k).ratio())
    top = score.nlargest(2)
    if top.iloc[0] >= FUZZY_MIN and (len(top) < 2 or top.iloc[0] - top.iloc[1] >= FUZZY_MARGIN):
        return cand.loc[top.index[0], "loc_code"], "fuzzy"
    return None, "none"


def match(samples: pd.DataFrame, gaz: pd.DataFrame) -> pd.DataFrame:
    """Unique (district, taluk, village) → loc_code / match kind."""
    out = []
    names = samples[["district", "taluk", "village"]].dropna().drop_duplicates()
    for (d, t), grp in names.groupby(["district", "taluk"], sort=False):
        cand = taluk_candidates(gaz, d, t)
        for v in grp["village"]:
            out.append((d, t, v, *match_village(cand, v)))
    return pd.DataFrame(out, columns=["district", "taluk", "village", "loc_code", "match"])


# --------------------------------------------------------------------------- build
def build(samples: pd.DataFrame, gaz: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    labels, drops, _ = build_with_villages(samples, gaz)
    return labels, drops


def build_with_villages(samples: pd.DataFrame, gaz: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """build() plus the per-village table (site, match kind, sample count)."""
    n_all = len(samples)
    s = samples[samples["state"].astype(str).str.strip() == "Karnataka"].copy()
    drops = {"malformed_rows": n_all - len(s)}
    m = match(s, gaz)
    drops.update({f"villages_{k}": int(v) for k, v in m["match"].value_counts().items()})
    s = s.merge(m, on=["district", "taluk", "village"], how="left")
    unplaced = s["loc_code"].isna()
    drops["samples_unplaced"] = int(unplaced.sum())
    s = s[~unplaced]

    lab = s["authority"].astype(str).str.strip()
    ph = common.num(s["ph"])
    soc = common.num(s["oc"]) * 10.0
    k = (common.num(s["av_k"]) / K_MG_PER_CMOL).where(lab.isin(K_LABS))
    long = pd.concat([
        pd.DataFrame({"loc_code": s["loc_code"], "target": "ph", "value": ph,
                      "method_id": SOURCE + "/pH_H2O_" + lab, "method_family": "ph_h2o"}),
        pd.DataFrame({"loc_code": s["loc_code"], "target": "soc", "value": soc,
                      "method_id": SOURCE + "/OC_WB_" + lab, "method_family": "walkley_black"}),
        pd.DataFrame({"loc_code": s["loc_code"], "target": "k", "value": k,
                      "method_id": SOURCE + "/avK_NH4OAc_" + lab, "method_family": "unknown"}),
    ], ignore_index=True).dropna(subset=["value"])
    ok = pd.Series(True, index=long.index)
    for t, (lo, hi) in schema.VALID_RANGE.items():
        ok &= (long["target"] != t) | long["value"].between(lo, hi)
    drops["sample_values_out_of_range"] = int((~ok).sum())
    long = long[ok]

    agg = (long.groupby(["loc_code", "target", "method_id", "method_family"], sort=False)["value"]
           .agg(value="median", n_samples="size").reset_index())
    vals = agg.assign(site_key=agg["loc_code"], layer_key=agg["loc_code"] + "/0-15",
                      depth_top=DEPTH[0], depth_bot=DEPTH[1])

    kind = m.dropna(subset=["loc_code"]).groupby("loc_code")["match"].agg(
        lambda x: "exact" if (x == "exact").all() else "fuzzy")
    g = gaz.drop_duplicates("loc_code").set_index("loc_code").loc[kind.index]
    floor = kind.map(PRECISION_FLOOR_M)
    sites = pd.DataFrame({
        "site_key": kind.index, "lon": g["lon"].values, "lat": g["lat"].values,
        "coord_source": ("village centroid (Census 2001 polygon), " + kind + " name match").values,
        "coord_precision_m": np.maximum(g["rmax_m"].values, floor.values).round(0),
    })
    labels, qa = common.finalize(vals, sites, source=SOURCE, prefix=PREFIX, license=LICENSE)
    drops.update(qa)
    n = long.groupby("loc_code").size()      # sample values kept (all targets)
    villages_out = sites.assign(n_samples=sites["site_key"].map(s.groupby("loc_code").size()),
                                n_values=sites["site_key"].map(n), match=kind.values,
                                name=g["name"].values, taluk=g["taluk"].values)
    return labels, drops, villages_out


def load(with_villages: bool = False):
    path = common.fetch(URL, RAW)
    gaz_path = common.fetch(GAZ_URL, GAZ_RAW)
    cols = ["state", "district", "taluk", "village", "authority", "ph", "oc", "av_k"]   # never farmer_name
    samples = pd.read_csv(path, sep="\t", dtype=str, usecols=cols)
    gaz = villages(json.loads(Path(gaz_path).read_text()))
    out = build_with_villages(samples, gaz)
    return out if with_villages else out[:2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    out = ap.parse_args().out
    labels, drops, vt = load(with_villages=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    vt.to_parquet(out.with_name(f"{SOURCE}_villages.parquet"), index=False)
    common.write(labels, drops, out)


if __name__ == "__main__":
    main()
