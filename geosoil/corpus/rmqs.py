"""France RMQS1 composite analyses, campaign 1 2000–2009 (doi:10.15454/QSXKGA, Etalab 2.0) → canonical labels.

RMQS (Réseau de Mesures de la Qualité des Sols) samples ~2,170 sites on a 16 km × 16 km grid over metropolitan
France. File: `RMQS1_analyses_composites_18_11_2021_virgule` (datafile 122620 on Recherche Data Gouv; 68 columns,
4,148 rows), one row per site and composite layer. Each composite mixes 25 auger cores from a 20 m × 20 m plot.

Columns used (names as published; numbers in the names are the lab method codes):
  id_site, no_couche (1 = 0–30 cm or the ploughed layer, 2 = the layer below to 50 cm, 3 = an occasional deeper
  layer, 9 = holorganic forest/meadow layer above the mineral surface, with negative depths: skipped),
  profondeur_hz_sup / profondeur_hz_inf (cm), date_complete (dd/mm/yyyy), x_theo / y_theo (Lambert-93).
  argile, limon_fin, limon_grossier, sable_fin, sable_grossier: g/kg, French limits 0–2 / 2–20 / 20–50 / 50–200 /
  200–2000 µm, pipette (NF X 31-107). clay = argile / 10 %; sand = (sable_fin + sable_grossier) / 10 %, i.e.
  50–2000 µm, the USDA sand limit. Family pipette.
  carbone_16_5_1: organic C g/kg by dry combustion with carbonate correction (NF ISO 10694) → dry_combustion.
  ph_eau_6_1: pH in water (NF ISO 10390) → ph_h2o.
  cec_40_1: CEC by the Metson method (1 M NH4OAc at pH 7), cmol(+)/kg → cec_ph7.
  ca_ech_40_3, mg_ech_40_3, k_ech_40_3: exchangeable bases in the Metson extract, cmol(+)/kg → nh4oac_ph7.
  Not used: mat_org_0 (organic matter derived from C), Na/Mn/Al/Fe, trace elements. No bulk density in this file.
Missing values are written "ND". The published CSV may use ';' or ',' as separator and ',' as decimal mark
("virgule"); `read_raw` sniffs both, and the Windows-1252 encoding.
Coordinates: x_theo / y_theo are the THEORETICAL grid-cell centre; the real site is confidential and lies at the
centre or, when that spot is unsuitable, within about 1 km of it. coord_precision_m = 1000. Rows with
site_officiel False (replacement sites) carry the same theoretical node.
RMQS1 is not in WoSIS; check it against the compendium in dedup (the compendium does not list RMQS as a source).

    python -m geosoil.corpus.rmqs
"""
from __future__ import annotations

import argparse
import io
import time
import urllib.request
from pathlib import Path

import pandas as pd
from pyproj import Transformer

from .. import config as C
from . import common

SOURCE = "rmqs"
DOI = "doi:10.15454/QSXKGA"
FILE_ID = 122620
URL = f"https://entrepot.recherche.data.gouv.fr/api/access/datafile/{FILE_ID}?format=original"
RAW = common.RAW / "rmqs" / "RMQS1_analyses_composites_18_11_2021_virgule.csv"
LICENSE = "Etalab-2.0 (RMQS1, Gis Sol / INRAE, doi:10.15454/QSXKGA)"
CRS = "EPSG:2154"                  # Lambert-93
COORD_PRECISION_M = 1000.0
SKIP_LAYERS = {9}                  # holorganic layer above the mineral surface
NUMERIC = ["no_couche", "profondeur_hz_sup", "profondeur_hz_inf", "x_theo", "y_theo", "argile",
           "sable_fin", "sable_grossier", "carbone_16_5_1", "ph_eau_6_1", "cec_40_1",
           "ca_ech_40_3", "mg_ech_40_3", "k_ech_40_3"]


def read_raw(path: Path) -> pd.DataFrame:
    """Read the published CSV whatever its separator, decimal mark and encoding; every column as text."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    header = text.split("\n", 1)[0]
    sep = max((";", "\t", ","), key=header.count)
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, na_values=["ND", "NA", ""], keep_default_na=False)


def _num(s: pd.Series) -> pd.Series:
    """Numbers that may use a decimal comma."""
    return common.num(s.astype("string").str.strip().str.replace(",", ".", regex=False))


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n_all = len(df)
    v = {c: _num(df[c]) for c in NUMERIC}
    keep = ~v["no_couche"].isin(SKIP_LAYERS)
    df = df[keep]
    v = {c: s[keep] for c, s in v.items()}
    site_key = df["id_site"].astype(str).str.strip()
    layer = v["no_couche"].astype("Int64").astype(str)
    base = pd.DataFrame({"site_key": site_key, "layer_key": site_key + "/" + layer,
                         "depth_top": v["profondeur_hz_sup"], "depth_bot": v["profondeur_hz_inf"]})
    parts = [
        common.long_rows(base, "clay", v["argile"] / 10.0, "rmqs/argile", "pipette"),
        common.long_rows(base, "sand", (v["sable_fin"] + v["sable_grossier"]) / 10.0,
                         "rmqs/sable_fin+sable_grossier", "pipette"),
        common.long_rows(base, "soc", v["carbone_16_5_1"], "rmqs/carbone_16_5_1", "dry_combustion"),
        common.long_rows(base, "ph", v["ph_eau_6_1"], "rmqs/ph_eau_6_1", "ph_h2o"),
        common.long_rows(base, "cec", v["cec_40_1"], "rmqs/cec_40_1_metson", "cec_ph7"),
        *[common.long_rows(base, t, v[f"{t}_ech_40_3"], f"rmqs/{t}_ech_40_3_metson", "nh4oac_ph7")
          for t in ("ca", "mg", "k")],
    ]
    vals = pd.concat(parts, ignore_index=True)
    x, y = v["x_theo"].to_numpy(dtype=float), v["y_theo"].to_numpy(dtype=float)
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(x, y)
    year = pd.to_numeric(df["date_complete"].astype("string").str.extract(r"((?:19|20)\d\d)")[0], errors="coerce")
    sites = pd.DataFrame({
        "site_key": site_key,
        "lon": lon, "lat": lat,
        "coord_source": "x_theo/y_theo (Lambert-93 theoretical grid node)",
        "coord_precision_m": COORD_PRECISION_M,
        "year": year.to_numpy(),
    })
    labels, drops = common.finalize(vals, sites, source=SOURCE, prefix="rmqs", license=LICENSE)
    return labels, {"holorganic_layer_rows": int(n_all - keep.sum()), **drops}


def download(attempts: int = 5, wait_s: float = 30.0) -> Path:
    """The host often times out: retry with a growing pause between rounds of common.fetch.

    A cached file is used as is; otherwise a 15 s probe fails fast when the host is down, so a corpus build does not
    spend an hour in fetch's own long-timeout retries (2026-11-18).
    """
    cached = sorted(RAW.glob("*")) if RAW.exists() else []
    if not any(p.suffix != ".json" for p in cached):
        try:
            urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=15).close()
        except (OSError, IOError) as e:
            raise OSError(f"rmqs host unreachable: {e}") from e
    for i in range(attempts):
        try:
            return common.fetch(URL, RAW)
        except (OSError, IOError) as e:          # URLError and timeouts are OSErrors
            if i == attempts - 1:
                raise
            print(f"rmqs download failed ({e}); retry {i + 2}/{attempts} in {wait_s * (i + 1):.0f} s")
            time.sleep(wait_s * (i + 1))
    raise AssertionError("unreachable")


def load() -> tuple[pd.DataFrame, dict]:
    return build(read_raw(download()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=C.DATA_PROC / f"labels_{SOURCE}.parquet")
    common.write(*load(), ap.parse_args().out)


if __name__ == "__main__":
    main()
