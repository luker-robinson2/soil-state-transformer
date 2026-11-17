"""Land-use history for every labelled site: how long and how intensively it has been farmed (Phase 2).

Cultivation strips soil carbon and compacts soil (Sanderman et al. 2017, PNAS 114:9575: a 116 Pg C
"soil carbon debt" from land use), so a site's history matters for SOC and bulk density even when
today's land cover looks the same.

Source: HYDE 3.2.1 (Klein Goldewijk et al. 2017), DANS data station, CC0. 5 arc-minute ESRI ASCII
grids of km^2 per cell, decadal 1700-2000 and annual to 2017. The archive is a single 5.3 GB zip;
only the members needed are pulled, by HTTP range reads of the zip (the signed download URL is
re-resolved on each run because it expires after an hour). Members are Deflate64-compressed,
which needs the zipfile-deflate64 package. Values are turned into fractions using
the exact area of the 5' cell at each site's latitude.

Columns (prefix hist_):
  hist_crop_<year>      cropland fraction in 1850, 1900, 1950, 2000, 2017
  hist_graz_<year>      grazing-land fraction in 1950, 2017
  hist_crop_max         peak cropland fraction 1700-2017
  hist_crop_years       years with cropland fraction > 0.1 since 1700 (decadal resolution)
  hist_years_since_conv years since cropland first exceeded 0.1 (NaN = never)

    python -m geosoil.features.static --source landuse
"""
from __future__ import annotations

import io
import math
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..corpus import common
from . import shards
from .rasters import Layer, sample_layer

DATAFILE = "https://archaeology.datastations.nl/api/access/datafile/5490328"   # HYDE3_2_1-baseline.zip
DECADES = list(range(1700, 2001, 10)) + [2010, 2017]
CROP_YEARS = (1850, 1900, 1950, 2000, 2017)
GRAZ_YEARS = (1950, 2017)
THRESH = 0.1
EARTH_R_KM = 6371.0072
CELL_DEG = 5 / 60


def member(kind: str, year: int) -> str:
    return f"baseline/asc/{year}AD_lu/{kind}{year}AD.asc"


def cell_area_km2(lat: np.ndarray) -> np.ndarray:
    """Exact area of the 5-arc-minute cell containing each latitude."""
    lo = np.floor(np.asarray(lat) / CELL_DEG) * CELL_DEG
    hi = lo + CELL_DEG
    return EARTH_R_KM ** 2 * math.radians(CELL_DEG) * np.abs(np.sin(np.radians(hi)) - np.sin(np.radians(lo)))


class _HTTPRange(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP Range requests (for zipfile)."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}))
        self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        d = urllib.request.urlopen(urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})).read()
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def _signed_url() -> str:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    try:
        urllib.request.build_opener(NoRedirect).open(urllib.request.Request(DATAFILE, method="HEAD"))
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return e.headers["Location"]
        raise
    return DATAFILE


def fetch_members(names: list[str], dest: Path = common.RAW / "landuse") -> dict[str, Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for n in names:
        p = dest / Path(n).name
        (out.__setitem__(n, p) if p.exists() and p.stat().st_size > 0 else todo.append(n))
    if todo:
        import zipfile_deflate64  # noqa: F401  the HYDE zip uses Deflate64 (method 9), patched into zipfile
        z = zipfile.ZipFile(io.BufferedReader(_HTTPRange(_signed_url()), buffer_size=1 << 22))
        for n in todo:
            p = dest / Path(n).name
            tmp = p.with_suffix(".part")
            with z.open(n) as src, open(tmp, "wb") as dst:
                while chunk := src.read(1 << 22):
                    dst.write(chunk)
            tmp.rename(p)
            out[n] = p
    return out


def derive(crop: dict[int, np.ndarray], graz: dict[int, np.ndarray]) -> pd.DataFrame:
    """crop/graz: year -> fraction per site."""
    years = sorted(crop)
    stack = np.vstack([crop[y] for y in years]).T                     # sites x years
    out = {f"hist_crop_{y}": crop[y] for y in CROP_YEARS if y in crop}
    out.update({f"hist_graz_{y}": graz[y] for y in GRAZ_YEARS if y in graz})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)              # all-NaN rows: sites off the grid
        out["hist_crop_max"] = np.nanmax(stack, axis=1)
    above = stack > THRESH
    # each decadal step represents the 10 years that follow it (the last steps: to 2017 / 2025)
    step = np.diff(years + [years[-1] + 8])
    out["hist_crop_years"] = (above * step).sum(axis=1).astype(float)
    first = np.where(above.any(axis=1), np.asarray(years)[above.argmax(axis=1)], np.nan)
    out["hist_years_since_conv"] = np.where(np.isnan(first), np.nan, 2017 - first)
    allnan = np.isnan(stack).all(axis=1)
    df = pd.DataFrame(out)
    df.loc[allnan, :] = np.nan
    return df


def extract(sites: pd.DataFrame, out_root: Path = C.DATA_PROC / "features") -> dict:
    t0 = time.time()
    want = [member("cropland", y) for y in DECADES] + [member("grazing", y) for y in GRAZ_YEARS]
    files = fetch_members(want)
    lon, lat = sites["lon"].to_numpy(float), sites["lat"].to_numpy(float)
    area = cell_area_km2(lat)
    def frac(name):
        v = sample_layer(str(files[name]), Layer("x", str(files[name]), scale=1.0, offset=0.0), lon, lat)
        return np.clip(v / area, 0, 1)
    crop = {y: frac(member("cropland", y)) for y in DECADES}
    graz = {y: frac(member("grazing", y)) for y in GRAZ_YEARS}
    df = derive(crop, graz)
    df.insert(0, "site_id", sites["site_id"].to_numpy())
    shards.write(df, out_root / "landuse", 0, 1, {
        "source": DATAFILE, "licence": "CC0 (HYDE 3.2.1, Klein Goldewijk et al. 2017)",
        "members": want, "file_sha256": {Path(k).name: shards.sha256(v) for k, v in files.items()},
        "failed_blocks": [], "coverage": float(df["hist_crop_max"].notna().mean()), "sites": int(len(df)),
        "seconds": round(time.time() - t0, 1)})
    return {"source": "landuse", "sites": len(df), "coverage": float(df["hist_crop_max"].notna().mean()),
            "seconds": round(time.time() - t0, 1)}
