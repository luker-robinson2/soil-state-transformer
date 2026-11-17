"""Shared pieces for corpus source loaders: download cache, long-row building, QA."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from . import schema

RAW = C.REPO / "data" / "raw"
# Carbon in CaCO3 is 12/100 of its mass; SOC = total C - 0.12 * CaCO3 (both in %).
CACO3_C_FRACTION = 0.12


def fetch(url: str, dest: Path) -> Path:
    """Download once into data/raw/<source>/, recording url + sha256 in a sidecar json."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(6):                     # resume partial downloads (slow servers, dropped links)
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "geosoil-corpus/0.1"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                resumed = have and r.status == 206
                with open(tmp, "ab" if resumed else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            break
        except urllib.error.HTTPError as e:
            if e.code == 416:                    # already complete
                break
            if attempt == 5:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if attempt == 5:
                raise
    tmp.rename(dest)
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    meta = dict(url=url, sha256=sha, bytes=dest.stat().st_size,
                fetched=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    dest.with_suffix(dest.suffix + ".json").write_text(json.dumps(meta, indent=2))
    return dest


def num(s: pd.Series) -> pd.Series:
    """Numeric coercion that treats blanks and '<x' style flags as missing."""
    return pd.to_numeric(s, errors="coerce")


def long_rows(base: pd.DataFrame, target: str, value: pd.Series, method_id, method_family) -> pd.DataFrame:
    """One row per non-null value. `base` carries site_key, layer_key, depth_top, depth_bot.

    method_id / method_family may be a scalar, a Series aligned to `base`, or (family only)
    a function applied to method_id.
    """
    keep = value.notna()
    mid = method_id[keep] if isinstance(method_id, pd.Series) else pd.Series(method_id, index=base.index[keep])
    if callable(method_family):
        fam = mid.map(method_family)
    elif isinstance(method_family, pd.Series):
        fam = method_family[keep]
    else:
        fam = method_family
    out = base.loc[keep, ["site_key", "layer_key", "depth_top", "depth_bot"]].copy()
    out["target"] = target
    out["value"] = value[keep]
    out["method_id"] = mid.astype(str).values
    out["method_family"] = fam if np.isscalar(fam) else np.asarray(fam)
    return out


def finalize(vals: pd.DataFrame, sites: pd.DataFrame, *, source: str, prefix: str,
             license: str) -> tuple[pd.DataFrame, dict]:
    """QA-filter long values, attach site coordinates and return the canonical table.

    vals:  site_key, layer_key, depth_top, depth_bot, target, value, method_id, method_family
           [, license]
    sites: site_key, lon, lat [, coord_source, coord_precision_m, year, license]
    """
    n0 = len(vals)
    drops = {}

    bad_depth = vals["depth_top"].isna() | vals["depth_bot"].isna() | (vals["depth_bot"] <= vals["depth_top"])
    drops["bad_depth"] = int(bad_depth.sum())
    vals = vals[~bad_depth]

    in_range = pd.Series(True, index=vals.index)
    for t, (lo, hi) in schema.VALID_RANGE.items():
        in_range &= (vals["target"] != t) | vals["value"].between(lo, hi)
    drops["out_of_range"] = int((~in_range).sum())
    vals = vals[in_range]

    s = sites.copy()
    for col, default in (("coord_source", pd.NA), ("coord_precision_m", np.nan), ("year", pd.NA)):
        if col not in s:
            s[col] = default
    ok = s["lat"].between(-90, 90) & s["lon"].between(-180, 180) & ~((s["lat"] == 0) & (s["lon"] == 0))
    s.loc[~ok, ["lat", "lon"]] = np.nan
    site_cols = ["site_key", "lon", "lat", "coord_source", "coord_precision_m", "year"]
    if "license" in s and "license" not in vals:
        site_cols.append("license")
    vals = vals.merge(s[site_cols].drop_duplicates("site_key"), on="site_key", how="left")
    no_coord = vals["lat"].isna()
    drops["no_coordinates"] = int(no_coord.sum())
    vals = vals[~no_coord]

    vals = vals.drop_duplicates(["layer_key", "target", "method_id"])
    year = pd.to_numeric(vals["year"], errors="coerce").astype("Int64")
    out = pd.DataFrame({
        "site_id": prefix + ":" + vals["site_key"].astype(str),
        "layer_id": prefix + ":" + vals["layer_key"].astype(str),
        "source": source,
        "lon": vals["lon"],
        "lat": vals["lat"],
        "coord_source": vals["coord_source"],
        "coord_precision_m": vals["coord_precision_m"],
        "depth_top": vals["depth_top"],
        "depth_bot": vals["depth_bot"],
        "target": vals["target"],
        "value": vals["value"],
        "unit": vals["target"].map(C.TARGET_UNITS),
        "method_id": vals["method_id"],
        "method_family": vals["method_family"],
        "year": year.where(year.between(1900, pd.Timestamp.now().year)),
        "license": vals["license"] if "license" in vals else license,
    })
    out = schema.conform(out)
    schema.validate(out)
    drops["kept"] = len(out)
    drops["input_values"] = n0
    return out, drops


def carbon_family_from_text(text) -> str:
    """Classify a free-text carbon method description (WoSIS method_options etc.)."""
    t = "" if pd.isna(text) else str(text).lower()
    if re.search(r"walkley|dichromate|wet oxidation|wet combustion|tyurin|wet digestion", t):
        return "walkley_black"
    if re.search(r"dry combustion|dry oxidation|elemental|dumas|cn analy|c/n analy|leco", t):
        return "dry_combustion"
    if re.search(r"loss on ignition|\bloi\b|ignition", t):
        return "loss_on_ignition"
    return "unknown"


def write(labels: pd.DataFrame, drops: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out, index=False)
    print(f"{out}: {len(labels):,} values, {labels['site_id'].nunique():,} sites")
    print("dropped:", drops)
    print(labels.groupby(["target", "method_family"]).size().to_string())
