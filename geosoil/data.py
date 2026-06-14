"""Data layer: harmonize multiple ground-truth sources + modalities into one
master table, build spatial-block CV folds, and serve modality-aware tensors.

Sources on disk (CONUS, public):
  - OpenLandMap labels + AlphaEarth   (alphaearth-soc/future_work/data/soil_unified.csv)
  - Sentinel-2 spectral (paired)      (alphaearth-soc/future_work/data/s2_features.csv)
  - KSSL lab-measured truth + AE      (alphaearth-soc/future_work/data/kssl_with_features.csv)
Optional (added by GEE extraction, joined when present in data/processed/):
  - terrain/DEM static                (gee_static.parquet)
  - ERA5 climate + MODIS veg 12-mo    (gee_temporal.parquet)
  - RaCA independent SOC truth        (raca.parquet)
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Harmonization                                                               #
# --------------------------------------------------------------------------- #
def _key(df: pd.DataFrame, prec: int = 4) -> pd.Series:
    return df["lon"].round(prec).astype(str) + "_" + df["lat"].round(prec).astype(str)


def load_openlandmap() -> pd.DataFrame:
    """OpenLandMap targets + AlphaEarth, with Sentinel-2 joined where available."""
    df = pd.read_csv(C.DATA_RAW / "soil_unified.csv")
    out = pd.DataFrame({"lon": df["lon"], "lat": df["lat"]})
    out[list(C.AE_COLS)] = df[list(C.AE_COLS)]
    # Canonical targets + unit conversion
    out["soc"] = df["soc"]                # g/kg
    out["ph"] = df["ph_x10"] / 10.0       # pH x10 -> pH
    out["sand"] = df["sand_pct"]
    out["clay"] = df["clay_pct"]
    out["bd"] = df["bd"] / 100.0          # cg/cm3 -> g/cm3
    if "land_cover" in df:
        out["land_cover"] = df["land_cover"]
    # Join Sentinel-2 spectral (paired subset)
    s2 = pd.read_csv(C.DATA_RAW / "s2_features.csv")
    s2 = s2.drop_duplicates(subset=["lon", "lat"])
    out["_k"] = _key(out)
    s2["_k"] = _key(s2)
    out = out.merge(s2[["_k", *C.S2_COLS]], on="_k", how="left").drop(columns="_k")
    out["source"] = "openlandmap"
    out["split"] = "train"
    return out


def load_kssl() -> pd.DataFrame:
    """KSSL lab-measured truth + AlphaEarth (no Sentinel-2). Predefined split."""
    df = pd.read_csv(C.DATA_RAW / "kssl_with_features.csv")
    out = pd.DataFrame({"lon": df["lon"], "lat": df["lat"]})
    out[list(C.AE_COLS)] = df[list(C.AE_COLS)]
    out["soc"] = df["soc_g_kg"]
    out["ph"] = df["ph_h2o"]
    out["sand"] = df["sand_pct"]
    out["clay"] = df["clay_pct"]
    out["bd"] = df["bd_g_cm3"]
    out["source"] = "kssl"
    out["split"] = df.get("split", "test")
    return out


def load_raca() -> Optional[pd.DataFrame]:
    p = C.DATA_PROC / "raca.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["source"] = "raca"
    df["split"] = "test"
    return df


def _merge_gee(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join GEE-extracted terrain + temporal covariates when present."""
    for fname in ("gee_static.parquet", "gee_temporal.parquet", "gee_soil.parquet", "gee_dynamics.parquet"):
        p = C.DATA_PROC / fname
        if not p.exists():
            continue
        gee = pd.read_parquet(p)
        df["_k"] = _key(df)
        gee["_k"] = _key(gee)
        gee = gee.drop_duplicates(subset=["_k"])          # one row per join key (no fan-out)
        cols = [c for c in gee.columns if c not in ("lon", "lat", "_k")]
        df = df.merge(gee[["_k", *cols]], on="_k", how="left").drop(columns="_k")
    return df


def build_master() -> pd.DataFrame:
    """Concatenate all sources into one harmonized master table."""
    frames = [load_openlandmap(), load_kssl()]
    raca = load_raca()
    if raca is not None:
        frames.append(raca)
    master = pd.concat(frames, ignore_index=True)
    master = _merge_gee(master)
    # spatial block id for grouped CV (1deg x 1deg by default)
    b = C.TrainConfig.block_deg
    master["block"] = (np.floor(master["lon"] / b).astype(int).astype(str) + "_"
                       + np.floor(master["lat"] / b).astype(int).astype(str))
    return master


# --------------------------------------------------------------------------- #
# Modality availability                                                        #
# --------------------------------------------------------------------------- #
def modality_presence(master: pd.DataFrame) -> Dict[str, bool]:
    """Which optional modalities are populated in this master table."""
    def has_static(cols):
        return all(c in master for c in cols) and master[list(cols)].notna().any().any()
    has = {
        "s2": has_static(C.S2_COLS),
        "terrain": all(c in master for c in C.TERRAIN_COLS),
        "baresoil": has_static(C.BARESOIL_COLS),
        "sar": has_static(C.SAR_COLS),
        "precip": has_static(C.PRECIP_COLS),
        "cdl": has_static(C.CDL_COLS),
        "climate": all(f"{v}_00" in master for v in C.CLIMATE_VARS),
        "veg": all(f"{v}_00" in master for v in C.VEG_VARS),
        "moisture": all(f"{v}_00" in master for v in C.MOISTURE_VARS),
    }
    return has


# --------------------------------------------------------------------------- #
# Normalization                                                                #
# --------------------------------------------------------------------------- #
class Normalizer:
    """Z-score normalizer fit on training rows only (no leakage)."""

    def __init__(self):
        self.mean: Dict[str, float] = {}
        self.std: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, cols: List[str]) -> "Normalizer":
        for c in cols:
            v = pd.to_numeric(df[c], errors="coerce")
            m, s = float(np.nanmean(v)), float(np.nanstd(v))
            self.mean[c], self.std[c] = m, (s if s > 1e-8 else 1.0)
        return self

    def transform(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        out = np.zeros((len(df), len(cols)), dtype=np.float32)
        for j, c in enumerate(cols):
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32)
            out[:, j] = (v - self.mean[c]) / self.std[c]
        return out


def transform_targets(df: pd.DataFrame, norm: Normalizer) -> tuple[np.ndarray, np.ndarray]:
    """Return (y, mask) for the 5 targets, with log1p on skewed ones, z-scored."""
    n = len(df)
    y = np.zeros((n, len(C.TARGETS)), dtype=np.float32)
    mask = np.zeros((n, len(C.TARGETS)), dtype=np.float32)
    for j, t in enumerate(C.TARGETS):
        v = pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=np.float64)
        if t in C.LOG_TARGETS:
            v = np.log1p(np.clip(v, 0, None))
        key = f"_tgt_{t}"
        y[:, j] = ((v - norm.mean[key]) / norm.std[key]).astype(np.float32)
        mask[:, j] = (~np.isnan(v)).astype(np.float32)
    np.nan_to_num(y, copy=False)
    return y, mask


def fit_target_norm(df: pd.DataFrame, norm: Normalizer) -> Normalizer:
    for t in C.TARGETS:
        v = pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=np.float64)
        if t in C.LOG_TARGETS:
            v = np.log1p(np.clip(v, 0, None))
        m, s = float(np.nanmean(v)), float(np.nanstd(v))
        norm.mean[f"_tgt_{t}"], norm.std[f"_tgt_{t}"] = m, (s if s > 1e-8 else 1.0)
    return norm


def invert_target(y_norm: np.ndarray, t: str, norm: Normalizer) -> np.ndarray:
    """Map a normalized target column back to physical units."""
    v = y_norm * norm.std[f"_tgt_{t}"] + norm.mean[f"_tgt_{t}"]
    if t in C.LOG_TARGETS:
        v = np.expm1(v)
    return v


# --------------------------------------------------------------------------- #
# Torch dataset                                                                #
# --------------------------------------------------------------------------- #
def make_tensors(df: pd.DataFrame, norm: Normalizer, has: Dict[str, bool]) -> dict:
    """Build a dict of numpy modality arrays + presence masks + targets."""
    import numpy as np
    n = len(df)
    d: dict = {}

    d["ae"] = norm.transform(df, list(C.AE_COLS))
    # geo as raw normalized lon/lat in [-1,1]-ish (Fourier handles the rest)
    geo = np.stack([df["lon"].to_numpy(np.float32) / 180.0,
                    df["lat"].to_numpy(np.float32) / 90.0], axis=1)
    d["geo"] = geo.astype(np.float32)

    def block(cols, name):
        present = np.array([[1.0]] * n, dtype=np.float32)
        if has.get(name, False):
            arr = norm.transform(df, list(cols))
            pres = (~pd.to_numeric(df[cols[0]], errors="coerce").isna()).to_numpy(np.float32)[:, None]
            np.nan_to_num(arr, copy=False)
            return arr, pres
        return np.zeros((n, len(cols)), np.float32), np.zeros((n, 1), np.float32)

    d["s2"], d["s2_present"] = block(C.S2_COLS, "s2")
    d["terrain"], d["terrain_present"] = block(C.TERRAIN_COLS, "terrain")
    d["baresoil"], d["baresoil_present"] = block(C.BARESOIL_COLS, "baresoil")
    d["sar"], d["sar_present"] = block(C.SAR_COLS, "sar")
    d["precip"], d["precip_present"] = block(C.PRECIP_COLS, "precip")
    d["cdl"], d["cdl_present"] = block(C.CDL_COLS, "cdl")

    def seq(vars, name):
        if not has.get(name, False):
            return np.zeros((n, C.SEQ_LEN, len(vars)), np.float32), np.zeros((n, 1), np.float32)
        arr = np.zeros((n, C.SEQ_LEN, len(vars)), np.float32)
        for k, v in enumerate(vars):
            cols = [f"{v}_{i:02d}" for i in range(C.SEQ_LEN)]
            arr[:, :, k] = norm.transform(df, cols)
        np.nan_to_num(arr, copy=False)
        pres = (~pd.to_numeric(df[f"{vars[0]}_00"], errors="coerce").isna()).to_numpy(np.float32)[:, None]
        return arr, pres

    d["climate"], d["climate_present"] = seq(C.CLIMATE_VARS, "climate")
    d["veg"], d["veg_present"] = seq(C.VEG_VARS, "veg")
    d["moisture"], d["moisture_present"] = seq(C.MOISTURE_VARS, "moisture")

    y, ymask = transform_targets(df, norm)
    d["y"], d["y_mask"] = y, ymask
    return d


def fit_normalizer(train_df: pd.DataFrame, has: Dict[str, bool]) -> Normalizer:
    norm = Normalizer()
    cols = list(C.AE_COLS)
    if has["s2"]:
        cols += list(C.S2_COLS)
    if has["terrain"]:
        cols += list(C.TERRAIN_COLS)
    if has.get("baresoil"):
        cols += list(C.BARESOIL_COLS)
    if has.get("sar"):
        cols += list(C.SAR_COLS)
    if has.get("precip"):
        cols += list(C.PRECIP_COLS)
    if has.get("cdl"):
        cols += list(C.CDL_COLS)
    if has["climate"]:
        cols += [f"{v}_{i:02d}" for v in C.CLIMATE_VARS for i in range(C.SEQ_LEN)]
    if has["veg"]:
        cols += [f"{v}_{i:02d}" for v in C.VEG_VARS for i in range(C.SEQ_LEN)]
    if has.get("moisture"):
        cols += [f"{v}_{i:02d}" for v in C.MOISTURE_VARS for i in range(C.SEQ_LEN)]
    norm.fit(train_df, cols)
    fit_target_norm(train_df, norm)
    return norm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    C.DATA_PROC.mkdir(parents=True, exist_ok=True)
    master = build_master()
    out = C.DATA_PROC / "master.parquet"
    master.to_parquet(out, index=False)
    has = modality_presence(master)
    print(f"master.parquet: {master.shape} -> {out}")
    print("sources:", master["source"].value_counts().to_dict())
    print("modalities present:", has)
    print("targets non-null by source:")
    for src, g in master.groupby("source"):
        print(f"  {src}:", {t: int(pd.to_numeric(g[t], errors='coerce').notna().sum()) for t in C.TARGETS})
