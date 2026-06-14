"""Consolidate all result JSONs into one markdown summary (for the README)."""
from __future__ import annotations

import json

from . import config as C

TARGETS = list(C.TARGETS)


def _load(name):
    p = C.RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def r2row(label, d):
    cells = []
    for t in TARGETS:
        v = d.get(t, {}).get("r2") if d else None
        cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "—")
    return f"| {label} | " + " | ".join(cells) + " |"


def main():
    print("### Spatial-block CV R² (out-of-fold, OpenLandMap labels)\n")
    print("| model | " + " | ".join(TARGETS) + " |")
    print("|" + "---|" * (len(TARGETS) + 1))
    base = _load("baselines.json") or {}
    for name in ("ridge", "rf", "lightgbm", "xgboost", "catboost", "tabpfn"):
        if name in base:
            print(r2row(name, base[name]))
    hyb = _load("hybrid.json") or {}
    if "z_only" in hyb:
        print(r2row("hybrid (z→LGBM)", hyb["z_only"]))
        print(r2row("hybrid (z+feat→LGBM)", hyb["z_plus_features"]))
    cv = _load("cv_jepa.json")
    if cv:
        print(r2row("**GeoSoil-JEPA**", cv["targets"]))
    ebm = _load("ebm.json")
    if ebm and "targets" in ebm:
        print(r2row("GeoSoil-EBM head", ebm["targets"]))

    ver = _load("verification_jepa.json")
    if ver:
        print("\n### Multi-truth validation\n")
        kt = ver.get("kssl_transfer", {}).get("targets", {})
        ki = ver.get("kssl_internal", {})
        print(r2row("OpenLandMap→KSSL (transfer)", kt))
        if "lightgbm" in ki:
            print(r2row("KSSL→KSSL in-domain (LGBM, AE)", ki["lightgbm"]))
        print("\n### Frozen-latent probe R²\n")
        pr = ver.get("probe", {})
        if "linear" in pr:
            print(r2row("linear probe", pr["linear"]))
            print(r2row("kNN probe", pr["knn"]))
        print("\n### Cross-modal retrieval & calibration\n")
        print("retrieval:", ver.get("retrieval"))
        print("calibration:", ver.get("calibration"))


if __name__ == "__main__":
    main()
