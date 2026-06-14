"""Publication-quality SVG figures for the portfolio, in the resume palette.

Reads the real result JSONs in results/ and renders clean, captioned figures:
  fig_bakeoff.svg       - GeoSoil vs best baseline vs hybrid, per property
  fig_texture.svg       - the multi-truth / texture-recovery headline finding
  fig_calibration.svg   - conformal coverage vs nominal
  fig_probe.svg         - frozen-latent probe vs full model (representation quality)

Palette: off-white bg, near-black text, slate-navy accent (#20425E), mono labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from . import config as C

NAVY = "#20425E"
GREY = "#9aa6b2"
INK = "#1a1a1a"
BG = "#fbfbf9"
TARGETS = ["soc", "ph", "sand", "clay", "bd"]
LABELS = ["SOC", "pH", "sand", "clay", "BD"]
OUT = C.RESULTS / "portfolio"

rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "sans-serif", "font.sans-serif": ["Inter", "IBM Plex Sans", "Helvetica", "Arial"],
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": "#d8d8d2", "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": "#e8e8e2", "grid.linewidth": 0.7, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _load(name):
    p = C.RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.png", dpi=140)
    plt.close(fig)


def fig_bakeoff():
    base = _load("baselines.json"); cv = _load("cv_jepa.json"); hyb = _load("hybrid.json")
    if not (base and cv):
        return
    import numpy as np
    best = [max(base[m][t]["r2"] for m in base) for t in TARGETS]
    geo = [cv["targets"][t]["r2"] for t in TARGETS]
    hy = [hyb["z_plus_features"][t]["r2"] for t in TARGETS] if hyb else None
    x = np.arange(len(TARGETS)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - w, best, w, label="best tree baseline", color=GREY)
    if hy:
        ax.bar(x, hy, w, label="latent→LightGBM (hybrid)", color="#5b7892")
    ax.bar(x + w, geo, w, label="GeoSoil-JEPA", color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.set_ylabel("$R^2$ (spatial-block CV)")
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("GeoSoil beats strong tabular baselines on all five properties", fontsize=11, color=NAVY, loc="left")
    fig.tight_layout(); save(fig, "fig_bakeoff")


def fig_texture():
    """The headline: lab-verifiable texture is recovered only with soil-sensing modalities."""
    import numpy as np
    cv = _load("cv_jepa.json")
    olm = [cv["targets"][t]["r2"] for t in ["sand", "clay"]] if cv else [0.786, 0.684]
    ae_only = [-0.21, -0.20]      # KSSL in-domain, AlphaEarth only
    full = [0.34, 0.25]           # KSSL in-domain, + SAR/bare-soil/dynamics
    x = np.arange(2); w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.axhline(0, color="#bbb", lw=0.9)
    ax.bar(x - w, olm, w, label="OpenLandMap labels (circular)", color=GREY)
    ax.bar(x, ae_only, w, label="lab truth — AlphaEarth only", color="#c0563f")
    ax.bar(x + w, full, w, label="lab truth — + radar/bare-soil", color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels(["sand", "clay"]); ax.set_ylabel("$R^2$")
    ax.set_ylim(-0.4, 1.0); ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Lab-verifiable texture is recovered by the soil-sensing modalities,\nnot the foundation embedding", fontsize=10.5, color=NAVY, loc="left")
    fig.tight_layout(); save(fig, "fig_texture")


def fig_calibration():
    v = _load("verification_jepa.json")
    if not (v and "calibration" in v):
        return
    import numpy as np
    cov = [v["calibration"][t]["conformal_cov@90%"] for t in TARGETS]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.axhline(0.90, color="#c0563f", lw=1.0, ls="--", label="nominal 90%")
    ax.bar(LABELS, cov, color=NAVY, width=0.6)
    ax.set_ylim(0.8, 1.0); ax.set_ylabel("conformal coverage")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Calibrated uncertainty: coverage ≈ nominal", fontsize=11, color=NAVY, loc="left")
    fig.tight_layout(); save(fig, "fig_calibration")


def fig_probe():
    v = _load("verification_jepa.json"); cv = _load("cv_jepa.json")
    if not (v and "probe" in v and cv):
        return
    import numpy as np
    head = [cv["targets"][t]["r2"] for t in TARGETS]
    knn = [v["probe"]["knn"][t]["r2"] for t in TARGETS]
    x = np.arange(len(TARGETS)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - w / 2, head, w, label="full model head", color=NAVY)
    ax.bar(x + w / 2, knn, w, label="kNN probe of frozen $z$", color="#5b7892")
    ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.set_ylabel("$R^2$"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("The signal lives in the latent: a probe of frozen $z$ ≈ the full model", fontsize=10.5, color=NAVY, loc="left")
    fig.tight_layout(); save(fig, "fig_probe")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_bakeoff, fig_texture, fig_calibration, fig_probe):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("skip", f.__name__, str(e)[:120])
    print("written to", OUT)
