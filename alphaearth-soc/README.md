# AlphaEarth → Soil Organic Carbon

> Do AlphaEarth Foundations embeddings carry usable signal for Soil Organic
> Carbon (SOC), despite never being trained on a soil target?

Originally a STAT 5000 graduate-statistics project (April 2026). Built entirely on
**public data** and **safe to publish**.

## Result
- Several of the 64 AlphaEarth dims correlate with log(1+SOC) up to **|r| = 0.61**
  (BCa 95% CI `[-0.628, -0.581]`).
- Mean log-SOC differs significantly between **Croplands and Grasslands**
  (Welch's two-sample t).
- A spatial-block cross-validated **Random Forest reaches R² = 0.75** out-of-fold —
  the embeddings encode subsurface-relevant signal beyond any single dimension.

## Layout
```
alphaearth-soc/
├── paper/        main.tex · main.pdf · references.bib   ← the write-up
├── analysis/     soil_analysis.Rmd · soil_analysis.pdf · 01_extract_data.py
├── figures/      EDA, PCA, bootstrap, RF importance/prediction plots
├── data/         soil_samples.csv  (n=3000, AlphaEarth + OpenLandMap + MODIS)
└── future_work/  3 follow-on directions (multi-target, S2 baseline, KSSL truth)
```

## Reproduce
1. `analysis/01_extract_data.py` — pulls the n=3000 sample from Google Earth Engine
   (requires a GEE-authenticated Google account).
2. `analysis/soil_analysis.Rmd` — knit in R to regenerate every figure and the
   statistics in the paper.
3. `paper/main.tex` — compile with `pdflatex` + `bibtex`.

## Data sources (all public)
- **AlphaEarth Foundations** 2024 annual embedding (Google DeepMind, in GEE)
- **OpenLandMap** SOC at 0 cm (ISRIC)
- **MODIS** IGBP land cover

## `future_work/`
Extends the single-target SOC result into a publishable multi-target study —
see [`future_work/README.md`](future_work/README.md). Headline so far
(`future_work/data/phaseF_headline.csv`): AlphaEarth-only RF on OpenLandMap labels
R² ≈ 0.75, vs ≈ 0.32 when validated against noisier WoSIS point labels — a useful
lesson on label quality.
