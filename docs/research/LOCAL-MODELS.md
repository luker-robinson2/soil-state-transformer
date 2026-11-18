# Model tiers: global, regional, edaphic zones, local

GeoSoil is one foundation with several tiers of models built on it. Soil varies at every scale, and a model
that is good at continental extrapolation is not automatically the best model where labels are dense. The
tiers below share the same features, labels and code. They differ in **which data they train on** and
**which question their evaluation answers**.

| Tier | Trains on | Answers | Evaluated with |
|---|---|---|---|
| **T0 Global foundation** | every labelled site, plus unlabelled AlphaEarth pixel-years for pretraining | "What is the soil here, with no labels nearby?" | 1° block CV (~100 km), the fixed 1° test blocks |
| **T1 Regional (radius)** | training sites within R km of a region (R = 150–500 km), alone or as a correction on T0 | "How well can a model specialised to this region do?" | 0.25° block CV (~25 km); skill by distance to training |
| **T2 Edaphic zones** | sites grouped into zones of similar soil-forming conditions; one head per zone, distilled from T0 | "Does a model per pedo-climatic zone beat one model for everything?" | zone-blocked CV; skill per zone |
| **T3 Local** | a site's own neighbourhood (≤ 5 km): neighbouring labels, residual interpolation | "How much do labels a few km away add?" | 0.05° block CV (~5 km); skill by distance |

## Why the evaluation changes per tier

Spatial CV with large blocks (1°) puts validation sites tens of kilometres from any training label. That is
the right test for T0, and it is why neighbouring-label features add nothing there (round 4: SOC 0.543 →
0.543, pH 0.602 → 0.603): at that distance they repeat what the satellite and climate features already say.
Smaller blocks leave labels closer, so scores rise. Those scores are not inflated or wrong, but they answer a
different question. Every result therefore states its block scale, and local skill is reported **by
distance to the nearest training site** (bins 0–1, 1–5, 5–25, 25–100, 100–300, > 300 km), following Roberts
et al. 2017 (Ecography 40:913) on block CV for spatially structured data.

## What is built (2026-11-17)

- `geosoil/corpus/splits.py`: folds at 1° (`fold`), 0.25° (`fold_025`) and 0.05° (`fold_005`), all inside
  the same training split; the test blocks stay the 1° ones.
- `geosoil/local_eval.py`:
  - `scales`: the global model at each block scale, with and without neighbour features
  - `distance`: out-of-fold R² by distance to training
  - `regional`: radius models per 2° anchor cell, scored beside the global model and a 50/50 blend on the same rows
- `scripts/slurm/local_eval.sbatch` runs all three on Alpine.

## Plan

1. **Measure** (now): the scale curve, skill by distance, and regional vs global at R = 150 and 400 km.
   If regional models beat the global one in some regions and lose in others, that is the case for T1/T2.
2. **Regional correction**: instead of a standalone regional model, fit a regional model to the global
   model's out-of-fold residuals (inner CV so the residuals are honest). This is the usual way a broad
   model is specialised locally (regression kriging and its ML variants; Hengl et al. 2004, Geoderma 120:75).
3. **Edaphic zones** from published methods only: cluster sites and unlabelled pixels in the space of the
   foundation embedding plus climate, lithology and terrain (fuzzy k-means, as in McBratney & de Gruijter
   1992, J. Soil Sci. 43:159; pedoregions as in Hole 1978 and the USDA MLRA concept). Choose the number of
   zones by held-out skill, not by cluster indices.
4. **Distillation**: per-zone and per-region students trained on the global teacher's predictions over the
   unlabelled pixels in their zone (the 39.7M pixel-years already extracted), then calibrated on that zone's
   labels. Small, fast models per zone that carry the foundation's knowledge.
5. **Local tier**: neighbour features and residual interpolation, evaluated by distance. This is where
   dense sampling pays off, and where field-scale use lives.

## Scope

This repository uses public data and published methods only. Product-specific zone delineation or sampling
design from any commercial platform is out of scope here and must not be reproduced; see CONTRIBUTING.md.
