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

**Serving rule:** a more specific tier is used only where it beats its parent on held-out data in that place
(regional over global, zone over regional, local over zone). Otherwise the parent's prediction stands. Every
tier reports a predictive interval with its measured coverage, not just a point value.

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
- `scripts/slurm/local_eval.sbatch` runs all three on Alpine; `--parts correction` runs the regional tier as a
  correction on the global model (honest inner-CV residuals; plan step 2).
- `geosoil/zones.py` (T2): zones by k-means on PCA of robust-scaled site covariates (k = 8, 16, 32); per-zone models,
  zone-as-feature, the blend, and leave-one-zone-out transfer (k = 8), all on the 1° folds. `scripts/slurm/zones.sbatch`.

## Metrics every tier reports

- R², RMSE and RPIQ, as now.
- **Interval coverage:** the share of held-out labels inside the stated 90% interval (split-conformal on spatial
  folds; Vovk et al. 2005).
- **Class agreement:** the share of predictions in the same class as the lab value, using public class bands:
  USDA pH reaction classes, SOC classes, and texture class from sand and clay. A prediction is useful when it lands
  in the right class, which R² alone does not show.
- **Held-out axes:** by block scale (above), leave-one-zone-out once zones exist, and leave-one-region-out.

## First results (round 4, 2026-11-17; `geosoil/results/local_eval_round4.json`)

Same rows and features (all groups, 150k rows per target), cross-validated at three block sizes. The median
distance from a validation site to the nearest training site is in brackets.

| | 1° (~31 km) | 0.25° (~12 km) | 0.05° (~5 km) | 0.05° + neighbours |
|---|---|---|---|---|
| SOC | 0.536 | 0.560 | 0.579 | 0.586 |
| pH | 0.601 | 0.623 | 0.644 | 0.651 |
| Sand | 0.412 | 0.459 | 0.495 | 0.513 |
| Clay | 0.365 | 0.418 | 0.462 | 0.473 |
| BD | 0.471 | 0.504 | 0.529 | 0.540 |

R² by distance to the nearest training site (0.05° folds):

| | 0–1 km | 1–5 km | 5–25 km | 25–100 km | 100–300 km |
|---|---|---|---|---|---|
| SOC | **0.71** | 0.62 | 0.52 | 0.49 | 0.51 |
| pH | 0.57 | 0.61 | 0.67 | 0.68 | 0.55 |
| Sand | **0.57** | 0.51 | 0.49 | 0.38 | 0.28 |
| Clay | **0.54** | 0.51 | 0.42 | 0.38 | 0.29 |
| BD | **0.62** | 0.55 | 0.51 | 0.50 | 0.46 |

Radius models (0.25° folds; only cells with ≥ 1,500 training rows within R, which is 15–30% of rows):

| | Global | Regional only | 50/50 blend |
|---|---|---|---|
| SOC, 150 km | 0.646 | 0.642 | **0.656** |
| pH, 400 km | 0.436 | 0.416 | **0.439** |
| Sand, 400 km | 0.451 | 0.462 | **0.470** |
| Clay, 150 km | 0.490 | 0.500 | **0.514** |
| BD, 150 km | 0.364 | 0.386 | **0.411** |

What this says:
- **Nearby labels are worth a lot, most of all for texture.** Moving from ~31 km to ~5 km from the nearest
  label adds +0.04 (pH, SOC, BD) to +0.10 (clay) R². Within 1 km, SOC reaches 0.71 and BD 0.62. This is the
  evidence for the local tier (T3).
- **Neighbour features start to help below ~10 km** (+0.007 to +0.018 at 5 km), and add nothing at 1° blocks.
- **pH is the exception:** it is hardest to predict right next to other labels (0.57 within 1 km vs 0.68 at
  25–100 km). The likely cause is management, such as liming, varying between neighbouring fields; this is not yet
  tested.
- **A regional model on its own is no better than the global one; blended with it, it is better everywhere**
  (+0.003 for pH to +0.047 for BD). T1 should be a correction on top of T0, which is plan step 2 below.

### Regional correction (round 5, job 35871129)

Plan step 2, learning the global model's honest residuals per region, does **not** beat the plain blend:

| (0.25° folds) | Global | Residual correction | 50/50 blend |
|---|---|---|---|
| SOC, 400 km | 0.659 | 0.659 | **0.671** |
| pH, 400 km | 0.454 | 0.447 | **0.459** |
| Sand, 400 km | 0.457 | 0.469 | **0.476** |
| Clay, 150 km | 0.503 | 0.489 | **0.521** |
| BD, 150 km | 0.358 | 0.286 | **0.394** |
| CEC, 400 km | 0.447 | 0.454 | **0.466** |

The correction helps sand and CEC only, and overfits where regional data are thin (BD at 150 km). The blend wins
everywhere, but a blend is also an ensemble of two models, so part of its gain may be plain ensembling rather than
locality. The next run adds a control: the global model blended with a model trained on a random subset of the
same size. Only the margin over that control counts as regional information.

### Edaphic zones (round 5, job 35871203)

Zones by k-means on 16 principal components of the site covariates, scored on the 1° folds:

| | Global | Zone as feature | Per-zone models (k=8) | Blend (k=8) | **Leave one zone out (k=8)** |
|---|---|---|---|---|---|
| SOC | 0.539 | 0.540 | 0.533 | 0.548 | **0.448** |
| pH | 0.608 | 0.608 | 0.602 | 0.615 | **0.489** |
| Sand | 0.425 | 0.426 | 0.425 | 0.440 | **0.244** |
| Clay | 0.376 | 0.375 | 0.370 | 0.389 | **0.248** |
| BD | 0.473 | 0.471 | 0.455 | 0.479 | **0.322** |

- Per-zone models lose to the global model, and lose more with more zones (k=16, 32). The zone as a feature adds
  nothing: the trees already split on the covariates the zones are built from.
- The blend gains (+0.004 to +0.015) match the regional blend and are probably ensembling; the control decides.
- **Transfer to an unseen pedo-climate is the real weakness:** holding out a whole zone costs 0.09 (SOC) to 0.18
  (sand) R². The model interpolates well inside the environments it has seen and extrapolates poorly outside them.
  Data acquisition should therefore weight new sites by how under-sampled their environment is, not just count them.
- So T2 becomes a distillation and serving tier (small per-zone students of the global model), not separately trained
  models; and zone coverage becomes a data-acquisition target.

### Regional blend control (round 7, job 35874901): the gain is locality

Blending the global model with a model trained on a random subset of the same size makes it **worse** for every
target; blending with the regional model makes it better. The difference is regional information:
SOC +0.020, pH +0.024, sand +0.059, clay +0.056, BD +0.092, CEC +0.049 (0.25° folds, 150 or 400 km; full table in
`geosoil/results/round7_summary.json`). T1 is therefore a blend of the global model with a radius model where
labels are dense, not a residual correction and not plain ensembling.

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

## Selective tiers (round 7, `geosoil/tiers.py`)

The rule is nested: use the regional blend in a held-out 0.25° cell only if its local training density meets a
threshold chosen on the other folds. Gains are MSE reductions vs global, with 95% cell-bootstrap intervals.

- **150 km:** significant for SOC (4.9% [3.7, 6.0]), pH (2.1%), sand (3.8%), clay (4.1%), CEC (3.4%) and K (5.8%).
  BD reaches 7.7% but is noisy (p = 0.08).
- **400 km:** weaker for clay, BD and pH, where the interval includes 0.
- **Locality:** the blend beats the same-size random-subset control by 5–16% on blended cells, so the gain comes
  from locality, not from averaging two models.
- **The selected rule blends in 84–100% of evaluated cells.** These cells are already the ones dense enough for
  a regional model, so density hardly discriminates among them.

Serving rule:
- 150 km regional blend wherever a regional model can be fitted (enough labels within the radius).
- Global model everywhere else.
- Sand may use 400 km.
- Never the residual correction or per-zone models, both of which lost.
