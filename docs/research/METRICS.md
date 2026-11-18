# How GeoSoil models are evaluated

**Short answer: R² is the right metric for one job only, ranking models on exactly the same test rows.** On a
fixed test set the total variance is a constant, so R² = 1 − MSE/s² orders models exactly as MSE does. Across
different rows, R² mixes "the model changed" with "the test set's variance changed", and that misleads. GeoSoil
therefore reports skill in physical units against the lab noise floor, and compares models by paired error
differences with spatial uncertainty. Code: `geosoil/metrics.py` (functions) and `geosoil/scorecard.py` (applies
them to saved out-of-fold predictions).

## Where R² misled us

| Case | What happened | Why |
|---|---|---|
| Same model, two row sets | identical predictions scored pH 0.62 on all rows and 0.40 on the dense-cell subset | R² divides by the subset's variance; the dense subset has less pH spread |
| Rounds 5 → 7 | CEC 0.405 → 0.385 and K 0.491 → 0.436 as rows doubled | rows, label methods and test variance all changed; R² cannot attribute the drop |
| The "ceiling" | "0.47 of 0.92" | the noise-floor R² divides by US-only KSSL variance, the model R² by global variance: two different baselines |
| Distance bins | pH best at 25–100 km, worst within 1 km | without RMSE per bin, a low R² may just be a low-variance bin |

## The suite, one metric per purpose

| Purpose | Metric | Why |
|---|---|---|
| Is model B better than model A? | **Paired difference in MSE** on shared rows, with a 1° block-bootstrap CI and a sign-flip randomization p-value over blocks (`paired_compare`) | Same rows, so the comparison is exact; blocks because errors are spatially correlated, so rows are not iid. Multiple ablations: Holm or Benjamini–Hochberg |
| How accurate, in units people use? | **RMSE, MAE, mean error** with block-bootstrap CIs (`point_metrics`, `block_bootstrap`) | Comparable across regions and runs; ME exposes bias that squared Pearson r hides |
| How close to what is achievable? | **Floor skill** (s² − MSE)/(s² − τ²) on the same rows, and **excess RMSE** √(MSE − τ²) (`floor_skill`) | τ² is the lab + micro-scale noise from co-located replicates; uses the test rows' own s², fixing the ceiling mismatch |
| Is the prediction usable? | **Class agreement** (exact, within one) and **quadratic-weighted kappa** on public bands (`weighted_kappa`) | Agronomic decisions use classes; kappa corrects for chance and penalises far misses more |
| Is the uncertainty honest? | **Coverage** of split-conformal 90% intervals and the **interval (Winkler) score** (`interval_score`) | Coverage alone rewards infinitely wide intervals; the interval score is a proper scoring rule (Gneiting & Raftery 2007) |
| What kind of error? | **Bias / scale / correlation** split of the MSE (Theil), Lin's **CCC**, calibration slope | Says whether to fix bias, spread or signal |

R² (model efficiency) is still reported, alongside these, and used for ranking within one row set. Squared Pearson
r is never used alone: it is blind to bias and scale.

## Statistical practice

- **Confidence intervals** resample whole 1° blocks (spatial block bootstrap). As a robustness check, resample by
  source × country too.
- **Training variability** (a different fold assignment would give a different model) is not in a test-set
  bootstrap. When it matters, repeat CV with re-hashed blocks and use the Nadeau–Bengio corrected resampled t.
  The naive t over 5 folds has 4 degrees of freedom and overstates confidence.
- **Log targets** (SOC, OM, CEC, K, Ca, Mg). expm1 of the prediction is the conditional median, which is biased low
  for the mean. Rank models on the log scale. Report classes and MAE with the median, and anything that sums
  (carbon stocks) with the Duan-smeared mean. Intervals transform back exactly.
- **Spatial CV scale is part of the claim.** Every score states its block size (1°, 0.25°, 0.05°). Local skill is
  reported by distance to the nearest training label (LOCAL-MODELS.md).
- **Map accuracy** over a region needs a probability sample with design weights; RaCA (held out, CONUS) is that
  sample for SOC and BD.

## Lineage

This is standard regression diagnostics and resampling (residual analysis, heteroscedasticity, transformation and
back-transformation bias, bootstrap and randomization inference, family-wise error control), applied to spatial
data. Digital-soil-mapping validation literature: Wadoux et al. 2021 (Ecological Modelling 457) and 2022 (Geoderma
405); Piikki et al. 2021 (Soil Use and Management 37:7); Brus, Kempen & Heuvelink 2011 (EJSS 62:394); Meyer &
Pebesma 2021, 2022; Lin 1989 (CCC); Duan 1983 (smearing); Nadeau & Bengio 2003; Gneiting & Raftery 2007.
Operational soil-mapping practice reached the same place: class agreement as the headline, skill measured against
lab error rather than R², and stated intervals checked for coverage.

## Round 7 scorecard (what the suite showed)

`geosoil/results/scorecard_round7.json`. 95% intervals come from resampling 1° blocks; SOC, OM, CEC and bases are on
the log scale.

| Target | RMSE [95% CI] | R² [95% CI] | Lin's CCC | Floor skill | Weighted κ |
|---|---|---|---|---|---|
| SOC | 0.673 [0.660, 0.687] log | 0.553 [0.520, 0.584] | 0.708 | 0.63 | 0.69 |
| pH | 0.790 [0.778, 0.802] | 0.618 [0.600, 0.636] | 0.761 | 0.68 | 0.75 |
| Sand | 18.3 [18.0, 18.6] % | 0.469 [0.445, 0.487] | 0.629 | 0.51 | 0.52 |
| Clay | 12.4 [12.1, 12.6] % | 0.398 [0.379, 0.417] | 0.559 | 0.44 | 0.55 |
| BD | 0.218 [0.213, 0.224] g/cm³ | 0.505 [0.484, 0.527] | 0.671 | 0.59 | 0.60 |
| CEC | 0.571 [0.557, 0.584] log | 0.385 [0.356, 0.413] | 0.550 | 0.44 | — |
| K | 0.276 [0.265, 0.286] log | 0.436 [0.415, 0.460] | 0.609 | 0.51 | — |

What it says that R² alone did not:

- **No bias, good calibration, under-dispersed on purpose.** Bias and scale together are under 0.3% of the MSE
  for every target, and the calibration slope is 1.00–1.05. All remaining error is unexplained signal. CCC sits
  well below r because MSE-optimal predictions shrink toward the mean. Maps that must show realistic spread
  (class maps) need a different estimator, not a better one.
- **Floor skill 0.44–0.68.** Even pH, the best target, captures two thirds of the error reduction that repeat
  lab measurements would allow.
- **The same model scores very differently by region.** SOC R² is 0.60 in South America, 0.51 in CONUS, 0.39 in
  Canada (RMSE nearly double) and 0.20–0.21 in South Asia and Russia. Clay R² is 0.12 in Mexico and Central
  America.
- **Paired tests settle the method question.** On standard-method rows, training on standard-method rows only
  lowers MSE by 1.4–1.7% for CEC, K, Ca and Mg. Each block sign-flip test gives p ≤ 0.013 and each CI excludes 0.
  The raw R² (Mg 0.469 vs 0.529) looked like a 6-point jump only because the row sets differ.
