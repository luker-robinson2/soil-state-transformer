# Direction 1 — Multi-Target Soil Property Extension

**Question:** Do AlphaEarth embeddings encode the full soil property space,
or only SOC?

**Hypothesis:** AlphaEarth's training inputs (Sentinel-1, Sentinel-2,
Landsat, climate, GEDI) carry information about surface roughness,
moisture, vegetation, and topography — all of which correlate with pH,
texture, and bulk density via long-established pedological relationships
(jenny 1941; mcbratney 2003). Predictive skill should be highest for
properties most strongly tied to surface signals (pH via vegetation
community, sand/clay via reflectance + drainage) and lowest for those
most decoupled (deep bulk density, CEC).

## Targets

Five OpenLandMap surfaces, all available as GEE assets and all at 0 cm
depth (or 0–30 cm where the asset is depth-stratified):

| Target          | OpenLandMap asset                                                | Units  | Expected $R^2$ |
|-----------------|------------------------------------------------------------------|--------|----------------|
| SOC             | `OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02`             | g/kg   | 0.75 (baseline)|
| pH (H2O)        | `OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02`                   | pH × 10| 0.55–0.65      |
| Sand %          | `OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02`           | %      | 0.65–0.75      |
| Clay %          | `OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02`           | %      | 0.55–0.65      |
| Bulk density    | `OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02`         | kg/m³  | 0.40–0.55      |

The expected $R^2$ values are guesses based on the pedological priors
above and on the literature for similar Sentinel-2-only DSM benchmarks
(Vaudour 2022; Žížala 2022). They're worth pre-registering as a sanity
check on the model's behavior.

## Plan

### Step 1 — Extend the data extractor (~30 lines)

Modify `future_work/scripts/01_extract_multi.py` (copy of
`../../01_extract_data.py`) to add the four new soil bands to the GEE
stack. Re-run with `seed=42` so that the same 3000 points are sampled.

```python
ph   = ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02").select(["b0"], ["ph"])
sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select(["b0"], ["sand_pct"])
clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select(["b0"], ["clay_pct"])
bd   = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select(["b0"], ["bd"])
stack = ae.addBands([soc, ph, sand, clay, bd, lc])
```

Output: `future_work/data/soil_multi_target.csv`, ~3000 rows × ~72 cols.

### Step 2 — Per-target analysis (~150 lines of R)

`future_work/scripts/multi_target_analysis.Rmd`:

1. **Distribution and transform per target** — pH is roughly normal
   (no transform); sand/clay are bounded [0,100] (logit transform);
   BD is normal-ish; SOC keeps `log1p`.
2. **Univariate correlation matrix** — 64-dim AlphaEarth × 5 targets.
   Reproduce the bar chart from the original paper, faceted by target.
3. **Same hypothesis test as the original paper** — Welch's two-sample
   $t$-test of each target between Grasslands and Croplands. Tabulate
   $t$, $p$, Cohen's $d$ for all five targets.
4. **Per-target spatial-block-CV Random Forest** — fit one RF per target
   on the 64-dim embedding. Report mean and 95% CI of $R^2$ across folds.
5. **Multi-output Random Forest** — fit `ranger` with multiple outputs
   (or use `MultiOutputRegressor` in Python equivalent). Compare to the
   per-target RFs to see whether joint training helps.

### Step 3 — Headline figure & paper section

A 5-panel figure: predicted-vs-observed scatter for each target on the
same scale, ordered by descending $R^2$. The story this tells:
"AlphaEarth's signal generalises beyond SOC, with predictable strength
ordering — sand > SOC > pH > clay > BD."

## Production-integration notes

- **Schema extension** — a single-nutrient-per-column table would extend with
  `pH_H2O`, `BulkDensity_kg_m3`, `Sand_pct`, `Clay_pct` columns, plus unit
  handling in the lab-result normalization step for pH (no scaling) and bulk
  density (kg/m³ vs g/cm³).
- **Trainer extension** — a trainer that fits one model per target sequentially
  could move to a multi-output RF (a small change), though per-target ensembles
  are probably cleaner. Either way the spatial GroupKFold logic ports unchanged.
- **Predictor / API** — multi-target prediction needs a `property_name` field on
  outputs so the API isn't implicitly SOC-only.

## Risks & mitigations

- **Skewed targets violate normality for the $t$-test.** Sand/clay/BD all
  have bounded or asymmetric distributions; apply the appropriate
  transform (logit for fractions, log for SOC) before any $t$-test, and
  bootstrap CIs as a backstop.
- **Targets are highly correlated** (e.g., sand and clay sum to ~95%).
  Don't double-count. Either drop clay or report sand/(sand+clay) as a
  single texture summary.
- **OpenLandMap has variable native resolution per band.** All five are
  250 m, but the publication dates differ; the script samples all at
  scale=250 m which is correct.

## Estimated effort

- **Step 1 (extract):** 30 min coding, 30 min re-running GEE.
- **Step 2 (analysis):** 4 hours including paper-quality figures.
- **Step 3 (write-up):** 2 hours.
- **Total:** ~1 day. Lowest-risk of the three directions; do this first.
