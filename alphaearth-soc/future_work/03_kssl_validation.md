# Direction 3 — KSSL Ground-Truth Validation

**Question:** When we replace OpenLandMap (an interpolated machine-learning
product) with USDA KSSL (actual lab-measured pedons), does AlphaEarth's
predictive skill survive?

**Why this matters most:** The original paper's most uncomfortable
limitation is that both predictors and labels are model outputs. A
field-truth result is what makes this work scientifically defensible
rather than a model-vs-model alignment study. It's also the hardest of
the three directions — most of the engineering effort goes into
properly ingesting and harmonizing the KSSL data.

## Why KSSL, not SSURGO or WoSIS

From the data-source research:

- **SSURGO** is **polygon-based, expert-aggregated estimates** — not lab
  measurements. Using it as ground truth would be circular if AlphaEarth
  was trained on similar interpolated surfaces. Useful only as a
  stratification covariate.
- **WoSIS** is largely a re-ingestion of KSSL for CONUS, with extra
  harmonization metadata but a CC BY 4.0 license that complicates
  publication and adds no new measurements.
- **KSSL** (Kellogg Soil Survey Laboratory, part of the NCSS Soil
  Characterization Database) is the **gold-standard CONUS source of
  point-located lab measurements** — Walkley-Black or dry-combustion
  organic C, particle size, pH, bulk density, by horizon, ~45-55k
  pedons in CONUS, public domain.

## Plan

### Step 1 — KSSL ingestion module (~250 lines)

`future_work/scripts/03_ingest_kssl.py` (parallel to
`01_extract_data.py`):

1. **Download the KSSL SQLite snapshot** from the NCSS Lab Data Mart:
   <https://ncsslabdatamart.sc.egov.usda.gov/datadownload.aspx>. One
   file, ~500 MB. Cache locally; do not re-download per run.
2. **Filter to clean pedons.** Apply these conjunctive filters:
   - In CONUS bounding box (already used in
     `01_extract_data.py:conus_geometry()`).
   - **Sample year ≥ 2010** (minimize temporal mismatch with AlphaEarth
     2024).
   - **Coordinate quality flag = GPS** (drop PLSS-derived coords with
     ±100 m–±1 km uncertainty; check the relevant location-quality
     column — verify exact column name against the schema docs).
   - **Dry-combustion SOC method** (Walkley-Black is biased low and
     would require a conversion factor we'd rather avoid).
   - Has at least one horizon spanning 0–30 cm.

   Expected yield: 5,000–15,000 pedons (verify after filtering).

3. **Depth-harmonize each pedon to 0–30 cm** using thickness-weighted
   means. For each property (SOC %, BD, sand, clay, pH):

   ```
   prop_0_30 = sum(prop_h * thickness_h) / 30   # over horizons
                                                 # intersecting [0, 30 cm]
   ```

   This is the standard mass-preserving aggregation. A spline-based
   alternative (Bishop et al. 1999, equal-area splines) is more accurate
   but more code; defer unless results are noisy.

4. **Compute SOC stock (Mg C / ha)** from SOC % × BD × 30 cm × (1 −
   coarse-fragment fraction). This is the more interpretable target than
   raw SOC %.

5. **Output `future_work/data/kssl_points.csv`** with columns:
   `pedon_key, lon, lat, sample_year, soc_pct_0_30, soc_stock_mg_ha,
   ph_h2o_0_30, bd_0_30, sand_pct_0_30, clay_pct_0_30,
   coord_quality, soc_method`.

### Step 2 — Pair KSSL with AlphaEarth in GEE

`future_work/scripts/04_extract_kssl_alphaearth.py`:

1. Upload the KSSL CSV as a **GEE FeatureCollection** asset (one-time
   upload via the Code Editor or `earthengine upload table`).
2. `sampleRegions` over the AlphaEarth 2024 image at scale=10 m —
   **higher than the 250 m used for OpenLandMap**, since KSSL points are
   exact, not pixel-aggregated.
3. For comparison, also extract the S2 feature stack from #2 and
   OpenLandMap SOC at the same points.

Output: `future_work/data/kssl_with_features.csv`,
~5–15k rows × ~94 cols (lon/lat + 64 AE + 24 S2 + 5 KSSL targets +
OpenLandMap SOC for reference).

### Step 3 — Validation analysis

`future_work/scripts/kssl_validation.Rmd`:

1. **OpenLandMap-vs-KSSL agreement check.** Plot OpenLandMap SOC at the
   KSSL points against KSSL-measured SOC. Compute $R^2$ and bias. This
   is the "how good is OpenLandMap really?" baseline that the original
   paper implicitly assumed at $R^2 = 1$.
2. **Refit the original RF on KSSL labels.** Same 5-fold spatial-block
   CV, same 64-dim feature space, but now the target is KSSL SOC.
   Report $R^2$, RMSE.
3. **Headline comparison table:**

   | Target | Predictor | OpenLandMap label $R^2$ | KSSL label $R^2$ | Δ |
   |--------|-----------|-------------------------|------------------|---|
   | SOC    | AlphaEarth| 0.75 (orig.)            | (new)            |   |
   | SOC    | S2 (#2)   | (#2 result)             | (new)            |   |
   | SOC    | S2 + AE   | (#2 result)             | (new)            |   |

   The expected narrative: $R^2$ drops from 0.75 to ~0.40–0.55 when
   moving to KSSL, because KSSL has real measurement noise and
   sub-grid heterogeneity that OpenLandMap smooths over. **The
   *ranking* of the three feature stacks should remain stable** — that
   would be the strong robustness claim.
4. **Spatial residual analysis** — Moran's $I$ on KSSL-residuals to
   check whether the spatial-block CV adequately broke autocorrelation.
   This is the "Moran's $I$" test from the original proposal that
   didn't end up in the STAT 5000 paper.
5. **Multi-target on KSSL** — repeat #1's per-target RFs, but with
   KSSL labels for SOC, pH, BD, sand, clay. Five-panel figure as in #1
   but with field-measured labels. This is the headline result if it
   works.

## Production-integration notes

- **KSSL ingestion module** — a `kssl_pedon` loader parallel to an existing
  SSURGO loader, with the same caching pattern (download once to a reference
  cache, lookup-by-pedon-key afterward).
- **Validation harness** — a nearest-neighbor SSURGO join pattern adapts
  directly to KSSL (nearest-neighbor from prediction grid to KSSL pedon with an
  explicit distance threshold); KSSL is just a different table.
- **Reference data path** — write the harmonized KSSL points to a standard
  `_reference/validation_truth/kssl_v1/` location so every model evaluation can
  pull the same external validation set.
- **Per-target accuracy reports** — a trainer that writes `metrics.json` per run
  can also emit `validation_kssl.json` with R²/RMSE against the external set
  (a small addition).
- **WoSIS as an extension** — once KSSL ingestion is in, WoSIS for
  global validation is a smaller delta and uses the same depth-
  harmonization code.

## Risks & mitigations

- **Coordinate quality lookup column is uncertain.** I haven't verified
  the exact KSSL schema column for GPS-vs-PLSS provenance — the
  research brief flagged this. First task on starting this direction
  is to download the SQLite, inspect the schema, and confirm the
  filter column name.
- **Walkley-Black vs. dry-combustion bias.** The 1.3× WB→DC conversion
  factor is debated in the literature. Filter to dry-combustion only
  to avoid the conversion, even at the cost of a smaller sample.
- **Pedon-pixel mismatch.** A 10 m AlphaEarth pixel may not represent the
  pedon's actual sampling location well in heterogeneous fields. Use
  a 30 m circular buffer median rather than the single nearest pixel,
  and report sensitivity to buffer size.
- **Sample density geographic bias.** KSSL is dense in IA/IL/NE,
  sparse in the Intermountain West and AK. Restrict the 5-fold
  spatial-block CV to CONUS and report per-MLRA performance to surface
  this.
- **Effort overrun.** This is the largest direction. Time-box Step 1
  (ingestion) at 2 days; if blocked, fall back to a smaller hand-curated
  subset (e.g., RaCA, the USDA Rapid Carbon Assessment, ~6,000 points
  with already-clean lat/lon and 0–30 cm SOC).

## Estimated effort

- **Step 1 (KSSL ingest + harmonize):** 2-3 days, mostly schema
  archaeology and depth math.
- **Step 2 (pair with GEE):** 0.5 day.
- **Step 3 (analysis & write-up):** 1–2 days.
- **Total:** 4–6 days. Highest scientific payoff but most engineering
  per insight; do this third, after #1 and #2 have produced the
  comparison framework.
