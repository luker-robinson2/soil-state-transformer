# Direction 2 — Sentinel-2-Only Baseline

**Question:** Does the AlphaEarth 64-dim embedding actually beat what you
can extract from the raw Sentinel-2 imagery it was trained on?

**Why this matters:** AlphaEarth is opaque (no per-band physical meaning),
expensive in storage, and a closed Google product. Before recommending it
for a production stack, we need to know how much it adds over
the standard, open Sentinel-2 + spectral-indices recipe that the Digital
Soil Mapping (DSM) literature has been using for a decade.

## Comparison design

Three feature stacks, all evaluated on the same n=3000 CONUS points and
the same 5-fold spatial-block CV from the original paper:

1. **S2 baseline** — 24 bare-soil-composite + spectral-index features
   from Sentinel-2 SR Harmonized (see Step 1 below).
2. **AlphaEarth** — 64-dim embedding (the original paper's setup).
3. **S2 + AlphaEarth concatenated** — 88 features, to test whether the
   embedding is *additive* over the explicit features or merely
   reproduces them.

The cleanest contrast is **(1) vs. (2)**: same imagery in spirit, very
different representation. **(3)** is the ablation that asks "what does
AlphaEarth add that you couldn't extract with hand-engineered features?"

To keep the contrast clean, **terrain and climate covariates are excluded
from both arms**. AlphaEarth was trained on SRTM, ERA5, GEDI, etc., so
its embedding already encodes them; adding them to the S2 arm would
artificially level the comparison. They become a separate ablation if
needed.

## Plan

### Step 1 — Build the S2 feature stack in GEE

Following Safanelli et al. (2020), Loiseau et al. (2019/2020), and the
broader S2-BSC tradition. `future_work/scripts/02_extract_s2.py`
(parallel to `01_extract_data.py`):

**Bare-soil composite (BSC), 2024**
- Filter `COPERNICUS/S2_SR_HARMONIZED` to 2024.
- Mask clouds with **Cloud Score+** (`GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`,
  `cs >= 0.6`); fall back to SCL classes 4, 5, 6, 7, 11 only.
- Compute NDVI and NBR2 per scene; restrict to pixels where
  `NDVI < 0.25 AND NBR2 < 0.075` (the canonical bare-soil window).
- Take the **per-band median** over all retained scenes.
- Bands kept: B2, B4, B6, B8, B11, B12. (6 features.)
- Compute on the BSC: **BSI, NDTI, STI, SATVI, BI**. (5 features.)

**Annual all-observations composite, 2024**
- Same date range, Cloud Score+ masking, no NDVI restriction.
- Compute per-scene: NDVI, EVI, NDWI(Gao), NBR.
- Take **median** of each. (4 features.)
- Take **p10 and p90** of NDVI and NBR (phenology amplitude). (4 features.)

**Seasonal medians**
- DJF and JJA NDVI medians; (JJA − DJF) NDVI difference. (3 features.)
- DJF and JJA B11 (SWIR1) medians. (2 features.)

Total: **24 features**. Sample at the same 3000 lat/lon points at
scale=20 m to align with S2's 20 m bands and avoid implicit pyramiding.

**Reference formulas** (verified standard forms):

```
NDVI  = (B8 - B4) / (B8 + B4)
EVI   = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1)
NDWI  = (B8 - B11) / (B8 + B11)         # Gao
NBR   = (B8 - B12) / (B8 + B12)
NBR2  = (B11 - B12) / (B11 + B12)        # also called STI
BSI   = ((B11+B4) - (B8+B2)) / ((B11+B4) + (B8+B2))
NDTI  = (B11 - B12) / (B11 + B12)        # tillage; same as NBR2
STI   = B11 / B12
SATVI = ((B11 - B4) / (B11 + B4 + 1)) * 1.5 - (B12 / 2)
BI    = sqrt((B4^2 + B3^2) / 2)          # brightness
```

(NDTI ≡ NBR2 ≡ STI numerator in some conventions; we keep all three under
their literature names but de-duplicate when fitting the model.)

Output: `future_work/data/s2_features.csv`, 3000 rows × 26 cols
(24 features + lon/lat).

### Step 2 — Joint analysis notebook

`future_work/scripts/s2_vs_alphaearth.Rmd`:

1. **Sanity check the BSC** — for each point, count the number of
   bare-soil retained scenes. Drop points with fewer than 5 (cropland in
   permanent vegetation, lakes, etc.). Expect to lose ~10-20%.
2. **Univariate correlations** — Pearson $r$ of each S2 feature vs.
   log-SOC; compare top-5 to AlphaEarth's top-5.
3. **Three RFs** under identical 5-fold spatial-block CV:
   - `rf_s2`: log-SOC ~ 24 S2 features
   - `rf_ae`: log-SOC ~ 64 AlphaEarth features (paper baseline)
   - `rf_both`: log-SOC ~ 88 features
4. **Bootstrap CIs on $\Delta R^2$** — paired bootstrap over points to
   compute the 95% CI for $R^2_{\text{AE}} - R^2_{\text{S2}}$. This is the
   headline statistic.
5. **Permutation feature importance** for `rf_both` — which AE dimensions
   stay important when S2 features are also available?

### Step 3 — Headline figure & paper section

Three-bar chart with paired-bootstrap error bars: $R^2$ for S2, AE, and
S2+AE, with a callout for the $\Delta$ between AE and S2.

If $\Delta R^2$ is small (say <0.05) and the CI crosses zero, the
publishable finding is **negative-result-but-honest**: "AlphaEarth's
embedding does not materially outperform a 24-feature S2 baseline for
SOC; its main practical value is in eliminating preprocessing rather
than in raw predictive power."

If $\Delta$ is large and the CI excludes zero, the finding is the
positive headline: "AlphaEarth adds X% $R^2$ over a strong S2 baseline."

Either result is publishable.

## Production-integration notes

- **GEE pipeline reuse** — a pipeline that already pulls Sentinel-2 NDVI can
  reuse the same auth/export plumbing; the 24-feature stack is a strict superset,
  exposed behind the same `(geometry, year) -> features` interface.
- **Cloud Score+ adoption** — Cloud Score+ masking is materially better than
  simple cloud logic for bare-soil compositing (no false positives on dark wet
  soil) and worth adopting generally.
- **Bare-soil compositing** — a reusable BSC utility benefits any soil-property
  target, not just this benchmark.
- **Ensemble fairness** — running all three feature stacks through the *same*
  ensemble (RF/GBM/XGB) gives an apples-to-apples production-style benchmark,
  alongside the ranger-only RF used in the original paper.
- **Storage** — write the S2 feature stack into the same GCS layout as the
  ee-export output so it's discoverable by downstream steps.

## Risks & mitigations

- **Bare-soil composite empty over evergreen forests.** Many CONUS
  points (Pacific Northwest, Appalachians) will have <5 bare-soil
  observations. Document the drop rate; report the comparison both on
  the full set (with NaN-imputed BSC features) and on the bare-soil-rich
  subset for robustness.
- **Cloud Score+ recency.** The asset only goes back to mid-2017;
  fine for 2024 data but worth noting if we extend backward.
- **Computation time.** A 24-feature stack across 3000 points and a
  full year of S2 is ~50× more GEE compute than the original
  AlphaEarth pull. Use chunked extraction (250 points/call) like the
  original script, and use `tileScale=4` to stay under per-call limits.
  Expect ~30–60 min total.
- **Index redundancy.** NDTI ≡ NBR2 ≡ (B11–B12)/(B11+B12) under different
  names; STI = B11/B12. Keep all in the feature set; tree-based RF
  handles redundancy gracefully and removing it would require taking a
  side in a literature-naming dispute.

## Estimated effort

- **Step 1 (S2 extractor):** 4 hours coding (Cloud Score+, BSC mask,
  per-season composites), 1 hour debugging GEE, 30–60 min runtime.
- **Step 2 (analysis):** 4 hours including paired bootstrap.
- **Step 3 (write-up):** 3 hours.
- **Total:** ~2 days. Run after #1 because it shares the same n=3000
  CSV anchor.
