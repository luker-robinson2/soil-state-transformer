# Future Work — Expanding the AlphaEarth/SOC Project

This directory expands the scope of the STAT 5000 project (`../main.tex`,
`../soil_analysis.Rmd`) into three follow-on research directions, and notes for
each how it could fold into a production soil-analytics pipeline.

The original paper used $n = 3000$ CONUS points to show that AlphaEarth
Foundations 64-dim embeddings predict OpenLandMap SOC at $R^2 \approx 0.75$
on spatial-block CV. Three directions extend that result:

1. **Multi-target extension** — predict pH, sand %, clay %, bulk density
   alongside SOC. Tests whether AlphaEarth encodes the full soil property
   space or just SOC. *Plan:* [`01_multi_target.md`](01_multi_target.md).
2. **Sentinel-2-only baseline** — fit the same RF on a standard
   bare-soil-composite + spectral-index feature stack, compare to
   AlphaEarth. Quantifies the marginal value of the foundation-model
   embedding over the raw imagery it was trained on. *Plan:*
   [`02_sentinel2_baseline.md`](02_sentinel2_baseline.md).
3. **KSSL ground-truth validation** — replace OpenLandMap labels with USDA
   KSSL/NCSS lab-measured pedon profiles (post-2010, GPS coordinates).
   Converts the project from model-vs-model into a true accuracy assessment.
   *Plan:* [`03_kssl_validation.md`](03_kssl_validation.md).

## Unified narrative

Each direction answers a different reviewer question that the original
paper invites:

| Reviewer's question | Answered by |
|---|---|
| "Does AlphaEarth encode soil chemistry generally, or did you cherry-pick SOC?" | #1 Multi-target |
| "Does AlphaEarth actually beat the Sentinel-2 imagery it was trained on?" | #2 S2 baseline |
| "Your labels are themselves an ML product — is this real?" | #3 KSSL validation |

If all three land, the combined story is publishable: *"Foundation-model
embeddings provide modest but consistent uplift over a strong S2 baseline
across multiple soil properties, validated against field-measured pedons."*

## Sequencing & effort estimates

```
Week 1  ──────  #1 Multi-target extension     (1-2 days, low risk)
Week 1  ──────  #2 Sentinel-2 baseline         (2-3 days, medium risk)
Week 2-3 ─────  #3 KSSL ground-truth           (4-6 days, highest risk)
```

The first two reuse the existing `01_extract_data.py` GEE pipeline;
extending the script costs maybe 30 lines each. #3 requires building a
KSSL ingestion module and harmonizing horizon depths to 0–30 cm, so it's
materially harder.

I recommend running #1 and #2 in parallel (they share the same n=3000 CSV
backbone), then promoting the most promising signal into #3 against KSSL.

## Integration with a production pipeline

A typical production soil-analytics pipeline already provides most of the
plumbing these directions need. The reusable pieces and the new code each
direction requires:

### Reusable directly

- **GEE auth** — a service-account-with-ADC-fallback init pattern; the
  future-work scripts reuse the same Earth Engine credentials.
- **GCS export convention** — a bucket with a stable path layout for exported
  features; the KSSL validation set lands in a parallel
  `_reference/validation_truth/` subtree.
- **Sentinel-2 in GEE** — any existing `COPERNICUS/S2_HARMONIZED` NDVI
  extraction generalizes; the S2 baseline (#2) extends it to a full 24-feature
  stack rather than just NDVI.
- **SSURGO offline cache** — a MUKEY → component join is useful for
  *stratifying* the validation set by soil order in #3 (we do **not** use SSURGO
  as ground truth — it's modeled, not measured).
- **GroupKFold spatial CV** — a GroupKFold-by-field trainer takes the 1°×1°
  spatial-block CV from the original paper as a custom `groups=` argument.
- **Ensemble trainer** — an RF/GBM/XGB ensemble is the right shared comparison
  engine; the S2 baseline (#2) runs through it to keep comparisons fair.
- **WoSIS pre-train data** — the [foundation-model](../../foundation-model/)
  service already prepares WoSIS; if we go beyond CONUS in a v2, that's the
  on-ramp.

### Needs to be built

- **KSSL ingestion module** — new code. Plan in
  [`03_kssl_validation.md`](03_kssl_validation.md): a `kssl_pedon` loader
  parallel to an SSURGO loader.
- **AlphaEarth feature extractor** — wiring AlphaEarth into an operational
  pipeline as an `alphaearth_features` processor; the work in this project is
  the natural prototype.
- **Multi-target schema** — extend a one-nutrient-per-column table to (SOC, pH,
  BD, sand, clay) and the trainer to fit one model per target (or a
  multi-output RF).
- **Bare-soil compositing in GEE** — the SOC literature (Safanelli 2020,
  Loiseau 2019) prefers bare-soil composites over simple median composites for
  soil-property prediction; a ~50-line addition to the GEE extractor.

### Config awareness

Any production code should take the GCP project ID and bucket from config
(the convention in `config.yml`) so the same scripts run against a different
project or a local-only setup with a one-line change.

## Repository layout

```
future_work/
├── README.md                       # this file
├── 01_multi_target.md              # direction 1 plan
├── 02_sentinel2_baseline.md        # direction 2 plan
├── 03_kssl_validation.md           # direction 3 plan
├── (later) data/                   # extended CSVs from each direction
└── (later) scripts/                # extraction + analysis code
```

The original `01_extract_data.py` and `soil_samples.csv` stay where they
are; new scripts will write to `future_work/data/` and `future_work/scripts/`
to keep the original paper's artifacts pristine.

## Success metrics for the combined effort

A single summary table that the next-paper draft can lead with:

| Target  | OpenLandMap label | KSSL label |
|---------|-------------------|------------|
|         | S2 / AE / S2+AE   | S2 / AE / S2+AE |
| SOC     | (#2 result)       | (#3 result) |
| pH      | (#1 + #2)         | (#3 stretch) |
| Sand %  | (#1 + #2)         | (#3 stretch) |
| Clay %  | (#1 + #2)         | (#3 stretch) |
| BD      | (#1 + #2)         | (#3 stretch) |

Bold any cell where AlphaEarth beats S2 by more than its CI; that's the
publishable headline.
