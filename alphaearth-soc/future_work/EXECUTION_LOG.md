# Future-Work Execution Log

Live log of execution progress. Each phase appends a section as it completes.

## Phase A — Unified extraction

- Script: `scripts/01_extract_unified.py`
- Anchor: same n=3000 CONUS random points (seed=42) used by the original
  paper's `../01_extract_data.py`.
- New OpenLandMap targets pulled alongside SOC: pH-H₂O, sand %, clay %,
  bulk density.
- Output: `data/soil_unified.csv` (~3000 × 73 columns).
- Status: **(running)**.

## Phase A — S2 24-feature extraction

- Script: `scripts/02_extract_s2_features.py`
- Approach: Cloud Score+ (cs ≥ 0.6) masking on `S2_SR_HARMONIZED`, bare-soil
  composite (NDVI<0.25 ∧ NBR2<0.075), per-band median + 5 BSC indices,
  4 annual median indices, 4 phenology percentiles, 5 seasonal features.
  Sampled at scale=20 m at the same 3000 lon/lat.
- Output: `data/s2_features.csv` (~3000 × 26 columns), joined into
  `soil_unified.csv`.
- Status: **(pending Phase A finish)**.

## Phase A — Sanity check

- After unified+S2 extraction, refit the original AE→log-SOC RF on
  `soil_unified.csv`. Pass criterion: spatial-block-CV $R^2 \in [0.74, 0.76]$.
- Status: **(pending)**.

## Phase B — Multi-target analysis

- Notebook section in `scripts/99_followup_analysis.Rmd`.
- Per-target: distribution + transform; Welch's t (Grass vs Crop); 64-dim
  univariate barplot; spatial-CV RF with bootstrap CI on R²; pred-vs-obs.
- Status: **(pending)**.

## Phase C — Sentinel-2 baseline

- Three RFs (S2 24-feature, AE 64-dim, S2+AE 88-feature) on identical
  spatial-block CV. Paired BCa bootstrap (B=10000) on $\Delta R^2$.
- Status: **(pending)**.

## Phase D — KSSL ingestion

- Script: `scripts/03_ingest_kssl.py`.
- **Manual download required**: visit
  <https://ncsslabdatamart.sc.egov.usda.gov/database_download.aspx>
  and click "Tabular and Spatial SQLite". The page uses `__doPostBack`
  ASP.NET form submission so it cannot be cleanly scripted. Place the
  resulting `.sqlite` file in `data/kssl_raw/`. Then run the script.
- Once present, the script auto-discovers the schema and writes
  `data/kssl_points.csv` (filtered, depth-harmonized).
- Status: **(blocked on manual download)**.

## Phase E — KSSL feature extraction

- Script: `scripts/04_extract_kssl_features.py` (already drafted).
- Pulls AE 2024 at scale=10 m, S2 24-feature stack at scale=20 m, and
  OpenLandMap SOC at every KSSL pedon.
- Status: **(pending Phase D)**.

## Phase F — KSSL validation analysis

- Notebook section in `99_followup_analysis.Rmd` (already drafted).
- OpenLandMap-vs-KSSL agreement, refit RFs on KSSL labels, headline
  3×2 comparison table, Moran's *I* on residuals.
- Status: **(pending Phase E)**.

## Phase G — Follow-on LaTeX paper

- `paper/main.tex` skeleton already in place with placeholders to fill
  from notebook outputs.
- Status: **(pending all analysis)**.
