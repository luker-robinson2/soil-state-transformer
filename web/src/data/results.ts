/*
  Real, verifiable results lifted directly from geosoil/results/*.json.
  Nothing here is invented. Each block cites its source file.
  Spatial-block CV (1°×1° GroupKFold), n=3000 unless noted.
*/

export type Target = 'soc' | 'ph' | 'sand' | 'clay' | 'bd';

export const TARGETS: { key: Target; label: string; unit: string; long: string }[] = [
  { key: 'soc', label: 'SOC', unit: 'g/kg', long: 'Soil organic carbon' },
  { key: 'ph', label: 'pH', unit: '', long: 'Soil pH (H₂O)' },
  { key: 'sand', label: 'Sand', unit: '%', long: 'Sand fraction' },
  { key: 'clay', label: 'Clay', unit: '%', long: 'Clay fraction' },
  { key: 'bd', label: 'BD', unit: 'g/cm³', long: 'Bulk density' },
];

export type Row = Record<Target, number>;

// ── Bake-off: R² against OpenLandMap labels, spatial-block CV ──────────────
// Source: geosoil/results/baselines.json, cv_transformer.json, hybrid.json
export type ModelKind = 'tabular' | 'frozen-probe' | 'geosoil';

export interface ModelRow {
  key: string;
  name: string;
  note: string;
  kind: ModelKind;
  r2: Row;
}

export const BAKEOFF: ModelRow[] = [
  { key: 'ridge', name: 'Ridge', note: 'linear baseline', kind: 'tabular',
    r2: { soc: 0.672, ph: 0.897, sand: 0.597, clay: 0.475, bd: 0.785 } },
  { key: 'rf', name: 'Random Forest', note: 'tabular baseline', kind: 'tabular',
    r2: { soc: 0.692, ph: 0.909, sand: 0.718, clay: 0.620, bd: 0.805 } },
  { key: 'catboost', name: 'CatBoost', note: 'gradient boosting', kind: 'tabular',
    r2: { soc: 0.708, ph: 0.916, sand: 0.735, clay: 0.624, bd: 0.828 } },
  { key: 'lightgbm', name: 'LightGBM', note: 'gradient boosting', kind: 'tabular',
    r2: { soc: 0.719, ph: 0.917, sand: 0.751, clay: 0.642, bd: 0.829 } },
  { key: 'xgboost', name: 'XGBoost', note: 'strongest tabular', kind: 'tabular',
    r2: { soc: 0.723, ph: 0.917, sand: 0.751, clay: 0.639, bd: 0.833 } },
  { key: 'probe', name: 'Frozen z → linear', note: 'linear probe of latent', kind: 'frozen-probe',
    r2: { soc: 0.685, ph: 0.902, sand: 0.709, clay: 0.568, bd: 0.807 } },
  { key: 'hybrid', name: 'Frozen z → GBM', note: 'latent + features → GBM', kind: 'frozen-probe',
    r2: { soc: 0.735, ph: 0.916, sand: 0.752, clay: 0.631, bd: 0.829 } },
  { key: 'geosoil', name: 'GeoSoil', note: 'multi-modal fusion, end-to-end', kind: 'geosoil',
    r2: { soc: 0.762, ph: 0.926, sand: 0.789, clay: 0.687, bd: 0.858 } },
];

// ── The multi-truth finding ────────────────────────────────────────────────
// Same model family, three different ground truths.
// Source: cv_jepa.json (OpenLandMap), verification_jepa.json (kssl_internal, ridge),
//         cv_jepa_kssl.json (multi-modal recovery on lab source).
export interface TruthSeries {
  key: string;
  label: string;
  sub: string;
  n: number | string;
  tone: 'good' | 'warn' | 'bad';
  r2: Row;
}

export const MULTI_TRUTH: TruthSeries[] = [
  {
    key: 'openlandmap',
    label: 'OpenLandMap',
    sub: 'modelled label product · spatial CV',
    n: 3000,
    tone: 'good',
    r2: { soc: 0.762, ph: 0.926, sand: 0.789, clay: 0.687, bd: 0.858 },
  },
  {
    key: 'kssl-ae',
    label: 'KSSL lab — AlphaEarth only',
    sub: 'train on lab, test on lab · in-domain control',
    n: 157,
    tone: 'bad',
    // verification_jepa.json → kssl_internal.ridge (AlphaEarth features)
    r2: { soc: 0.083, ph: 0.535, sand: -0.131, clay: -0.178, bd: 0.091 },
  },
  {
    key: 'kssl-mm',
    label: 'KSSL lab — multi-modal',
    sub: 'radar + bare-soil + terrain added · spatial CV',
    n: 2983,
    tone: 'warn',
    // cv_jepa_kssl.json
    r2: { soc: 0.133, ph: 0.546, sand: 0.339, clay: 0.250, bd: 0.168 },
  },
];

// ── Calibration: interval coverage of the heteroscedastic head + conformal ──
// Source: verification_jepa.json → calibration
export interface CalRow {
  key: Target;
  cov68: number;
  cov95: number;
  conformal90: number;
}
export const CALIBRATION: CalRow[] = [
  { key: 'soc', cov68: 0.649, cov95: 0.907, conformal90: 0.884 },
  { key: 'ph', cov68: 0.644, cov95: 0.923, conformal90: 0.890 },
  { key: 'sand', cov68: 0.612, cov95: 0.905, conformal90: 0.897 },
  { key: 'clay', cov68: 0.608, cov95: 0.894, conformal90: 0.919 },
  { key: 'bd', cov68: 0.643, cov95: 0.923, conformal90: 0.899 },
];

// ── Representation quality: cross-modal retrieval ──────────────────────────
// Source: verification_jepa.json → retrieval (n=1000)
export const RETRIEVAL = {
  n: 1000,
  recall1: 0.055,
  recall5: 0.1985,
  recall10: 0.3145,
  randomBaseline10: 0.01,
};

// ── Data modalities ────────────────────────────────────────────────────────
export interface Modality {
  group: string;
  source: string;
  role: string;
  temporal?: boolean;
}
export const MODALITIES: Modality[] = [
  { group: 'Foundation embedding', source: 'AlphaEarth Foundations 64-d', role: 'core modality' },
  { group: 'Optical spectral', source: 'Sentinel-2 (6 bands + indices)', role: 'modality' },
  { group: 'Radar', source: 'Sentinel-1 SAR (VV/VH)', role: 'soil-sensing' },
  { group: 'Terrain', source: 'Copernicus DEM', role: 'modality' },
  { group: 'Climate', source: 'ERA5-Land monthly ×12', role: 'temporal', temporal: true },
  { group: 'Vegetation', source: 'MODIS NDVI/EVI ×12', role: 'temporal', temporal: true },
  { group: 'Dynamics', source: 'SMAP · CHIRPS · CDL', role: 'micro-events' },
  { group: 'Labels (modelled)', source: 'OpenLandMap SOC/pH/texture/BD', role: 'training labels' },
  { group: 'Labels (measured)', source: 'KSSL / NCSS lab pedons', role: 'independent validation' },
];

// ── The six training objectives ────────────────────────────────────────────
export const OBJECTIVES: { name: string; blurb: string }[] = [
  { name: 'Heteroscedastic NLL', blurb: 'Masked Gaussian negative log-likelihood over five targets — predicts μ and σ per property.' },
  { name: 'Cross-modal JEPA', blurb: 'Predict one modality’s embedding from the others against an EMA target encoder. No augmentation.' },
  { name: 'InfoNCE', blurb: 'Contrastive alignment of paired modalities, e.g. AlphaEarth ↔ Sentinel-2.' },
  { name: 'Masked modeling', blurb: 'Denoising reconstruction of masked input features.' },
  { name: 'VICReg', blurb: 'Variance / covariance regularisation that prevents latent collapse.' },
  { name: 'Temporal forecasting', blurb: 'Predict held-out future steps of climate / vegetation sequences in latent space.' },
];

export const HEADLINE = {
  latentDim: 256,
  embeddingDim: 64,
  nModalities: 9,
  nTargets: 5,
  bestSOC: 0.762,
};
