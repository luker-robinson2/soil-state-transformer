# Phase 0 — Data extraction

The on-disk static data (AlphaEarth, Sentinel-2, OpenLandMap targets, KSSL lab
truth) is already harmonized by `geosoil/data.py`. This directory adds the
**richer multi-modal + multi-truth layer** that unlocks the temporal models
(climate / vegetation sequences → the Mamba branch) and extra ground truth.

## 1. Earth Engine covariates (temporal climate + vegetation + terrain)

Needs a personal Earth-Engine-enabled GCP project (free for non-commercial use):

```bash
# one-time
.venv/bin/earthengine authenticate

# extract at every point in master.parquet
.venv/bin/python -m data.extraction.extract_gee --project YOUR_GEE_PROJECT --year 2022
# -> data/processed/gee_temporal.parquet  (ERA5 t2m/precip/soil-temp/soil-moist 12mo + MODIS NDVI/EVI 12mo)
# -> data/processed/gee_static.parquet    (Copernicus DEM elevation/slope/aspect/TWI)

# fold into the master table
.venv/bin/python -m geosoil.data --build
```

After this, `geosoil/data.py` auto-detects the new modalities, the temporal
encoders (GRU / Transformer / **Mamba**) activate, and the bake-off can compare
them on real climate/vegetation sequences:

```bash
.venv/bin/python -m geosoil.train --variant mamba --device mps
```

## 2. Additional ground truth (RaCA, OSSL) — independent cross-reference

These give independent lab-measured truths beyond KSSL. Both ship coordinates +
lab properties; to score *our* AlphaEarth-based model on them, extract AlphaEarth
+ covariates at their points (extend `extract_gee.py` with their lon/lat), then
add them as extra `source=` rows in the master table.

- **RaCA** (USDA Rapid Carbon Assessment, ~6k US sites): https://agdatacommons.nal.usda.gov/ — `soilDB::fetchRaCA()` (R) or CSV.
- **OSSL** (Open Soil Spectral Library, >100k samples): `soilspecdata` (Python) or https://soilspectroscopy.org/.

## Notes
- `extract_gee.py` column names are kept in sync with `geosoil/config.py`
  (`CLIMATE_VARS`, `VEG_VARS`, `TERRAIN_COLS`). The harmoniser joins on rounded
  (lon, lat); missing modalities are masked, never imputed silently.
- All sources are public (CC-BY / public domain). See `docs/research/` for the
  full dataset survey and licensing.
