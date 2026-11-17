# GeoSoil data report: what goes into the model, from where, and what helps

Written 2026-11-17. This covers the C1 labelled set and every input feature built so far. It combines
measured results from Alpine runs with evidence from the digital-soil-mapping literature. Results still
being computed are marked **(pending)** and get filled in when their job finishes.

## 1. The model's job

Predict five lab-measured topsoil properties at a location:
- **SOC** (g/kg, modelled as log1p)
- **pH** (in water)
- **sand** (%)
- **clay** (%)
- **bulk density** (BD, g/cm³)

Inputs are what can be known about that place from public data. Evaluation is always against lab
measurements, using spatially blocked cross-validation.

## 2. Labels (the targets)

| Source | Training sites kept | Region | Targets | Licence | How we get it |
|---|---|---|---|---|---|
| WoSIS 2023 (ISRIC) | 120,668 | Global; brings in Australia (CSIRO), Mexico (INEGI), Chile, African profiles (AfSP), SISLAC | all 5 | CC-BY / CC-BY-NC per profile | Anonymous zip, `geosoil/corpus/wosis.py`. US-NCSS rows dropped, since they are KSSL. |
| KSSL Lab Data Mart (USDA) | 47,347 | US | all 5, at depth | Public domain | labtaxa mirror of the LDM, `kssl.py` |
| Brazil SoilData 2024 | 9,123 | Brazil | all 5 | CC-BY / CC-BY-NC | Dataverse, `brazil.py` |
| Canada NPDB v3 | 8,061 | Canada | all 5 | OGL-Canada | `npdb.py` |
| USGS DS 801 | 4,782 | CONUS (probability sample) | SOC | Public domain | `usgs_ds801.py` |
| OSSL v1.2 (non-KSSL, non-LUCAS) | 494 | Africa, global reference profiles | all 5 | CC-BY / ODbL | `ossl.py` |
| NAPESHM | 92 sites, 1.5k plots | US/Mexico | all 5 | CC-BY | `napeshm.py` |
| **RaCA** (hold-out, never trained on) | 6,367 | US | SOC, BD | CC0 | `raca.py` |

- **Totals:** 190,567 deduplicated training sites (58,388 in CONUS), plus 6,367 hold-out sites and 3.07M
  lab values.
- **Topsoil training sites per target:** SOC 132.6k, pH 124.2k, clay 141.9k, sand 109.9k, BD 25.5k.
- **Licences:** 18% of values are under non-commercial licences. That's fine for this project; filter
  them before any redistribution.

How the labels are processed:
- **Dedup.** A site is dropped only when its source republishes another source (WoSIS and OSSL are
  compilations). Native sources win over their WoSIS copies.
- **Harmonization.** Values are thickness-weighted onto standard depths. Each target keeps one method:
  - CaCl₂ pH converted to water pH (R² 0.96)
  - Walkley-Black converted ×1.019, since KSSL's values are already corrected
  - oven-dry bulk density converted to third-bar (R² 0.69)

  Converted rows are flagged.
- **Splits.** Hashed 1° blocks: 5 CV folds, with 10% of blocks held out as test and never used for
  model selection.
- **Noise floor** (best achievable R², from co-located KSSL pedons): SOC 0.87, pH 0.90, clay 0.87,
  sand 0.92, BD 0.82.

## 3. Input features

| Group | Columns | Source | Resolution | Coverage of C1 sites | Licence | Code |
|---|---|---|---|---|---|---|
| **AlphaEarth embeddings** | 64 per year, 2017–2025; the model uses the mean of 8 years (2024 held out), renormalised | Google DeepMind Satellite Embedding v1 (Sentinel-1/2, Landsat, climate, terrain, distilled), S3 COGs | 10 m | 183.6k sites with coordinates precise to ≤1 km | CC-BY 4.0 | `features/alphaearth.py` |
| **Terrain** | elevation, slope, aspect sin/cos, curvature, TPI 100/500 m, roughness 100 m, relief 500 m | Copernicus DEM GLO-30, AWS | 30 m, ±500 m window | 196.3k sites (31 ocean or unpublished tiles) | Copernicus DEM licence (free, attribution) | `features/terrain.py` |
| **Climate (annual)** | MAP, MAT, PET, AET, water deficit, soil moisture, VPD, solar radiation, temperature range, aridity (P/PET), rainfall seasonality (CV) | TerraClimate 1991–2020 normals | ~4 km | 99.96% | CC0 | `features/climate.py` |
| **Climate (monthly)** | 9 variables × 12 months (108 columns) | same | ~4 km | 99.96% | CC0 | same |
| **Radiometrics** | K %, eU, eTh, Th/K, U/Th, source flag | USGS NURE Bayesian grids (CONUS, 2 km); Radmap v4 (Australia, 100 m) | 2 km / 100 m | CONUS + Australia only **(pending: exact %)** | Public domain / CC-BY 4.0 | `features/radiometrics.py` |
| **Location** | lon, lat | — | — | all | — | — |
| **Depth** | interval midpoint | — | — | all | — | — |

**Unlabelled corpus (Phase 3 pretraining seed).** 39.7M AlphaEarth pixel-years: 64 pixels per block, kept
from every block read for the labelled sites, 2017–2025. Stored as int8 with UTM coordinates in
`data/processed/features/alphaearth/y*/unlabelled-*.parquet` on Alpine.

**Where everything lives on Alpine.** All under `/projects/luro4974/soil-state-transformer/data/`:
- `raw/<source>/`: cached downloads, each with a sha256 sidecar.
- `processed/labels_*.parquet`: the label tables.
- `processed/features/<source>/`: feature shards, each with a provenance manifest (source, licence,
  hashes, code version, failures).
- `processed/train_c1.parquet`: the joined training table.

## 4. What has proven useful so far (measured)

LightGBM, 5-fold spatial-block CV, lab labels, topsoil. R² on held-out folds:

| Features | SOC | pH | Sand | Clay | BD |
|---|---|---|---|---|---|
| Terrain only | 0.48 | 0.52 | 0.34 | 0.27 | 0.42 |
| AlphaEarth 2020 only | 0.53 | 0.57 | 0.37 | 0.32 | 0.48 |
| **AlphaEarth 8-year mean** | **0.54** | **0.58** | **0.40** | **0.34** | **0.49** |
| + climate + radiometrics, all groups **(pending)** | | | | | |
| *Noise-floor ceiling* | *0.86* | *0.90* | *0.92* | *0.87* | *0.82* |

For comparison, the model before C1 (trained on ~3k sites) got lab SOC R² of 0.13–0.22 in-domain, and
−0.44 when the OpenLandMap-trained model was transferred.

What the numbers so far say:
1. **Labels mattered most.** Going from ~3k to ~120k lab sites per target took lab SOC from ~0.2 to 0.54.
2. **AlphaEarth beats 30 m terrain on every target**, by about +0.05 R².
3. **Averaging AlphaEarth over years helps a little** (+0.01–0.02), most for sand and clay (+0.02).
4. **Texture is furthest from its ceiling** (sand 0.40 vs 0.92; clay 0.34 vs 0.87). The surface-dominated
   inputs don't see parent material or landscape position at the right scale.
5. **Feature-group importance per target:** drop-group ablation, permutation and SHAP **(pending: job
   35829185)**.

## 5. What the literature says is useful (and what we're missing)

| Signal | Helps | Evidence | Status in GeoSoil |
|---|---|---|---|
| **Multi-scale terrain context** (Gaussian-pyramid elevation, slope, curvature, TPI at ~0.5–20 km) | **sand, clay** | Behrens et al. 2018: clay R² 0.38 → 0.68, silt 0.16 → 0.63 | **Missing.** Our windows stop at 500 m. Top priority. |
| **Parent material / lithology** (GLiM global; SGMC for the US) | sand, clay, pH | Consistently among the top texture and pH covariates in SoilGrids and continental studies | **Missing.** Cheap; covers the regions with no radiometrics. |
| **Bare-soil reflectance** (SYSI / GEOS3 multi-year bare-soil median; SWIR clay ratio, ferric iron) | clay, sand, SOC | Safanelli 2020; Demattê 2025 GEOS3 | **Missing.** AlphaEarth's all-season embedding likely dilutes it. |
| **Water balance P−PET (signed) + months with P > PET** | **pH** | Slessarev 2016 (Nature, ~60k sites): pH flips to acid where MAP > PET | Partly (aridity ratio). Signed version is cheap to add. |
| Climate (MAT, MAP, GDD, growing season) | **SOC** | Top global SOC driver (Wiesmeier 2019; SoilGrids) | Have normals; GDD and frost days are cheap. |
| Gamma radiometrics (K, Th, U; ratios; weathering index) | clay, pH | Strong in Australian and US studies | Have K/U/Th + ratios (US, AU). Weathering index not yet. |
| Valley-floor / wetness (MRVBF, TWI, HAND, distance to stream) | clay, SOC, BD | Regional top predictors | Missing. MERIT Hydro / Hydrography90m are open. |
| MODIS LST diurnal range (thermal inertia) | sand, clay | Regional: sand RMSE 10.7% | Missing. |
| Cultivation history (HYDE / LUH2, ESA CCI) | SOC, BD | Sanderman 2017 | Missing. |
| Regolith thickness, depth to bedrock | texture, BD | Pelletier 2016, Shangguan 2017 | Missing. |
| Foundation-model embeddings vs handcrafted covariates | all | 2025–26 abstracts: hybrids beat either alone, modestly | We run the hybrid. |

**Avoid:**
- **SoilGrids predictions as features.** They were trained on WoSIS, so using them is label leakage.
- **Neighbour target means**, unless recomputed inside each fold.
- **Extra coordinate tricks** (oblique coordinates, distance fields). Milà et al. 2024 found them neutral
  or harmful where samples are sparse; embeddings already encode location.

**How to select features:** forward selection or drop-group ablation inside spatial folds. Our
`feature_selection.py` does drop-group, permutation and SHAP. Random-CV importance, RFE or Boruta outside
spatial folds would favour location proxies.

## 6. Recommended next feature additions (ranked by expected gain per effort)

1. **Multi-scale terrain pyramid** from the Copernicus DEM we already read. Targets sand and clay. Medium effort.
2. **GLiM lithology class** (and SGMC for the US), target-encoded inside folds. Targets texture and pH. Low effort.
3. **P−PET signed + wet-month count, GDD, frost days**, from TerraClimate (already staged). Targets pH and SOC. Low effort.
4. **MERIT Hydro HAND / distance to stream + TWI.** Targets texture, SOC and BD. Low effort.
5. **MODIS LST diurnal range.** Targets texture. Low–medium effort.
6. **Bare-soil composite** (check GEOS3 access; otherwise Landsat SYSI). Targets clay. Medium–high effort.
7. **Cultivation history (HYDE 3.3) + NPP.** Targets SOC. Medium effort.

Each addition is accepted only if drop-group ablation shows it beats fold-to-fold noise on at least one target.
