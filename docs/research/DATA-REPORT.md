# GeoSoil data report: what goes into the model, from where, and what helps

Written 2026-11-17. This covers the C1 labelled set and every input feature built so far. It combines
measured results from Alpine runs with evidence from the digital-soil-mapping literature. Results still
being computed are marked **(pending)** and get filled in when their job finishes.

## 1. The model's job

Predict lab-measured topsoil properties at a location:
- **SOC** (g/kg, modelled as log1p)
- **pH** (in water)
- **sand** (%)
- **clay** (%)
- **bulk density** (BD, g/cm³)
- **organic matter** (g/kg; derived from harmonized SOC × 1.724, measured OM only where SOC is missing)
- **CEC** (cmol(+)/kg at pH 7; pH 8.2 sum-of-cations converted, r² 0.82; effective CEC dropped, r² 0.41)
- **exchangeable K, Ca, Mg** (cmol(+)/kg, NH4OAc pH 7; Mehlich-3 K converted, r² 0.91). Multiply by 391, 200.4
  and 121.5 for mg/kg (ppm).

Topsoil training sites (added 2026-11-17): OM 160.7k, CEC 96.9k, K 52.3k, Ca 51.8k, Mg 52.3k. Their
noise-floor R² ceilings (log scale) are 0.83 (K) to 0.91 (Ca).

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
| **Radiometrics** | K %, eU, eTh, Th/K, U/Th, source flag | USGS NURE Bayesian grids (CONUS, 2 km); Radmap v4 (Australia, 100 m); USGS OFR 2005-1413 (Canada + Alaska, 2 km) | 2 km / 100 m | ~39% (CONUS, Australia, Canada, Alaska) | Public domain / CC-BY 4.0 | `features/radiometrics.py` |
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

Latest run (job 35828308, 2026-11-17). Every group is scored on the **same rows** and includes
lon/lat + depth. This is why AlphaEarth-alone reads slightly lower here than in earlier runs, which
dropped rows with no embedding.

| Features (+ lon/lat, depth) | SOC | pH | Sand | Clay | BD |
|---|---|---|---|---|---|
| Radiometrics only (37% coverage) | 0.44 | 0.47 | 0.34 | 0.25 | 0.34 |
| Terrain only | 0.48 | 0.52 | 0.34 | 0.28 | 0.43 |
| Climate only (annual + monthly normals) | 0.51 | 0.55 | 0.36 | 0.30 | 0.42 |
| AlphaEarth (8-year mean) only | 0.52 | 0.56 | 0.38 | 0.33 | 0.49 |
| Terrain + AlphaEarth | 0.53 | 0.56 | 0.39 | 0.34 | 0.49 |
| **All groups** | **0.55** | **0.59** | **0.43** | **0.36** | **0.51** |
| *Noise-floor ceiling* | *0.86* | *0.90* | *0.92* | *0.87* | *0.82* |
| Gain of all over terrain + AlphaEarth | +0.02 | +0.03 | **+0.04** | +0.03 | +0.02 |

**Round 2** (job 35833568, 2026-11-17): adds multi-scale terrain, lithology, land-surface temperature, HAND and land-use history.

| Features (+ lon/lat, depth) | SOC | pH | Sand | Clay | BD |
|---|---|---|---|---|---|
| Multi-scale terrain only | 0.47 | 0.50 | 0.34 | 0.27 | 0.40 |
| Lithology only | 0.45 | 0.50 | 0.35 | 0.27 | 0.35 |
| Land-surface temperature only | 0.50 | 0.55 | 0.35 | 0.30 | 0.44 |
| HAND only | 0.45 | 0.49 | 0.32 | 0.26 | 0.36 |
| Land-use history only | 0.46 | 0.49 | 0.32 | 0.25 | 0.38 |
| **All groups, round 2** | **0.55** | **0.60** | **0.45** | **0.39** | **0.52** |
| Gain over round 1 (all groups) | +0.003 | +0.013 | **+0.024** | **+0.024** | +0.007 |

**Round-2 feature-group importance** (job 35833569). Unique contribution = R² lost when the group is
removed and the model retrained; ± is the fold-to-fold sd.

| Group | SOC | pH | Sand | Clay | BD |
|---|---|---|---|---|---|
| AlphaEarth | **+0.021** | **+0.012** | **+0.018** | **+0.026** | **+0.034** |
| Depth | **+0.084** | +0.004 | 0 | **+0.010** | **+0.023** |
| Lithology (new) | +0.003 | **+0.006 ± 0.002** | **+0.014 ± 0.004** | **+0.009 ± 0.004** | +0.003 |
| Radiometrics | +0.003 | +0.001 | **+0.012 ± 0.007** | +0.001 | −0.001 |
| Land-surface temperature (new) | +0.001 | **+0.005 ± 0.001** | +0.001 | +0.003 | +0.001 |
| Climate (monthly) | +0.005 | +0.004 | +0.007 | +0.004 | −0.001 |
| Multi-scale terrain (new) | +0.001 | +0.003 | +0.004 | +0.003 | 0 |
| HAND (new) | 0 | +0.001 | −0.001 | −0.002 | +0.001 |
| Land-use history (new) | +0.001 | +0.002 | +0.001 | −0.001 | −0.002 |
| Terrain 30 m, coordinates | ≤ 0.004 | ≤ 0.002 | ≤ 0 | ≤ 0 | ≤ 0.002 |

Top individual columns by SHAP share:
- **SOC:** depth, September and April daytime LST, April VPD, **P−PET** (new), climatic water deficit.
- **pH:** **aridity index** (0.115, the single largest), MAP, LST diurnal range, **1950 cropland
  fraction**, GLiM carbonate rock.
- **Sand:** **radiometric Th/K** (0.084), Th, GLiM carbonate rock, **20 km relief** (multi-scale), then
  AlphaEarth dimensions.
- **Clay:** latitude, depth, Th, September LST, AlphaEarth dimensions.
- **BD:** LST diurnal range, depth, June tmax, July night LST.

**Decisions from round 2:**
- **Keep:** lithology, radiometrics, LST, monthly climate, multi-scale terrain. Multi-scale terrain is
  marginal as a group, but its 20 km relief is a top sand column.
- **Drop or demote:** HAND and land-use history show no unique value beyond noise for any target. Keep
  hist_crop_1950 only as a pH probe. 30 m terrain and coordinates add nothing on top of the rest.
- **Highest-value next step:** extend radiometrics coverage, since Th/K is the top sand feature but
  covers 37% of sites. **Done 2026-11-17:** Canada and Alaska via USGS OFR 2005-1413 (+~3,350 sites,
  about +2 points). Also added **surficial materials** (USGS DS-425: till, loess, dune sand, alluvium,
  with grain size), because bedrock lithology misses the transported deposits most US pedons formed in.
  Both are measured in round 3.
  Then improve lithology resolution beyond GLiM's 1:3.75M (national geological maps), and the
  bare-soil composite.

**Round 3** (baseline job 35844337, feature selection 35844338, 2026-11-17; `geosoil/results/round3_summary.json`).
Adds Canada/Alaska radiometrics, the restored Australian K grid, surficial materials (USGS DS-425) and depth to
bedrock. Same 190k-site label set as round 2.

| | SOC | pH | Sand | Clay | BD |
|---|---|---|---|---|---|
| Round 2, all groups | 0.552 | 0.601 | 0.453 | 0.387 | 0.516 |
| **Round 3, all groups** | **0.555** | **0.602** | **0.458** | **0.391** | **0.514** |

Unique contribution (drop-group ΔR², mean of 3 seeds; ± is about 0.002–0.006):

| Group | SOC | pH | Sand | Clay | BD |
|---|---|---|---|---|---|
| AlphaEarth | **+0.021** | **+0.013** | **+0.015** | **+0.026** | **+0.037** |
| Depth | **+0.082** | +0.005 | +0.001 | +0.009 | **+0.022** |
| Lithology | +0.003 | +0.005 | **+0.014** | **+0.012** | +0.002 |
| Radiometrics (now 47% of sites) | +0.001 | +0.002 | +0.007 | +0.005 | +0.003 |
| Land-surface temperature | +0.003 | +0.004 | 0.000 | +0.006 | −0.001 |
| Monthly climate | +0.004 | +0.003 | +0.005 | +0.005 | −0.001 |
| Surficial materials (new) | +0.002 | 0.000 | +0.002 | +0.002 | 0.000 |
| Depth to bedrock (new) | 0.000 | 0.000 | −0.002 | 0.000 | −0.001 |
| HAND, land-use history, annual climate, 30 m terrain | ≤ +0.001 | ≤ 0 | ≤ 0 | ≤ +0.001 | ≤ 0 |

**Decisions from round 3:**
- The static map layers are saturating: the four new sources together moved texture by about +0.005,
  which is within seed noise. Each new geology layer mostly repeats lithology, radiometrics and climate.
- **Drop from the default set:** depth to bedrock, HAND, land-use history (keep `hist_crop_1950` as a pH
  probe), annual climate (redundant with monthly), 30 m terrain. This removes about 40 columns at no cost.
- **Keep:** AlphaEarth, depth, lithology, radiometrics, LST, monthly climate, multi-scale terrain;
  surficial materials stays on probation until a US-only texture run.
- **Where the gains are now:** new labels (SISLAC, +28k sites, round 4), neighbouring-label features
  (round 4; locally +0.07–0.18 over location alone), and the two Earth Engine features that observe
  something new, bare-soil reflectance and AlphaEarth landscape context (round 5).

Earlier runs, which dropped rows without an embedding: AlphaEarth 2020 SOC 0.53, pH 0.57, sand 0.37,
clay 0.32, BD 0.48; 8-year mean SOC 0.54, pH 0.58, sand 0.40, clay 0.34, BD 0.49.

For comparison, the model before C1 (trained on ~3k sites) got lab SOC R² of 0.13–0.22 in-domain, and
−0.44 when the OpenLandMap-trained model was transferred.

What the numbers so far say:
1. **Labels mattered most.** Going from ~3k to ~120k lab sites per target took lab SOC from ~0.2 to 0.54.
2. **AlphaEarth beats 30 m terrain on every target**, by about +0.05 R².
3. **Averaging AlphaEarth over years helps a little** (+0.01–0.02), most for sand and clay (+0.02).
4. **Climate is nearly as informative as AlphaEarth on its own** (pH 0.55 vs 0.56; SOC 0.51 vs 0.52), and
   the two are complementary. Adding climate and radiometrics gives +0.02–0.04 on every target, most for
   **sand (+0.04)**.
5. **Radiometrics adds signal where it exists.** It covers only 37% of sites, so its global effect is
   diluted; its value shows up in the US/Australia subset (feature-selection job).
6. **Texture is still furthest from its ceiling** (sand 0.43 vs 0.92; clay 0.36 vs 0.87). The inputs
   still don't see parent material or landscape position at the right scale.
7. **Feature-group importance** (job 35829185; drop-group = unique contribution, permutation = how much
   the model leans on the group):
   - **AlphaEarth has the largest unique contribution for every target** (drop-group +0.024 to +0.056).
   - **Radiometrics matter uniquely for sand** (+0.014 ± 0.006, the second-largest unique sand
     contribution), even at 37% coverage. Where present they carry heavy weight (permutation 0.20).
   - **Climate is used heavily but is redundant.** Shuffling the monthly normals costs 0.25–0.49 R², yet
     dropping them costs at most 0.010: annual normals and AlphaEarth carry the same information. Keep one
     form (monthly, for the temporal encoder), but don't expect more climate to help.
   - **Depth is big for SOC (+0.085) and BD (+0.022).** Topsoil SOC falls steeply with depth.
   - **Location adds nothing** (≤ 0.002). Good: the model isn't leaning on coordinates as a spatial proxy.
   - **Terrain at 30 m and ±500 m adds little uniquely** (≤ 0.008). That supports building coarse-scale
     terrain context rather than more fine-scale derivatives.

**soc** (all groups R² 0.539 ± 0.030, n 150,000)

| group | drop-group ΔR² | permutation ΔR² | SHAP share |
|---|---|---|---|
| depth | +0.085 ± 0.009 | 0.157 | 0.10 |
| alphaearth | +0.027 ± 0.005 | 0.099 | 0.26 |
| climate_monthly | +0.006 ± 0.003 | 0.488 | 0.47 |
| terrain | +0.005 ± 0.003 | 0.022 | 0.05 |
| location | +0.001 ± 0.001 | 0.010 | 0.01 |
| radiometrics | +0.001 ± 0.002 | 0.007 | 0.02 |
| climate_annual | +0.000 ± 0.002 | 0.042 | 0.08 |

**ph** (all groups R² 0.579 ± 0.036, n 150,000)

| group | drop-group ΔR² | permutation ΔR² | SHAP share |
|---|---|---|---|
| alphaearth | +0.024 ± 0.004 | 0.126 | 0.30 |
| depth | +0.004 ± 0.002 | 0.007 | 0.01 |
| climate_monthly | +0.004 ± 0.004 | 0.309 | 0.40 |
| radiometrics | +0.002 ± 0.003 | 0.009 | 0.02 |
| terrain | +0.001 ± 0.003 | 0.016 | 0.04 |
| location | +0.000 ± 0.003 | 0.007 | 0.01 |
| climate_annual | +0.000 ± 0.001 | 0.197 | 0.21 |

**sand** (all groups R² 0.423 ± 0.019, n 150,000)

| group | drop-group ΔR² | permutation ΔR² | SHAP share |
|---|---|---|---|
| alphaearth | +0.030 ± 0.005 | 0.146 | 0.35 |
| radiometrics | +0.014 ± 0.006 | 0.199 | 0.15 |
| climate_monthly | +0.007 ± 0.004 | 0.254 | 0.37 |
| terrain | +0.006 ± 0.003 | 0.033 | 0.06 |
| location | +0.002 ± 0.002 | 0.023 | 0.03 |
| climate_annual | +0.002 ± 0.003 | 0.015 | 0.03 |
| depth | +0.000 ± 0.004 | 0.002 | 0.01 |

**clay** (all groups R² 0.353 ± 0.011, n 150,000)

| group | drop-group ΔR² | permutation ΔR² | SHAP share |
|---|---|---|---|
| alphaearth | +0.044 ± 0.004 | 0.163 | 0.36 |
| depth | +0.011 ± 0.004 | 0.022 | 0.04 |
| climate_monthly | +0.010 ± 0.009 | 0.222 | 0.38 |
| terrain | +0.008 ± 0.002 | 0.053 | 0.07 |
| radiometrics | +0.005 ± 0.005 | 0.038 | 0.05 |
| location | +0.001 ± 0.005 | 0.052 | 0.07 |
| climate_annual | +0.000 ± 0.004 | 0.019 | 0.04 |

**bd** (all groups R² 0.502 ± 0.038, n 58,603)

| group | drop-group ΔR² | permutation ΔR² | SHAP share |
|---|---|---|---|
| alphaearth | +0.056 ± 0.016 | 0.189 | 0.34 |
| depth | +0.022 ± 0.004 | 0.047 | 0.05 |
| terrain | +0.002 ± 0.001 | 0.027 | 0.07 |
| radiometrics | +0.001 ± 0.004 | 0.017 | 0.04 |
| climate_annual | -0.002 ± 0.002 | 0.019 | 0.05 |
| climate_monthly | -0.002 ± 0.007 | 0.342 | 0.43 |
| location | -0.003 ± 0.007 | 0.015 | 0.03 |

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
