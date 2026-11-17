# GeoSoil data and feature pipeline: design

Written 2026-11-16 at the start of Phase 2. The design takes the patterns that hold up in
production geospatial pipelines and fixes the weaknesses those pipelines commonly have. All code here
is written for GeoSoil, on public data only.

## Shape

```
data/raw/<source>/                downloads, cached; each has a .json sidecar (url, sha256, time)
        │  geosoil/corpus/<source>.py   one loader per label source → canonical long table
        ▼
data/processed/labels_<source>.parquet
        │  dedup → harmonize → splits → noise floor        (python -m geosoil.corpus.build_c1)
        ▼
data/processed/labels_c1 / labels_harmonized / splits      the label side ("silver")
        │
        │  geosoil/features/<source>.py  one extractor per input source
        ▼
data/processed/features/<source>/<run>/sites-<task>-of-<n>.parquet  (+ .json manifest)
data/processed/features/<source>/<run>/unlabelled-<task>-of-<n>.parquet   Phase 3 seed
        │  join on (site_id, year)
        ▼
training tables / Zarr shards for the model                the "gold" side
```

## Principles

1. **One module per source, same shape.**
   - Each label source has `load() -> (labels, drops)`.
   - Each feature source has `plan` / `extract` / `status`.
   - Adding a source never touches the others, and one failing source can't break the rest.
2. **Stage data we control; don't call live point APIs at run time.** Read cloud-optimized files (COG,
   Zarr) over HTTPS or S3, or stage them to `/scratch`. Third-party uptime and quotas then never decide
   whether a job finishes. This is why Earth Engine is out of the critical path.
3. **Read by block, never by scattered point.** For tiled sources the unit of work is a storage block.
   - A single AlphaEarth pixel costs ~38 MB to read, while its whole 1024² block costs ~37 B per pixel.
   - Sites are grouped by (tile, block) and each block is read once.
   - Every block read can also yield unlabelled pixels for pretraining at no extra cost.
4. **Scale-aware fan-out.** Coarse sources (ERA5-Land 9 km, SMAP 9 km, MODIS 500 m) are extracted once
   per unique cell and joined to sites. Only 10 m sources are read per pixel.
5. **One point-sampling function per raster type** (`geosoil/features/rasters.py` for every static grid). It reprojects points into the raster's CRS, maps
   nodata to NaN, and documents its point-support policy:
   - 10 m embeddings: the pixel containing the point
   - terrain derivatives: a stated window
   - coarse grids: the containing cell
6. **Point-in-time keys.** Time-varying features are keyed by `(site_id, year)`, never "latest", so
   backtests and the temporal hold-out (2024) can't leak future data.
7. **Idempotent shards plus reconcile.**
   - Work is split into hashed shards (`task` of `ntasks`). A finished shard file is never rewritten:
     writes go to `.tmp`, then an atomic rename.
   - `status` lists missing tasks and failed blocks, and re-running the array job finishes only those.
   - The hash assignment is stable, so a re-run never reshuffles work.
8. **Provenance on every shard.** The manifest records:
   - source URL and licence
   - sha256 of the source index and of the plan
   - code version (git sha), thread count and timing
   - failed blocks with their errors
9. **Fail loud, count failures, keep what worked.**
   - A failed block is logged in the manifest and the rest of the shard is still written.
   - Loaders return explicit `drops` counts: bad depth, out of range, no coordinates.
   - Nothing is silently returned empty.
10. **Additive schemas.** New columns are added; existing ones are never renamed or dropped in place.
    The canonical label schema (`geosoil/corpus/schema.py`) is validated on every write.
11. **Fixture-first tests.** No test touches the network. Each loader is tested on synthetic rows that
    use the source's real column names; the block extractor on a synthetic bottom-up COG.

## Where this improves on typical production pipelines

| Common weakness | GeoSoil |
|---|---|
| Per-point fetching in thread pools (tens of seconds per point) | Block reads in Slurm array jobs. All C1 sites × 9 years ≈ 624k blocks, run as 64 shards × 16 threads. |
| No per-row provenance | A manifest per shard: source, index and plan hashes, code version |
| Appends that duplicate rows on re-run | Shards are write-once; reruns skip what exists |
| Partitioning declared but never implemented | Run directory per year range; shards hashed by block |
| Two implementations of the same source drifting apart | One extractor per source, used for labelled sites and for unlabelled pixels |
| Single least-cloudy scene for imagery | Planned: SCL / cloud-probability masking plus seasonal median composites (bare-soil composite is a separate, documented rule) |
| Inconsistent point support across sources | A point-support policy written down per source (principle 5) |
| Coupling to a hosted export queue | No hosted compute in the loop; everything is a file read |

## Feature sources, in build order

| # | Source | Access (anonymous) | Unit of work | Status |
|---|---|---|---|---|
| 1 | AlphaEarth embeddings, 64-d, 2017–2025 | S3 COGs (`us-west-2.opendata.source.coop/tge-labs/aef`) | 1024² block | **built** (`geosoil/features/alphaearth.py`) |
| 2 | Terrain: elevation, slope, aspect, curvature, TPI (100 / 500 m), roughness, relief | Copernicus DEM 30 m COGs on AWS (`copernicus-dem-30m`) | 1° tile (whole tile if ≥10 sites, else windows) | **built** (`geosoil/features/terrain.py`); TWI needs flow accumulation, later |
| 3 | Climate: 1991–2020 monthly normals (ppt, tmax, tmin, pet, aet, def, soil, vpd, srad) + annual summaries (MAP, MAT, aridity, seasonality) | TerraClimate climatology NetCDF (CC0), staged | containing 4 km cell | **built** (`geosoil/features/climate.py`); monthly time series for the temporal encoder later |
| 4 | Gamma radiometrics K / U / Th + Th/K, U/Th | USGS NURE Bayesian grids, 2 km (CONUS, public domain); Radmap v4, 100 m (Australia, CC-BY) | containing cell | **built** (`geosoil/features/radiometrics.py`); Canada (OFR 2005-1413) waits on a projection check |
| 4b | Multi-scale terrain: DEV, TPI, relief, slope at 0.5/1/2/5/10/20 km + max-DEV scale | Copernicus DEM GLO-90 on AWS (`copernicus-dem-90m`, tileList.txt) | 1° tile + 8 neighbours, summed-area tables | **built** (`geosoil/features/terrain_ms.py`) |
| 4c | Lithology: GLiM 16 classes (global) + USGS SGMC generalized lithology (US), one-hot | GLiM FileGDB (Hamburg); SGMC FileGDB (ScienceBase) | point-in-polygon | **built** (`lithology.py`) |
| 4d | Land-surface temperature: monthly day/night, diurnal range, amplitude | OpenLandMap MOD11A2 1 km climatology (CC-BY-SA) | containing cell | **built** (`thermal.py`) |
| 4e | Depth to bedrock | ISRIC / Shangguan 2017, 250 m | containing cell | **built** (`regolith.py`) |
| 4f | Height above nearest drainage (cell, 270 m mean, 270 m min) | ASF GLO-30 HAND COGs (CC0) | 1° tile | **built** (`hand.py`) |
| – | Deferred: Hydrography90m (CC-BY-NC, ~0.4 MB/s server), Geomorpho90m (~80 GB/variable; covered by terrain_ms), HYDE 3.2.1 cultivation history and MODIS NPP (bigger processing), bare-soil composite (no global grid; must be computed) | | | planned |
| 5 | Sentinel-2 bare-soil + seasonal medians; Sentinel-1 | Planetary Computer / Earth Search STAC | MGRS tile × season | planned |
| 6 | MODIS NDVI / LST, SMAP | Planetary Computer, `earthaccess` | unique cell | planned |
| 7 | TESSERA embeddings (2024 only so far) | `geotessera` | tile | planned |
| 8 | EMIT minerals (drylands) | LP DAAC | granule | planned |

## Running on Alpine

```bash
sbatch scripts/slurm/corpus_build.sbatch                      # C1 + harmonize + splits + AlphaEarth plan
YEARS=2020 sbatch scripts/slurm/features_aef.sbatch           # 64 shards; NEIGHBORS=64 unlabelled px per block
sbatch scripts/slurm/features_terrain.sbatch                  # 16 shards, Copernicus DEM
sbatch scripts/slurm/features_static.sbatch                   # climate + radiometrics (one shard)
python -m geosoil.features.alphaearth status --ntasks 64 --years 2020   # then resubmit missing tasks
```

Throughput from the read-cost run: one block ≈ 7.7 s per reader, 66 MB/s per S3 stream. 64 tasks ×
16 threads keeps many reads in flight, so one year of C1 (~69k blocks, ~2.6 TB) should take hours, not
days. The first run will measure the real aggregate rate.
