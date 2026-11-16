# Scaling GeoSoil into a soil foundation model on Alpine

Draft 2026-11-16, revised the same day against the 2024–2026 literature. Companion docs:
- CURC-PLAN.md: access, policy, allocations
- CORPUS-PLAN.md: data sources, sampling, feature selection

## The goal and the rule

The goal is **the largest soil model that still improves on held-out, lab-measured soil**.
Parameters are not the target in themselves. Size is decided by experiment: go up a rung only while the
held-out probe keeps improving (see "Overfitting guards").

## What changed in this revision, and why

| Old plan | New plan | Evidence |
|---|---|---|
| Chinchilla's ~20 tokens per parameter, with one location-day as a token | Count **distinct location-years**. Daily steps at one place are highly autocorrelated, so they are not independent tokens. | TESSERA v2 (Feng et al. 2026, arXiv 2607.03949), 395 runs: the compute-optimal split for pixel time series leans heavily toward data (encoder ∝ C^0.36, data ∝ C^0.63), because one pixel series "carries far less independent signal than a text token". |
| Ladder up to 1.5B parameters | 10M → 30M → 100M → 300M, gated. A 1B **teacher** is optional, and only for distilling a smaller student. | MITRE (Wickrema et al. 2025, arXiv 2512.23903): scaling the model 86M → 1.8B gave no measurable gain on EO data. TESSERA v2: a 44M student distilled from a 2B teacher beat every embedding product tested, AlphaEarth included. |
| About 3,000 daily steps fed to Mamba | Daily data stored, fed as **weekly patches**: about 52 tokens per year, 100–400 per multi-year window. Mamba-2 stays as an ablation arm. | PatchTST (arXiv 2211.14730). TESSERA found 8–16 observations per view beat 32–64. Under about 1k tokens, Mamba's cost advantage doesn't matter (arXiv 2403.11144; Dumeur et al. 2026, arXiv 2603.24109). |
| Choose checkpoints by pretraining loss | Choose by **soil-probe skill under spatial CV** | TESSERA v2: pretraining loss barely predicts downstream skill (r = −0.18). Choosing by loss costs 2–5× the compute for the same skill. |
| Scale only GeoSoil's own encoder | Also use open EO foundation models (Galileo, TESSERA, Presto) as **baselines to beat, as frozen embeddings, and as initializations** | See CORPUS-PLAN.md §B. |

## Data unit and corpus tiers

The unit is one **location-year**. It holds:
- 52 weekly temporal tokens: ERA5-Land, MODIS, SMAP
- about 12 static or annual tokens: AlphaEarth and TESSERA embeddings, terrain, radiometrics, bare-soil composite, parent material
- so about **64 tokens per location-year**.

Years 2017–2024 are for training. **2025 is held out** as a temporal test, since AlphaEarth and TESSERA both cover 2017–2025.

| Tier | Locations | Location-years | Tokens (×64) | Feature store on disk | Labels |
|---|---|---|---|---|---|
| C0 today | 43k CONUS lab rows (KSSL + WoSIS) plus 3k OpenLandMap | ~4e4 | 3e6 | MBs | 43k |
| C1 labelled, global, deduplicated | ~250k (KSSL LDM, WoSIS CC-BY, OSSL, LUCAS, RaCA, national sets) | 2e6 | 1.3e8 | ~4 GB | ~250k |
| C2 CONUS unlabelled | 2M (stratified / cLHS sample) | 1.6e7 | 1e9 | ~30 GB | C1 |
| C3 global unlabelled | 10M | 8e7 | 5e9 | ~140 GB | C1 |

These counts assume about 1.7 KB per location-year at float16/int8. The feature store fits in `/projects`
(250 GB). The **raw** rasters are what need `/scratch/alpine` and staged downloads (see CORPUS-PLAN.md).

## Compute check

Training cost is about 6 · N · D FLOPs. D counts the tokens processed, capped at 4 epochs
(Muennighoff et al. 2023: up to about 4 epochs of repeats is nearly as good as fresh data).
Assume an A100 delivers 1e14 FLOP/s effective in bf16 and an H200 about 2.5×. Small models with heavy
data loading often reach only half that, so **budget 2–3× the numbers below**.

| Rung | N | Corpus | D (4 epochs) | FLOPs | A100-h | Fits in |
|---|---|---|---|---|---|---|
| R0 | 0.4–31M (current presets) | C0 | — | — | 4×4 | gpu-testing / Trailhead |
| R1 | 30M | C1 | 5e8 | 9e16 | 0.3 | Trailhead |
| R2 | 100M | C2 | 4e9 | 2.4e18 | 7 | Trailhead |
| R3 | 300M | C3 | 2e10 | 3.6e19 | 100 | Trailhead, about 10 h on 4×H200 |
| R4 (teacher, optional) | 1B | C3 | 2e10 | 1.2e20 | 330 | Trailhead stretch, or Ascent |

With up to about 400 tokens per sequence and a width of 512 or more, attention adds about 10% on top of 6·N·D.

**Conclusion: compute is not the binding constraint, even on free Trailhead.** The limits are:
1. how many distinct, well-spread locations we can build features for (an I/O and engineering problem),
2. how many lab labels we have to probe with,
3. overfitting.

Ascent becomes worth it for repeated seeds, ablations, and the R4 teacher. It does not buy a much larger model.

## Model

**Backbone:**
1. A per-modality patch embedding (weekly temporal patches; one token per static group).
2. A **Transformer** encoder over (time × modality) tokens, with factorized time/modality attention once sequences pass about 400 tokens.
3. GeoSoil's existing cross-modal fusion, which produces the latent z.
4. Separate heads per target and per lab method (see CORPUS-PLAN.md §E).

**Ablation arms at equal FLOPs (R1):** GRU, Transformer, and Mamba-2 on the same patched inputs. Add an
unpatched daily Mamba arm on 4–8-year windows to test the long-sequence hypothesis directly.

**Self-supervised objectives** (all already in the repo):
- masked time-span JEPA
- next-k-step forecasting (`geosoil/forecasting.py`)
- cross-modal InfoNCE
- **modality dropout**: drop whole modalities at random during pretraining. Inference then tolerates missing inputs, and it doubles as a per-modality importance measure.

**External foundation models:**
- **Baselines to beat** at every rung: frozen embeddings + linear probe from AlphaEarth, TESSERA, Galileo and Presto.
- **Extra input modality:** the TESSERA 128-d embedding next to AlphaEarth's 64-d.
- **Initialization:** start from Galileo's pixel-time-series encoder. Its inputs (S1, S2, ERA5, TerraClimate, SRTM, Dynamic World) closely match ours. Compare against training from scratch at R1.

## Overfitting guards (they decide "as large as possible")

**What must be held out:**
- **Pretraining validation:** held-out spatial blocks (1° cells), plus the year 2025.
- **Probe and fine-tune evaluation:**
  - 5-fold spatial-block CV on lab labels only
  - leave-region-out
  - transfer tests: KSSL → LUCAS, and RaCA held out entirely
  - area-of-applicability masks (Meyer & Pebesma 2021)
- **Noise floor:** KSSL lab replicates set the ceiling on achievable R². Never report skill above it.

**Rules:**
- **Rung gate:** move to the next rung only if the probe improves by at least 0.02 R² on lab SOC, or on the mean over the five targets, across 3 seeds, at the same corpus tier. If the gain is under that, the data is the limit: grow the corpus, not the model.
- **Epoch cap:** stop at 4 epochs, then add new locations instead of repeating.
- **No model-product labels in evaluation, ever.** OpenLandMap, POLARIS and SoilGrids may appear only as auxiliary pretraining targets, and only as an ablation, because copying them is the failure mode this project already found.

## Engineering checklist

- **Environment:**
  - Keep the cu121 environment for A100 and H200.
  - Add `environment-cu128.yml` (torch ≥2.7) for RTX Pro 6000.
  - Use `module load miniforge`, with conda envs in `/projects`.
- **Parallelism:** DDP on one node: 3 GPUs on `aa100`, 4 on `ah200`/`artxpro6000`. FSDP only above about 2B parameters, which this plan never reaches. No multi-node training (no InfiniBand).
- **Training settings:**
  - bf16, `torch.compile`, activation checkpointing for windows over 400 tokens
  - DataLoader workers: about 6–8 per GPU, within the 21/32 cores-per-GPU limit
- **Data format:** Zarr or WebDataset shards in `/projects` (feature store) or `/scratch/alpine` (raw). At job start, stage to `$SLURM_SCRATCH`.
- **Checkpoint and resume:** checkpoint every 30 min. Use `#SBATCH --signal=B:USR1@300` with a trap that saves and requeues, since gpu-normal caps jobs at 24 h and gpu-long at 7 days.
- **Testing and sweeps:** smoke tests on `gpu-testing` (A100 MIG slice, 10% billing); sweeps as array jobs.
- **Tracking:** wandb offline (sync from a login node), or a CSV logger.
- **Policy:** public data only, open results, and record the license of every source in `data/corpus/sources.yaml`.

## Order of work

1. **R0 now:** run the bootstrap, then `scaling_sweep.sbatch` (4 sizes × 2 label sources). This is the baseline, and it shows the kernels work.
2. **C1:** labelled corpus with dedup and method harmonization (CORPUS-PLAN.md C1). Then R1: architecture bake-off, external-FM baselines, Galileo initialization versus scratch.
3. **C2:** CONUS unlabelled corpus. Then R2. Fit loss and probe curves against N and D.
4. **Apply for Ascent** (or ACCESS Explore with an advisor as co-PI), using the R1/R2 job numbers.
5. **C3 + R3.** Then R4 teacher → distilled student, only if the R2 → R3 gain passes the rung gate.
6. **Write-up:** STAT 5010 / CSCI 5622 material, the white paper's "future work" made real, and released weights and embeddings (CC-BY, data licenses permitting).
