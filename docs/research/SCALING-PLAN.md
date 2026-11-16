# Scaling GeoSoil into a soil foundation model on Alpine

Draft 2026-11-16, revised the same day against the 2024–2026 literature. Companion docs:
- CURC-PLAN.md: access, policy, allocations
- CORPUS-PLAN.md: data sources and acquisition
- EXECUTION-PLAN.md: the step-by-step schedule

## The goal and the rule

The goal is **the largest soil model that still improves on held-out, lab-measured soil**. Size is
predicted from the size of the unlabelled corpus (below), then confirmed by experiment. A rung is only
kept if the held-out probe improves.

## What changed in this revision, and why

| Old plan | New plan | Evidence |
|---|---|---|
| Chinchilla's ~20 tokens per parameter, with one location-day as a token | Count **distinct location-years**. Size the model with TESSERA v2's compute-optimal law. | TESSERA v2 (Feng et al. 2026, arXiv 2607.03949), 395 runs: encoder ∝ C^0.36, data ∝ C^0.63. One pixel-year series "carries far less independent signal than a text token". |
| Ladder to 1.5B on 3e10 tokens | ~30M → ~100M → ~250M → **0.5–1B on a dense C4 corpus** | The law predicts 1B only at ~4e9 location-years (1 epoch) or ~1e9 (4 epochs). MITRE (arXiv 2512.23903) saw no gain from 86M → 1.8B without more data. |
| About 3,000 daily steps fed to Mamba | **Weekly patches**: ~52 tokens per year. Mamba-2 stays as an ablation arm. | PatchTST (arXiv 2211.14730). TESSERA uses ~12 tokens per view. Under ~1k tokens, attention is fine (arXiv 2403.11144, 2603.24109). |
| Choose checkpoints by pretraining loss | Choose by **soil-probe skill under spatial CV** | TESSERA v2: loss vs downstream r = −0.18. Choosing by loss costs ~3.5× the compute (CI 2.6–5.5×) for the same skill. |
| Ship the biggest model | Train the biggest model as a teacher; **distill to a 20–50M student** for embeddings and inference | TESSERA v2: 7–44M students lose ≤0.015 vs their 2B teacher, after distilling on ~0.6–1.2B location-years. |

## How big a model each dataset supports

**Plug-in law.** TESSERA v2 prints exponents only. This anchor is our reconstruction from their 2B
teacher trained on 1.4e10 location-years:

    N_opt ≈ 2.06e9 × (D / 1.4e10)^0.571        D = unique location-years, 1 epoch
    with up to ~4 epochs of repetition, multiply N_opt by ~2.2 (4^0.571)

**Caveats:**
- The law was fit on pixel-wise Barlow Twins over S1/S2.
- Our inputs are multimodal and our objectives differ.
- Independent, well-spread locations carry more signal than TESSERA's within-patch pixels, which argues
  for the upper end of each range.
- Treat the law as the starting guess; the rung gate below makes the final call.

| Tier | What it is | Unique location-years | N_opt (1 → 4 epochs) | Feature store | Model trained there |
|---|---|---|---|---|---|
| C0 | today's labelled rows | 4e4 | 1.4 → 3M | <1 GB | **R0:** current presets. xl (31M) is already ~10× oversized. |
| C1 | all labelled sites, global, deduplicated (~300k) × 7 years | 2e6 | 13 → 29M | 3.4 GB | **R1: ~30M** |
| C2 | 2M scattered CONUS points × 8 years | 1.6e7 | 43 → 95M | 27 GB | **R2: ~100M** |
| C3 | 10M scattered global points × 8 years | 8e7 | 108 → 238M | 136 GB | **R3: ~250M** |
| C4 | dense: ~11k AlphaEarth tile-years × ~10k pixels each | 1e9 | 457M → 1B | 1.7 TB (in `/scratch`) | **R4: 0.5–1B teacher**, then a distilled student |

Model sizes on labelled data alone are small (C0 → ~3M). **Labels set the probe, not the model size.**
To make the model large we need many unlabelled locations, and C4 is the step that gets there. It is
feasible because:
- coarse modalities are shared across nearby points
- the 10 m embeddings are read a whole tile at a time (CORPUS-PLAN.md §B1).

## Compute check

The unit is one location-year: 52 weekly temporal tokens (ERA5-Land, MODIS, SMAP) plus ~12
static/annual tokens (embeddings, terrain, radiometrics, minerals, bare soil, parent material), so
**~64 tokens**.

Cost ≈ 6 · N · (tokens processed). Assume an A100 delivers 1e14 FLOP/s effective in bf16, and an H200
about 2.5×. Small models often reach half that, so **budget 2–3× these numbers**.

| Rung | N | Corpus, epochs | FLOPs | A100-h | H200-h | SU (H200, raw) | SU budget (×2) | Fits in |
|---|---|---|---|---|---|---|---|---|
| R0 | 0.4–31M | C0 | — | 16 (sweep) | — | — | — | Trailhead |
| R1 | 30M | C1 × 4 | 9e16 | 0.3 | — | — | — | Trailhead |
| R2 | 100M | C2 × 4 | 2.4e18 | 7 | 3 | 1k | 2k | Trailhead |
| R3 | 250M | C3 × 4 | 3e19 | 82 | 33 | 12k | 25k | Trailhead (~8 h on 4×H200) |
| R4a | 650M | C4 × 2 | 5e20 | 1,400 | 560 | 210k | ~420k | **Ascent** (450k SU) |
| R4b | 1B | C4 × 4 | 1.5e21 | 4,300 | 1,700 | 630k | ~1.3M | **Peak** (to 7.5M SU) or ACCESS Discover |
| Distill | 20–50M student | 0.6–1B location-years of cached teacher embeddings | small | <50 | | | | Trailhead |

Hours and SU are raw; the budget column doubles them. With ~400 tokens per sequence and width ≥512, attention
adds ~10% on top of 6·N·D.

**Conclusions:**
- Through R3, compute is not the constraint, even on free Trailhead. The constraints are the number of
  distinct locations we can build features for, the lab labels we probe with, and overfitting.
- R4 is the one step that needs a real allocation, and it is the step that makes the model "large".

## Model

**Backbone:**
1. A per-modality patch embedding (weekly temporal patches; one token per static group).
2. A **Transformer** over (time × modality) tokens, with factorized time/modality attention past ~400 tokens.
3. GeoSoil's cross-modal fusion, which produces the latent z.
4. Heads per target and per lab method.

**Ablation arms at equal FLOPs (R1):**
- GRU, Transformer and Mamba-2 on patched inputs
- an unpatched daily Mamba on 4–8-year windows, to test the long-sequence hypothesis

**Objectives** (in the repo): masked time-span JEPA, next-k-step forecasting, cross-modal InfoNCE, and
**modality dropout** (tolerates missing inputs; gives per-modality importance). For R4, add the TESSERA
recipe: train the teacher, cache its embeddings, then distill Matryoshka students with a cosine loss
over embedding prefixes.

**External foundation models:**
- **Baselines** at every rung: AlphaEarth, TESSERA, Galileo, Presto frozen embeddings + linear probe.
- **Extra input:** the TESSERA embedding.
- **Initialization:** Galileo's encoder vs training from scratch, compared at R1.

## Overfitting guards (they decide "as large as possible")

**What must be held out:**
- **Pretraining validation:** held-out 1° spatial blocks, plus the latest year both embedding sources cover.
- **Probe and fine-tune evaluation:**
  - 5-fold spatial-block CV on lab labels only
  - leave-region-out
  - KSSL → LUCAS and US → Brazil transfer
  - RaCA held out entirely
  - area-of-applicability masks (Meyer & Pebesma 2021)
- **Noise floor:** KSSL lab replicates. Never report skill above it.

**Rules:**
- **Rung gate:** keep the larger rung only if the probe improves by at least 0.02 R² on lab SOC, or on
  the 5-target mean, across 3 seeds. If not, the data is the limit: grow the corpus tier, not the model.
- **Epoch cap:** 4 epochs.
- **No model-product labels in evaluation, ever.**

## Engineering checklist

- **Environments:** `environment-cuda.yml` (cu121) for A100/H200; `environment-cu128.yml` for RTX Pro 6000.
  Use `module load miniforge` with envs in `/projects`.
- **Parallelism:** DDP on one node (3 GPUs on `aa100`; 4 on `ah200`/`artxpro6000`). FSDP only for R4b at
  1B, and only if memory requires it (4×H200 = 564 GB). No multi-node (no InfiniBand).
- **Training settings:** bf16, `torch.compile`, activation checkpointing above 400 tokens. 6–8 DataLoader
  workers per GPU.
- **Data:** Zarr / WebDataset shards; stage to `$SLURM_SCRATCH` (5.4 TB on H200/RTX nodes). The C4 feature
  store (1.7 TB) lives in `/scratch/alpine`, which purges after 90 days, so keep the build script
  rerunnable.
- **Checkpointing:** every 30 min, with `#SBATCH --signal=B:USR1@300` and a save-and-requeue trap.
- **Testing and tracking:** smoke tests on `gpu-testing`; sweeps as array jobs; wandb offline.
- **Policy:** public data, open results, every source in `data/corpus/sources.yaml`.
