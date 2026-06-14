# GeoSoil Frontend — Plan

Aligned to `portfolio_spec.md`. The site's job (per the spec and the resume review):
**convert Luke's unverifiable resume claims into clickable, falsifiable evidence**
in under three minutes. The reviewer's exact gap was *"no paper, demo, or production
metric I can click."* GeoSoil now supplies all three: a white paper, real
spatial-CV results, and an interactive research tool.

## Goals (from the spec)
- **Show the science, not the result.** Lead with method: baselines, spatial CV,
  ablations, uncertainty, and honest limitations — all of which GeoSoil has.
- **Every claim clickable or falsifiable.** A number sits next to the figure that
  proves it.
- **Minimal, refined, figure-led.** Figures *are* the design; restraint over motion.
- **IP-safe.** All public data; the SoilMetrix-derived code is already sanitized.

## Where it fits
This is flagship case study **2A (Soil Spectral / State Transformer)** in the
4-page site (Home · Projects · Case studies · About). The GeoSoil case-study page
is the centerpiece; an embedded **interactive research tool** is the differentiator
that makes it an *exhibit you operate*, not read.

## Visual design (from spec §5 — match the resume)
- Near-black text on off-white; one accent **slate-navy `#20425E`**. No gradients/neon.
- Body **Inter / IBM Plex Sans**; **JetBrains Mono / IBM Plex Mono** for metrics, axis
  labels, code. The mono accents read "engineer."
- Single ~680px prose column; figures full-width, crisp SVG/high-DPI, labeled & captioned.
- KaTeX for real notation only (Gaussian NLL, InfoNCE, the VICReg/JEPA losses).
- Subtle fade-ins at most.

## Case-study page structure (spec §3 template → GeoSoil content)
1. **TL;DR** — thesis + the headline figure (bake-off or pred-vs-obs).
2. **Motivation** — representations vs point predictions; label circularity.
3. **Data** — 11 public modalities + two ground truths; harmonization; spatial-block split.
4. **Method** — architecture SVG; the six objectives; the LoRA/efficiency angle; KaTeX losses.
5. **Experiments** — baselines (LightGBM/XGBoost/CatBoost), the GRU/Transformer/Mamba
   comparison, modality ablations.
6. **Results** — R²/RMSE/RPIQ tables; pred-vs-obs with 1:1 line; calibration; retrieval.
7. **The multi-truth finding** — the standout: texture R²<0 on lab truth with AlphaEarth
   alone, recovered to ~0.3 with radar/bare-soil. (This *is* the spec's "spatial-CV
   rigor" signal, taken further.)
8. **Limitations & frontier** — proximal sensing for SOC/texture; the honest ceiling.
9. **Reproducibility** — link the white paper PDF + (sanitized) repo; "what I'd do next."

## The interactive research tool
Two tiers. **Everything in Tier A needs no backend** — it ships as a static site and
is reliable and free, which the spec prioritizes ("pick the lowest-friction stack
you'll actually finish").

### Tier A — precomputed, static, instant (build this)
- **CONUS prediction map** — run the model offline over a grid; serve per-property
  μ and σ as raster tiles (COG/PMTiles) or a quantized lookup. Click a point → soil
  report card with **uncertainty bands**. Layer toggles + legend. (MapLibre GL, no token.)
- **Latent explorer** — precomputed UMAP of the 256-d `z`, brushable; selecting points
  highlights them on the map and vice-versa; color by SOC / land cover / crop class.
- **Multi-truth panel** — click a KSSL site → prediction vs OpenLandMap vs lab value,
  with the distance shown. Visualizes the headline finding interactively.
- **Static science figures** — pred-vs-obs, calibration curve, retrieval, ablations,
  architecture diagram: exported as SVG from the real Python (no faked JS numbers).

### Tier B — live single-point inference ("very cool", optional flex)
*Difficulty: moderate. Recommended as a v2 add-on behind an explicit "compute live"
button, not the default path.*

- The model is tiny (<100 ms/point on CPU), so inference is trivial. **The hard part
  is fetching the 11 modalities at an arbitrary clicked point**, which requires Google
  Earth Engine at request time (AlphaEarth, S2, S1, ERA5, MODIS, SMAP, CHIRPS, CDL,
  DEM). That means:
  - a hosted backend (FastAPI) with a **GEE service account** (not free static hosting),
  - **~10–30 s latency per click** (dominated by GEE `reduceRegion` calls), and
  - GEE quota / cold-start management.
- **Verdict:** feasible but not instant. Best as a gated "Fetch live from satellite
  (~20 s)" button with a spinner that proves the pipeline is real, *alongside* the
  instant precomputed path. Do **not** make live-fetch the primary interaction —
  latency would hurt the <3-minute reviewer experience.
- **Cheaper middle option:** pre-cache a few dozen "try me" points (run the real GEE
  pipeline offline, store inputs+outputs) so a visitor can trigger genuine end-to-end
  inference on those points instantly, demonstrating the live path without a live backend.

## Stack
- **Astro** (or Next.js) + **MDX** — mix prose, KaTeX, code, and embedded charts/maps.
- **MapLibre GL JS** for the map (open, no token); **PMTiles** for static prediction tiles.
- Figures exported from the existing Python (`geosoil/results/*.png`, regenerated as SVG).
- Host **Vercel / GitHub Pages**, custom domain; site repo public (doubles as a code sample).
- Tier B backend (optional): **FastAPI** + `earthengine-api` + the trained model.

## Artifacts already in hand
- White paper PDF (`docs/whitepaper/main.pdf`), all result JSON (`geosoil/results/`),
  pred-vs-obs + UMAP figures, the SOTA survey and data roadmap. The case-study prose
  can be lifted largely from the white paper.

## Build order (ship fast)
1. Case-study page (MDX) with the static figures + white paper — this alone satisfies
   the spec's flagship requirement and is the highest-credibility unit.
2. Tier-A interactive map + latent explorer + multi-truth panel (precomputed).
3. Framing pages (Home, Projects index, About/Resume).
4. (Optional v2) Tier-B live-fetch backend + the "try me" pre-cached points.
5. Custom domain + deploy.

> Recommendation: build Tier A (static, instant, free) first — it fully delivers the
> "interactive research tool featuring the science." Add Tier B live inference as a
> visible-but-optional flex once the core ships.
