---
version: 0.3.0
date: 2026-03-01
status: complete
---

# Scan Results: Local Pipeline (Rijksmuseum + Met)

## Overview

Expanded the Colab prototype to a local pipeline (`scan.py`) combining Rijksmuseum and Met Open Access. Tested two feature configurations.

## Dataset

| Group | v1 count | v2 count | Source breakdown |
|-------|----------|----------|-----------------|
| Autograph | 29 | 29 | 19 RIJ + 10 MET |
| Circle/disputed | 18 | 18 | 2 RIJ + 16 MET |
| Pupils | 33 | 32 | 27 RIJ + 6 MET |
| Other Dutch | 28 | 22 | 8 RIJ + 20 MET |
| **Total** | **108** | **101** | 56 RIJ + 52 MET |

v2 lost 7 paintings due to Rijksmuseum IIIF SSL failures on very large native-resolution images.

Circle/disputed group expanded from N=2 (prototype) to N=18 — the primary goal of the Met expansion.

## v1: Low-res, mean-only (1536d)

**Config:** 2000px images, 224×224 tiles (~60 tiles/painting), DINOv2 ViT-B/14, mean CLS + mean patch = 1536d.

| Metric | Value |
|--------|-------|
| Auto↔Auto sim | 0.8462 |
| Auto↔Circle sim | 0.8478 |
| Auto↔Pupil sim | 0.8118 |
| Auto↔Other sim | 0.8165 |
| MW p (vs circle) | **0.805** — no signal |
| MW p (vs pupils) | **2.03×10⁻¹¹** — strong |
| MW p (vs other) | **3.40×10⁻¹⁰** — strong |
| KNN K=3 | 73.1% |
| KNN K=5 | 75.9% |
| KNN K=7 | 77.8% |
| Baseline | 73.1% |

## v2: High-res, mean+std (3072d)

**Config:** Native-resolution images (capped 8000px), 192–696 tiles/painting, DINOv2 ViT-B/14, mean+std of CLS and patch = 3072d.

| Metric | Value |
|--------|-------|
| Auto↔Auto sim | 0.9064 |
| Auto↔Circle sim | 0.9062 |
| Auto↔Pupil sim | 0.9078 |
| Auto↔Other sim | 0.8870 |
| MW p (vs circle) | **0.335** — no signal |
| MW p (vs pupils) | **0.146** — signal lost |
| MW p (vs other) | **7.36×10⁻¹⁵** — strongest yet |
| KNN K=3 | 67.3% |
| KNN K=5 | 68.3% |
| KNN K=7 | 68.3% |
| Baseline | 71.3% |

## Comparison: Prototype → v1 → v2

| Metric | Prototype (RIJ only) | v1 (RIJ+MET) | v2 (high-res) |
|--------|---------------------|---------------|---------------|
| N paintings | 70 | 108 | 101 |
| N circle | 2 | 18 | 18 |
| Circle p-value | 0.088 (N=2) | 0.805 | 0.335 |
| Pupils p-value | 0.001 | 2.03e-11 | 0.146 |
| Other p-value | 0.002 | 3.40e-10 | 7.36e-15 |
| KNN best | 82.9% | 77.8% | 68.3% |
| KNN baseline | 72.9% | 73.1% | 71.3% |

## Key Findings

1. **Circle/autograph separation does not exist in DINOv2 ViT-B/14 features.** At N=18 with two feature configs, the circle paintings are statistically indistinguishable from autographs. This is either because (a) "Style of Rembrandt" paintings genuinely look like Rembrandts to a vision model, or (b) ViT-B/14 lacks the resolution to capture the differences art historians use.

2. **High-res hurts more than it helps.** All similarities pushed toward 0.90, collapsing the between-group discrimination that existed at 2000px. The std features add noise. More tiles per painting capture shared Dutch Golden Age texture (canvas, craquelure, varnish) that drowns out artist-specific style.

3. **Pupil/other separation is real and robust (v1).** p=2e-11 for pupils and 3e-10 for other Dutch masters. DINOv2 can reliably tell "different artist" but not "same artist, different quality level."

4. **The scanner is a "different artist" detector, not an authenticator.** Useful for catching misattributed pupils or other-artist works sold as Rembrandt. Not useful for the harder circle/autograph question without better features.

## Top Candidates (v2, by similarity to autograph centroid)

| Rank | Sim | Source | Title | Creator |
|------|-----|--------|-------|---------|
| 1 | 0.9844 | MET | Old Woman Cutting Her Nails | Style of Rembrandt |
| 2 | 0.9754 | MET | Portrait of a Man ("The Auctioneer") | Follower of Rembrandt |
| 3 | 0.9688 | MET | Man with a Steel Gorget | Style of Rembrandt |
| 4 | 0.9672 | MET | Man with a Beard | Style of Rembrandt |
| 5 | 0.9646 | MET | Head of Christ | Style of Rembrandt |

## v1-L: ViT-L/14 backbone (2048d)

**Config:** Same as v1 (2000px, mean-only) but DINOv2 ViT-L/14 (1024d per token → 2048d concatenated). Batch size 8 (vs 16 for ViT-B) to fit 8 GB unified memory.

| Metric | Value |
|--------|-------|
| Auto↔Auto sim | 0.8650 |
| Auto↔Circle sim | 0.8709 |
| Auto↔Pupil sim | 0.8366 |
| Auto↔Other sim | 0.8450 |
| MW p (vs circle) | **0.831** — no signal (worse than ViT-B) |
| MW p (vs pupils) | **3.59×10⁻¹¹** — strong |
| MW p (vs other) | **8.85×10⁻⁸** — strong (weaker than ViT-B) |
| KNN K=3 | 75.9% |
| KNN K=5 | 75.9% |
| KNN K=7 | 70.4% |
| Baseline | 73.1% |

## Comparison: Prototype → v1 → v2 → v1-L

| Metric | Prototype (RIJ only) | v1 ViT-B | v2 (high-res) | v1-L ViT-L |
|--------|---------------------|----------|---------------|------------|
| N paintings | 70 | 108 | 101 | 108 |
| N circle | 2 | 18 | 18 | 18 |
| Circle p-value | 0.088 (N=2) | 0.805 | 0.335 | **0.831** |
| Pupils p-value | 0.001 | 2.03e-11 | 0.146 | 3.59e-11 |
| Other p-value | 0.002 | 3.40e-10 | 7.36e-15 | 8.85e-08 |
| KNN best | 82.9% | 77.8% | 68.3% | 75.9% |
| KNN baseline | 72.9% | 73.1% | 71.3% | 73.1% |
| Embed dim | 1536 | 1536 | 3072 | 2048 |

## Key Findings

1. **Circle/autograph separation does not exist in DINOv2 features — regardless of model size.** Tested ViT-B/14 (768d) and ViT-L/14 (1024d, 3.5× larger). Circle p-value went from 0.805 → 0.831 (worse). The problem is not model capacity.

2. **ViT-L shifts all similarities upward uniformly.** Every pairwise similarity increased by ~0.02–0.03. The bigger model sees Rembrandt-school paintings as more alike, not less. It captures shared style more strongly but doesn't resolve within-school differences.

3. **ViT-L weakens other-Dutch separation.** p went from 3.4e-10 → 8.9e-8 — still strong, but two orders of magnitude weaker. The extra capacity dilutes the between-artist signal while amplifying the within-school similarity.

4. **High-res hurts more than it helps (v2).** All similarities pushed toward 0.90, collapsing the between-group discrimination that existed at 2000px. The std features add noise. More tiles per painting capture shared Dutch Golden Age texture (canvas, craquelure, varnish) that drowns out artist-specific style.

5. **Pupil/other separation is real and robust (v1 ViT-B).** p=2e-11 for pupils and 3e-10 for other Dutch masters. DINOv2 can reliably tell "different artist" but not "same artist, different quality level."

6. **The scanner is a "different artist" detector, not an authenticator.** Useful for catching misattributed pupils or other-artist works sold as Rembrandt. Not useful for the harder circle/autograph question without fundamentally different features.

## Eliminated Options

| Option | Result | Conclusion |
|--------|--------|------------|
| C: ViT-L/14 | Circle p=0.831, worse than ViT-B | Model capacity is not the bottleneck |
| v2: High-res + std | All sims→0.90, lost pupil signal | More tiles + std features add noise |

## Next Steps

| # | Approach | Hypothesis | Effort |
|---|----------|-----------|--------|
| D | **Entropy-weighted tiles** | Weight complex tiles (brushwork) higher, suppress flat areas (sky, background) | ~1 hr code + rerun |
| E | **Per-tile classification** | Instead of mean pooling, classify individual tiles and vote | ~2 hr |
| F | **Linear probe** | Thin supervised layer on top of frozen DINOv2 | ~2 hr |

Recommendation: D next — the aggregation method (mean pooling) may be washing out discriminative tiles. If D doesn't help, move to F (supervised signal).

## Files

| File | What |
|------|------|
| `cache/embeddings/embeddings.npz` | v1 ViT-B embeddings (1536d, 108 paintings) |
| `cache/embeddings/embeddings_v2.npz` | v2 ViT-B embeddings (3072d, 101 paintings) |
| `cache/embeddings/embeddings_vitl.npz` | v1-L ViT-L embeddings (2048d, 108 paintings) |
| `cache/results.json` | v1 ViT-B metrics |
| `cache/results_v2.json` | v2 metrics |
| `cache/results_vitl.json` | v1-L ViT-L metrics |
| `cache/plots/` | UMAP, cosine, heatmap for latest run |
