---
version: 0.4.0
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

## v1-D: Entropy-weighted tile aggregation (1536d)

**Config:** Same as v1 (2000px, ViT-B/14) but tiles weighted by DINOv2 patch token variance instead of uniform mean. Variance = visual complexity proxy — high for brushwork/faces/hands, low for flat sky/background. Weight function: z-score → softmax (self-calibrating, no temperature hyperparameter). Degenerate guard: uniform fallback if std < 1e-6.

**Diagnostics (first 3 paintings):** Variance range ~1.0–2.3, max/uniform weight ratios 3.6–10.5×. Weighting is meaningfully non-uniform.

| Metric | Value |
|--------|-------|
| Auto↔Auto sim | 0.7921 |
| Auto↔Circle sim | 0.7945 |
| Auto↔Pupil sim | 0.7857 |
| Auto↔Other sim | 0.7681 |
| MW p (vs circle) | **0.615** — no signal (improved from 0.805, still nowhere near significant) |
| MW p (vs pupils) | **3.70×10⁻²** — weak (degraded from 2e-11) |
| MW p (vs other) | **1.02×10⁻⁶** — strong (degraded from 3e-10) |
| KNN K=3 | 75.0% |
| KNN K=5 | 78.7% |
| KNN K=7 | 75.9% |
| Baseline | 73.1% |

## Comparison: Prototype → v1 → v2 → v1-L → v1-D

| Metric | Prototype (RIJ only) | v1 ViT-B | v2 (high-res) | v1-L ViT-L | v1-D Entropy |
|--------|---------------------|----------|---------------|------------|--------------|
| N paintings | 70 | 108 | 101 | 108 | 108 |
| N circle | 2 | 18 | 18 | 18 | 18 |
| Circle p-value | 0.088 (N=2) | 0.805 | 0.335 | 0.831 | **0.615** |
| Pupils p-value | 0.001 | 2.03e-11 | 0.146 | 3.59e-11 | 3.70e-02 |
| Other p-value | 0.002 | 3.40e-10 | 7.36e-15 | 8.85e-08 | 1.02e-06 |
| KNN best | 82.9% | 77.8% | 68.3% | 75.9% | 78.7% |
| KNN baseline | 72.9% | 73.1% | 71.3% | 73.1% | 73.1% |
| Embed dim | 1536 | 1536 | 3072 | 2048 | 1536 |

## Key Findings

1. **Circle/autograph separation does not exist in DINOv2 features — regardless of model size or aggregation method.** Tested ViT-B/14, ViT-L/14 (3.5× larger), and entropy-weighted aggregation. Best circle p=0.615 (entropy). Four experiments, none below 0.3.

2. **Entropy weighting trades pupil signal for marginal circle improvement.** Circle p improved 0.805→0.615 but pupil p collapsed from 2e-11→0.037 (five orders of magnitude worse). The weighting amplifies noise in complex tiles, not style signal. All similarities dropped ~0.05 uniformly.

3. **Aggregation is not the bottleneck.** Mean pooling wasn't washing out discriminative signal — the signal for circle/autograph separation simply isn't in generic DINOv2 embeddings. Confirmed by entropy weighting making things worse overall despite being non-trivially different (max/uniform weight ratios up to 10.5×).

4. **High-res hurts more than it helps (v2).** All similarities pushed toward 0.90, collapsing between-group discrimination. The std features add noise.

5. **ViT-L shifts all similarities upward uniformly.** Bigger model sees Rembrandt-school paintings as more alike, not less. Doesn't resolve within-school differences.

6. **Pupil/other separation is real and robust — but only with v1 ViT-B baseline.** p=2e-11 for pupils and 3e-10 for other Dutch masters. Every variation tested (v2, ViT-L, entropy) degrades this signal. DINOv2 ViT-B mean pooling is the sweet spot for "different artist" detection.

7. **The scanner is a "different artist" detector, not an authenticator.** Four feature configurations all confirm this. The circle/autograph question requires supervised learning, not better unsupervised features.

## Eliminated Options

| Option | Result | Conclusion |
|--------|--------|------------|
| v2: High-res + std | All sims→0.90, lost pupil signal | More tiles + std features add noise |
| C: ViT-L/14 | Circle p=0.831, worse than ViT-B | Model capacity is not the bottleneck |
| D: Entropy-weighted tiles | Circle p=0.615, pupil p collapsed 2e-11→0.037 | Aggregation is not the bottleneck |

## Next Steps

| # | Approach | Hypothesis | Effort |
|---|----------|-----------|--------|
| F | **Linear probe** | Thin supervised layer on frozen DINOv2 embeddings learns circle/autograph boundary | ~2 hr |
| E | **Per-tile classification** | Instead of mean pooling, classify individual tiles and vote | ~2 hr |

Recommendation: F next — unsupervised features exhausted (v2, C, D all failed). Supervised signal is the next logical step.

## Files

| File | What |
|------|------|
| `cache/embeddings/embeddings.npz` | v1 ViT-B embeddings (1536d, 108 paintings) |
| `cache/embeddings/embeddings_v2.npz` | v2 ViT-B embeddings (3072d, 101 paintings) |
| `cache/embeddings/embeddings_vitl.npz` | v1-L ViT-L embeddings (2048d, 108 paintings) |
| `cache/embeddings/embeddings_entropy.npz` | v1-D entropy-weighted embeddings (1536d, 108 paintings) |
| `cache/results.json` | v1 ViT-B metrics |
| `cache/results_v2.json` | v2 metrics |
| `cache/results_vitl.json` | v1-L ViT-L metrics |
| `cache/results_entropy.json` | v1-D entropy-weighted metrics |
| `cache/plots/` | UMAP, cosine, heatmap for latest run |
