---
version: 0.6.0
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

## Comparison: Prototype → v1 → v2 → v1-L → v1-D → v1-F → v1-H

| Metric | Prototype | v1 ViT-B | v2 (high-res) | v1-L ViT-L | v1-D Entropy | v1-F Probe | v1-H Non-linear |
|--------|-----------|----------|---------------|------------|--------------|------------|-----------------|
| N paintings | 70 | 108 | 101 | 108 | 108 | 47 (auto+circle) | 47 (auto+circle) |
| N circle | 2 | 18 | 18 | 18 | 18 | 18 | 18 |
| Circle metric | p=0.088 | p=0.805 | p=0.335 | p=0.831 | p=0.615 | **72.3% LOO, p=0.015** | **72.3% LOO, p=0.015** |
| Pupils p-value | 0.001 | 2.03e-11 | 0.146 | 3.59e-11 | 3.70e-02 | — | — |
| Other p-value | 0.002 | 3.40e-10 | 7.36e-15 | 8.85e-08 | 1.02e-06 | — | — |
| KNN best | 82.9% | 77.8% | 68.3% | 75.9% | 78.7% | — | — |
| KNN baseline | 72.9% | 73.1% | 71.3% | 73.1% | 73.1% | — | — |
| Embed dim | 1536 | 1536 | 3072 | 2048 | 1536 | 1536→PCA 20 | 1536→PCA 10–20 |
| Method | unsupervised | unsupervised | unsupervised | unsupervised | unsupervised | **supervised** | **supervised** |

## Key Findings

1. **Frozen DINOv2 features are tapped out at 72.3%.** Three classifiers (logistic, SVM RBF, MLP) all plateau at or below 72.3% LOO accuracy. Non-linear classifiers extract no additional signal from PCA-reduced embeddings. The bottleneck is data quantity, not classifier capacity.

2. **Supervised signal exists for circle/autograph — unsupervised methods can't find it.** The linear probe (Option F) achieves 72.3% LOO accuracy with permutation p=0.015. Four unsupervised configs (v1, v2, ViT-L, entropy) all failed (best p=0.615). The autograph/circle boundary is a learned linear combination across PCA components, not a cosine distance.

3. **The signal is real but weak.** 72.3% is 20 points above the null (51.9%) — statistically significant, not practically sufficient. At N=47 with 1536d features, even PCA-reduced logistic regression is operating at the edge.

4. **Entropy weighting trades pupil signal for marginal circle improvement.** Circle p improved 0.805→0.615 but pupil p collapsed from 2e-11→0.037. The weighting amplifies noise in complex tiles, not style signal.

5. **Aggregation and model capacity are not the bottleneck.** Mean pooling (v1), entropy weighting (D), ViT-L (C), and high-res (v2) all fail at circle/autograph. The information is in the features but requires a learned projection to access.

6. **High-res hurts more than it helps (v2).** All similarities pushed toward 0.90, collapsing between-group discrimination. The std features add noise.

7. **Pupil/other separation is real and robust — but only with v1 ViT-B baseline.** p=2e-11 for pupils and 3e-10 for other Dutch masters. DINOv2 ViT-B mean pooling is the sweet spot for "different artist" detection.

8. **The scanner is a "different artist" detector, not an authenticator — but supervised methods partially close the gap.** Unsupervised features separate distinct artists but not quality levels within the Rembrandt school. A thin supervised layer finds some of this boundary.

## v1-F: Linear probe on frozen embeddings (supervised)

**Config:** Filter to autograph (N=29) + circle (N=18) = 47 paintings. PCA dimensionality reduction → L2-regularized logistic regression → leave-one-out cross-validation. Grid search over PCA dims [10, 20] and C=[0.001, 0.01, 0.1, 1.0, 10.0]. Permutation test (1000 shuffles) on best config for proper p-value.

**Why PCA first:** At 1536 features and 47 samples, logistic regression can perfectly separate any random labeling. PCA to 10–20 dims compresses noise dimensions, making LOO accuracy meaningful.

### Grid search results

| PCA dims | C=0.001 | C=0.01 | C=0.1 | C=1.0 | C=10.0 |
|----------|---------|--------|-------|-------|--------|
| 10 | 0.617 | 0.681 | 0.681 | 0.660 | 0.660 |
| 20 | 0.617 | 0.681 | **0.723** | 0.681 | 0.660 |

### Best config

| Metric | Value |
|--------|-------|
| PCA dims | 20 |
| Regularization C | 0.1 |
| LOO accuracy | **72.3%** |
| Permutation p-value | **0.015** |
| Null mean ± std | 0.519 ± 0.085 |

### Interpretation

The permutation test is definitive: p=0.015 means only 1.5% of random label shuffles scored ≥72.3%. The supervised signal is real. DINOv2 *does* encode autograph/circle differences — but the boundary is a learned linear combination across 20 PCA components, not a simple cosine distance.

72.3% accuracy is moderate — well above the null (51.9%) but below the 85%+ needed for practical screening. The gap between null and real accuracy (~20 percentage points) suggests signal exists but is weak relative to the noise floor at N=47.

## v1-H: Non-linear probes on frozen embeddings (supervised)

**Config:** Same autograph (N=29) + circle (N=18) = 47 paintings, same PCA + LOO-CV framework. Three classifiers compared side-by-side: Logistic Regression, SVM RBF, and MLP (32 hidden units). Grid search over PCA dims [10, 20] and per-classifier hyperparameters. Permutation test on overall best.

### Classifier comparison

| Classifier | Best PCA | Best param | LOO acc |
|------------|----------|------------|---------|
| Logistic | 20 | C=0.1 | **0.723** |
| SVM RBF | 10 | C=1.0 | 0.660 |
| MLP (32) | 10 | alpha=10.0 | **0.723** |

### Interpretation

SVM RBF scored *worse* (66.0%) than logistic regression — the RBF kernel finds no useful non-linear structure in PCA-reduced DINOv2 features. MLP ties logistic at 72.3% with heavy regularization (alpha=10.0), effectively collapsing to near-linear behavior. The frozen embeddings contain no untapped non-linear signal.

This eliminates classifier capacity as a variable. The 72.3% ceiling is a feature limitation, not a model limitation. More data (Option G) is the clear next step.

## Eliminated Options

| Option | Result | Conclusion |
|--------|--------|------------|
| v2: High-res + std | All sims→0.90, lost pupil signal | More tiles + std features add noise |
| C: ViT-L/14 | Circle p=0.831, worse than ViT-B | Model capacity is not the bottleneck |
| D: Entropy-weighted tiles | Circle p=0.615, pupil p collapsed 2e-11→0.037 | Aggregation is not the bottleneck |
| F: Linear probe | 72.3% LOO, perm p=0.015 | Signal exists but weak — not enough for practical screening at N=47 |
| H: Non-linear probes | SVM RBF 66.0%, MLP 72.3% — no gain over logistic | Classifier capacity is not the bottleneck |

## Next Steps

| # | Approach | Hypothesis | Effort |
|---|----------|-----------|--------|
| E | **Per-tile classification** | Classify individual tiles and vote — bypasses mean-pooling information loss | ~2 hr |
| G | **More data** | N=47 is marginal. Expand circle corpus (Wallace Collection, National Gallery, Louvre) to N=50+ | ~4 hr |
| I | **Fine-tune DINOv2** | LoRA or last-layer fine-tuning on authentication labels — but N=47 is dangerously small | ~8 hr |

Recommendation: G next. Options C, D, F, and H have systematically eliminated model capacity, aggregation strategy, and classifier capacity as bottlenecks. The 72.3% ceiling is a data limitation. More circle paintings would both improve accuracy and make the permutation test more powerful.

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
| `cache/results_probe.json` | v1-F linear probe metrics |
| `cache/plots/` | UMAP, cosine, heatmap for latest run |
