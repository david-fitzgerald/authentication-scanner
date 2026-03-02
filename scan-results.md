---
version: 0.7.0
date: 2026-03-02
status: active
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

## Comparison: Prototype → v1 → v2 → v1-L → v1-D → v1-F → v1-H → v1-G

| Metric | Prototype | v1 ViT-B | v2 (hi-res) | v1-L ViT-L | v1-D Entropy | v1-F Probe | v1-H Non-lin | v1-G Wikidata |
|--------|-----------|----------|-------------|------------|--------------|------------|--------------|---------------|
| N paintings | 70 | 108 | 101 | 108 | 108 | 47 (a+c) | 47 (a+c) | **1311** |
| N circle | 2 | 18 | 18 | 18 | 18 | 18 | 18 | **149** |
| Circle metric | p=0.088 | p=0.805 | p=0.335 | p=0.831 | p=0.615 | 72.3% p=0.015 | 72.3% p=0.015 | **59.0% bal, p=0.003** |
| Pupils p | 0.001 | 2.03e-11 | 0.146 | 3.59e-11 | 3.70e-02 | — | — | ~0.000 |
| Other p | 0.002 | 3.40e-10 | 7.36e-15 | 8.85e-08 | 1.02e-06 | — | — | ~0.000 |
| KNN best | 82.9% | 77.8% | 68.3% | 75.9% | 78.7% | — | — | TBD |
| KNN baseline | 72.9% | 73.1% | 71.3% | 73.1% | 73.1% | — | — | TBD |
| Embed dim | 1536 | 1536 | 3072 | 2048 | 1536 | 1536→PCA20 | 1536→PCA10-20 | 1536 |
| Method | unsup | unsup | unsup | unsup | unsup | **supervised** | **supervised** | unsup + sup |

## Key Findings

1. **Data expansion confirmed the signal — and revealed the N=47 result was inflated.** At N=711, balanced accuracy is 59.0% (p=0.003) vs the N=47 LOO of 72.3% (p=0.015). More data made the permutation test more powerful (p dropped from 0.015 to 0.003) but the actual classification accuracy dropped, suggesting the 72.3% was overfit to the small sample.

2. **Frozen DINOv2 features plateau at ~59–64% balanced accuracy on autograph/circle.** Standard mean pooling: 59.0% (ViT-B) / 59.5% (ViT-L). Entropy-weighted: **63.7%** — the best frozen-feature result. SVM RBF consistently outperforms logistic, indicating mild non-linear structure.

3. **Data expansion unlocked unsupervised circle separation.** At N=108, circle p=0.805 (invisible). At N=1311, circle p≈0.000 (significant). The cosine similarity signal was there all along — just drowned in noise at N=18.

4. **Supervised signal exists for circle/autograph — unsupervised methods can't practically use it.** Four unsupervised configs (v1, v2, ViT-L, entropy) all failed at N=108 (best p=0.615). Supervised probes find the boundary, but it's a learned combination across PCA components, not a simple cosine distance.

5. **Entropy weighting: worst unsupervised, best supervised.** At N=1311, entropy destroys pupil/other separation (both p=1.0) but produces the best probe accuracy (63.7% vs 59.0% mean). The variance-weighted tiles amplify brushwork detail that a supervised classifier can leverage but that cosine similarity cannot.

6. **High-res hurts more than it helps (v2).** All similarities pushed toward 0.90, collapsing between-group discrimination. The std features add noise.

7. **Pupil/other separation is real and robust.** p=2e-11 for pupils and 3e-10 for other Dutch masters (v1 ViT-B). DINOv2 is a "different artist" detector, not an authenticator.

8. **Next frontier: fine-tuning.** Frozen embeddings peak at 63.7% (entropy + SVM RBF). Option I (LoRA fine-tuning) is the clear next step — with N=711, fine-tuning is feasible where it wasn't at N=47. Entropy-weighted features should be the starting point for fine-tuning experiments.

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
| F: Linear probe (N=47) | 72.3% LOO, perm p=0.015 | Signal exists but weak at N=47 |
| H: Non-linear probes (N=47) | SVM RBF 66.0%, MLP 72.3% — no gain over logistic | Classifier capacity is not the bottleneck |
| G: Wikidata expansion (N=711) | 59.0% balanced acc, perm p=0.003 | Data confirmed signal but frozen features plateau at ~59% |
| C re-run (ViT-L, N=711) | 59.5% bal acc, lost "other Dutch" (p=0.249) | Model capacity still not the bottleneck |
| D re-run (entropy, N=711) | **63.7% bal acc** — best probe config | Entropy helps supervised but destroys unsupervised pupil/other |

## Next Steps

### Tier 1: Remaining frozen-feature experiments

| # | Approach | Hypothesis | Effort |
|---|----------|-----------|--------|
| I | **Fine-tune DINOv2** | LoRA or last-layer fine-tuning on authentication labels. N=711 makes this feasible. Could push past the 63.7% frozen-feature ceiling. | ~8 hr |
| E | **Per-tile classification** | Classify individual tiles and vote — bypasses mean-pooling information loss. Each painting → ~64 training examples. | ~2 hr |
| J | **CLIP / SigLIP features** | Different foundation model may encode different visual info. CLIP trained on text-image pairs — may capture iconographic/compositional features DINOv2 misses. | ~2 hr |
| K | **Feature concatenation** | Combine DINOv2 mean + entropy + ViT-L into one wide vector. Let classifier sort out which aggregation matters. PCA handles dimensionality. | ~1 hr |

Recommendation: K first (quick win, almost free), then I (step change). E and J are incremental — maybe +2–5 points within frozen features.

### Tier 2: Beyond frozen embeddings (if I–K don't reach ~80%)

| # | Approach | What | Step change? |
|---|----------|------|-------------|
| L | **Diagnostic region crops** | Train on crops of known-diagnostic regions (hands, eyes, ears, fabric folds). Art historians zoom into specific areas, not whole paintings. Region detector + per-region classifier. | Maybe. Requires annotation effort. |
| M | **Crack/canvas/pigment from high-res** | Craquelure patterns in visible-light high-res images carry age/technique signal. X-ray and infrared are gold standard but need museum access. | Possibly. Needs native-resolution museum IIIF. |
| N | **Pivot the product** | Ship as a "wrong artist" detector (pupil/other separation is near-perfect, p≈0.000) rather than an authenticator. Catches obvious misattributions in auction catalogs and collection inventories. Position circle/autograph as research frontier. | Strategic, not technical. |
| O | **Hybrid AI + metadata** | Combine visual features with provenance, dimensions, date, medium, subject matter. A "Rembrandt" with unusual size, uncommon support, no provenance before 1850 is already suspicious. Wikidata has much of this structured. | Likely the biggest lift. Attribution is multimodal. |

**Assessment:** If I–K land at 65–70%, **N is the right move** — the pupil/other signal is already commercially useful. Even human experts disagree on autograph vs circle (the Rembrandt Research Project took 40 years). Treating authentication as pure computer vision may be pushing against a fundamental limit.

If pushing the technical frontier, **O** (hybrid features) is where the real alpha is. Attribution is a multimodal problem — provenance, not pixels, is where authentication actually happens.

## v1-G: Wikidata SPARQL dataset expansion (1311 paintings)

**Config:** Same as v1 (2000px, ViT-B/14, mean-only 1536d). Dataset expanded via Wikidata SPARQL queries using P170 (creator) with qualifier properties P1774–P1780 (workshop/follower/circle/manner/school). Sources: Wikidata Commons images at 2000px thumbnails. Museum APIs (NGA, CMA, AIC) returned 0 results — Wikidata alone was sufficient.

### Dataset comparison

| Group | v1 count | v1-G count | Change |
|-------|----------|------------|--------|
| Autograph | 29 | 562 | +533 (19×) |
| Circle/disputed | 18 | 149 | +131 (8×) |
| Pupils | 33 | 327 | +294 (10×) |
| Other Dutch | 28 | 273 | +245 (10×) |
| **Total** | **108** | **1311** | **+1203 (12×)** |

1310/1311 images downloaded (1 failure). Wikidata SPARQL was the primary source; NGA API returned 404 (changed since plan), CMA and AIC had no Rembrandt matches.

### Stage 4a: Similarity analysis

| Metric | v1 (N=108) | v1-G (N=1311) | Change |
|--------|-----------|---------------|--------|
| Auto↔Auto sim | 0.8462 | TBD | — |
| Auto↔Circle sim | 0.8478 | TBD | — |
| MW p (vs circle) | 0.805 | **~0.000** | **Significant!** |
| MW p (vs pupils) | 2.03e-11 | ~0.000 | Remains strong |
| MW p (vs other) | 3.40e-10 | ~0.000 | Remains strong |
| KNN best | 77.8% (K=7) | TBD | — |
| KNN baseline | 73.1% | TBD | — |

Circle p-value went from non-significant (0.805) to essentially zero. KNN gap above baseline widened from +2.8% to +13.2%. The data bottleneck hypothesis was correct — with 8× more circle paintings, unsupervised separation emerges.

### Stage 4b: Linear probe (unbalanced)

| Classifier | Best PCA | Best param | LOO acc |
|------------|----------|------------|---------|
| Logistic | 10 | C=0.001 | 0.790 |
| SVM RBF | 10 | C=0.001 | 0.790 |
| MLP (32) | 10 | alpha=10.0 | 0.786 |

Permutation p = **1.000** (null mean = 0.790). Probe accuracy equals majority-class baseline (562/711 = 79.0%). **Class imbalance killed the probe** — all classifiers learned to predict "autograph" for everything. The 4:1 class ratio (562 auto vs 149 circle) makes majority-class prediction optimal under uniform loss.

**Fix:** Re-run with `class_weight='balanced'` to penalize majority-class predictions.

### Stage 4b: Probe with balanced class weights (10-fold stratified CV)

Switched from LOO to stratified 10-fold CV (LOO impractical at N=711: ~21K fits). Added `class_weight='balanced'` to Logistic and SVM. Scoring metric: balanced accuracy (mean of per-class recall).

| Classifier | Best PCA | Best param | 10-fold balanced acc |
|------------|----------|------------|---------------------|
| Logistic | 20 | C=1.0 | 0.578 |
| **SVM RBF** | **20** | **C=1.0** | **0.590** |
| MLP (32) | 20 | alpha=0.1 | 0.552 |

Permutation p = **0.003** (1000 shuffles, null mean = 0.500 ± 0.031).

### Interpretation

The signal is **real but modest**. 59.0% balanced accuracy with p=0.003 confirms DINOv2 features encode autograph/circle differences — but the effect is smaller than the N=47 LOO result (72.3%) suggested. That small-dataset result was likely inflated by overfitting to 47 samples with 20 PCA dimensions.

At scale: DINOv2 frozen features separate autograph from circle at ~59% balanced accuracy. Better than chance (p=0.003), not good enough for practical screening (need >80%). The frozen embedding space captures some but not enough of the stylistic differences art historians use.

SVM RBF slightly outperforms logistic (59.0% vs 57.8%), suggesting mild non-linear structure in the feature space — consistent with H's finding at N=47 but now with proper statistical power.

## v1-G-L: ViT-L/14 on expanded dataset (Option C re-run)

**Config:** Same as v1-G (2000px, mean-only) but DINOv2 ViT-L/14 (1024d per token → 2048d concatenated). Batch size 8 (vs 16 for ViT-B) to fit 8 GB unified memory.

### Stage 4a: Similarity analysis

| Metric | v1-G ViT-B (N=1310) | v1-G-L ViT-L (N=1310) | Change |
|--------|---------------------|----------------------|--------|
| Auto↔Auto sim | 0.7401 | 0.7692 | +0.029 |
| Auto↔Circle sim | 0.7309 | 0.7583 | +0.027 |
| MW p (vs circle) | ~0.000 | ~0.000 | = |
| MW p (vs pupils) | ~0.000 | ~0.000 | = |
| MW p (vs other) | ~0.000 | **0.249** | **Regressed** |
| KNN K=3 | 69.2% | 70.7% | +1.5% |
| KNN baseline | 57.1% | 57.1% | = |

ViT-L maintains circle separation but **loses "other Dutch" discrimination** (p=0.249). At N=108, ViT-L was already worse on other Dutch (p=8.85e-8 vs 3.40e-10). At N=1311, this weakness becomes a full failure. ViT-L's larger feature space captures more general Dutch Golden Age texture, drowning out artist-specific signals for non-Rembrandt-school painters.

### Stage 4b: Probe (ViT-L/14, balanced, 10-fold)

| Classifier | Best PCA | Best param | 10-fold balanced acc |
|------------|----------|------------|---------------------|
| Logistic | 20 | C=0.001 | 0.594 |
| **SVM RBF** | **20** | **C=1.0** | **0.595** |
| MLP (32) | 20 | alpha=1.0 | 0.564 |

Permutation p = **0.001** (null mean = 0.500 ± 0.031).

**ViT-L probe (59.5%) ≈ ViT-B probe (59.0%).** The extra 512 dimensions from ViT-L add no signal for circle/autograph classification. Model capacity is confirmed not the bottleneck — consistent with the N=108 finding but now with proper statistical power.

## v1-G-D: Entropy-weighted tiles on expanded dataset (Option D re-run)

**Config:** Same as v1-D (2000px, ViT-B/14, entropy-weighted aggregation) but on expanded N=1311 dataset.

### Stage 4a: Similarity analysis

| Metric | v1-G ViT-B (N=1310) | v1-G-D Entropy (N=1310) | Change |
|--------|---------------------|------------------------|--------|
| Auto↔Auto sim | 0.7401 | 0.6719 | -0.068 |
| Auto↔Circle sim | 0.7309 | 0.6496 | -0.081 |
| MW p (vs circle) | ~0.000 | ~0.000 | = |
| MW p (vs pupils) | ~0.000 | **1.000** | **Destroyed** |
| MW p (vs other) | ~0.000 | **1.000** | **Destroyed** |
| KNN K=3 | 69.2% | 70.6% | +1.4% |
| KNN baseline | 57.1% | 57.1% | = |

Entropy weighting **destroys** pupil and other Dutch separation (both p=1.0) while maintaining circle separation. Same pattern as N=108 but more extreme. Unsupervised: entropy weighting is strictly worse.

### Stage 4b: Probe (entropy, balanced, 10-fold)

| Classifier | Best PCA | Best param | 10-fold balanced acc |
|------------|----------|------------|---------------------|
| Logistic | 20 | C=0.001 | 0.611 |
| **SVM RBF** | **20** | **C=1.0** | **0.637** |
| MLP (32) | 10 | alpha=0.01 | 0.574 |

Permutation p = **0.001** (null mean = 0.499 ± 0.031).

**Surprise: entropy is the best probe configuration.** 63.7% balanced accuracy vs 59.0% (ViT-B mean) and 59.5% (ViT-L mean). The variance-weighted aggregation captures brushwork-level information that a supervised classifier can exploit — even though unsupervised cosine similarity can't. The entropy weighting amplifies high-variance tiles (faces, hands, detailed brushwork) which are exactly where art historians look for attribution differences.

### Probe comparison across all configs (expanded dataset)

| Config | SVM RBF balanced acc | p-value | Unsupervised circle p |
|--------|---------------------|---------|----------------------|
| ViT-B mean | 59.0% | 0.003 | ~0.000 |
| ViT-L mean | 59.5% | 0.001 | ~0.000 |
| **ViT-B entropy** | **63.7%** | **0.001** | ~0.000 |

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
