---
version: 0.1.0
date: 2026-03-06
status: assessment (not yet started)
---

# Adjacent Angles: Grounded Assessment

Assessment of 10 proposed extensions to the authentication pipeline, grounded against current codebase state, data availability, and commercial use case.

## Current State (context for assessment)

- **Best result**: 63.7% balanced accuracy (entropy SVM RBF, frozen DINOv2 ViT-B/14)
- **Dataset**: 1,311 Rembrandt-focused (562 auto, 149 circle) + 5,860 transfer (6 artists)
- **Sources**: 6 APIs (Rijksmuseum, Met, Wikidata, NGA, CMA, AIC) — Wikidata dominant (~92%)
- **Metadata captured**: obj_id, source, title, creator, date, image_url, artist_group, attribution, label_confidence
- **Metadata available but unused**: dimensions, medium, provenance (Met), conservation (Rijks LA), exhibition history (Wikidata)
- **Robustness checks**: Perceptual dedup, holdout-source (ready), strict-labels (ready), nested CV, permutation tests
- **Not yet checked**: Source confounds, calibration, augmentation robustness, hard-negative performance

---

## Tier 1: Do Now (high signal, buildable today)

### 3. Confounder Audits

**Priority: HIGHEST. Do before anything else.**

You have 6 sources with different photography pipelines. A source classifier on current embeddings takes 30 min to build and will tell you how much of the 63.7% is "museum photography style" vs "artist attribution style."

- Train a simple classifier to predict `source` from the same 1536d embeddings used for attribution
- If source prediction > 80%, the result is substantially confounded
- Also test: resolution/compression prediction, watermark/frame presence
- If shortcut signal is strong, the 63.7% may be partially or mostly explained by source artifacts

**Implementation:**
- Load embeddings from `embeddings_entropy.npz`
- Train logistic regression on `source` column instead of `artist_group`
- Compare accuracy to chance (1/6 = 16.7%)
- If high: rebalance sources before attribution training, or use domain-adversarial approach

**Effort**: 2h
**Impact**: Could invalidate or validate the entire 63.7% number

---

### 10. Red-Team the Pipeline

JPEG artifacts, color cast, mild blur, crop shifts, frame inclusion — apply `torchvision.transforms` augmentations at inference time, check prediction stability.

- Apply 5 augmentation regimes to test images: JPEG compression (quality=30), Gaussian blur (sigma=2), color jitter (brightness=0.2), 10% center crop, horizontal flip
- Re-run inference, compare predictions to clean versions
- If predictions flip under benign transforms, the model is fragile and unsuitable for deployment

**Implementation:**
- Augment cached images in-memory, re-tile, re-embed, re-predict
- Report per-augmentation accuracy delta and prediction flip rate
- Require <5% flip rate under each augmentation before promoting candidates

**Effort**: 4h
**Impact**: Essential before deploying capital. Trivially implementable.

---

### 7. Calibration and Abstention

Currently using SVM RBF (no native probabilities). Need calibrated confidence and an abstain class.

- Switch to `SVC(probability=True)` or logistic regression for probability output
- Fit Platt scaling or isotonic calibration on held-out fold
- Add abstain threshold: if P(autograph) in [0.35, 0.65], output "insufficient evidence"
- Report: "X% accuracy on the Y% of cases we choose to call"

**Implementation:**
- `CalibratedClassifierCV` from sklearn wrapping the best SVM
- Reliability diagram (calibration curve) in results
- Abstention rate vs accuracy tradeoff curve

**Effort**: 4h
**Impact**: Moves from "63.7% accuracy" to a defensible confidence band. Essential for legal/reputational risk in the fund.

---

## Tier 2: Build Next (moderate effort, high strategic value)

### 1. Multimodal Attribution

Met API already returns `dimensions`, `medium`, `provenance` (text), `objectDate`. Currently thrown away.

- Concat a ~10-feature metadata vector with 1536d embedding:
  - log(area), is_canvas, provenance_text_length, date_precision_score, has_inscription, support_type, earliest_record_year, num_exhibitions, conservation_mention_count
- Train on the concatenated feature vector
- Don't expect a revolution — these are weak signals individually — but provenance_length alone may separate circle from autograph (circle works have shorter provenance chains)

**Implementation:**
- Extend Stage 1 to capture additional Met/Rijks fields
- Build `metadata_features()` function that returns normalized vector
- Concat with embeddings before PCA + probe

**Effort**: 1 day
**Expected impact**: +1-3% balanced accuracy. Real value is interpretability — "flagged because short provenance + borderline embedding score."

---

### 4. Hard-Negative Mining

Current "autograph vs circle" boundary is the easy version. Commercial value is in paintings that *look* autograph but aren't.

- Filter circle set to only paintings with cosine similarity > 0.85 to nearest autograph
- Train/eval on this hard subset only
- If accuracy drops to ~52%, the model has no commercial edge on the cases that matter

**Implementation:**
- Compute pairwise cosine similarity between autograph and circle embeddings
- Select circle paintings with max similarity to any autograph > threshold
- Re-run probe on hard subset
- Report: "on the N paintings that most closely resemble autographs, accuracy is X%"

**Effort**: 4h
**Impact**: This is the "would this actually work on a real auction candidate?" test.

---

### 2. Domain-Shift Stress Testing

`--holdout-source` is built but not yet run. Bigger gap: zero auction catalog images.

- Run `--holdout-source` for each source (met, rijksmuseum, wikidata)
- Scrape ~50 images from Christie's/Sotheby's past lot results (public pages)
- Run inference on auction images, compare to museum-trained predictions
- If predictions are garbage on auction images, museum-trained model won't transfer to commercial setting

**The fund buys at auction, not from museums. Auction images are the deployment domain.**

**Implementation:**
- Phase 1: Run existing `--holdout-source` for all sources (immediate)
- Phase 2: Manual scrape of 50 auction lot images, add as new source
- Phase 3: Report per-domain accuracy table

**Effort**: 1 day
**Impact**: Determines whether museum training transfers to the commercial setting at all.

---

## Tier 3: Build When Signal Proven

### 5. Graph-Based Provenance

Wikidata has ownership chains via P127 (owned by) and P276 (location). Building a knowledge graph embedding (TransE/RotatE) is a real project.

- Value: anomaly detection — "this painting looks autograph but has a suspicious 80-year gap in provenance"
- Build after image signal is validated, not before
- Requires: SPARQL queries for ownership chains, graph embedding library (PyKEEN), anomaly scoring

**Effort**: 2-4 weeks
**When**: After commercial candidate list exists to score

---

### 6. Decision-Theoretic Layer

Expected value = P(reattribution) x price_uplift x P(scholar_agreement) - purchase_price - research_cost - legal_risk.

- This is the business layer, not the ML layer
- Build as a spreadsheet first, then code when you have 20+ scored candidates
- Inputs: model P(autograph), estimated market value delta, scholar availability, legal jurisdiction

**Effort**: 3 days
**When**: After scoring pipeline works and you have candidates to rank

---

### 9. Human-in-the-Loop Evidence UI

- GradCAM / attention rollout on DINOv2 to show which tiles drove prediction
- Nearest-neighbor retrieval: "this looks most like [authenticated work X]"
- Metadata factors driving score
- This is the expert adoption layer

**Effort**: 1-2 weeks
**When**: After you have a scholar partner who will use it. Not before.

---

### 8. Temporal/Restoration Robustness

- Find before/after restoration photos of same painting (manual research task)
- Some museums publish conservation reports with photos
- Small N, high value per case
- If predictions are unstable across captures of the same work, penalize reliability score

**Effort**: 2+ weeks (mostly research, not code)
**When**: Ongoing — collect opportunistically when found

---

## Execution Order

```
Week 1 (immediate):
  1. Confounder audit (source classifier on embeddings)       — 2h
  2. Red-team (augmentation robustness)                       — 4h
  3. Hard-negative filter (cosine-similar subset eval)         — 4h

If confounders are clean:
  4. Calibration + abstention                                  — 4h
  5. Multimodal metadata features                              — 1d
  6. Domain-shift (holdout-source + auction images)            — 1d

If confounders are dirty:
  → Fix first: source-balanced training, domain adversarial,
    or report source-stratified results honestly
  → The 63.7% may fall to ~55% after deconfounding —
    and that's the real number

Skip entirely (for now):
  - Graph provenance (premature — no candidate list)
  - Evidence UI (no user yet)
  - Decision-theoretic layer (spreadsheet first, not code)
```

## Key Insight

The confounder audit is the single highest-leverage thing. You could be at 63.7% because Wikidata circle paintings are photographed differently than Rijksmuseum autographs, not because the model sees brushwork differences. 2 hours of work to find out.
