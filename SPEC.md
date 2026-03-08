---
version: 0.1.0
created: 2026-03-09
updated: 2026-03-09
---

# authentication — spec

## Objective

Collection-scale art screening tool that flags potential misattributions in open museum collections for expert review, using DINOv2 vision embeddings.

## Non-goals

- Authenticating individual works (we screen and flag, never authenticate)
- Replacing human expert opinion or connoisseurship
- Building a customer-facing product or API
- Manuscript stylometry (separate vertical if pursued)
- Real-time or interactive analysis
- Supporting non-open-access collections (Tier 3–4 access)
- Insurance or guarantee products

## User Interface

CLI pipeline via `scan.py`:

```
python scan.py                        # Full pipeline (fetch → embed → analyze)
python scan.py --stage N              # Run specific stage
python scan.py --hires                # High-res tile mode
python scan.py --model vitl14         # ViT-L/14 instead of ViT-B/14
python scan.py --corpus transfer      # Multi-artist transfer corpus
python scan.py --lora                 # LoRA fine-tuning experiment
python scan.py --lora-transfer        # Leave-artist-out LoRA
python scan.py --lora-curriculum      # Two-phase curriculum transfer
python scan.py --transfer-probe       # Frozen transfer probe
```

## Data Model

- **Paintings:** metadata (artist, title, date, attribution level, source museum) + cached image + DINOv2 embedding
- **Attribution levels:** autograph, circle, pupil, school, dutch_other (from Wikidata + museum APIs)
- **Sources:** Rijksmuseum, Met, Wikidata SPARQL, NGA, CMA, AIC — all CC0, no keys
- **Storage:** Three-tier file cache: `cache/metadata/` → `cache/images/` → `cache/embeddings/`

## Architecture

Four-stage pipeline, all in `scan.py`:

1. **Fetch** — Query museum APIs + Wikidata SPARQL for painting metadata
2. **Download** — Retrieve high-res images via IIIF, cache locally
3. **Embed** — DINOv2 ViT-B/14 (or ViT-L/14) tile embeddings, mean/entropy aggregation
4. **Analyze** — Unsupervised (permutation tests, clustering) + supervised (SVM, logistic, LoRA) probes

GPU deployment: `deploy-gpu.sh` → GCE T4 VM → `startup-gpu.sh` → results to GCS.

## Decisions

- In the context of feature extraction, facing need for fast signal validation, we chose DINOv2 frozen embeddings to achieve zero training time, accepting a ceiling on discriminative power.
- In the context of data sourcing, facing need for large diverse corpus, we chose Wikidata SPARQL as primary source to achieve 1311 paintings across attribution levels, accepting noisy labels.
- In the context of tile aggregation, facing multiple strategies (mean, std, entropy), we chose entropy-weighted mean to achieve best supervised accuracy (63.7%), accepting worst unsupervised performance.
- In the context of classification, facing diminishing returns from model complexity, we chose SVM RBF on frozen features to achieve interpretable baseline, accepting 63.7% ceiling.
- In the context of compute, facing cost constraints, we chose GCE Spot T4 to achieve GPU access at minimal cost, accepting preemption risk (mitigated by on-demand fallback).

## Acceptance Criteria

| Criterion | Measurement | Pass |
|-----------|-------------|------|
| Pipeline runs end-to-end | `pytest tests/ -x -q` — 78 tests pass | All green |
| Statistically significant signal | Permutation test on autograph vs circle | p < 0.05 |
| Balanced accuracy above chance | 10-fold CV balanced accuracy | > 50% (3-class) |
| GPU portability | Same code runs on MPS + CUDA | No device-specific branches |
| Reproducible embeddings | Same image → same embedding across runs | Cosine similarity > 0.999 |

## Failure Modes

| Failure | Impact | Mitigation |
|---------|--------|------------|
| Museum API changes or rate limits | Data fetch fails | Three-tier cache; `--refetch` only when needed |
| DINOv2 encodes source museum, not style | Confounded results | Confounder audit (done: within-wikidata probe 62.1% confirms real signal) |
| Embedding space saturates | No accuracy gain from more data | Confirmed: frozen features plateau at ~63.7% regardless of data or model size |
| Hard negatives undetectable | No commercial value for disputed works | Confirmed: 55.1% on hardest 25% (p=0.228) — model has no edge here |
| GPU VM preempted mid-experiment | Lost compute time | On-demand fallback in `europe-west4-a`; checkpoint results to GCS |
