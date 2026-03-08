---
version: 0.3.3
status: building
harness: L1
updated: 2026-03-08
---

# Authentication — Claude Instructions

AI-driven art authentication via DINOv2 embeddings. Collection-scale screening — position as "flag for expert review," never "authenticate."

## Quick Reference

| Action | Command |
|--------|---------|
| Run local pipeline | `python scan.py` |
| Run specific stage | `python scan.py --stage N` |
| High-res mode (v2) | `python scan.py --hires` |
| ViT-L/14 model | `python scan.py --model vitl14` |
| Re-fetch all data | `python scan.py --refetch` |
| Transfer corpus (fetch+embed) | `python scan.py --corpus transfer` |
| Transfer corpus (stage 1 only) | `python scan.py --corpus transfer --stage 1` |
| Exp A: Frozen transfer probe | `python scan.py --transfer-probe` |
| Exp B: LoRA Rembrandt-only | `python scan.py --lora` |
| Exp C: LoRA leave-artist-out | `python scan.py --lora-transfer` |
| Exp D: Two-phase curriculum | `python scan.py --lora-curriculum` |

## Environment

- **Compute:** MPS (local), GCE on-demand T4 (deploy-gpu.sh + startup-gpu.sh, europe-west4-a)
- **Data:** Rijksmuseum APIs, Met Open Access, Wikidata SPARQL, NGA/CMA/AIC APIs. All free, no API keys.
- **Storage:** Three-tier cache: metadata → images → embeddings (~1.5 GB for prototype)
- **Repo:** Private GitHub: `david-fitzgerald/authentication-scanner`

## Architecture

| File | What |
|------|------|
| `scan.py` | Local pipeline. Rijksmuseum + Met + Wikidata SPARQL + museum APIs. Three-tier cache (metadata→images→embeddings). CUDA/MPS/CPU auto-detect. |
| `deploy-gpu.sh` | GCE Spot T4 launcher. Creates VM, uploads cache, runs experiments, self-destructs. |
| `startup-gpu.sh` | VM startup script. Installs deps, runs experiments, uploads results to GCS. |
| `prototype.ipynb` | Phase 1 prototype. DINOv2 embedding pipeline on Rijksmuseum Rembrandt corpus. Run on Colab Pro (T4). |
| `adjacent-angles.md` | Grounded assessment of 10 extensions. Tiered by priority. Confounder audit is #1. |
| `scan-results.md` | Local pipeline results. v1 vs v2 comparison, key findings, next steps (C/D/E/F). |
| `prototype-results.md` | Colab prototype results. Original Rijksmuseum-only metrics + API discoveries. |
| `research.md` | Opportunity landscape — art authentication + lost manuscripts, six gaps, monetisation paths |
| `business-plan.md` | Attribution Alpha fund. Buy misattributed works, reattribute, sell. |
| `execution.md` | Detailed execution plan — MVP pipeline, week-by-week, competitors, capital requirements |

## Status

**Best result: 63.7% balanced accuracy (entropy SVM RBF, frozen features).** All Tier 1 exhausted, all Tier 2 adjacent experiments complete. Tier 3 (LoRA transfer) running on T4 VM. Key finding: hard-negative performance is weak (55.1%, p=0.228) — model has no edge on the commercially relevant cases.

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-26 | Start with DINOv2 embeddings over custom CNN | Fastest to validate signal. Days not weeks. No training needed. |
| 2026-02-26 | Multi-artist from the start | More signals per scan. Style embedding space > single-artist classifier. |
| 2026-02-26 | Rijksmuseum first | Largest open collection, best Rembrandt corpus, high-res IIIF API. |
| 2026-02-26 | Prototype result: weak-to-moderate signal | p=0.001 pupils, 82.9% KNN. Data bottleneck not method bottleneck. |
| 2026-02-26 | Met expansion: circle still inseparable | N=18 circle, p=0.80. DINOv2 ViT-B is "different artist" detector, not authenticator. |
| 2026-02-26 | High-res + std features (v2) hurt | All sims→0.90, lost pupil signal. Low-res mean-only (v1) is better. |
| 2026-03-01 | ViT-L/14 (option C) eliminated | Circle p=0.831 (worse). Model capacity not the bottleneck. |
| 2026-03-01 | Entropy-weighted tiles (option D) eliminated | Circle p=0.615 (marginal). Pupil p collapsed. Aggregation not the bottleneck. Next: linear probe (F). |
| 2026-03-01 | Linear probe (option F): signal confirmed | 72.3% LOO, perm p=0.015. Supervised signal exists in frozen DINOv2 features. Weak but real. |
| 2026-03-01 | Non-linear probes (option H): features tapped out | SVM RBF 66.0%, MLP 72.3% — no gain over logistic. Bottleneck is data quantity, not classifier capacity. |
| 2026-03-01 | Option G: Wikidata SPARQL dataset expansion | 1311 paintings (was ~108). Circle 149, autograph 562, pupil 327, dutch_other 273. Wikidata primary source. |
| 2026-03-02 | Option G probe: frozen features plateau at 59% | Balanced 10-fold: SVM RBF 59.0%, p=0.003. N=47 LOO (72.3%) was overfit. Unsupervised circle p went from 0.805→~0.000 with 8× more data. |
| 2026-03-02 | C/D re-run on expanded data | ViT-L: 59.5% (no gain). Entropy: **63.7%** — best frozen-feature probe. Entropy worst unsupervised but best supervised. |
| 2026-03-02 | K: Concat embeddings | 62.3% — diluted entropy signal. PCA too aggressive on 5120d. |
| 2026-03-03 | I: Fine-tune eliminated | 60.0% ± 5.0% — overfits by epoch 10, doesn't beat frozen entropy (63.7%). N=568 too small for 14M params. |
| 2026-03-03 | E: Per-tile classification eliminated | 60.2% vote bal acc — tile labels too noisy, majority voting can't recover signal. |
| 2026-03-03 | J: CLIP ViT-L/14 eliminated | 62.9% bal acc — different foundation model, similar result. Overlapping style info. |
| 2026-03-03 | **Tier 1 exhausted** | All frozen-feature options (C/D/E/I/J/K) converge at ~59–64%. Ceiling is 63.7% (entropy SVM RBF). |
| 2026-03-03 | Tier 3: Multi-artist transfer + LoRA | 6 artists (Rubens, Cranach, Van Dyck, Titian, Hals, Rembrandt). 4 experiments: A (frozen probe), B (LoRA Rembrandt), C (LoRA leave-artist-out), D (curriculum transfer). peft installed. |
| 2026-03-05 | GCE Spot T4 deployed | scan.py CUDA-portable (10 edits). deploy-gpu.sh + startup-gpu.sh created. Transfer corpus: 5,859 paintings embedded. |
| 2026-03-05 | EXP B (LoRA Rembrandt): 60.9% ± 3.5% | Worse than frozen entropy (63.7%). LoRA with 148K params didn't help. |
| 2026-03-06 | EXP C crashed (CUDA error) | Cranach fold only (53.4%). Spot VM preempted on retry. Redeployed on-demand `europe-west4-a`. |
| 2026-03-07 | Deploy fix: CUDA_LAUNCH_BLOCKING removed | Was causing silent hang (3.5h no output). Replaced with PYTHONUNBUFFERED=1. |
| 2026-03-07 | Deploy fix: SSH access | VM missing `finance-scanner` network tag for IAP firewall rule. Added `--tags=auth-experiments` to deploy-gpu.sh. |
| 2026-03-08 | Confounder audit: MIXED | Source classifier 99.6% → embeddings encode source. Within-wikidata probe: 62.1% (vs 63.7% full). Signal is real but partially confounded. |
| 2026-03-08 | Robustness test: PASS | All 5 augmentations <5% flip rate. Predictions are stable under benign transforms. |
| 2026-03-08 | Calibration + abstention: MARGINAL | At |d(x)|≥0.5 threshold: 68.4% bal acc on 72.4% coverage. Modest improvement, not a step change. |
| 2026-03-08 | Hard-negative mining: WEAK | Top-25% hardest circle subset: 55.1% bal acc (p=0.228). Model has no edge on commercially relevant cases. |
| 2026-03-08 | Domain-shift holdout: VARIABLE | Per-source holdout varies widely. Reverse (wikidata→rest): 62.5%. Autograph signal more robust than circle signal. |
| 2026-03-08 | Multimodal probe: NEUTRAL | Embed+meta 64.5% vs embed-only 63.7% (+0.7pp, p=0.005). Date range is strongest metadata feature. Marginal gain, interpretability value. |
| 2026-03-08 | **All Tier 2 adjacent experiments complete** | Confounder, robustness, calibration, hard-negatives, domain-shift, multimodal — all done. Waiting on T4 for Tier 3 (EXP C/D/A). |

## Verification

```bash
ruff check . && pytest tests/ -x -q --tb=short
```

## Conventions

- Never say "authenticate" — always "screen and flag for expert review"
- Business/research context in `research.md`, `business-plan.md`, `execution.md`

## Next Session

### Waiting on T4 VM (EXP C/D/A)
Deployed 2026-03-07. Running LoRA leave-artist-out (C), curriculum (D), frozen transfer probe (A).

```bash
# Check VM status
gcloud compute instances describe auth-experiments --zone=europe-west4-a --project=perpstrader --format='value(status)' 2>&1
# Check GCS results
gsutil ls gs://auth-ml-cache/results/
# Pull results
gsutil -m rsync -r gs://auth-ml-cache/results/ cache/
```

### Tier 3 results so far
- **EXP 0** (transfer embed): Done. 5,859 paintings embedded.
- **EXP B** (LoRA Rembrandt): **60.9% ± 3.5%** — worse than frozen 63.7%.
- **EXP C/D/A**: Running on T4.

### All Tier 2 adjacent experiments complete
| Experiment | Verdict | Key finding |
|-----------|---------|-------------|
| Confounder audit | MIXED | Source classifier 99.6%. Within-wikidata: 62.1%. Signal real but partially confounded. |
| Robustness | PASS | All augmentations <5% flip rate. |
| Calibration | MARGINAL | 68.4% at 72% coverage. Modest gain. |
| Hard-negatives | WEAK | 55.1% on hardest 25% (p=0.228). No commercial edge. |
| Domain-shift | VARIABLE | Reverse direction 62.5%. Autograph signal > circle signal. |
| Multimodal | NEUTRAL | +0.7pp (p=0.005). Date range dominant. Interpretability value only. |

### Remaining options
- Auction image scrape (50 Christie's/Sotheby's lots) — tests deployment domain
- Tier 4: cross-domain transfer (Chinese painting) — only if Tier 3 caps out

---
Screen and flag, never authenticate. No API keys needed.
