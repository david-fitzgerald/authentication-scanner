---
version: 0.1.0
created: 2026-02-26
updated: 2026-03-09
---

# Authentication — Decisions Log

Append-only record of key technical decisions and findings.

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
| 2026-03-08 | **All Tier 2 adjacent experiments complete** | Confounder, robustness, calibration, hard-negatives, domain-shift, multimodal — all done. |
| 2026-03-09 | EXP D (curriculum): 59.0% ± 8.1% | Worse than frozen 63.7%. High variance (48.6%–70.0%). Transfer learning adds no value. |
| 2026-03-09 | EXP A skipped | Transfer embeddings lost when first VM crashed. B/C/D all failed — running A wouldn't change conclusion. |
| 2026-03-09 | **Tier 3 dead** | All LoRA variants (B: 60.9%, C: crashed, D: 59.0%) underperform frozen 63.7%. Ceiling confirmed. |
