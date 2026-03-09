---
version: 0.1.0
created: 2026-03-09
---

# Authentication — Executive Briefing

## One-liner

DINOv2 frozen embeddings can detect "wrong artist" misattributions (p=2e-11) but cannot distinguish master from skilled imitator — the commercially valuable question. Ceiling confirmed at 63.7% balanced accuracy after exhaustive 3-tier search. Project complete.

## Timeline

Feb 26 – Mar 9, 2026 (~12 days)

## What we built

A CLI pipeline (`scan.py`) that fetches paintings from 6 museum APIs + Wikidata SPARQL, computes DINOv2 vision embeddings, and runs unsupervised + supervised classification probes. 1,311 paintings across 4 attribution levels (autograph, circle, pupil, dutch_other). 78 tests, CUDA/MPS portable, GCE T4 deployment scripts.

## What we learned

### The model is a "different artist" detector, not an authenticator

| Task | Result | Verdict |
|------|--------|---------|
| Autograph vs pupil/other Dutch | p=2e-11, p=3e-10 | Strong — reliably detects different artist |
| Autograph vs circle | p=0.80 (unsupervised), 63.7% (supervised) | Weak — can't separate master from skilled imitator |
| Hard negatives (top 25% hardest circle) | 55.1% balanced accuracy, p=0.228 | Dead — no edge on commercially relevant cases |

### Exhaustive search — nothing left to try with frozen embeddings

| Tier | What we tried | Best result |
|------|---------------|-------------|
| **Tier 1: Frozen features** | ViT-B mean (59.0%), ViT-L (59.5%), entropy (63.7%), CLIP (62.9%), concat (62.3%), per-tile voting (60.2%), fine-tune (60.0%) | 63.7% |
| **Tier 2: Diagnostics** | Confounder audit, robustness, calibration, hard-negatives, domain-shift, multimodal | Signal real but marginal; no step change |
| **Tier 3: LoRA + transfer** | LoRA Rembrandt (60.9%), LoRA leave-artist-out (crashed), curriculum transfer (59.0%) | All underperform frozen 63.7% |

### Why the competitors can do it (and we can't with this approach)

Art Recognition and Hephaestus train **custom supervised models per artist** on brushstroke-level features. They don't use frozen foundation model embeddings. Our approach traded training time for signal strength — fast validation, but the ceiling is real.

Training a custom model would require: months of work, significant GPU compute, tile-level brushstroke annotation, and competing head-to-head with companies that have a 7-year head start + art world relationships + (in Hephaestus's case) an insurance product.

## Costs

- Compute: <$100 (GCE Spot/on-demand T4, ~10 GPU-hours total)
- Data: $0 (all museum APIs are CC0, no keys)
- Time: ~12 days part-time

## Decision

**Complete.** Technical exploration exhausted. The question "can frozen vision embeddings authenticate art?" is definitively answered: no, not at commercially useful accuracy. The research, pipeline, and 33-entry decisions log have reference value for future vision projects.

## Key artifacts

| File | What |
|------|------|
| `docs/decisions.md` | 33-entry chronological decisions log |
| `docs/scan-results.md` | Full experimental results with statistics |
| `docs/research.md` | Market opportunity landscape |
| `docs/execution.md` | Competitor deep-dive, pricing benchmarks, risk analysis |
| `SPEC.md` | Retroactive build contract |
| `docs/graduation-tracking.md` | Gate documentation |
