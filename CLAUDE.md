---
version: 0.2.0
status: building
harness: L0
updated: 2026-03-01
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

## Environment

- **Compute:** MPS (local), Google Colab Pro (T4 GPUs)
- **Data:** Rijksmuseum APIs (free, no API key needed). Met Open Access. Other CC0 collections via IIIF.
- **Storage:** Three-tier cache: metadata → images → embeddings (~1.5 GB for prototype)
- **Repo:** Private GitHub: `david-fitzgerald/authentication-scanner`

## Architecture

| File | What |
|------|------|
| `scan.py` | Local pipeline. Rijksmuseum + Met Open Access. Three-tier cache (metadata→images→embeddings). MPS device. |
| `prototype.ipynb` | Phase 1 prototype. DINOv2 embedding pipeline on Rijksmuseum Rembrandt corpus. Run on Colab Pro (T4). |
| `scan-results.md` | Local pipeline results. v1 vs v2 comparison, key findings, next steps (C/D/E/F). |
| `prototype-results.md` | Colab prototype results. Original Rijksmuseum-only metrics + API discoveries. |
| `research.md` | Opportunity landscape — art authentication + lost manuscripts, six gaps, monetisation paths |
| `business-plan.md` | Attribution Alpha fund. Buy misattributed works, reattribute, sell. |
| `execution.md` | Detailed execution plan — MVP pipeline, week-by-week, competitors, capital requirements |

## Status

**Unsupervised DINOv2 features exhausted.** Circle/autograph inseparable across all four configs (v1, v2, ViT-L, entropy). Next: linear probe (option F) — supervised learning on frozen embeddings.

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

## Conventions

- Never say "authenticate" — always "screen and flag for expert review"
- Business/research context in `research.md`, `business-plan.md`, `execution.md`

---
Screen and flag, never authenticate. No API keys needed.
