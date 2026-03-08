# authentication/ — AI Art Authentication

AI-driven art authentication via DINOv2 embeddings. Collection-scale screening — position as "flag for expert review," never "authenticate." Best result: 63.7% balanced accuracy (entropy SVM RBF, frozen features). All tiers exhausted — frozen DINOv2 is the ceiling.

**Status:** building — all gates passed (retroactive) | **Harness:** L1

## Quick Reference

| Action | Command |
|--------|---------|
| Run local pipeline | `python scan.py` |
| Run specific stage | `python scan.py --stage N` |
| High-res mode (v2) | `python scan.py --hires` |
| ViT-L/14 model | `python scan.py --model vitl14` |
| Re-fetch all data | `python scan.py --refetch` |
| Transfer corpus (fetch+embed) | `python scan.py --corpus transfer` |
| Exp A: Frozen transfer probe | `python scan.py --transfer-probe` |
| Exp B: LoRA Rembrandt-only | `python scan.py --lora` |
| Exp C: LoRA leave-artist-out | `python scan.py --lora-transfer` |
| Exp D: Two-phase curriculum | `python scan.py --lora-curriculum` |
| Lint + test | `ruff check . && pytest tests/ -x -q --tb=short` |

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
| `prototype.ipynb` | Phase 1 prototype. DINOv2 embedding pipeline on Rijksmuseum Rembrandt corpus. |
| `tests/` | 78 tests (59 local, 19 skip without GPU/network) covering pipeline stages, embedding, classification, and GPU deployment. |
| `docs/` | Research, business plan, execution plan, experiment results, decisions log. |
| `SPEC.md` | Build contract — objective, non-goals, acceptance criteria, failure modes. |

## Testing

78 tests via pytest (59 local, 19 skip without GPU/network). Cover pipeline stages (fetch, download, embed, analyze), classification probes, GPU portability, and data integrity.

```bash
pytest tests/ -x -q --tb=short
```

## Verification

```bash
ruff check . && pytest tests/ -x -q --tb=short
```

## Conventions

- Never say "authenticate" — always "screen and flag for expert review"
- Decisions log in `docs/decisions.md` (append-only)
- Research/business/results docs live in `docs/`

---
Screen and flag, never authenticate. No API keys needed.
