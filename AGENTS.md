# authentication/ — AI Art Authentication

AI-driven art authentication via DINOv2 embeddings. Collection-scale screening — position as "flag for expert review," never "authenticate." Best result: 63.7% balanced accuracy (entropy SVM RBF, frozen features). All tiers exhausted — frozen DINOv2 is the ceiling.

## Serves
- **Primary:** `research/lost-treasure/` — authentication infrastructure for lost-art recovery
- **Secondary:** `fi/` — payoff angle if commercially viable (ruled out at G5)

## Status — COMPLETE

Ceiling confirmed at 63.7% balanced accuracy. All 3 tiers exhausted — frozen DINOv2 can detect "wrong artist" (p=2e-11) but cannot distinguish master from skilled imitator. Commercially unviable. See `README.md` for full summary. Completed Mar 9, 2026.

## Docker

```bash
.docker/run.sh projects/complete/authentication
```

One-time venv setup inside container: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`

## Quick Reference

| Action | Command |
|--------|---------|
| Run local pipeline | `python scan.py` |
| Run specific stage | `python scan.py --stage N` |
| Re-fetch all data | `python scan.py --refetch` |
| Lint + test | `ruff check . && pytest tests/ -x -q --tb=short` |

Full command list (including completed experiment modes) in git history — see `docs/experiment-results.md` for what was run.

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

78 tests (59 local, 19 skip without GPU/network). Cover pipeline, classification probes, GPU portability, data integrity. Verify: `ruff check . && pytest tests/ -x -q --tb=short`.

## Conventions

- Never say "authenticate" — always "screen and flag for expert review"
- Decisions log in `docs/decisions.md` (append-only)
- Research/business/results docs live in `docs/`

---
Screen and flag, never authenticate. No API keys needed.
