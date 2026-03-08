---
version: 0.1.0
created: 2026-03-09
updated: 2026-03-09
---

# authentication — graduation tracking

Retroactive — documenting gates that were passed informally before the gate standard existed. Authentication graduated from `ideas/anomalies/` on first code commit (2026-02-26).

## G1: Kill Screen

| Check | Result |
|-------|--------|
| Incumbents | PASS — 2 funded competitors (Art Recognition, Hephaestus). Neither offers collection-scale screening. |
| Platform dependency | PASS — No platform dependency. Open museum APIs (CC0). |
| Core technical assumption | PASS — DINOv2 embeddings produce separable style clusters (p=0.001 on Rijksmuseum prototype). |
| Real P&L | PASS — Art Recognition charges $2,200/work. Hephaestus charges 60bps of certified value. Market exists. |
| Automation-to-money path | PASS — Screening → flag misattributions → expert review → reattribution → value increase. |

**Verdict:** PASS (5/5)

## G2: Spike

**Question:** Can DINOv2 embeddings distinguish Rembrandt autograph works from circle/school?
**Finding:** Yes — weak-to-moderate signal. p=0.001 on pupils, 82.9% KNN accuracy on Rijksmuseum corpus (Feb 26). Data bottleneck, not method bottleneck.
**Evidence:** `prototype.ipynb`, `docs/prototype-results.md`

## G3: Specify

See `SPEC.md` — all 8 sections filled (retroactive, created 2026-03-09).

## G4: Walking Skeleton

**Scope:** `scan.py` stages 1–4: fetch metadata → download images → compute embeddings → analyze (unsupervised + supervised probes).

**Cut list:**
- Auction screening product
- Manuscript stylometry vertical
- Multi-model ensemble
- Insurance/guarantee product
- Web UI / API endpoint
- Per-work authentication reports

**Unknowns (resolved):**
- Whether DINOv2 can separate circle from autograph → ceiling at 63.7%, insufficient for commercial use
- Whether LoRA/fine-tuning improves on frozen features → no (Tier 3 dead)
- Whether transfer learning across artists helps → no (EXP D: 59.0%)

## G5: Verify & Harness

**Smoke test:** `ruff check . && pytest tests/ -x -q --tb=short`

**Verification plan:**

| Criterion | Test Method | When to Run |
|-----------|------------|-------------|
| Pipeline runs end-to-end | 78 pytest tests covering all stages | Pre-commit hook |
| Lint passes | `ruff check .` | Pre-commit hook |
| Embedding quality | Permutation test p-values in scan-results | Manual after data changes |
| Classification accuracy | 10-fold balanced accuracy | Manual after model changes |

**Failure tests:**
- API returns empty results → pipeline handles gracefully (tested)
- Cache corruption → re-fetch with `--refetch` flag (tested)
- CUDA unavailable → auto-fallback to MPS/CPU (tested)

**HARNESS L1:**
- [x] Git repo with remote (GitHub private)
- [x] CLAUDE.md with status line + required sections
- [x] .gitignore covering language artifacts
- [x] Linter (ruff) configured and passing
- [x] Package manifest (pyproject.toml)
- [x] 78 passing tests
- [x] Pre-commit hook (ruff + pytest)
- [x] Verification section in CLAUDE.md
