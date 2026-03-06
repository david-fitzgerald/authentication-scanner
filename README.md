# Authentication Scanner

AI-driven art authentication screening at collection scale. Uses DINOv2 vision embeddings to detect style differences between confirmed autographs and disputed attributions.

## Status

**Best result: 63.7% balanced accuracy** (entropy-weighted SVM RBF, frozen DINOv2 features). All frozen-feature options exhausted (Tier 1). Tier 3 multi-artist transfer + LoRA experiments running on GCE T4.

## What it does

1. Queries museum APIs (Rijksmuseum, Met, Wikidata SPARQL, NGA, CMA, AIC) for paintings by target artists
2. Downloads images, tiles into 224x224 patches
3. Embeds with DINOv2 ViT-B/14 or ViT-L/14 (multiple aggregation strategies)
4. Classifies autograph vs circle/workshop/pupil via supervised probes (logistic, SVM, MLP, LoRA fine-tuning)

## Quick start

```bash
# Local pipeline (MPS/CUDA/CPU auto-detect)
python scan.py

# Specific stage
python scan.py --stage N

# High-res mode
python scan.py --hires

# Transfer corpus (6 artists, ~6K paintings)
python scan.py --corpus transfer

# LoRA experiments
python scan.py --lora              # B: LoRA Rembrandt-only
python scan.py --lora-transfer     # C: LoRA leave-artist-out
python scan.py --lora-curriculum   # D: Two-phase curriculum
python scan.py --transfer-probe    # A: Frozen transfer probe
```

## GPU deployment

```bash
# Deploy to GCE T4 (on-demand, 36h cap, auto-deletes)
bash deploy-gpu.sh

# Monitor
gcloud compute ssh auth-experiments --zone=europe-west4-a -- tail -f /var/log/auth-experiments.log

# Pull results
gsutil -m rsync -r gs://auth-ml-cache/results/ cache/
```

## Corpus

| Source | Data |
|--------|------|
| Rijksmuseum | Rembrandt autograph, circle, workshop, school |
| Met Open Access | Rembrandt + attributed works |
| Wikidata SPARQL | 6 artists (Rubens, Cranach, Van Dyck, Titian, Hals, Rembrandt) — ~6K paintings |
| NGA, CMA, AIC | Supplementary attributions |

## Results summary

| Approach | Balanced Acc | Notes |
|----------|-------------|-------|
| Frozen DINOv2 ViT-B mean (v1) | 59.0% | Baseline probe |
| Frozen DINOv2 entropy-weighted | **63.7%** | Best frozen result |
| ViT-L/14 | 59.5% | Model capacity not the bottleneck |
| CLIP ViT-L/14 | 62.9% | Different foundation, similar ceiling |
| Fine-tune (14M params) | 60.0% | Overfits by epoch 10 |
| LoRA Rembrandt (148K params) | 60.9% | Worse than frozen entropy |

## Architecture

```
scan.py          — Local pipeline. Multi-museum fetch, three-tier cache, CUDA/MPS/CPU.
deploy-gpu.sh    — GCE T4 launcher. Uploads cache, creates VM, auto-destructs.
startup-gpu.sh   — VM startup. Installs deps, runs experiments, uploads results to GCS.
prototype.ipynb  — Phase 1 Colab prototype (historical).
```

## Framing

This is **screening**, not authentication. Flags candidates for expert review. Never claims to authenticate.
