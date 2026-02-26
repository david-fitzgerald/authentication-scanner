# Authentication Scanner

AI-driven art authentication screening at collection scale. Uses DINOv2 vision embeddings to detect style differences between confirmed autographs and disputed attributions.

## Status

**Phase 1: Prototype** — validating whether pre-trained DINOv2 captures style signal on Rijksmuseum Rembrandt corpus.

## What it does

1. Queries Rijksmuseum APIs for paintings by Rembrandt (autograph), his circle/workshop/school, pupils (Bol, Flinck, Lievens), and control Dutch masters (Hals, Vermeer)
2. Downloads images via IIIF, tiles into 224x224 patches
3. Embeds with DINOv2 ViT-B/14 (CLS + mean patch tokens, 1536d per painting)
4. Tests for style clustering: UMAP, cosine similarity distributions, KNN classification, candidate ranking

## Quick start

Open in Colab (requires Colab Pro for T4 GPU):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/david-fitzgerald/authentication-scanner/blob/main/prototype.ipynb)

Runtime: ~10 minutes end-to-end. Three-tier cache (metadata/images/embeddings) survives session restarts.

## Signal assessment

The notebook auto-classifies the result:

| Signal | Criteria | Next step |
|--------|----------|-----------|
| **Strong** | UMAP separates, p < 0.01, KNN > 85% | Scale to multi-artist, multi-collection |
| **Weak** | Partial separation, p < 0.05, KNN 75-85% | Full-res tiles or fine-tune linear probe |
| **None** | No separation, p > 0.05, KNN ~75% | Custom CNN or CLIP with style prompts |

## Architecture

```
Rijksmuseum Search API → Object IDs
  → Linked Art resolver (la-framed) → metadata (title, creator, date)
  → EDM resolver (edm-framed) → IIIF image identifier
    → IIIF Image API (iiif.micr.io) → 2000px images
      → 224x224 tiles → DINOv2 ViT-B/14
        → CLS token (768d) + mean patch tokens (768d)
          → painting embedding (1536d)
            → UMAP / cosine similarity / KNN / candidate ranking
```

## Framing

This is **screening**, not authentication. Flags candidates for expert review. Never claims to authenticate.
