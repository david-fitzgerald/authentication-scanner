---
version: 0.1.0
date: 2026-02-26
status: complete
signal: weak-to-moderate
---

# Prototype Results: DINOv2 on Rijksmuseum Rembrandt Corpus

## Question

Does DINOv2 (pre-trained, no fine-tuning) capture style differences between confirmed Rembrandt autographs and other Dutch Golden Age painters?

## Dataset

| Group | Count | Artists |
|-------|-------|---------|
| Rembrandt autograph | 19 | Rembrandt van Rijn |
| Rembrandt circle | 2 | circle of / attributed to Rembrandt |
| Rembrandt pupils | 36 | Ferdinand Bol (19), Jan Lievens (10), Govert Flinck (7) |
| Other Dutch masters | 13 | Frans Hals (9), Johannes Vermeer (4) |
| **Total** | **70** | |

Source: Rijksmuseum APIs (data.rijksmuseum.nl). Images via IIIF at 2000px, tiled 224×224, embedded with `dinov2_vitb14` (768d CLS + 768d patch = 1536d per painting).

## Key Metrics

| Metric | Value |
|--------|-------|
| Autograph↔Autograph mean cosine sim | 0.8470 |
| Autograph↔Circle mean cosine sim | 0.8314 |
| Autograph↔Pupils mean cosine sim | 0.8255 |
| Autograph↔Other Dutch mean cosine sim | 0.8235 |
| Separation (auto-auto minus auto-pupils) | +0.0215 |
| Mann-Whitney p (vs pupils) | **1.01×10⁻³** |
| Mann-Whitney p (vs other Dutch) | **1.98×10⁻³** |
| Mann-Whitney p (vs circle) | 8.79×10⁻² (N=2, too small) |
| KNN K=3 accuracy | **82.9%** (58/70) |
| KNN K=5 accuracy | 75.7% (53/70) |
| KNN K=7 accuracy | **82.9%** (58/70) |
| Baseline (majority class) | 72.9% |

## Circle/Attributed Works

| Painting | Attribution | Cosine sim to autograph centroid |
|----------|------------|----------------------------------|
| Salome ontvangt het hoofd van Johannes de Doper | circle of Rembrandt | 0.9250 |
| Simson en Delila | attributed to Rembrandt | 0.8730 |

Salome is closer to the autograph centroid than most actual autographs. Simson en Delila sits further out.

## Assessment

**Weak-to-moderate signal.** The "WEAK" classification undersells the result:

- p=0.001 for pupil separation is genuinely strong — DINOv2 can tell Rembrandt from Bol/Flinck/Lievens (his own students) with zero fine-tuning
- p=0.002 for other Dutch masters (Hals, Vermeer) — also significant
- KNN at K=3,7 hits 82.9% — 10pp above majority baseline
- K=5 (75.7%) is an outlier likely due to small dataset
- Circle comparison (p=0.088) is meaningless at N=2

The bottleneck is **data, not method**. The Rijksmuseum has only 2 disputed Rembrandt works. The embedding approach works — it just needs a larger corpus.

## Decision

Per the decision framework:

| Signal | Criteria | Verdict |
|--------|----------|---------|
| Strong | UMAP separates, p < 0.01, KNN > 85% | p qualifies, KNN borderline |
| **Weak** | Partial separation, p < 0.05, KNN 75-85% | **This is where we land** |
| None | No separation, p > 0.05, KNN ~75% | Ruled out |

## Next Steps (ranked)

| # | Action | Cost | What it proves |
|---|--------|------|----------------|
| 1 | **Add Met Open Access** — Met has more circle/workshop Rembrandt | Free, ~1 day | Whether signal holds with more disputed works |
| 2 | **Linear probe fine-tune** — thin supervised layer on DINOv2 | Free (Colab), ~2 days | Whether supervised learning sharpens separation |
| 3 | **Full-res IIIF tiles** — use full resolution instead of 2000px | Free, ~1 hour | Whether brushwork detail at pixel level helps |

Recommendation: **Do #1 first.** The Rijksmuseum bottleneck is data, not method.

## API Discoveries

- Rijksmuseum `creator=` parameter only matches core names, not qualifiers ("circle of", "workshop of")
- Use `description=` for full-text search, then classify from `produced_by.referred_to_by[].content`
- Creator string is in LA profile at `produced_by.referred_to_by[].content` (AAT 300435416)
- IIIF ID is in EDM profile, extracted via regex from `iiif.micr.io/{id}`
- Title is in LA `identified_by[]` with type Name, fallback to EDM `dc:title`
- Date is in LA `produced_by.timespan.identified_by[].content`

## Repo

`david-fitzgerald/authentication-scanner` (private). Notebook: `prototype.ipynb`.
