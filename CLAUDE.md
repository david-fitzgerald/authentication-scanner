# Authentication — Claude Instructions

## Scope

AI-driven art authentication and lost manuscript discovery. Collection-scale screening — the gap nobody fills.

## Status

**Local pipeline complete (v1 + v2).** Met expansion done — circle N=2→18. Core finding: DINOv2 ViT-B/14 is a "different artist" detector (pupils p=2e-11, other Dutch p=3e-10) but **cannot separate circle/style-of from autograph** (p=0.80 at N=18). High-res + std features (v2) made things worse — collapsed all similarities toward 0.90. Next: try ViT-L/14 (option C) and entropy-weighted tiles (option D).

## Thesis

Style similarity shows up in vision embeddings. A single scan of open museum collections can flag misattributed works worth investigating. Position as screening/flagging, never "authentication."

## Technical Approach

**Phase 1 (current): DINOv2 embedding prototype**
- Embed painting tiles with DINOv2 (pre-trained, no custom training needed)
- Test on Rijksmuseum Rembrandt corpus (confirmed autograph vs "circle of")
- Key question: does style clustering work on generic vision embeddings?
- Compute: Google Colab Pro
- If signal exists → scale to multi-artist, multi-collection

**Phase 2: Multi-artist style embedding space**
- Contrastive/metric learning on top of DINOv2
- Build reference signatures for 200-500 well-documented artists
- Stream 500K+ paintings via IIIF, store only embeddings (~75 GB)
- FAISS index for cross-artist nearest-neighbour search

**Phase 3: Full scan + cross-reference**
- Score every painting against every artist signature
- Flag disagreements (current attribution ≠ model's top match)
- Filter: medium, period, conservation, statistical threshold
- Cross-reference against catalogue raisonné + provenance
- Output: ranked candidate list for expert review

## Funnel (expected)

```
800K paintings scanned
 → 500K sufficient resolution
   → 200K with reference signatures available
     → 10-20K show disagreement with current attribution
       → 1-2K statistically significant
         → 200-500 survive filters
           → 50-100 meaningful financial upside
             → 10-20 survive provenance check
               → 3-5 credible candidates for art historian review
```

## Monetisation (three paths)

1. **Collection-scale screening** — museum pilot, $50-200/work at volume. The structural gap.
2. **Auction intelligence** — screen "attributed to" / "circle of" lots pre-sale. Subscription to art funds.
3. **AI-only authentication service** — $1-2K/work (undercut Art Recognition at $2,200).

## Competitors

| Competitor | Model | Gap |
|---|---|---|
| Art Recognition (Zurich) | $2,200/work, one at a time, needs 100s of training images | No collection-scale offering |
| Hephaestus/ArtDiscovery (London/NYC) | 0.6% of certified value, insured guarantee | Premium only, no collection-scale |

## Legal Framing

Never "authenticate." Only "screen and flag for expert review." This reduces liability dramatically. E&O insurance from day one if commercialised.

## Key Files

| File | Contents |
|---|---|
| `prototype.ipynb` | **Phase 1 prototype.** DINOv2 embedding pipeline on Rijksmuseum Rembrandt corpus. Run on Colab Pro (T4). |
| `scan.py` | **Local pipeline.** Rijksmuseum + Met Open Access. Three-tier cache (metadata→images→embeddings). MPS device. Run with `python scan.py` or `--stage N`. `--hires` for v2 (high-res + std features). |
| `scan-results.md` | **Local pipeline results.** v1 vs v2 comparison, key findings, next steps (C/D/E/F). |
| `prototype-results.md` | **Colab prototype results.** Original Rijksmuseum-only metrics + API discoveries. |
| `business-plan.md` | **Attribution Alpha fund.** Buy misattributed works, reattribute, sell. Phased capital deployment. |
| `research.md` | Opportunity landscape — art authentication + lost manuscripts, six gaps, monetisation paths |
| `execution.md` | Detailed execution plan — MVP pipeline, week-by-week, competitors, capital requirements, risk factors |

## Repo

Private GitHub repo: `david-fitzgerald/authentication-scanner`. Contains `prototype.ipynb` + README.

## Infrastructure

- **Compute:** Google Colab Pro (T4 GPUs)
- **Data:** Rijksmuseum APIs (free, no API key needed — legacy key system retired Jan 2026). Search API + IIIF Image API + Linked Data Resolver. Docs: data.rijksmuseum.nl/docs/. Also: Met Open Access, other CC0 collections via IIIF.
- **Storage:** Embeddings only (~75 GB full scale, ~1.5 GB for Rijksmuseum prototype)

## Capital Requirements

| Phase | Cost | Timeline |
|---|---|---|
| Prototype (DINOv2 + Rijksmuseum) | ~$10 compute | 1-2 weeks |
| Multi-artist expansion | ~$50 compute | 2-4 weeks |
| Expert validation | $5-10K (art historian advisor) | Month 3-4 |
| First revenue | $10-31K total | Month 4-8 |

## Second Vertical: Lost Manuscripts

Stylometry for authorship attribution. Zero competition. Grant-fundable. Builds credibility. Deferred until art authentication prototype validates.

## Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-02-26 | Start with DINOv2 embeddings (Option C) over custom CNN | Fastest to validate signal. Days not weeks. No training needed. |
| 2026-02-26 | Multi-artist from the start | More signals per scan. Style embedding space > single-artist classifier. |
| 2026-02-26 | Rijksmuseum first | Largest open collection, best Rembrandt corpus, high-res IIIF API. |
| 2026-02-26 | Colab Pro for compute | User has 50 GB local. Colab Pro gives T4 GPUs, enough for prototype + full scan. |
| 2026-02-26 | Prototype result: weak-to-moderate signal | p=0.001 pupils, 82.9% KNN. Data bottleneck not method bottleneck. Add Met next. |
| 2026-02-26 | Met expansion: circle still inseparable | N=18 circle, p=0.80. DINOv2 ViT-B is "different artist" detector, not authenticator. |
| 2026-02-26 | High-res + std features (v2) hurt | All sims→0.90, lost pupil signal. Low-res mean-only (v1) is better. Revert to v1 as baseline. |
| 2026-02-26 | Next: ViT-L/14 (C) then entropy-weighted tiles (D) | Systematic options before concluding ViT-B is ceiling. |
