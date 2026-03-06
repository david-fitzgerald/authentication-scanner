---
version: 0.1.0
date: 2026-02-24
status: research
---

# Historical Archives / Art Authentication: Execution Plan

Companion to `lost-works-authentication.md` (what the opportunity is). This file is **how you'd actually build it**.

## 1. MVP Pipeline — Art Authentication

### What Images Do You Need?

For CNN-based brushstroke authentication, you need:
- **High-resolution photographs** of the painting surface (minimum ~300 DPI equivalent, ideally 600+)
- **Full painting image** + **macro detail shots** (brushstroke-level)
- Standard lighting, no heavy post-processing
- Ideally: raking light, UV fluorescence, infrared reflectography for deeper analysis (but these are bonus, not MVP)

Art Recognition requires **several hundred training images** of confirmed authentic works per artist.
Hephaestus (Pictology AI) claims it needs only **30 images** per artist.

### Available Open Museum Collections

| Museum | Collection Size | Open Access Images | Resolution | API | License |
|--------|----------------|-------------------|------------|-----|---------|
| **Metropolitan Museum of Art** | 470K+ artworks | 406K+ images | High-res JPEG, downloadable | REST API (JSON) | CC0 |
| **Rijksmuseum** | 800K+ objects | 800K+ (most with images) | Very high-res via IIIF | REST API + IIIF | CC0 |
| **National Gallery of Art (DC)** | 150K+ | ~52K images | High-res | REST API | CC0 |
| **Art Institute of Chicago** | 300K+ | Open access subset | High-res IIIF | IIIF API | CC0 |
| **Smithsonian** | 4.5M+ | 4.5M+ | Varies | Open Access API | CC0 |
| **Cleveland Museum of Art** | 61K+ | Open access | High-res | REST API | CC0 |
| **Yale University Art Gallery** | Thousands | Open access subset | High-res | IIIF | CC0 |

**Total accessible paintings with high-res images for free: ~500K-1M+**

### Could You Build a Prototype?

**Yes.** The Rijksmuseum alone has enough Rembrandt paintings (~400 works attributed to Rembrandt/workshop/circle) plus thousands of Dutch Golden Age paintings for training a Rembrandt-specific model.

### Minimum Viable Pipeline

```
Rijksmuseum IIIF API / Met Open Access API
    → Download all paintings by target artist (confirmed authentic, from catalogue raisonne)
    → Download "circle of" / "attributed to" / "school of" works
    → Tile images into 100x100px or 256x256px patches (brushstroke level)
    → Train CNN (ResNet-50 or EfficientNet) on authentic vs. known-not-authentic tiles
    → Siamese network or contrastive learning for style similarity scoring
    → Apply to "attributed to" / "circle of" works → rank by probability of autograph
    → Cross-reference with provenance data, condition reports, historical documentation
```

### Hardware and Models

| Component | What | Cost |
|-----------|------|------|
| Training data | Open access museum APIs | Free |
| Image tiling + preprocessing | Python (Pillow, OpenCV) | Free |
| Model training | ResNet-50 or EfficientNet, fine-tuned. ~2-4 hours on single GPU | $5-20 cloud GPU |
| Inference | CPU-viable for individual works | Free |
| Alternative: CLIP / DINOv2 embeddings | Pre-trained vision transformer embeddings, then clustering/classification | Free (open models) |

**Total MVP cost: effectively $0-$50** in compute. The data is free.

### Week-by-Week MVP

| Week | Task | Output |
|------|------|--------|
| 1 | Download Rijksmuseum collection via API. Index all Rembrandt + "circle/school/manner of Rembrandt" works. Build tile preprocessing pipeline. | Training dataset |
| 2 | Train CNN classifier: Rembrandt autograph vs. known copies/school works. Evaluate with held-out set. | Working model |
| 3 | Run model across all "attributed to" / "circle of" Rembrandt works in Rijksmuseum + Met. Rank by authenticity probability. | Candidate list |
| 4 | Cross-reference top candidates against catalogue raisonne (Bredius/Gerson, RRP). Identify any "circle of" works the model flags as potentially autograph. | Research findings |

## 2. Lost Manuscripts — Minimum Viable Stylometry

### Pipeline

```
Digitized manuscript corpus (e.g., all anonymous plays in Biblioteca Nacional de Espana)
    → HTR/OCR: Transkribus (pre-trained models for period scripts)
    → Clean text extraction
    → Feature extraction: function word frequencies, sentence length, vocabulary richness,
      character n-grams, syntactic patterns (POS tag sequences)
    → Train stylometric model on known-authorship corpus (same period, language)
    → Apply to anonymous/unattributed manuscripts
    → Rank by authorship probability
    → Cross-reference with historical records of lost works
```

### Most Accessible Archives for Unattributed Manuscripts

| Archive | Scale | Digitized | Searchable Text | Access |
|---------|-------|-----------|-----------------|--------|
| **Gallica** (BnF) | 14M+ docs | Many millions | Mixed (good for printed, poor for MS) | Free, open |
| **British Library Digitised Manuscripts** | 30K+ manuscripts digitized | Growing | Image-only (needs HTR) | Free, open |
| **BnF-BL Polonsky Project** | 800 medieval manuscripts | Fully digitized | Image-only | Free, open, IIIF |
| **Biblioteca Nacional de Espana** | Massive | Significant portion | Text available for some | Free |
| **e-codices** (Switzerland) | 2,600+ manuscripts | All | Image-only | Free, CC |
| **Internet Archive** | 47M+ texts | All | OCR (quality varies) | Free |
| **HathiTrust** | 17M+ volumes | All | OCR + full text search | Free (in-copyright: limited) |

**Best entry point for stylometry MVP:** Pick a specific author with (a) a large authenticated corpus and (b) known lost works. The Lope de Vega precedent (ETSO project) worked because Golden Age Spanish drama has ~350 playwrights with digitized, attributed corpora, and thousands of anonymous plays in the same archives.

**Other high-value targets:**
- **Shakespeare**: 2+ lost plays, extensive authenticated corpus, Early English Books Online (EEBO) has thousands of anonymous Elizabethan plays
- **Chaucer**: authenticated corpus + many anonymous Middle English texts
- **Anonymous medieval lyrics/romances**: vast unattributed corpus in BnF/BL

## 3. First Customer / First Dollar

### Who Pays for Authentication?

| Customer | What They Need | Price Sensitivity | Access Difficulty |
|----------|----------------|-------------------|-------------------|
| **Auction houses** (Christie's, Sotheby's, Bonhams, Germann) | Pre-sale authentication, reduce risk | Low (authentication costs are tiny vs. sale value) | Medium — relationships required |
| **Private collectors** | Authentication before purchase or for insurance | Low for high-value works | Easy — direct |
| **Museums** | Collection-wide attribution review, deaccessioning decisions | High (budget-constrained) | Medium — slow institutional processes |
| **Insurance companies** (AXA Art, Hiscox) | Authentication for underwriting, claims validation | Medium | Medium |
| **Art funds / investment vehicles** | Due diligence on acquisitions | Low | Easy — direct |
| **Law firms** (art disputes) | Expert opinion for litigation | Very low | Easy — they come to you |
| **Art dealers / galleries** | Authentication to increase sale price | Low for high-value works | Easy |

### Fee Structures (Benchmarks)

| Provider | Service | Price |
|----------|---------|-------|
| **Art Recognition** (AI only) | AI authentication report | ~$2,200 per work. Two tiers: basic (yes/no) and detailed (probability %) |
| **Hephaestus/ArtDiscovery** (AI + scientific + provenance) | Full authentication with insured guarantee | **60 basis points (0.6%) of certified value.** For a $1M painting = $6,000. For a $100M painting = $600,000. |
| **Traditional authentication** (connoisseur opinion) | Expert letter | $2,000-$25,000+ depending on expert prestige and work complexity |
| **Scientific analysis** (pigment, canvas, dendrochronology) | Lab testing | $5,000-$50,000 depending on methods required |

### Fastest Path to First Dollar

**Option A: AI-Only Authentication Service (Art Recognition competitor)**
- Price: $1,000-$3,000 per work (undercut Art Recognition)
- Target: private collectors, small dealers, online art marketplace sellers
- Revenue potential: 50 works/month at $2,000 = $100K/month
- **Problem:** Credibility. You need a track record before anyone trusts your AI.

**Option B: Collection-Scale Screening (the gap nobody fills)**
- Pitch to museums: "We'll scan your entire open-access collection and flag works where our AI disagrees with current attribution. Free pilot on 100 works. Pay for the full scan."
- Price: $50-$200 per work at collection scale (volume discount), or flat project fee ($50K-$200K for a major museum's painting collection)
- **This is the gap.** Nobody offers this. Art Recognition does one painting at a time. Museum budgets are tight but grant-fundable.

**Option C: Art Fund / Collector "Alpha" Service**
- Pitch: "We systematically screen auction catalogues for undervalued misattributed works before each sale. We flag 'school of' works that our model scores as potentially autograph. You buy them."
- Revenue model: success fee (% of value increase upon re-authentication) or subscription ($10K-$50K/year)
- **This is the most exciting commercial angle.** A Rembrandt "circle of" painting bought for $15K and confirmed as autograph is worth $5M-$15M. You take 10-20% of the upside.

### The Salvator Mundi Benchmark

- Bought in 2005 for **$1,175** at a New Orleans auction
- Authenticated as Leonardo da Vinci
- Sold in 2017 for **$450.3 million**
- Value increase: **383,000x**
- If reattributed to workshop: drops to ~$1.5M
- The authentication process (not AI) included microscopic pigment analysis, expert consensus, and restoration

Other examples:
- $139 thrift shop painting → verified Jackson Pollock → valued at $10M
- $30 "unknown German 19th century" drawing → confirmed Albrecht Durer → valued at $46M
- Caravaggio found in French attic (2014) → authenticated → valued at $100M+

## 4. Competitor Deep Dive

### Art Recognition (Zurich, est. 2019)

| Dimension | Details |
|-----------|---------|
| **Founders** | Dr. Carina Popovici, Christiane Hoppe-Oehl |
| **Technology** | Deep CNN trained on photographs of authentic works. Analyzes brushstrokes, proportions, color. Needs several hundred training images per artist. |
| **Pricing** | ~$2,200/work. Two tiers. |
| **Milestone** | Nov 2024: First auction (Germann, Zurich) sold artwork authenticated solely by AI — von Werefkin watercolor at 2x high estimate ($17K). Louise Bourgeois ink drawing sold for $31,600. |
| **Limitations** | Needs 100s of training images (fails for artists with small oeuvres like Vermeer, Van Eyck). Two AI models analyzing the same Raphael produced different results. Van Eyck analysis was challenged by leading scholars. |
| **Funding** | Undisclosed (small, self-funded or angel). Listed on Azure Marketplace. |

### Hephaestus Analytical / ArtDiscovery (London/NYC, est. 2018)

| Dimension | Details |
|-----------|---------|
| **Founder** | Denis Moiseev |
| **Technology** | "Pictology" AI: needs only 30 images per artist (vs. Art Recognition's hundreds). Analyzes brushstroke variation, curvature, motor skill signatures. Combined with SEM-EDX, Raman spectroscopy, FTIR, radiocarbon dating, provenance research. |
| **Pricing** | 60 basis points (0.6%) of certified value for insured guarantee. |
| **Milestone** | 2025: Merged with ArtDiscovery (NYC lab). Launched "world's first insured authenticity guarantee for artworks" — backed by a major insurer. Certificate travels with artwork as transferable warranty. |
| **Key advisor** | Noah Charney (Pulitzer finalist, art crime expert) joined Art Recognition (not Hephaestus) as advisor in Jan 2025. |
| **Strengths** | Full-stack (AI + scientific + provenance + insurance). The insured guarantee is a game-changer — collectors can use it for non-recourse lending. |
| **Limitations** | Still small. Premium pricing limits market to high-value works. No collection-scale offering. |

### What You'd Do Differently

1. **Collection-scale, not single-work.** Both competitors authenticate one work at a time on commission. Nobody offers "scan your entire collection." This is the structural gap.
2. **Open access data advantage.** 500K+ paintings are freely available in high-res from museums. Train on everything, not just commission by commission.
3. **Auction screening as a product.** Systematically scan every upcoming auction's "attributed to" / "circle of" lots. Flag probable misattributions. Sell this as intelligence to art funds.
4. **Manuscripts as a second vertical.** Neither Art Recognition nor Hephaestus does text. Stylometry for lost manuscripts is a completely uncontested space.
5. **Lower price point, higher volume.** $500/work for AI-only screening (vs. $2,200 Art Recognition). Makes collection-scale economically viable.

## 5. Legal and Reputational Risk

### What Happens When Your AI Says "Fake"?

This is the existential risk for any authentication business.

**The legal landscape:**
- Authentication can subject you to liability **even when you don't use the word "authenticated"** — any comment on authorship carries legal risk
- The Knoedler Gallery scandal (2011): $80M in fake Rothkos, Pollocks, etc. Multiple lawsuits, criminal charges. Authenticators are routinely sued.
- Several artist foundations (Warhol, Basquiat, Haring) **shut down their authentication boards** specifically to avoid litigation
- In the US, the Federal Rules of Evidence don't have established frameworks for admitting AI conclusions as evidence

**Specific risks:**

| Risk | Scenario | Mitigation |
|------|----------|------------|
| **False negative** — AI says fake, it's real | Owner sues for devaluation / lost sale | Frame all output as "statistical analysis, not authentication opinion." Require human expert review. Insurance. |
| **False positive** — AI says real, it's fake | Buyer sues after purchase based on your report | Same framing. Limit liability in contracts. E&O insurance. |
| **Conflicting AI results** | Two AI systems disagree (already happened with Raphael) | Position as "screening tool" not "final authority." Transparency about confidence intervals. |
| **Art world politics** | Established experts disagree with your AI finding | This WILL happen. Art authentication is a political/social process, not purely technical. Build relationships with scholars, not against them. |

### The Hephaestus Model

Hephaestus/ArtDiscovery's insured guarantee (backed by a major insurer) is the smartest structural innovation. It transfers the liability from the authenticator to an insurance company. The insurer prices the risk based on the thoroughness of the authentication process.

**Implication for you:** Don't compete on authentication opinions. Compete on **screening and flagging at scale.** "Our AI identifies works that merit further investigation by human experts." You never authenticate — you triage. This dramatically reduces legal exposure.

## 6. Institutional Access

### How Do You Get Museums to Let You Scan?

**You don't need permission for open-access collections.** The Met (406K images), Rijksmuseum (800K), and others have CC0 open access APIs. Download and analyze freely.

**For non-open-access collections:**

| Approach | How | Timeline |
|----------|-----|----------|
| **Research partnership** | Approach museum's research/curatorial department. Offer joint publication. Academic affiliation helps enormously. | 3-12 months |
| **Free pilot** | "We'll screen 100 works from your collection at no cost and deliver a report." Low-risk for the museum. | 1-3 months to agree |
| **Grant-funded project** | Apply for AHRC, NEH, Getty Foundation, or Mellon Foundation grant with museum as partner. Museum gets free analysis; you get access + credibility. | 6-18 months |
| **Student / visiting researcher** | Embed in a museum's research program. Many museums have visiting scholar programs. | 3-6 months |
| **University partnership** | Partner with a university art history department. They have existing institutional relationships. | 1-3 months |

### The Access Hierarchy

1. **Tier 1 — Immediate (no permission needed):** Open-access museum APIs. 500K+ paintings available today.
2. **Tier 2 — Easy (request access):** IIIF-published collections without full open access. Many museums publish images but require attribution.
3. **Tier 3 — Moderate (relationship required):** Non-digitized collections. Requires institutional partnership, usually through a research collaboration.
4. **Tier 4 — Hard (political):** Private collections, estate storage, institutional attics. The "Maine Rembrandt" scenario. Requires trust and often intermediaries.

**Start with Tier 1. Prove the model on freely available data before seeking access to anything restricted.**

## 7. Skills Needed

### Art History Knowledge Required?

**For the AI pipeline: No.** The CNN doesn't need to know art history. It learns style from images.

**For everything else: Somewhat.**

| Phase | Art History Needed? | What Specifically |
|-------|-------------------|-------------------|
| Building the model | No | Computer vision / ML skills |
| Selecting training data | Some | Need to know which works are reliably attributed (catalogue raisonne) |
| Interpreting results | Yes | Understanding attribution levels, provenance, conservation history |
| Selling to art world | Yes | Credibility, relationships, speaking the language |
| Defending findings | Yes | Art historical reasoning to support or contextualize AI output |

### Verdict

**You don't need an art history PhD, but you need access to one.** A part-time advisor (art historian specializing in the period/artist you're targeting) is essential for credibility and interpretation. Many academics would be enthusiastic collaborators — this is exactly the kind of digital humanities project that attracts grant funding and publications.

**Minimum viable team:**
- **Builder** (you): ML pipeline, data engineering, product, business
- **Art history advisor** (part-time): Academic, specializing in period of focus. 5-10 hrs/month.
- **For manuscripts:** Add a computational linguistics / NLP person or skill

## 8. Capital Requirements and Timeline

| Phase | Duration | Cost | Milestone |
|-------|----------|------|-----------|
| **Phase 1: Prototype** — Build CNN on Rijksmuseum open data. Rembrandt authentication model. | 4 weeks | ~$50-100 (compute) | Working model with accuracy metrics |
| **Phase 2: Validation** — Test against known attribution changes. Does the model correctly flag works that were reattributed? | 2-4 weeks | ~$0 | Validated accuracy on historical reattributions |
| **Phase 3: Collection scan** — Run model across all "attributed to" / "circle of" / "school of" works in open collections | 2-4 weeks | ~$100-500 (compute for full collection) | Candidate list of potential misattributions |
| **Phase 4: Expert review** — Engage art historian to evaluate top candidates | 4-8 weeks | $5K-10K (advisor fees) | Shortlist of credible reattribution candidates |
| **Phase 5: First revenue** — Either (a) approach auction houses/collectors with findings, or (b) publish + attract clients | 4-12 weeks | $5K-20K (legal, IP, marketing) | First paying customer or partnership |

### Total: $10K-$31K over 4-8 months

**This is dramatically cheaper than the metagenomic opportunity.** The data is free, the compute is cheap, and you don't need wet-lab validation. The bottleneck is credibility, not capital.

### If Going Bigger

- Seed ($250K-$1M): Hire art history advisor full-time, expand to multiple artists/periods, build collection-scale screening product, hire a salesperson with art world connections
- Series A ($3-10M): Multi-artist models, manuscript stylometry vertical, auction screening service, insured authentication product (requires insurance partnership)

## 9. Risk Factors — What Kills This

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Art world rejection** — traditional experts refuse to accept AI authentication | HIGH | This is already happening. Position as "screening tool that supports human experts" not "replacement." Partner with respected scholars. The Germann auction (Nov 2024) is a proof point that the market is opening. |
| **Liability from authentication opinions** | HIGH | Never "authenticate." Only "screen" and "flag for expert review." Robust disclaimers. E&O insurance. Learn from Hephaestus: eventually partner with an insurer. |
| **Small market** — how many works need authentication? | MEDIUM | Art forgery is $6B/year. But the number of truly high-value misattributed works is finite. Manuscript discovery adds a second vertical. Collection cataloguing (metadata enrichment) is a broader market. |
| **Art Recognition / Hephaestus dominate** | MEDIUM | They're doing single-work authentication. Collection-scale screening and auction intelligence are uncontested. Manuscripts are completely uncontested. |
| **Model produces nonsense** — CNN learns surface features, not true style | MEDIUM | Known problem. Two AI models disagreed on a Raphael. Mitigate with ensemble methods, multiple model architectures, and always requiring human expert review. |
| **Open access collections are biased** — museums share their best-documented works, not the ambiguous ones | LOW-MEDIUM | True. The "circle of" works that most need analysis are often the least photographed. This pushes you toward institutional partnerships (Tier 3-4 access). |
| **Someone sues** | MEDIUM | One lawsuit could sink a small company. Structure as LLC with limited liability. E&O insurance from day one. Legal review of all published findings. |

### What Kills It Dead

Two scenarios:

1. **You build a model, find "candidates," present them to the art world, and get laughed out of the room.** If the first few findings are embarrassingly wrong, you have no credibility to recover. The art world is small and reputation-driven. **Mitigation: validate on KNOWN reattributions first.** Run the model on works that were reattributed in the last 20 years. If the model correctly identifies them, you have a credible validation story.

2. **Liability kills you.** One false positive (AI says real, it's fake) or false negative (AI says fake, it's real) that results in a financial loss for someone who relied on your output. **Mitigation: never make authentication claims. Always position as screening/flagging. Contractual liability caps. Insurance.**

### The Structural Advantage

Unlike metagenomic enzymes, art authentication has a **self-reinforcing flywheel**: each successful identification builds credibility, attracts more clients, generates more data, improves the model. The Germann auction (Nov 2024) was the first turn of this flywheel for AI authentication. The market is at an inflection point.

The manuscript vertical is pure upside — zero competition, grant-fundable, generates publications and prestige that feeds back into the authentication business's credibility.
