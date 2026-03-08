---
version: 0.1.0
date: 2026-02-26
status: draft
---

# Attribution Alpha — Quantitative Art Fund

## One-liner

Systematic scanner finds misattributed artworks at auction. Buy low, reattribute, sell high.

## Thesis

Art attribution is subjective and slow. Computer vision is fast and systematic. The information asymmetry between a scanner that has analysed millions of works and an auction house cataloguer who eyeballs each lot is the alpha. This alpha decays on publication — so we exploit it, not sell it.

## How It Works

```
Train scanner on 20-30 high-value artists (authenticated corpus)
  → Scan auction archives (Artnet, Invaluable, house archives)
    → Flag "attributed to" / "circle of" / "style of" lots
      → Score: P(autograph) × price_discount × liquidity
        → Buy top candidates at auction
          → Commission reattribution research (art historians)
            → Sell reattributed works at 10-100x purchase price
```

## Target Artists (20-30)

| Tier | Artists | Training corpus | Autograph price | Entry price |
|------|---------|----------------|-----------------|-------------|
| 1. Large corpus + value | Rembrandt, Rubens, Van Dyck, Titian | 200-400 works | $5-50M | $50-500K |
| 2. Small corpus + extreme value | Vermeer, Caravaggio, Raphael, Velázquez | 30-80 works | $30-100M+ | $100K-1M |
| 3. Active auction market | Hals, El Greco, Tintoretto, Bellini, Giorgione, Cuyp, Ruisdael | 50-200 works | $1-10M | $20-200K |
| 4. Volume play | Monet, Renoir, Corot, Boucher, Dutch/Flemish broadly | 100-500+ works | $1-20M | $10-100K |

Tier 1 + 3 = sweet spot. Enough data to train, enough auction supply, affordable entry.

## Economics

| Scenario | Bear | Base | Bull |
|----------|------|------|------|
| Lots purchased (over 5 years) | 20 | 40 | 60 |
| Avg purchase price | $75K | $60K | $50K |
| Capital in art | $1.5M | $2.4M | $3M |
| Research costs (all lots) | $300K | $500K | $600K |
| Successfully reattributed | 0 | 2 | 5 |
| Avg reattributed sale price | — | $8M | $10M |
| Gross proceeds | $1.2M* | $16M | $50M |
| Total cost (incl. scanner build) | $1.9M | $3M | $3.7M |
| **Net return** | **-$700K** | **+$13M** | **+$46M** |
| **MOIC** | **0.6x** | **5.3x** | **13.5x** |

*Bear case: sell unreattributed lots at ~80% of purchase price.

Breakeven requires ~2% reattribution hit rate. Scanner's job is to push this above 5%.

## Phases

### Phase 1: Build (months 0-18) — $50K

- Expand scanner to 20-30 artists
- Scan all open-access museum collections (~200K high-res paintings)
- Scan auction archives (Artnet subscription + scraping)
- Backtest: does the scanner flag known historical reattributions?
- Deliverable: ranked watchlist of candidates at upcoming auctions

### Phase 2: Scout (months 12-30) — $500K

- Systematic auction monitoring (automated alerts)
- Recruit 2-3 art historians as advisors (per-lot retainer)
- Purchase first 5-10 lots where scanner confidence highest + price lowest
- Begin reattribution research on top candidates
- Overlap with Phase 1 — start buying before scanner is "finished"

### Phase 3: Scale (months 24-60) — $2-3M

- Portfolio of 30-50 works across 10+ artists
- Parallel reattribution research (3-5 active projects)
- First publications / reattributions announced
- Track record enables outside capital raise if needed

### Phase 4: Harvest (months 48-96)

- Sell reattributed works (auction or private sale)
- Reinvest or distribute
- IP increasingly valuable — auction houses want the scanner

### Phase 5: Exit (months 72-120)

- Options: sell company to auction house / art insurer, license technology, or continue as fund
- Remaining portfolio liquidated or transferred

## Capital Scenarios

| Budget | What's possible | Expected outcome |
|--------|----------------|-----------------|
| **$4-5K** (current) | Scanner prototype for 3-5 artists. Backtest only. No purchases. Proves concept. | Proof of concept + watchlist. No revenue. |
| **$50K** | Full scanner (20 artists), auction data, backtest. Maybe 1-2 cheap lots. | Validated system + first acquisitions. |
| **$500K** (seed) | Phase 1+2. Scanner + 10 purchases + research. | First reattribution attempt within 2 years. |
| **$3M** (full fund) | Phase 1-3. 40 lots, parallel research, real portfolio. | 2-5 reattributions over 5 years. Target 5x return. |

## Outside Capital Path

The $4-5K → $50K gap can be bridged with a compelling backtest. If the scanner correctly flags 3-5 historically reattributed works in its backtest, that's a fundable pitch:

1. Build scanner on $4-5K budget (current)
2. Backtest against known reattributions (free — just need the data)
3. Pitch deck: "Our scanner flagged X of Y known reattributions. Here's the watchlist."
4. Raise $500K-3M from art-adjacent investors (art funds, family offices, auction insiders)

Art world investors understand attribution risk intuitively. The pitch is: "Moneyball for Old Masters."

## Moat

1. **Secrecy** — alpha decays on publication. Never publish the method.
2. **Data flywheel** — every purchase generates training signal (confirmed or denied)
3. **Relationships** — art historian network is slow to build and essential for reattribution
4. **Head start** — first systematic scanner has years of advantage in training data

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Scanner doesn't generalise beyond Rembrandt | High | Backtest before deploying capital |
| Reattribution politics — scholars disagree | High | Multiple independent opinions, choose battles carefully |
| Capital lockup — art is illiquid (2-5 year holds) | Medium | Buy cheap lots, diversify across artists |
| Legal/ethical challenge from auction houses | Low | No insider info, no market manipulation — just better analysis |
| Someone else builds the same thing | Medium | Speed + secrecy. Don't publish. |
| Art market downturn | Medium | Reattributed works hold value better than market |

## Immediate Next Steps

1. **Improve scanner** — tile distribution features + high-res + multi-artist
2. **Backtest** — find 10-20 known reattributions, check if scanner would have flagged them
3. **Artnet trial** — assess auction archive data quality and coverage
4. **Draft pitch deck** — contingent on backtest results
