# Lost Works & Authentication: AI in Historical Archives

## The Opportunity

Hundreds of millions of items sit in archives worldwide — uncatalogued, miscatalogued, or catalogued with minimal metadata. AI can systematically find misattributed works, authenticate art, and discover lost manuscripts. The financial and scholarly payoffs are enormous.

## Two Angles

### 1. Art Authentication (Financial)

Attribution is the single largest variable in art valuation:

| Case | Before | After | Multiplier |
|------|--------|-------|------------|
| Rembrandt, *Adoration of the Kings* | $10-16K ("circle of Rembrandt," Christie's 2021) | $14M (confirmed Rembrandt, Sotheby's 2023) | **~1,000x** |
| Maine attic portrait (possible Rembrandt) | Found in attic | $1.4M at auction; could reach $5-15M if authenticated | TBD |
| Ingres, *Odalisque en Grisaille* (Met) | ~$1M (attributed to Ingres) | ~$100K (reattributed to apprentice) | **0.1x** |

Scale of the problem:
- Art forgery costs the market **$6B annually** (Fine Arts Expert Institute)
- Estimated **40-50% of artworks in circulation are fake or misattributed**
- "School of" vs confirmed autograph: typically **10-100x** value difference

**Who pays:** Auction houses, museums, private collectors, insurers. Christie's invested $4.7M in photogrammetry equipment alone. The Louvre fingerprinted 8,500 paintings for authentication reference in 2024.

**AI state of play:**
- **Art Recognition** (Zurich, 2019) — leading company, claims 95% forgery detection accuracy via brushstroke/proportion/colour analysis
- **Nov 2024:** First auction house (Germann, Zurich) sold artwork authenticated entirely by AI — Marianne von Werefkin watercolour at 2x high estimate
- Identified counterfeit Monets and Renoirs on eBay (2024)
- Challenged Van Eyck *Saint Francis* panels — but leading scholars pushed back (only ~25 Van Eycks exist for training)
- Two AI models analysed the same Raphael and got different results — technology not yet authoritative alone

**The gap nobody's filling:** No one is running systematic AI authentication across entire museum collections to proactively flag misattributions. Everyone authenticates individual works on commission. Reactive, not systematic.

### 2. Lost Manuscripts & Texts (Discovery)

**What's known to be lost:**
- **Aristotle**: wrote ~200 works, only **31 survive**. Including lost second book of *Poetics* (on comedy).
- **Sappho**: ~10,000 lines across 9 books at Alexandria. Only **650 lines remain**.
- **Sophocles**: 123 plays, 7 survive complete.
- **Shakespeare**: At least 2 lost plays (*Love's Labour's Won*, *Cardenio*). ~3,000 Elizabethan plays are lost — surviving plays = 1/6 of total production.

**Proven discoveries:**

*Lope de Vega's La francesa Laura* — The clearest example. The ETSO project (University of Vienna) ran stylometry across ~350 Spanish Golden Age playwrights at >99% accuracy. Identified an anonymous manuscript at Spain's National Library as a previously unknown Lope de Vega comedy (~1628-1630). Traditional philological analysis confirmed the AI finding. Needle in haystack, found by machine.

*Vesuvius Challenge (2024-25)* — ML/computer vision virtually unrolled carbonized Herculaneum scrolls (buried 79 AD). Three students won the $700K Grand Prize for revealing 2,000 characters — a previously unread Philoderus tract. Oxford's Bodleian announced first image inside scroll PHerc. 172 in Feb 2025. Pipeline: synchrotron CT scanning → 3D virtual unwrapping → AI ink detection.

*Dead Sea Scrolls dating* — AI model "Enoch" predicted radiocarbon dates with ~28-31 year accuracy. Key finding: many scrolls are older than thought — some dating to late 4th century BCE, ~100 years earlier than prior estimates.

## The Bottleneck: Access, Not Algorithms

"Digitised" does not mean "searchable." The critical pipeline:

```
Physical item → Scanned image → OCR/HTR → Searchable text → AI analysis
                     ↑                ↑                          ↑
              Most stop here    Fails on handwriting    Almost nobody here
```

- **Google Books**: 40M+ titles scanned, most OCR'd, but accuracy degrades badly pre-1800
- **Internet Archive**: 47M+ texts, older docs have poor OCR
- **Vatican Library**: 80,000 manuscripts, only ~30,000 digitised, many not text-searchable
- **Oxyrhynchus Papyri**: 500,000+ fragments, **1-2% processed**. AI character recognition (YOLOv8) being applied but scale dwarfs resources
- **Handwritten Text Recognition** (Transkribus) is the emerging solution but requires training per hand/script

## Technical Methods

| Method | Application | Maturity |
|--------|-------------|----------|
| **Stylometry** | Authorship attribution from writing patterns (function words, sentence length, n-grams) | Mature — >99% accuracy with sufficient training data |
| **HTR / Transkribus** | Reading handwritten manuscripts | Mature for trained scripts, weak on long tail |
| **CNN brushstroke analysis** | Art authentication at ~100x100px tile level | Working but not yet authoritative alone |
| **Multispectral imaging** | Palimpsest recovery (12+ wavelengths, UV to near-IR) | Mature but expensive and slow |
| **CT scanning + AI** | Reading sealed/carbonised scrolls (Herculaneum) | Proven but requires synchrotron facilities |
| **NLP cross-referencing** | Matching lost work descriptions against catalogue entries | Early / unexplored |

## The Six Gaps

1. **The uncatalogued majority.** Institutions hold massive uncatalogued material — "Box 47: miscellaneous manuscripts." AI-generated metadata for these collections is transformative but barely attempted.

2. **No systematic cross-archive search.** The Lope de Vega discovery was within one national tradition. Nobody is running stylometry across archives globally.

3. **Oxyrhynchus at scale.** 500K+ fragments, 1-2% done. AI fragment matching (joining torn pieces across boxes) could accelerate by orders of magnitude.

4. **Non-Western archives almost untouched.** Arabic/Persian/Ottoman, South Asian palm-leaf, Dunhuang, Ethiopian/Ge'ez, Central Asian Buddhist — vast manuscript traditions with minimal AI work.

5. **The "known unknown" match.** Nobody is systematically matching lists of known lost works against catalogue descriptions of anonymous/undated items using NLP.

6. **Art authentication at collection scale.** Everyone authenticates on commission. Nobody runs systematic AI across entire museum holdings to flag potential misattributions.

## Underexplored Archives

- **Vatican Apostolic Archive**: 85km of shelving, barely digitised
- **Timbuktu manuscripts**: Hundreds of thousands, many in private family collections
- **Cairo Geniza** (and other genizot): Cambridge holds most, but fragments remain unmatched
- **Latin American colonial archives**: Massive, minimally catalogued
- **Eastern European monastery collections**: Survived WWII, poorly catalogued
- **Institutional attics / country houses / estate storage**: The Maine Rembrandt is not unusual. No systematic approach exists.

## Monetisation

| Path | How | Payoff |
|------|-----|--------|
| **Art reattribution** | Find "school of X" works that are actually X. Broker authentication, take percentage. | 10-1000x value multiplier per work |
| **Forgery detection service** | Systematic screening for auction houses, insurers | $6B annual forgery market |
| **Literary discovery** | Find lost works, publish, academic prestige → grants, consulting | Lower direct financial payoff, high prestige |
| **Palimpsest recovery** | Identify and read overwritten texts at scale | Scholarly + potential IP on recovered content |
| **Catalogue enrichment** | AI-generated metadata for institutions → licensing, SaaS | Recurring revenue, builds moat through data access |
