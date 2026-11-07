# Where a prediction model fits — mapping the researcher's actual choices

Prepared 2026-11-07. Purpose: establish what wet labs currently do when they want to understand
genes they know little about, what each option costs in money and months, and where (if anywhere)
a perturbation-prediction model changes a real decision.

Every figure below carries its source. Figures marked ⚠️ come from vendors or single quotes and
should be confirmed before being put on a slide.

---

## 1. The decision map

```
START: "I care about gene X (or 500 candidate genes). What does it do?"
│
├─► STEP 1  Prior knowledge          $0        days
│   Literature, homology (BLAST/Pfam/AlphaFold), UniProt, GeneCards
│   └─ Enough? ──► STOP. Most well-studied genes end here.
│      For ~80% of genes, there is little or nothing to read.
│
├─► STEP 2  Public data mining       $0        days–weeks
│   DepMap co-essentiality · co-expression nets · STRING · IMPC mouse KO
│   · Human Protein Atlas · GWAS/ClinVar · Pharos
│   Output: a hypothesis, by guilt-by-association
│   └─ Enough to act? ──► Often yes. This is the free default.
│      Limitation: ALL correlational. Nothing about YOUR cell type/condition.
│
└─► STEP 3  Run an experiment. Which readout do you need?
    │
    ├─► 3a  BULK POOLED SCREEN           cheapest    ~3–5 months
    │   Readout: survival / growth / one FACS marker
    │   Output: ONE NUMBER per gene ("does this gene matter here?")
    │   Genome-wide is routine.
    │
    ├─► 3b  PERTURB-SEQ (single-cell)    10–25× 3a   ~3–5 months
    │   Readout: full transcriptome per perturbation
    │   Output: THOUSANDS OF NUMBERS per gene ("what changed, and how?")
    │   → enables clustering genes by function
    │   Genome-scale: $100k–$1M ⚠️
    │
    └─► 3c  ARRAYED SCREEN               high upfront  longer
        Readout: imaging / high-content, one perturbation per well
        Usually a FOLLOW-UP on hits from 3a/3b, in a more relevant model
        4–5 weeks for validation of a hit list

                            ▼
              STEP 4  Validate hits, publish, repeat
```

---

## 2. Costs and timelines, with sources

### Time

| Stage | Duration | Source |
|---|---|---|
| Generate the Cas9/dCas9 cell line (before anything) | ~4 weeks | STAR Protocols |
| The screen itself | 1–2 months | Harvard DRSC |
| Sequencing core turnaround | ~3–6 weeks ⚠️ (vendor) | Merck |
| **Genome-wide pooled screen, full service** | **12–20 weeks** | UT Health San Antonio core |
| Sub-library (focused) screen, full service | 8–16 weeks | same |
| Genome-scale protocol, design → data | 9–15 weeks, + 4–5 weeks validation | STAR Protocols |

A large share of this is waiting, not hands-on work. The core facility's queue is the main variable.

### Money

| Item | Figure | Source |
|---|---|---|
| **Genome-wide transcriptional screen, mammalian cells** | **$100,000 – $1,000,000** ⚠️ (investigator quote, wide range) | BioTechniques |
| Single-cell readout vs simple viability readout | **10–25× more per guide** | Addgene, 2026 |
| NGS deconvolution service (post-screen only) | from $320/sample, 3–5 wk | VectorBuilder |
| Per-cell sequencing at depth | ~$0.10–0.30 ⚠️ (low-confidence secondary source) | — |

### What drives the cost

- **Sequencing depth:** ≥20,000 reads/cell for the transcriptome + 2,000 reads/cell for guide
  identification (Illumina, NovaSeq X).
- **Cell count:** coverage scales with library size — ~5M cells per sample for a 100,000-guide
  library at 50× coverage.
- **Cost-reduction methods exist and are active research:** compressed Perturb-seq reports
  4–20× cost reduction; TAP-seq reports up to 50× less sequencing. **The cost of the thing we
  propose to predict is actively falling.**

---

## 3. Questioning every step — where the argument breaks

### ❌ "Researchers don't know which genes to screen"
Weak. They usually do: a pathway, a disease, a hypothesis. And when they don't, Step 2 is free.

### ⚠️ "Screens are too expensive, so predict instead" — true for Perturb-seq, false for bulk

**Correction (2026-11-07).** An earlier version of this document claimed pooled screens scale
sub-linearly, so adding genes is nearly free, and concluded the cost argument was weak. **That is
true for bulk screens and false for Perturb-seq**, which is the assay we predict.

| | Bulk pooled screen | Perturb-seq |
|---|---|---|
| Cells per perturbation | ~50 (coverage is enough) | **~100–800** (needed to average out single-cell noise) |
| Sequenced per cell | the guide barcode only | **full transcriptome: ~20k reads + 2k for the guide** |
| Cost of +1,000 genes | small | **~300,000 more cells, fully sequenced** |

For Perturb-seq, **cost scales roughly linearly with perturbation count**, because cells and
sequencing dominate and both scale with it. Sanity check: Replogle's 2.5M-cell genome-scale screen
at ~$0.10–0.30/cell ⚠️ lands at $250k–750k, consistent with the $100k–$1M figure from an
independent source.

What survives of the counter-argument:
- **Time does not scale.** 200 or 2,000 perturbations is the same 12–20 weeks. We make nobody faster.
- **Brute force exists.** Replogle 2022 already screened every expressed gene — in their cell lines.
- **The cost axis is eroding independently.** Compressed Perturb-seq claims 4–20× savings and
  TAP-seq up to 50× less sequencing, with no model involved.

### ❌ "Literature review takes a year, we skip it"
Partly true (target ID + validation runs 1–2 years), but our model doesn't read literature. The
right version is the **bias** argument: ~80% of genes are understudied, largely because of which
genes were convenient to work with in the 1980s–90s. For those, there is nothing to read, so
prediction and guilt-by-association are the only options.

### ✅ Where prediction has *irreducible* value — cases pooling cannot solve

| Case | Why experiments can't cover it |
|---|---|
| **Gene pairs / combinations** | ~200M pairs. No pooling strategy reaches that. |
| **Every cell context** | Each new cell type, timepoint or dose is a fresh 3–5 month screen. |
| **Cell-limited systems** | Pooled screens need millions of cells. Patient-derived, primary, or rare-population cells often can't supply them **at any price**. |
| **In vivo** | Cell numbers and logistics are far harder than in a flask. |

That third row is the sharpest: a cost argument can always be beaten by cheaper sequencing, but a
**cell-number** constraint can't. If a patient biopsy yields 50,000 cells, a 100,000-guide screen
is impossible regardless of budget.

---

## 4. The honest use case, if everything goes well

**Insertion point:** between Step 2 (free, correlational) and Step 3b (expensive, causal).

> A lab runs Perturb-seq on the genes it can justify — a focused library in its own cell system.
> The model extends those measurements to perturbations the lab cannot run: genes left out,
> combinations, or a context where cells are too scarce to screen.
> Output: predicted transcriptional responses that cluster alongside the measured ones, assigning
> likely function to genes with no literature.

**The comparison that decides whether it's worth anything** is three-way, not two:

| Approach | Input needed | Status |
|---|---|---|
| Co-essentiality clustering | DepMap (free) | **Incumbent.** Assigned function to 108 uncharacterised genes (Wainberg 2021) |
| Clustering on **measured** Perturb-seq | A screen you ran | **Ceiling.** The richer signal |
| Clustering on **predicted** responses | A model | What we'd add — must land near the ceiling to matter |

If predicted clustering only matches co-essentiality, we add nothing: the free method already
does that.

---

## 5. What we would have to prove, in order

1. **Predictions vary informatively between genes.** Low RMSE does not imply this — the
   training-mean baseline has decent RMSE and predicts the *same profile for every gene*, which
   has zero clustering value. None of our current metrics test this. **Test first; it's cheap and
   it can invalidate everything downstream.**
2. **Predicted responses recover measured cluster structure** on held-out genes.
3. **That clustering beats co-essentiality** on a fair gene set — one where DepMap actually has
   signal, not a set of non-essential genes where the incumbent is handicapped by construction.
4. **It transfers** to a case experiments can't reach (combinations or a new context). Only here
   does the value become irreducible.

We have evidence for none of these yet. Step 1 runs on predictions already saved to the Volume.

---

## 6. Current status of our evidence

| Claim | Status |
|---|---|
| Benchmark with honest baselines on Adamson | ✅ Done, reproducible |
| Standard metric is saturated (no-change scores 0.99 Pearson) | ✅ Measured |
| Foundation models beat baselines on responsive genes (0.27 vs 0.31 error) | ✅ Measured |
| Foundation models beat baselines overall | ❌ They lose (0.072 vs 0.0575) |
| Our JEPA v0 helps | ❌ No measurable difference |
| Cell-JEPA reimplementation helps | ⏳ Vincent's Track A, in progress |
| Predictions are useful for clustering | ❓ **Untested — the key gap** |
| Beats co-essentiality | ❓ Untested |

---

## Sources

- [Addgene, genome-wide KO libraries & relative screen costs (2026)](https://blog.addgene.org/julianna-and-jacquere-genome-wide-crispr-knockout-libraries)
- [BioTechniques, next-generation genetic screening ($100k–$1M quote)](https://www.biotechniques.com/crispr/introducing-the-next-generation-of-genetic-screening)
- [Harvard DRSC, "So you want to do a CRISPR pooled screen"](https://fgr.hms.harvard.edu/news/how-to-screen)
- [UT Health San Antonio TDC, screening service timelines](https://gccri.uthscsa.edu/services/tdc/service-and-pricing/)
- [VectorBuilder, pooled screen deconvolution pricing](https://en.vectorbuilder.com/products-services/service/library-construction/crispr-lib-service.html)
- [Illumina, Perturb-seq on NovaSeq X — sequencing depth](https://jp.illumina.com/content/dam/illumina/gcs/assembled-assets/marketing-literature/novaseq-x-perturb-seq-tech-note-m-gl-02890/novaseq-x-perturb-seq-tech-note-m-gl-02890.pdf)
- [Targeted Perturb-seq / TAP-seq (50× depth reduction)](https://pubmed.ncbi.nlm.nih.gov/32483332/)
- [Compressed Perturb-seq (4–20× cost reduction)](https://www.biorxiv.org/content/10.1101/2023.01.23.525200v1.full.pdf)
- [Pooled vs arrayed screening tradeoffs](https://synapse.patsnap.com/article/crispr-libraries-pooled-vs-arrayed-screening-pros-and-cons)
- [Sigma/Merck CRISPR & RNAi screening guide](https://b2b.sigmaaldrich.com/US/en/technical-documents/technical-article/genomics/functional-genomics-screening/crispr-rnai-screening-guide)
- [Wainberg et al., co-essential modules atlas (Nature Genetics 2021)](https://pubmed.ncbi.nlm.nih.gov/33859415/)
- [Replogle et al., genome-scale Perturb-seq (Cell 2022)](https://pesquisa.bvsalud.org/portal/resource/e/mdl-35688146)
- [Stoeger et al., understudied genes](https://pmc.ncbi.nlm.nih.gov/articles/PMC6759675)
- [Drug development timeline (target ID 1–2 yr)](https://intuitionlabs.ai/articles/drug-development-timeline-explained)

---

## 7. Side-by-side: today's decision tree vs ours (assuming every claim validates)

### Cost / time reference for every option

| Option | Money | Time | Output per gene | Hard constraint |
|---|---|---|---|---|
| Literature + homology | $0 | days | prose, if it exists | nothing to read for ~80% of genes |
| Public data mining (DepMap, co-expression, STRING) | $0 | days–weeks | correlational hypothesis | cancer lines only; not your cell type; correlational |
| Bulk pooled screen | cheapest | 12–20 wk genome-wide<br>8–16 wk sub-library | **one number** ("does it matter?") | one-dimensional readout |
| **Perturb-seq** | **10–25× bulk per gene**<br>$100k–$1M genome-scale<br>**scales ~linearly with perturbation count** | 12–20 wk<br>**flat in perturbation count** | **full transcriptome** → clustering, function assignment | **~100–800 cells per perturbation** |
| Arrayed follow-up | high upfront (automation) | +4–5 wk | per-well, imaging-capable | focused gene sets only |
| **Model prediction (ours)** | **$0 marginal** | **hours** | predicted transcriptome | **needs a screen in the same system to train on** |

Note the asymmetry in the Perturb-seq row: **cost scales with how many perturbations you include;
time does not.** That is what determines where a model can and cannot help — it can cut the bill
or the cell requirement, but not the calendar.

### TODAY

```
"What does gene X do?"  (X poorly characterised)
│
├─ 1. Literature / homology ................ $0 · days
│     └─ answer exists for well-studied genes → STOP
│        (for ~80% of genes there is nothing to read)
│
├─ 2. Public data mining ................... $0 · days–weeks
│     DepMap co-essentiality · co-expression · STRING · mouse KO
│     └─ correlational hypothesis — THE FREE DEFAULT, often where it ends
│        incumbent for function assignment: co-essentiality clustering
│        (Wainberg 2021 → 108 uncharacterised genes)
│
└─ 3. Need causal, context-specific evidence → run a screen
      │
      ├─ Bulk pooled ......... cheapest · 12–20 wk · ONE number per gene
      │
      ├─ Perturb-seq ......... 10–25×   · 12–20 wk · FULL profile per gene
      │     └─ cluster by effect → assign function
      │        ~100–800 cells/perturbation → cost AND cell count scale
      │        with the number of perturbations; time does not
      │
      └─ Arrayed follow-up ... high upfront · +4–5 wk · validate hits
```

### WITH THE MODEL (every claim validated)

Steps 1 and 2 are **unchanged** — still free, still first. Nothing we build displaces them.

```
└─ 3. Need causal, context-specific evidence
      │
      ├─ Run Perturb-seq on the subset you can afford / have cells for
      │        ........ SAME TIME as today, cost scales down with the subset
      │                    │
      │                    ▼
      │        ╔═══════════════════════════════════╗
      │        ║  MODEL EXTENDS IT   $0 · hours    ║
      │        ╚═══════════════════════════════════╝
      │                    │
      │        ┌───────────┼───────────────┐
      │        ▼           ▼               ▼
      │   genes you    gene PAIRS      other contexts
      │   left out     (~200M)         (cell types, doses)
      │        └───────────┴───────────────┘
      │                    ▼
      │    predicted responses clustered against measured ones
      │                    ▼
      │    ranked shortlist + functional hypotheses
      │                    ▼
      └─ Arrayed validation ... +4–5 wk, same as today
           but a better-chosen shortlist
```

### Worked example: functional data on 2,000 candidate genes

Perturb-seq cost is driven by **perturbations × cells-per-perturbation × reads-per-cell**, so it
scales roughly linearly with how many perturbations you include.

| | Perturbations screened | Cells | Sequencing ⚠️ | Time | Coverage |
|---|---|---|---|---|---|
| **Today** | 2,000 | ~600,000 | **$60k–180k** | 12–20 wk | 2,000 measured |
| **With model** | 200 | ~60,000 | **$6k–18k** | 12–20 wk | 200 measured + 1,800 predicted |
| **Difference** | — | **10× fewer** | **~90% less** | **none** | — |

Assumes ~300 cells per perturbation (Adamson used ~790; Replogle ~250) and ~22,000 reads per cell
(Illumina's recommendation: 20k transcriptome + 2k guide). ⚠️ The per-cell cost ($0.10–0.30) comes
from a weak secondary source — the *ratio* is solid, the absolute dollars need a real quote.

**The time column is the honest part: it does not move.** Growing cells and waiting on a sequencing
core is the critical path whether you run 200 perturbations or 2,000.

### Man-hours — a third axis, and it behaves differently again

Wall-clock is flat and cost is linear. **Labor is neither: it batches in steps**, because the
hands-on work is driven by the number of 10x *channels* you have to run, not by the cell count
directly.

The chain: cells → channels (~10,000–20,000 recovered per channel, GEM-X) → chip runs (8 channels
per chip) → library-prep batches (**8–12 hours of labor per batch**, per a core facility's costing;
a core typically preps up to 8 samples at a time).

| | Perturbations | Cells | Channels | Chip runs | **Bench labor** |
|---|---|---|---|---|---|
| **Today** | 2,000 | ~600,000 | 30–60 | 4–8 | **~32–96 h** |
| **With model** | 200 | ~60,000 | 3–6 | 1 | **~8–12 h** |
| **Difference** | — | 10× | 10× | 4–8× | **~4–8× less** |

⚠️ These are *composed* estimates — multiplying published component figures (cells/channel,
channels/chip, hours/batch), not a measured end-to-end study. The ratio is more trustworthy than
the absolute hours.

**Why the labor saving (4–8×) is smaller than the cost saving (10×):** batching. One chip run costs
the same labor whether you fill 3 channels or 8, so the small screen wastes capacity and pays a
fixed floor. Automation narrows the gap further — an automated library-prep workflow reports a
**75% hands-on reduction** across up to 48 reactions, which disproportionately helps the *large*
screen, since that is the one with enough batches to amortise the setup.

### Labor the model *adds*

The model side is not free, and two of these cut against us:

| Item | Scale |
|---|---|
| Building and validating the model | **One-time, large** — essentially this entire project |
| Running inference once trained | ~hours of compute, ~1 h human |
| Deciding whether to trust a prediction | Judgment, per use |
| **Interpreting 2,000 profiles instead of 200** | **Goes UP** — interpretation scales with what you look at, not with how you obtained it |

That last row matters: we do not reduce the analysis burden, we increase it. The lab now has 1,800
predicted profiles to make sense of. Whether that is a benefit or a burden depends entirely on
whether the predictions are trustworthy enough to act on — which is assumption 5 again.

### All three axes at once

| Axis | Today → with model | Why |
|---|---|---|
| **Wall-clock** | **No change** (12–20 wk) | Cell growth + core queue is the critical path regardless of scale |
| **Cost** | **~10× less** | Linear in perturbations: cells × reads |
| **Bench labor** | **~4–8× less** | Batches in steps of 8 channels; fixed floor per run |
| **Cells required** | **~10× fewer** | Linear; **the only axis no budget can substitute for** |
| **Analysis labor** | **Increases** | More profiles to interpret, plus trust judgments |


### What actually changes, ranked by how defensible it is

| Branch | Value | Honest strength |
|---|---|---|
| **Cell-limited systems** | A biopsy yielding 50,000 cells supports ~150 perturbations at 300 cells each. The rest are unreachable **at any budget**. Prediction is the only option. | **Strongest.** A cell-count limit cannot be bought away; a cost limit can |
| **Genes left out of a screen** | ~10× cost reduction for the same coverage (worked example above) | **Upgraded** — each skipped perturbation really does save ~300 cells of sequencing. But the cost axis is eroding anyway (compressed Perturb-seq: 4–20×) |
| **Gene pairs** | ~200M combinations; no screen reaches them. Train on singles, predict pairs. | **Strong in principle** — never tested by us (Adamson is single-gene only) |
| **New contexts** | Each cell type/timepoint is a fresh 12–20 wk screen | Strong in principle; known to be hard; untested |

### Assumptions this tree depends on

Every one is currently unproven:

1. Predictions carry enough functional signal to cluster usefully. **Partly tested** — real but
   weak: ~¼ of recoverable structure, and only after centring (`docs/clustering_finding.md`).
2. Predicted clustering beats free co-essentiality clustering. **Untested.**
3. Single-gene training transfers to pairs. **Untested.**
4. Training transfers across cell contexts. **Untested, known to be hard.**
5. A model trained on a *small* screen still works well enough to extend to a much larger gene
   set. **Untested, and the most load-bearing assumption here** — it underwrites both the
   cost-reduction and the cell-limited branches. Our models trained on 58 perturbations and gave
   only weak functional signal. Directly testable by retraining on fewer perturbations and
   re-running `scripts/clustering_structure.py`.

### The honest headline

> We do not remove a step and we do not make anyone faster — the screen still happens, and it still
> takes 12–20 weeks. What changes is that a fixed experimental budget covers roughly 10× more
> perturbations, with ~4–8× less bench labor, and that a cell-limited sample becomes workable at
> all. Analysis effort goes up, not down.

Three things to keep attached to that claim:

1. **Time is unchanged.** Anyone promising a speed-up from this is wrong.
2. **The cost axis is contested.** Compressed Perturb-seq (4–20×) and TAP-seq (up to 50× less
   sequencing) cut the same bill with no model. The cell-count argument is the durable one.
3. **It depends on training from a small screen.** Our models trained on 58 perturbations and
   produced only weak functional signal (`docs/clustering_finding.md`). Whether a 200-perturbation
   screen trains a model good enough to extend to 1,800 is the single most load-bearing untested
   assumption in this document.


### Sources added for the labor model

- [10x Chromium GEM-X technology (cells per channel, recovery)](https://www.10xgenomics.com/technology)
- [10x instrument comparison tech note](https://cdn.10xgenomics.com/image/upload/v1660261285/support-documents/CG000428_ChromiumInstrumentDataComparisonTN_RevC.pdf)
- [Core facility: ~8–12 h labor per library-prep batch, up to 8 samples at a time](https://biotech.illinois.edu/dna-services-core/dna-10x-single-cell-submission-requirements-and-other-information/)
- [Automated scRNA-seq library prep: 75% hands-on reduction, up to 48 reactions](https://pubmed.ncbi.nlm.nih.gov/35058211/)
