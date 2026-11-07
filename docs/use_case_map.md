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

### ❌ "Screens are too expensive, so predict instead"
**The strongest counter-argument to our own pitch, and it comes from how pooling works.**

Pooled screens scale *sub-linearly*. The library is one tube; the cells grow in one flask. Adding
500 more genes to a screen you're already running costs more guides, more cells and more
sequencing — but not 500× more work. The marginal cost of one more gene is small.

So "predict the genes you didn't screen" is weak whenever the researcher could simply have
included them. And Replogle 2022 already screened **every expressed gene** — the brute-force
option exists.

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
| **Perturb-seq** | **10–25× bulk per gene**<br>$100k–$1M genome-scale | 12–20 wk | **full transcriptome** → clustering, function assignment | **~500 cells per perturbation** |
| Arrayed follow-up | high upfront (automation) | +4–5 wk | per-well, imaging-capable | focused gene sets only |
| **Model prediction (ours)** | **$0 marginal** | **hours** | predicted transcriptome | **needs a screen in the same system to train on** |

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
      │        ~500 cells/perturbation, so cell count caps the gene count
      │
      └─ Arrayed follow-up ... high upfront · +4–5 wk · validate hits
```

### WITH THE MODEL (every claim validated)

Steps 1 and 2 are **unchanged** — still free, still first. Nothing we build displaces them.

```
└─ 3. Need causal, context-specific evidence
      │
      ├─ Run Perturb-seq on the subset you can afford / have cells for
      │        ........................ same cost and time as today
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

### What actually changes, ranked by how defensible it is

| Branch | Value | Honest strength |
|---|---|---|
| **Cell-limited systems** | A biopsy yielding 50,000 cells supports ~100 perturbations at 500 cells each. The other 19,900 genes are unreachable **at any budget**. Prediction is the only option. | **Strongest.** A cell-count limit cannot be bought away, unlike a cost limit |
| **Gene pairs** | ~200M combinations; no pooling strategy reaches them. Train on singles, predict pairs. | **Strong** — but we have never tested it (Adamson is single-gene only) |
| **New contexts** | Each cell type/timepoint is a fresh 12–20 wk screen | Strong in principle; known to be hard; untested by us |
| **Genes left out of a screen** | Predict what you didn't include | **Weakest.** Pooled screens scale sub-linearly — you could have added them for little extra |

### Assumptions this tree depends on

Every one is currently unproven:

1. Predictions carry enough functional signal to cluster usefully. **Partly tested** — real but
   weak: ~¼ of recoverable structure, and only after centring (`docs/clustering_finding.md`).
2. Predicted clustering beats free co-essentiality clustering. **Untested.**
3. Single-gene training transfers to pairs. **Untested.**
4. Training transfers across cell contexts. **Untested, known to be hard.**
5. A model trained on a small screen (the cell-limited case) still works. **Untested** — our
   models trained on 58 perturbations; a biopsy-scale screen might give far fewer.

### The honest headline

> We do not remove a step. The screen still happens. What changes is how much you get out of the
> one screen you could afford — and whether you can reach perturbations that no screen can.

