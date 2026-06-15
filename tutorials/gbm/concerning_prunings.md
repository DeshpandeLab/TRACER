# Concerning TRACER prunings — Slide 3

Cases where TRACER depleted a cell type's *own* canonical markers, after cross-
referencing the per-cell-type DE table
(`tutorials/gbm/output/slide3_profile_comparison/celltype_de.csv`) against the
curated marker list in `cell_type_markers.md`.

Significance filter applied: `FDR < 0.05` & `|log2FC| > 0.5` on Mann-Whitney U
of raw counts, BH-adjusted within cell type. `log2FC = log2((ft_mean + 0.1) /
(orig_mean + 0.1))` — negative = transcripts removed by TRACER.

The typical "depleted top 10" for any cell type is dominated by *contaminating*
markers (e.g. AQP4/GJA1/PTPRZ1 in T cells, CHI3L1 in microglia) — that is the
intended behavior. The flags below are cases where a cell's own tier-2/tier-3
markers are heavily depleted.

## Severity legend
- 🔴 **critical** — a marker that *defines* the cell type/state is being removed
- 🟠 **high** — a strong but not unique marker is heavily depleted
- 🟡 **watch** — a useful but promiscuous marker is depleted; could be contamination

## Flagged depletions

| Cell type | Gene | log2FC | orig_mean → ft_mean | Severity | Why it matters |
|---|---:|---:|---|:---:|---|
| **neuronal_IN** | GAD2 | −5.4 | 31.3 → 0.66 (47×) | 🔴 critical | Canonical GABA-synthesis enzyme; defines GABAergic neurons |
| **neuronal_IN** | SST | −4.8 | 17.2 → 0.53 (32×) | 🔴 critical | Defines an entire interneuron class (Tasic 2018) |
| **neuronal_IN** | TAC1 | −7.0 | 21.7 → 0.07 (293×) | 🔴 critical | Subtype-defining (interneuron / striatal projection) |
| **neuronal_IN** | CRHBP | −6.6 | 9.6 → 0.0006 | 🔴 critical | VIP/CCK interneuron subtype marker |
| **neuronal_IN** | RELN | −5.0 | 4.5 → 0.05 | 🟠 high | Cortical interneuron / Cajal-Retzius marker |
| **neuronal_IN** | CCK | −4.9 | 3.7 → 0.025 | 🟠 high | CCK+ interneuron subtype |
| **neuronal_IN** | BCL11B | −5.2 | 3.6 → 0.0003 | 🟠 high | Pan-neuronal TF |
| **neuronal_IN** | CNTN2 | −4.6 | 2.3 → 0.0001 | 🟠 high | Neuronal adhesion |
| **cancer_OPC** | OLIG1 | −5.6 | 12.95 → 0.165 (80×) | 🔴 critical | Defines the Neftel OPC-like state |
| **cancer_OPC** | OLIG2 | −4.3 | 10.09 → 0.402 (25×) | 🔴 critical | Defines the Neftel OPC-like state |
| **cancer_MES2** | NDRG1 | −8.3 | 35.4 → 0.011 (3000×) | 🔴 critical | The defining MES2 hypoxia gene |
| **cancer_MES2** | HMOX1 | −4.3 | 1.96 → 0.002 | 🟠 high | Hypoxia / oxidative stress (MES2 core) |
| **cancer_MES2** | VEGFA | (in n_down list) | — | 🟠 high | Hypoxia response (MES2 core) |
| **cancer_MES2** | GPNMB | −4.6 | 2.27 → 0.001 | 🟠 high | MES program component |
| **vascular_endothelial** | CD93 | −5.7 | 22.9 → 0.351 (65×) | 🔴 critical | Canonical capillary EC marker |
| **vascular_endothelial** | STAB1 | −5.1 | 8.97 → 0.170 (53×) | 🟠 high | Border-associated / sinusoidal EC |
| **vascular_endothelial** | APP | −4.4 | 22.2 → 0.93 (24×) | 🟡 watch | Promiscuous, but a major EC transcript |
| **neutrophil** | FCGR3B | −2.8 | 1.02 → 0.058 (18×) | 🔴 critical | The neutrophil-specific Fc receptor (CD16b) |
| **neutrophil** | SPI1 | −2.9 | 0.63 → 0 | 🟠 high | PU.1 — myeloid TF, retained in neutrophils |
| **neutrophil** | FCGR1A | −2.9 | 0.64 → 0 | 🟡 watch | More monocyte/macrophage, but also neutrophil |
| **myeloid_microglia** | BIN1 | −4.7 | 8.12 → 0.217 (37×) | 🟠 high | Microglia / Alzheimer's GWAS gene |
| **neuron_EN** | BIN1 | −6.9 | 17.4 → 0.04 (440×) | 🟠 high | Strong neuronal expression too (dual-marker gene) |
| **neuron_EN** | HS3ST2 | −6.5 | 10.7 → 0.019 (540×) | 🟠 high | Neuronal sulfotransferase |
| **neuron_EN** | TESPA1 | −5.8 | 5.5 → 0.003 | 🟠 high | Neuronal Ca²⁺ signaling |
| **neuron_EN** | EGR3 | −5.4 | 4.4 → 0.006 | 🟠 high | Activity-regulated neuronal IEG |
| **oligo** | APOE | −3.6 | 5.07 → 0.34 (15×) | 🟡 watch | Mostly astro/microglia, but mature oligo expression exists |
| **myeloid_macrophages** | CYP27A1 | −5.7 | 5.19 → 0 | 🟡 watch | TAM / border-mac marker (Sanin 2022) |

## Pattern

Worst overpruning concentrates on cells whose identity rests on a **small set of
high-abundance, spatially clustered transcripts**: interneurons, OPC-like cancer,
MES2 cancer, endothelial cells, neutrophils. When TRACER's NPMI flags those
clustered transcripts as suspect — because the cluster is spatially focal and
the genes do not co-occur globally with much else — it can strip the very
transcripts that define the cell.

The cleaner cases (T cells, macrophages, microglia overall) are dominated by
*contaminating* glial/cancer transcripts being correctly removed — that's
TRACER doing its job.

## Suggested follow-up

1. **Per-flagged-cell-type purity delta** — if `cell_purity` does not increase
   for cells where canonical markers were stripped, the trim is destroying
   signal rather than removing contamination. Strongest data-driven argument
   for/against overpruning.
2. **Stratified NPMI** — compute NPMI separately per coarse pre-call (oligo,
   neuronal, immune, vascular, cancer) so that focal/rare states are not
   penalized for being globally rare.
3. **Filter codewords from NPMI** — `Deprecated*`, `NegControl*`,
   `UnassignedCodeword*` (~28% of panel features, 0.32% of transcripts) are
   currently included in NPMI; filtering them is a cheap win.
4. **Adaptive prune threshold by cell transcript count** — require stronger
   evidence to prune from low-count cells, since each removed transcript is a
   larger fraction of that cell's identity.
