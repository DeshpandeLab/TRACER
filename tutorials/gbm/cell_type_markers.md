# GBM Xenium panel — canonical cell-type markers

Reference markers for cells expected in the GBM TME, restricted to genes present
in the Slide 3 Xenium panel (541 features incl. controls; ~390 real genes).

Tier legend:
- ★★★ canonical & near-exclusive for the cell type
- ★★ strong but shared with related lineages
- ★ useful but promiscuous

References: Tasic 2018 / Allen Brain (interneurons); Neftel 2019 (GBM cell states);
Couturier 2020, Greenwald 2024 (GBM TME); Bergles & Richardson 2016 (OPC);
Liddelow & Barres 2017 (reactive astro); Polański 2024 (myeloid); Mathys 2019,
Keren-Shaul 2017 (microglia); Sanin 2022 (TAM).

| Cell type | Tier-3 (★★★) | Tier-2 (★★) | Notes |
|---|---|---|---|
| **Astrocyte / cancer_AC** | AQP4, GJA1, FGFR3 | SOX9, ID4, APOE | GFAP not on panel |
| **Oligodendrocyte** | MOG, MOBP, MAG, OPALIN, MAL, CLDN11, MYRF, ERMN, KLK6 | CNDP1, UGT8, CNTN2, ST18, HHATL | MBP not on panel |
| **OPC / cancer_OPC** | PDGFRA, CSPG4, OLIG1, OLIG2, SOX10 | BCAN, PTPRZ1, CSPG5 | OLIG1/2 also define Neftel OPC-like |
| **Excitatory neuron (neuron_EN)** | SLC17A7, SLC17A6, RORB, CUX2, BCL11B | CNTN2, NRN1, HS3ST2, TESPA1 | TBR1 not on panel |
| **Inhibitory neuron (neuronal_IN)** | GAD1, GAD2, PVALB, SST, VIP, LAMP5, LHX6 | RELN, CCK, TAC1, CRHBP, NPY1R, SOX6 | Subtype markers cover MGE + CGE classes |
| **NPC / cancer_NPC** | DCX, SOX11, NES, SOX2, SOX4, HES6 | BCAN, BIN1*, CDK1, MKI67, TOP2A | BIN1 dual-marker (also microglia, neurons); proliferation overlap |
| **MES1 cancer** | CHI3L1, CD44, ANXA1 | TGFBI, S100A4, NDRG1, IFITM3 | CHI3L1 (YKL-40) also reactive astro |
| **MES2 cancer (hypoxia)** | NDRG1, HILPDA, HMOX1, VEGFA | IGFBP3, PLAUR, GPNMB | Defined by hypoxic stress program |
| **Microglia** | P2RY12, P2RY13, TMEM119, CX3CR1, GPR34 | TREM2, AIF1, BIN1, C3 | P2RY12/TMEM119 most specific |
| **Macrophage / TAM** | CD163, MRC1, LYVE1, STAB1 | CD68, CD14, MS4A6A | LYVE1/STAB1 mark perivascular/border-associated |
| **Monocyte** | FCN1, VCAN, S100A4, LYZ | CD14, FCGR3A, MGST1 | FCGR3A = CD16 (non-classical) |
| **cDC** | CD1A, FCER1A, ITGAX, CD86 | HLA-DMA, HLA-DMB, HLA-DQA1, HLA-DRB5 | CLEC9A not on panel |
| **Neutrophil** | FCGR3B, MGAM, S100P | CXCL3, IL1B, FCGR1A | CSF3R not on panel |
| **T cell (pan)** | CD2, CD3G, CD4, IL7R | BCL11B, ICOS, KLRB1 | CD3D/CD3E not on panel |
| **CD8 / cytotoxic** | GZMA, GZMB, NKG7, PRF1, GNLY, CTSW | KLRD1, IL2RB, TIGIT, CTLA4 | Treg markers (FOXP3) absent from panel |
| **Vascular endothelial** | PECAM1, FLT1, CD93, ESAM, CALCRL | NRP1, APLNR, ANGPT1 | CDH5/VWF not on panel |
| **Pericyte / mural** | ABCC9 | NR2F2, ANO3 | KCNJ8/RGS5/PDGFRB not on panel |
| **Proliferation (pan)** | MKI67, TOP2A, CDK1, CCNB2, CENPF | PCNA, GAS2L3, CCNA1 | Cell-cycle signature, not lineage-specific |

## Caveats

- **GBM cancer cells overlap normal lineage markers by design** (Neftel 2019). AC-like
  cancer cells express AQP4/GJA1; OPC-like cancer cells express OLIG1/2/PDGFRA;
  MES-like express astrocyte/microglia-leaning genes. Treat the cancer_* labels as
  *cell states*, not as fundamentally distinct cell types.
- **BIN1** is non-specific: highly expressed in microglia, but also in neurons and
  endothelial cells.
- **Several canonical markers are missing from the panel**: GFAP (astro), MBP (oligo),
  CD3D/CD3E (T), VWF/CDH5 (EC), KCNJ8/RGS5/PDGFRB (pericyte), FOXP3 (Treg). When
  evaluating cell identity in this panel, leaning on the in-panel substitutes
  listed above.
