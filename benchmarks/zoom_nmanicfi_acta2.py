#!/usr/bin/env python3
"""Plot the two SEG components of cell `nmanicfi-1` per-stage:

    ■  filled square  → Phase 1a → -1  (nuclear, gene in main seed)
    □  hollow square  → Phase 1b → -1  (nuclear, gene NOT in main seed)
    ●  filled circle  → Phase 1c → -1-1 (nuclear, gene in partial seed)
    ○  hollow circle  → Phase 1c-reassign → -1-1 (nuclear, gene NOT in
                                                    partial seed)
    ★  star           → cytoplasmic ACTA2

Stage classification is recomputed from first principles using the
production seed-greedy prune (`prune_genes_by_npmi_greedy`) at
threshold=PMI_THR=0.2 — the same threshold the pipeline used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tracer.pruning import prune_genes_by_npmi_greedy

ZOOM_DIR = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg"
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
PDAC = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
PMI_THR = 0.2
G_XY = 2.0
SENT = {"-1", "DROP", "UNASSIGNED", "nan"}


def _draw_grid(ax, xmin, xmax, ymin, ymax, G=G_XY, color="#cccccc", lw=0.5):
    x0 = np.floor(xmin / G) * G; x1 = np.ceil(xmax / G) * G
    y0 = np.floor(ymin / G) * G; y1 = np.ceil(ymax / G) * G
    for x in np.arange(x0, x1 + G * 0.5, G):
        ax.axvline(x, color=color, linewidth=lw, zorder=0)
    for y in np.arange(y0, y1 + G * 0.5, G):
        ax.axhline(y, color=color, linewidth=lw, zorder=0)


def _build_W(panel_path, all_genes):
    panel = pd.read_parquet(panel_path).rename(columns={"value": "NPMI"})
    panel["gene_i"] = panel["gene_i"].astype(str)
    panel["gene_j"] = panel["gene_j"].astype(str)
    g2i = {g: i for i, g in enumerate(all_genes)}
    G_ = len(all_genes)
    W = np.full((G_, G_), np.nan, dtype=np.float32)
    gi = panel["gene_i"].map(g2i); gj = panel["gene_j"].map(g2i)
    have = gi.notna() & gj.notna()
    gi = gi[have].to_numpy(np.int64); gj = gj[have].to_numpy(np.int64)
    v = panel.loc[have, "NPMI"].to_numpy(np.float32)
    W[gi, gj] = v; W[gj, gi] = v
    np.fill_diagonal(W, np.nan)
    return W, g2i


def _mean_pmi_to_seed(gene_idx, seed_indices, W):
    """Mean PMI of `gene_idx` vs `seed_indices`, self-excluded, NaN-skipped."""
    if gene_idx in seed_indices:
        rest = [g for g in seed_indices if g != gene_idx]
    else:
        rest = list(seed_indices)
    if not rest:
        return float("nan")
    vals = W[gene_idx, rest]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def main() -> int:
    zoom = pd.read_parquet(ZOOM_DIR / "zoom_worst_tx.parquet")
    feats = pd.read_parquet(PDAC, columns=["transcript_id", "feature_name"])
    nuc = pd.read_parquet(PDAC,
                          columns=["transcript_id", "overlaps_nucleus"])
    df = (
        zoom.merge(feats, on="transcript_id", how="left")
            .merge(nuc, on="transcript_id", how="left")
    )
    df["feature_name"] = df["feature_name"].astype(str)
    df["seg_lab"] = df["seg_lab"].astype(str)
    df["cell_id"] = df["cell_id"].astype(str)
    df["overlaps_nucleus"] = df["overlaps_nucleus"].astype(bool)

    cell = df[df["cell_id"] == "nmanicfi-1"].copy().reset_index(drop=True)
    main = cell[cell["seg_lab"] == "nmanicfi-1"].copy()
    part = cell[cell["seg_lab"] == "nmanicfi-1-1"].copy()
    unas = cell[cell["seg_lab"].isin(SENT)].copy()

    # Build W
    panel_raw = pd.read_parquet(PANEL)
    all_genes = sorted(set(panel_raw["gene_i"].astype(str))
                       | set(panel_raw["gene_j"].astype(str))
                       | set(df["feature_name"].unique()))
    W, g2i = _build_W(PANEL, all_genes)

    # ------------------------------------------------------------------
    # Phase 1a seed: greedy prune over MAIN-entity nuclear gene set
    # ------------------------------------------------------------------
    main_nuc_genes = sorted(main.loc[main["overlaps_nucleus"], "feature_name"].unique())
    main_nuc_idx = np.array([g2i[g] for g in main_nuc_genes if g in g2i],
                             dtype=np.int64)
    keep_mask_main = prune_genes_by_npmi_greedy(main_nuc_idx, W, PMI_THR)
    seed_main_idx = set(main_nuc_idx[keep_mask_main].tolist())
    seed_main_genes = {all_genes[i] for i in seed_main_idx}
    print(f"\nPhase 1a — main (-1) seed: {len(seed_main_genes)} / "
          f"{len(main_nuc_genes)} nuclear genes survived greedy prune at "
          f"threshold={PMI_THR}", flush=True)
    print(f"  seed genes: {sorted(seed_main_genes)}", flush=True)

    # ------------------------------------------------------------------
    # Phase 1c seed: greedy prune over rest-pile nuclear gene set.
    # Rest-pile = nuclear genes whose mean PMI vs main seed < PMI_THR.
    # ------------------------------------------------------------------
    part_nuc_genes = sorted(part.loc[part["overlaps_nucleus"], "feature_name"].unique())
    part_nuc_idx = np.array([g2i[g] for g in part_nuc_genes if g in g2i],
                             dtype=np.int64)
    # Just use the partial's nuclear-entity gene set as the rest-pile proxy.
    # These are the genes Phase 1c saw and ran its sub-seed greedy on.
    keep_mask_part = prune_genes_by_npmi_greedy(part_nuc_idx, W, PMI_THR)
    seed_part_idx = set(part_nuc_idx[keep_mask_part].tolist())
    seed_part_genes = {all_genes[i] for i in seed_part_idx}
    print(f"\nPhase 1c — partial (-1-1) sub-seed: "
          f"{len(seed_part_genes)} / {len(part_nuc_genes)} nuclear genes "
          f"survived sub-seed greedy at threshold={PMI_THR}", flush=True)
    print(f"  seed genes: {sorted(seed_part_genes)}", flush=True)

    # ------------------------------------------------------------------
    # Per-tx stage classification
    #   Phase 1a → -1:   nuclear, gene IS in main seed (anchored to seed)
    #   Phase 1b → -1:   nuclear, gene NOT in main seed, but mean PMI vs
    #                    main seed ≥ PMI_THR (admitted via Phase 1b test)
    #   Phase 1c → -1-1: nuclear, gene's main-seed mean PMI < PMI_THR
    #                    (was in rest-pile → admitted to sub-partial)
    #   Phase 1c-reassign → -1-1: nuclear, gene's main-seed mean PMI ≥
    #                    PMI_THR (would have been admitted to main →
    #                    moved by reassign because partial fit better)
    # ------------------------------------------------------------------
    def stage_for_tx(row):
        gene = row["feature_name"]
        is_nuc = bool(row["overlaps_nucleus"])
        ent = row["seg_lab"]
        if gene == "ACTA2" and not is_nuc:
            return "cyto_ACTA2"
        if not is_nuc:
            return "cyto_other"
        # Need mean PMI vs main seed to disambiguate
        if gene not in g2i:
            return "other"
        gidx = g2i[gene]
        m_main = _mean_pmi_to_seed(gidx, seed_main_idx, W)
        admitted_by_main = (m_main is not None
                            and not np.isnan(m_main)
                            and m_main >= PMI_THR)
        if ent == "nmanicfi-1":
            return "phase1a_main" if gene in seed_main_genes else "phase1b_main"
        if ent == "nmanicfi-1-1":
            return "reassign_part" if admitted_by_main else "phase1c_part"
        return "other"

    cell["stage"] = cell.apply(stage_for_tx, axis=1)
    print(f"\nStage breakdown for nmanicfi-1 cell footprint ({len(cell)} tx):")
    print(cell["stage"].value_counts().to_string())

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    pad = 2.0
    xmin = np.floor((cell["x"].min() - pad) / G_XY) * G_XY
    xmax = np.ceil((cell["x"].max() + pad) / G_XY) * G_XY
    ymin = np.floor((cell["y"].min() - pad) / G_XY) * G_XY
    ymax = np.ceil((cell["y"].max() + pad) / G_XY) * G_XY

    fig, ax = plt.subplots(figsize=(13, 12), dpi=140)
    _draw_grid(ax, xmin, xmax, ymin, ymax)

    MAIN_C = "#1f77b4"; PART_C = "#d62728"; ACTA_C = "#ffa500"

    # CAF gene panel: union of the nmanicfi-1-1 sub-seed genes plus the
    # canonical PDAC CAF panel. Anything else is "non-CAF".
    CAF_GENES = set(seed_part_genes) | {
        "COL1A1", "COL1A2", "COL3A1", "COL6A1", "COL6A2", "COL6A3",
        "DCN", "MMP2", "MMP9", "MMP11", "POSTN", "VCAN", "THY1",
        "PDGFRA", "PDGFRB", "TGFB1", "THBS2", "S100A4", "FAP", "VIM",
        "TIMP1", "BGN", "MFAP4", "SERPINH1",
    }

    # Classic housekeeping (broad-PMI-positive ubiquitous markers).
    # Only the 3 actually in the panel matter here.
    HK_GENES = {"ACTB", "TUBB", "SDC1"}

    # Region-wide tx: everything in the plot window, EXCLUDING tx already
    # plotted with the per-stage symbols above (Phase 1a/1c/reassign +
    # cyto ACTA2). For surrounding cells these are mostly cyto admits.
    region = df[
        df["x"].between(xmin, xmax) & df["y"].between(ymin, ymax)
    ].copy()
    # Drop the tx that are part of the nmanicfi-1 per-stage plot (i.e.,
    # nuclear tx of -1 or -1-1, plus cyto ACTA2 in nmanicfi-1 footprint).
    already_plotted_ids = set(cell.loc[
        cell["stage"].isin([
            "phase1a_main", "phase1b_main", "phase1c_part",
            "reassign_part", "cyto_ACTA2",
        ]), "transcript_id"
    ])
    region = region[~region["transcript_id"].isin(already_plotted_ids)]
    region["is_caf"] = region["feature_name"].isin(CAF_GENES)
    region["is_hk"] = region["feature_name"].isin(HK_GENES)
    region["host"] = region["seg_lab"].where(~region["seg_lab"].isin(SENT),
                                              other="unassigned")

    # Color by host: main/partial/unassigned/other-entity
    def _host_color(seg):
        if seg == "nmanicfi-1": return MAIN_C
        if seg == "nmanicfi-1-1": return PART_C
        if seg == "unassigned": return "#888888"
        return "#7f7f7f"   # other-entity gray
    region["color"] = region["host"].map(_host_color)

    # Housekeeping tx (ACTB / TUBB / SDC1) as dots — small filled points
    hk = region[region["is_hk"]]
    if len(hk):
        ax.scatter(hk["x"], hk["y"], s=40, c=hk["color"].to_list(),
                    marker=".", alpha=0.85, linewidths=0, zorder=3,
                    label=f"· housekeeping (ACTB/TUBB/SDC1) ({len(hk)})")

    # Non-CAF, non-housekeeping tx as x
    non_caf = region[~region["is_caf"] & ~region["is_hk"]]
    if len(non_caf):
        ax.scatter(non_caf["x"], non_caf["y"], s=70, c=non_caf["color"].to_list(),
                    marker="x", alpha=0.75, linewidths=1.2, zorder=3,
                    label=f"× non-CAF, non-HK tx in region ({len(non_caf)})")

    # CAF tx as triangle
    caf = region[region["is_caf"]]
    if len(caf):
        ax.scatter(caf["x"], caf["y"], s=110, c=caf["color"].to_list(),
                    marker="^", alpha=0.92, edgecolor="black", linewidths=0.6,
                    zorder=4,
                    label=f"▲ CAF tx in region ({len(caf)})")

    # 1. Phase 1a → main: filled squares
    g1a = cell[cell["stage"] == "phase1a_main"]
    ax.scatter(g1a["x"], g1a["y"], s=180, c=MAIN_C, marker="s",
                alpha=0.92, edgecolor="black", linewidths=0.7, zorder=4,
                label=f"■ Phase 1a → -1 main seed ({len(g1a)})")

    # 2. Phase 1c → partial: filled circles
    g1c = cell[cell["stage"] == "phase1c_part"]
    ax.scatter(g1c["x"], g1c["y"], s=180, c=PART_C, marker="o",
                alpha=0.92, edgecolor="black", linewidths=0.7, zorder=4,
                label=f"● Phase 1c → -1-1 sub-seed ({len(g1c)})")

    # 3. Phase 1b → main: hollow squares
    g1b = cell[cell["stage"] == "phase1b_main"]
    ax.scatter(g1b["x"], g1b["y"], s=180, facecolors="none",
                edgecolor=MAIN_C, marker="s", linewidths=1.8, zorder=4,
                label=f"□ Phase 1b → -1 admit ({len(g1b)})")

    # 4. Phase 1c-reassign → partial: hollow circles
    gra = cell[cell["stage"] == "reassign_part"]
    ax.scatter(gra["x"], gra["y"], s=180, facecolors="none",
                edgecolor=PART_C, marker="o", linewidths=1.8, zorder=4,
                label=f"○ Phase 1c-reassign → -1-1 ({len(gra)})")

    # 6. Cyto ACTA2: stars colored by current host entity
    gca = cell[cell["stage"] == "cyto_ACTA2"].copy()
    if len(gca):
        host = gca["seg_lab"]
        colors = np.where(host == "nmanicfi-1", MAIN_C,
                  np.where(host == "nmanicfi-1-1", PART_C, "#888888"))
        ax.scatter(gca["x"], gca["y"], s=560, c=colors,
                    marker="*", alpha=1.0, edgecolor="black",
                    linewidths=1.2, zorder=6,
                    label=f"★ cyto ACTA2 by host ({len(gca)})")

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title(
        f"nmanicfi-1 cell — per-stage Phase 1 admit trace\n"
        f"main seed: {len(g1a)} tx ({len(seed_main_genes)} genes)  ·  "
        f"main 1b admit: {len(g1b)} tx  ·  "
        f"partial sub-seed: {len(g1c)} tx ({len(seed_part_genes)} genes)  ·  "
        f"partial reassign: {len(gra)} tx  ·  "
        f"cyto ACTA2: {len(gca)}\n"
        f"G=2 µm grid",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    plt.tight_layout()
    out = ZOOM_DIR / "zoom_nmanicfi_acta2.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
