#!/usr/bin/env python3
"""Pick another cell in the 50µm ROI (not nloapcgp-1) that gets swept
into the over-merged epi blob under G=1, R=3 Stitch, and plot 3 panels:

    1. SEG (Xenium cell_id + Phase 1c) labels on this cell's tx
    2. NOSEG cascade pre-Stitch entities on this cell's tx
    3. Post-Stitch (G=1, R=3) final component label

Surrounding tx (other cells) are shown faded for context. Grid is drawn
at the per-panel binning (G=2 for SEG/NOSEG cascade origins, G=1 for the
new Stitch run). Marker shape distinguishes entities.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as _patheffects

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tracer.stitching import apply_stitching_to_transcripts_memory_efficient

PMI_THR = 0.2
SENT = {"-1", "DROP", "UNASSIGNED", "nan"}
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
PDAC = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
ZOOM = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg" / "zoom_worst_tx.parquet"
G_XY_DISPLAY = 1.0  # match the new Stitch's binning


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "p", "h", "<", ">", "H"]


def _draw_grid(ax, xmin, xmax, ymin, ymax, G, color="#dddddd", lw=0.3):
    x0 = np.floor(xmin / G) * G; x1 = np.ceil(xmax / G) * G
    y0 = np.floor(ymin / G) * G; y1 = np.ceil(ymax / G) * G
    for x in np.arange(x0, x1 + G * 0.5, G):
        ax.axvline(x, color=color, linewidth=lw, zorder=0)
    for y in np.arange(y0, y1 + G * 0.5, G):
        ax.axhline(y, color=color, linewidth=lw, zorder=0)


def _draw_labeled(ax, df, label_col, ident_map, title, xlim, ylim,
                   faded_df=None, G_grid=G_XY_DISPLAY):
    _draw_grid(ax, *xlim, *ylim, G=G_grid)
    if faded_df is not None and len(faded_df):
        ax.scatter(faded_df["x"], faded_df["y"], s=10, c="#dddddd",
                    alpha=0.45, linewidths=0, zorder=1,
                    label=f"surrounding tx ({len(faded_df)})")
    labels = sorted(df[label_col].astype(str).unique())
    for k, lab in enumerate(labels):
        if lab in SENT:
            sel = df[df[label_col].astype(str) == lab]
            ax.scatter(sel["x"], sel["y"], s=18, c="#999999", alpha=0.55,
                        linewidths=0, marker=".",
                        label=f"unassigned ({len(sel)})", zorder=2)
            continue
        c, m = ident_map[lab]
        sel = df[df[label_col].astype(str) == lab]
        short = lab[:14]
        ax.scatter(sel["x"], sel["y"], s=70, marker=m, c=[c], alpha=0.95,
                    edgecolor="black", linewidths=0.5, zorder=3,
                    label=f"{short} (n={len(sel)})")
        cx, cy = sel["x"].mean(), sel["y"].mean()
        ax.text(cx, cy, short, fontsize=6.5, ha="center", va="center",
                 color="black", zorder=5,
                 path_effects=[_patheffects.withStroke(linewidth=1.5,
                                                       foreground="white")])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    if len(labels) <= 12:
        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.88,
                   markerscale=0.7)


def main() -> int:
    zoom = pd.read_parquet(ZOOM)
    feats = pd.read_parquet(PDAC, columns=["transcript_id", "feature_name"])
    zcol = pd.read_parquet(PDAC, columns=["transcript_id", "z_location"]).rename(
        columns={"z_location": "z"})
    df = zoom.merge(feats, on="transcript_id", how="left").merge(
        zcol, on="transcript_id", how="left")
    df["feature_name"] = df["feature_name"].astype(str)
    df["noseg_lab"] = df["noseg_lab"].astype(str)
    df["seg_lab"] = df["seg_lab"].astype(str)
    df["cell_id"] = df["cell_id"].astype(str)
    df = df.reset_index(drop=True)

    # Run Stitch with G=1, R=3 over the full ROI
    df_in = df.copy()
    df_in["tracer_id"] = df_in["noseg_lab"]
    df_in.loc[df_in["tracer_id"].isin(SENT), "tracer_id"] = "-1"

    panel_raw = pd.read_parquet(PANEL)
    all_genes = sorted(set(panel_raw["gene_i"].astype(str))
                       | set(panel_raw["gene_j"].astype(str))
                       | set(df_in["feature_name"].unique()))
    g2i = {g: i for i, g in enumerate(all_genes)}
    G = len(all_genes)
    W = np.full((G, G), np.nan, dtype=np.float32)
    gi = panel_raw["gene_i"].astype(str).map(g2i)
    gj = panel_raw["gene_j"].astype(str).map(g2i)
    have = gi.notna() & gj.notna()
    gi = gi[have].to_numpy(np.int64); gj = gj[have].to_numpy(np.int64)
    v = panel_raw.loc[have, "value"].to_numpy(np.float32)
    W[gi, gj] = v; W[gj, gi] = v
    np.fill_diagonal(W, np.nan)

    df_s, _ = apply_stitching_to_transcripts_memory_efficient(
        df_final=df_in, aux={"W": W, "gene_to_idx": g2i},
        entity_col="tracer_id", gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        mode="count", threshold=PMI_THR, metric="pmi",
        penalize_simplicity=True, deltaC_min=0.03,
        c_union_bypass=0.9,
        dist_threshold=5.0, out_col="stitched", show_progress=False,
        candidate_source="grid", G=1.0, stitch_neighborhood="R3",
        G_z=1.0, z_neighbor_depth=1, min_local_tx_per_entity=3,
    )
    df_s["stitched"] = df_s["stitched"].astype(str)

    # Find a cell that got absorbed into the big over-merged epi blob,
    # i.e., a cell_id != nloapcgp-1 whose tx ended up in cascade_74606-1-2.
    big = "cascade_74606-1-2"
    in_big = df_s[df_s["stitched"] == big]
    by_cell = (
        in_big[in_big["cell_id"] != "nloapcgp-1"]
        .groupby("cell_id").size().reset_index(name="n_in_big")
    )
    by_cell_total = (
        df_s[df_s["cell_id"].isin(by_cell["cell_id"])]
        .groupby("cell_id").size().reset_index(name="n_total")
    )
    cands = by_cell.merge(by_cell_total, on="cell_id")
    cands["frac_in_big"] = cands["n_in_big"] / cands["n_total"]
    cands = cands[(cands["n_total"] >= 30) & (cands["frac_in_big"] >= 0.5)]
    if cands.empty:
        cands = by_cell.merge(by_cell_total, on="cell_id")
        cands["frac_in_big"] = cands["n_in_big"] / cands["n_total"]
    cands = cands.sort_values("n_in_big", ascending=False)
    print(f"Top candidates absorbed into {big}:")
    print(cands.head(10).to_string(index=False))
    target_cell = cands.iloc[0]["cell_id"]
    n_target = int(cands.iloc[0]["n_total"])
    n_swept = int(cands.iloc[0]["n_in_big"])
    print(f"\nPicked cell: {target_cell}  ({n_target} tx total, "
          f"{n_swept} swept into {big})", flush=True)

    sub = df_s[df_s["cell_id"] == target_cell].copy()
    pad = 3.0
    xmin, xmax = sub["x"].min() - pad, sub["x"].max() + pad
    ymin, ymax = sub["y"].min() - pad, sub["y"].max() + pad

    # Faded surrounding tx (NOT this cell, but in the same crop)
    surround = df_s[(df_s["cell_id"] != target_cell)
                    & (df_s["x"].between(xmin, xmax))
                    & (df_s["y"].between(ymin, ymax))].copy()

    # Build stable (color, marker) maps per column
    def make_ident(labels):
        labels = [l for l in labels if l not in SENT]
        cmap = plt.get_cmap("tab20", max(len(labels), 1))
        return {lab: (cmap(i % cmap.N), MARKERS[i % len(MARKERS)])
                for i, lab in enumerate(labels)}

    seg_labels = sorted(sub["seg_lab"].astype(str).unique())
    noseg_labels = sorted(sub["noseg_lab"].astype(str).unique())
    stitched_labels = sorted(sub["stitched"].astype(str).unique())
    seg_ident = make_ident(seg_labels)
    noseg_ident = make_ident(noseg_labels)
    stitch_ident = make_ident(stitched_labels)

    print(f"\nSEG entities for {target_cell}: {[l for l in seg_labels if l not in SENT]}")
    print(f"NOSEG cascade entities for {target_cell}: "
          f"{[l for l in noseg_labels if l not in SENT]}")
    print(f"Stitched (G=1,R=3) components: "
          f"{[l for l in stitched_labels if l not in SENT]}")

    fig, axes = plt.subplots(1, 3, figsize=(22, 8), dpi=140)
    _draw_labeled(axes[0], sub, "seg_lab", seg_ident,
                   f"SEG entities (Xenium + Phase 1c)\n{target_cell}: "
                   f"{n_target} tx → {len(seg_labels)-len([l for l in seg_labels if l in SENT])} entities",
                   (xmin, xmax), (ymin, ymax), faded_df=surround)
    _draw_labeled(axes[1], sub, "noseg_lab", noseg_ident,
                   f"NOSEG cascade (pre-Stitch)\n"
                   f"{len([l for l in noseg_labels if l not in SENT])} entities",
                   (xmin, xmax), (ymin, ymax), faded_df=surround)
    _draw_labeled(axes[2], sub, "stitched", stitch_ident,
                   f"Stitch (G=1, R=3) post-merge\n"
                   f"{len([l for l in stitched_labels if l not in SENT])} components; "
                   f"{n_swept}/{n_target} tx swept into {big[:20]}",
                   (xmin, xmax), (ymin, ymax), faded_df=surround)
    plt.suptitle(
        f"Cell {target_cell} — SEG vs NOSEG vs Stitch (G=1, R=3)\n"
        f"50µm ROI from PDAC; grid lines at G=1µm",
        fontsize=13, y=1.0,
    )
    plt.tight_layout()
    out = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg" / f"zoom_other_cell_{target_cell}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
