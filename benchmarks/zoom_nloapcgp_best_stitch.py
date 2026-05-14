#!/usr/bin/env python3
"""nloapcgp-1: SEG vs the BEST Stitch achieved on this branch.

"Best" = full-ROI re-stitch with:
    G=2 µm, stitch_neighborhood="8" (R=1)
    deltaC_min = -0.01   (loosened from production 0.03)
    c_union_bypass = 0.9 (newly added)
    capped min_local_tx_per_entity = 3 (newly capped at n_tx)
    penalize_simplicity = True

Tile-and-outline visualization at G=2 µm and G=1 µm. Compares SEG (2
entities) vs the best-stitched output for the same cell.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib import patheffects as _patheffects

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tracer.stitching import apply_stitching_to_transcripts_memory_efficient

ZOOM_DIR = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg"
PDAC = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
ZOOM = ZOOM_DIR / "zoom_worst_tx.parquet"
SENT = {"-1", "DROP", "UNASSIGNED", "nan"}
TARGET_CELL = "nloapcgp-1"


def _draw_panel(ax, df, label_col, title, xlim, ylim, G,
                 alpha_fill=0.18, dot_size=22, dot_alpha=0.85,
                 label_entities=True):
    labels = sorted(df[label_col].astype(str).unique())
    real_labels = [l for l in labels if l not in SENT]
    n = max(len(real_labels), 1)
    cmap = plt.get_cmap("tab20" if n <= 20 else "gist_ncar", n)
    color_for = {l: cmap(i % cmap.N) for i, l in enumerate(real_labels)}

    x0 = np.floor(xlim[0] / G) * G; x1 = np.ceil(xlim[1] / G) * G
    y0 = np.floor(ylim[0] / G) * G; y1 = np.ceil(ylim[1] / G) * G
    for x in np.arange(x0, x1 + G * 0.5, G):
        ax.axvline(x, color="#eeeeee", linewidth=0.4, zorder=0)
    for y in np.arange(y0, y1 + G * 0.5, G):
        ax.axhline(y, color="#eeeeee", linewidth=0.4, zorder=0)

    df_a = df[~df[label_col].astype(str).isin(SENT)].copy()
    df_a["xb"] = np.floor(df_a["x"].to_numpy() / G).astype(int)
    df_a["yb"] = np.floor(df_a["y"].to_numpy() / G).astype(int)
    for lab in real_labels:
        ent = df_a[df_a[label_col].astype(str) == lab]
        if len(ent) == 0:
            continue
        bins = set(zip(ent["xb"].tolist(), ent["yb"].tolist()))
        c = color_for[lab]
        for (xb, yb) in bins:
            ax.add_patch(Rectangle(
                (xb * G, yb * G), G, G,
                facecolor=c, alpha=alpha_fill, edgecolor="none", zorder=1,
            ))
        segs = []
        for (xb, yb) in bins:
            x0_, y0_ = xb * G, yb * G
            x1_, y1_ = x0_ + G, y0_ + G
            if (xb, yb + 1) not in bins:
                segs.append(((x0_, y1_), (x1_, y1_)))
            if (xb, yb - 1) not in bins:
                segs.append(((x0_, y0_), (x1_, y0_)))
            if (xb + 1, yb) not in bins:
                segs.append(((x1_, y0_), (x1_, y1_)))
            if (xb - 1, yb) not in bins:
                segs.append(((x0_, y0_), (x0_, y1_)))
        if segs:
            lc = LineCollection(segs, colors=[c], linewidths=1.6,
                                  alpha=0.95, zorder=3)
            ax.add_collection(lc)
        ax.scatter(ent["x"], ent["y"], s=dot_size, c=[c],
                    alpha=dot_alpha, linewidths=0, zorder=2,
                    label=f"{lab} ({len(ent)})")
        if label_entities:
            cx, cy = ent["x"].mean(), ent["y"].mean()
            ax.text(cx, cy, lab[-12:], fontsize=6.5, color="black",
                     ha="center", va="center", zorder=5,
                     path_effects=[_patheffects.withStroke(
                         linewidth=1.6, foreground="white")])

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title(f"{title}  ({len(real_labels)} entities, G={G} µm)",
                  fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    if len(real_labels) <= 14:
        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9,
                   markerscale=0.8)


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
    df["tracer_id"] = df["noseg_lab"]
    df.loc[df["tracer_id"].isin(SENT), "tracer_id"] = "-1"

    panel_raw = pd.read_parquet(PANEL)
    all_genes = sorted(set(panel_raw["gene_i"].astype(str))
                       | set(panel_raw["gene_j"].astype(str))
                       | set(df["feature_name"].unique()))
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

    print(f"Running BEST Stitch on full ROI: deltaC_min=-0.01, "
          f"c_union_bypass=0.9, capped witness=3, G=1 R=2 ...", flush=True)
    df_s, _ = apply_stitching_to_transcripts_memory_efficient(
        df_final=df, aux={"W": W, "gene_to_idx": g2i},
        entity_col="tracer_id", gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        mode="count", threshold=0.2, metric="pmi",
        penalize_simplicity=True,
        deltaC_min=-0.01,
        c_union_bypass=0.9,
        dist_threshold=5.0, out_col="stitched", show_progress=False,
        candidate_source="grid", G=1.0, stitch_neighborhood="R2",
        G_z=1.0, z_neighbor_depth=1, min_local_tx_per_entity=3,
    )
    df_s["stitched"] = df_s["stitched"].astype(str)

    # Find which stitched components contain ANY of nloapcgp-1's tx
    in_cell = df_s[df_s["cell_id"] == TARGET_CELL]
    stitched_for_cell = set(
        l for l in in_cell["stitched"].unique() if l not in SENT
    )
    print(f"\nStitched components touching {TARGET_CELL}: "
          f"{len(stitched_for_cell)}", flush=True)
    for sc in sorted(stitched_for_cell):
        sub = df_s[df_s["stitched"] == sc]
        in_c = sub[sub["cell_id"] == TARGET_CELL]
        out_c = sub[sub["cell_id"] != TARGET_CELL]
        pre_ents = sorted(p for p in sub["tracer_id"].unique() if p not in SENT)
        print(f"  {sc:>22s}  n_tx={len(sub):>3d}  in_cell={len(in_c):>3d}  "
              f"out_cell={len(out_c):>3d}  pre-merge entities: {pre_ents[:8]}"
              f"{'...' if len(pre_ents)>8 else ''}")

    # Subset df_s to those stitched components for the right panels
    stitched_df = df_s[df_s["stitched"].isin(stitched_for_cell)].copy()
    # And SEG side: nloapcgp-1 and nloapcgp-1-1
    seg_df = df_s[df_s["seg_lab"].isin({"nloapcgp-1", "nloapcgp-1-1"})].copy()

    union = pd.concat([seg_df, stitched_df])
    pad = 1.0
    xmin = np.floor((union["x"].min() - pad) / 2.0) * 2.0
    xmax = np.ceil((union["x"].max() + pad) / 2.0) * 2.0
    ymin = np.floor((union["y"].min() - pad) / 2.0) * 2.0
    ymax = np.ceil((union["y"].max() + pad) / 2.0) * 2.0

    fig, axes = plt.subplots(2, 2, figsize=(20, 20), dpi=140)
    _draw_panel(axes[0, 0], seg_df, "seg_lab",
                  f"SEG entities of {TARGET_CELL}",
                  (xmin, xmax), (ymin, ymax), G=2.0)
    _draw_panel(axes[0, 1], stitched_df, "stitched",
                  f"BEST Stitch (deltaC_min=-0.01, bypass=0.9) for {TARGET_CELL}",
                  (xmin, xmax), (ymin, ymax), G=2.0)
    _draw_panel(axes[1, 0], seg_df, "seg_lab",
                  f"SEG entities of {TARGET_CELL}",
                  (xmin, xmax), (ymin, ymax), G=1.0)
    _draw_panel(axes[1, 1], stitched_df, "stitched",
                  f"BEST Stitch for {TARGET_CELL}",
                  (xmin, xmax), (ymin, ymax), G=1.0)
    plt.suptitle(
        f"{TARGET_CELL}: SEG (2 entities) vs BEST Stitch on this branch\n"
        "G=1/R=2, deltaC_min=-0.01, c_union_bypass=0.9, capped min_local_tx=3\n"
        "left: SEG    right: BEST Stitch    top row: 2 µm tiles    bottom row: 1 µm tiles",
        fontsize=14, y=1.0,
    )
    plt.tight_layout()
    out = ZOOM_DIR / f"zoom_{TARGET_CELL}_best_stitch_g1r2.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
