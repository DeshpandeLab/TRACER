#!/usr/bin/env python3
"""Tile-based entity-extent visualization. For each entity, fill the
2 µm bins it occupies with a translucent color. Overlapping bins
(multiple entities sharing the same bin) blend visually.

Three panels for the worst-case 50µm ROI:
    1. SEG (Xenium + Phase 1c) — ground truth
    2. Stitch with G=2/R=1 (current production)
    3. Stitch with G=1/R=2 (proposed)

Per-tx dots are overlaid lightly so the eye can see local tx density
within each entity's footprint.
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

import os

PMI_THR = 0.2
SENT = {"-1", "DROP", "UNASSIGNED", "nan"}
G_TILE = float(os.environ.get("G_TILE", "2.0"))  # visualisation tile size
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
PDAC = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
ZOOM = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg" / "zoom_worst_tx.parquet"


def _run_stitch(df_in, aux, G_xy, neighborhood):
    df_in = df_in.copy()
    df_in["tracer_id"] = df_in["noseg_lab"].astype(str)
    df_in.loc[df_in["tracer_id"].isin(SENT), "tracer_id"] = "-1"
    df_s, _ = apply_stitching_to_transcripts_memory_efficient(
        df_final=df_in, aux=aux,
        entity_col="tracer_id", gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        mode="count", threshold=PMI_THR, metric="pmi",
        penalize_simplicity=True, deltaC_min=0.03,
        c_union_bypass=0.9,
        dist_threshold=5.0, out_col="stitched", show_progress=False,
        candidate_source="grid", G=G_xy, stitch_neighborhood=neighborhood,
        G_z=1.0, z_neighbor_depth=1, min_local_tx_per_entity=3,
    )
    return df_s["stitched"].astype(str)


def _draw_tile_panel(ax, df, label_col, title, xlim, ylim,
                      alpha_tile=0.30, dot_size=10, dot_alpha=0.6,
                      G=G_TILE):
    # Per-entity color assignment
    labels = sorted(df[label_col].astype(str).unique())
    real_labels = [l for l in labels if l not in SENT]
    n = max(len(real_labels), 1)
    cmap = plt.get_cmap("tab20" if n <= 20 else "gist_ncar", n)
    color_for = {l: cmap(i % cmap.N) for i, l in enumerate(real_labels)}

    # Grid lines (faint)
    x0 = np.floor(xlim[0] / G) * G; x1 = np.ceil(xlim[1] / G) * G
    y0 = np.floor(ylim[0] / G) * G; y1 = np.ceil(ylim[1] / G) * G
    for x in np.arange(x0, x1 + G * 0.5, G):
        ax.axvline(x, color="#dddddd", linewidth=0.4, zorder=0)
    for y in np.arange(y0, y1 + G * 0.5, G):
        ax.axhline(y, color="#dddddd", linewidth=0.4, zorder=0)

    # Compute per-entity bin sets, draw boundary outlines + a very light fill.
    # Boundary algorithm: for each bin in the entity's bin set, emit each of
    # its 4 sides that lies on the boundary (i.e., the neighboring bin in
    # that direction is NOT in the entity's bin set). Internal edges (shared
    # with the entity itself) are suppressed.
    df_assigned = df[~df[label_col].astype(str).isin(SENT)].copy()
    df_assigned["xb"] = np.floor(df_assigned["x"].to_numpy() / G).astype(int)
    df_assigned["yb"] = np.floor(df_assigned["y"].to_numpy() / G).astype(int)
    for lab in real_labels:
        ent = df_assigned[df_assigned[label_col].astype(str) == lab]
        if len(ent) == 0:
            continue
        bins = set(zip(ent["xb"].tolist(), ent["yb"].tolist()))
        c = color_for[lab]
        # Very light fill so overlap is still visible (optional).
        for (xb, yb) in bins:
            ax.add_patch(Rectangle(
                (xb * G, yb * G), G, G,
                facecolor=c, alpha=alpha_tile * 0.4, edgecolor="none",
                zorder=1,
            ))
        # Boundary segments
        segs = []
        for (xb, yb) in bins:
            x0, y0 = xb * G, yb * G
            x1, y1 = x0 + G, y0 + G
            if (xb, yb + 1) not in bins:
                segs.append(((x0, y1), (x1, y1)))   # top
            if (xb, yb - 1) not in bins:
                segs.append(((x0, y0), (x1, y0)))   # bottom
            if (xb + 1, yb) not in bins:
                segs.append(((x1, y0), (x1, y1)))   # right
            if (xb - 1, yb) not in bins:
                segs.append(((x0, y0), (x0, y1)))   # left
        if segs:
            lc = LineCollection(segs, colors=[c], linewidths=1.4,
                                  alpha=0.95, zorder=3)
            ax.add_collection(lc)

    # Light tx dots for local-density read
    for lab in real_labels:
        ent = df_assigned[df_assigned[label_col].astype(str) == lab]
        if len(ent) == 0:
            continue
        ax.scatter(ent["x"], ent["y"], s=dot_size, c=[color_for[lab]],
                    alpha=dot_alpha, linewidths=0, zorder=2)

    # Unassigned as light gray dots
    un = df[df[label_col].astype(str).isin(SENT)]
    if len(un):
        ax.scatter(un["x"], un["y"], s=dot_size, c="#cccccc", alpha=0.4,
                    linewidths=0, zorder=2)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    n_real = len(real_labels)
    ax.set_title(f"{title}  ({n_real} entities)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    zoom = pd.read_parquet(ZOOM)
    feats = pd.read_parquet(PDAC, columns=["transcript_id", "feature_name"])
    zcol = pd.read_parquet(PDAC, columns=["transcript_id", "z_location"]).rename(
        columns={"z_location": "z"}
    )
    df = zoom.merge(feats, on="transcript_id", how="left").merge(
        zcol, on="transcript_id", how="left"
    )
    df["feature_name"] = df["feature_name"].astype(str)
    df["noseg_lab"] = df["noseg_lab"].astype(str)
    df["seg_lab"] = df["seg_lab"].astype(str)
    df = df.reset_index(drop=True)

    # Build W for the two Stitch runs
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
    aux = {"W": W, "gene_to_idx": g2i}

    df["stitch_g2r1"] = _run_stitch(df, aux, G_xy=2.0, neighborhood="8")
    df["stitch_g1r2"] = _run_stitch(df, aux, G_xy=1.0, neighborhood="R2")

    # ROI bounds — full 50µm ROI
    pad = 1.0
    xmin = np.floor((df["x"].min() - pad) / G_TILE) * G_TILE
    xmax = np.ceil((df["x"].max() + pad) / G_TILE) * G_TILE
    ymin = np.floor((df["y"].min() - pad) / G_TILE) * G_TILE
    ymax = np.ceil((df["y"].max() + pad) / G_TILE) * G_TILE

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=140)
    _draw_tile_panel(axes[0], df, "seg_lab",
                      "SEG (Xenium + Phase 1c) — ground truth",
                      (xmin, xmax), (ymin, ymax))
    _draw_tile_panel(axes[1], df, "stitch_g2r1",
                      "Stitch G=2, R=1 (production)",
                      (xmin, xmax), (ymin, ymax))
    _draw_tile_panel(axes[2], df, "stitch_g1r2",
                      "Stitch G=1, R=2 (proposed)",
                      (xmin, xmax), (ymin, ymax))
    plt.suptitle(
        "Tile-based entity extent — overlapping bins blend (alpha=0.30).  "
        f"Tile size = {G_TILE} µm.",
        fontsize=13, y=1.0,
    )
    plt.tight_layout()
    g_tag = f"g{int(G_TILE)}" if G_TILE == int(G_TILE) else f"g{G_TILE}"
    out = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg" / f"zoom_entity_tiles_{g_tag}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}", flush=True)

    # Also report per-panel headline counts
    for col, label in [("seg_lab", "SEG"),
                        ("stitch_g2r1", "Stitch G=2/R=1"),
                        ("stitch_g1r2", "Stitch G=1/R=2")]:
        n_ent = df[col][~df[col].astype(str).isin(SENT)].nunique()
        print(f"  {label:<18s}  {n_ent} entities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
