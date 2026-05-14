#!/usr/bin/env python3
"""Tile-based entity-extent visualization restricted to entities
associated with cell `nloapcgp-1`.

    SEG side:    `nloapcgp-1` (main) + `nloapcgp-1-1` (Phase 1c partial)
    NOSEG side:  every cascade entity that has ≥1 tx in nloapcgp-1's
                 Xenium cell footprint. (These cascade entities may
                 extend beyond the cell — the full extent is shown.)

No other cells, no other entities, no surrounding tx.

Boundary outlines + faint fill, at both G=2 µm and G=1 µm tile sizes.
"""
from __future__ import annotations

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
ZOOM_DIR = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg"
PDAC = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
ZOOM = ZOOM_DIR / "zoom_worst_tx.parquet"
SENT = {"-1", "DROP", "UNASSIGNED", "nan"}


def _draw_panel(ax, df, label_col, title, xlim, ylim, G,
                 alpha_fill=0.18, dot_size=20, dot_alpha=0.8,
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
    df = zoom.merge(feats, on="transcript_id", how="left")
    df["feature_name"] = df["feature_name"].astype(str)
    df["seg_lab"] = df["seg_lab"].astype(str)
    df["noseg_lab"] = df["noseg_lab"].astype(str)
    df["cell_id"] = df["cell_id"].astype(str)

    # SEG entities for this cell: nloapcgp-1 + nloapcgp-1-1
    SEG_LABS = {"nloapcgp-1", "nloapcgp-1-1"}
    # NOSEG entities for this cell: every cascade label that has ≥1 tx
    # in nloapcgp-1's Xenium footprint
    in_cell = df[df["cell_id"] == "nloapcgp-1"]
    NOSEG_LABS = set(in_cell["noseg_lab"].astype(str).unique()) - SENT
    print(f"SEG entities for nloapcgp-1: {len(SEG_LABS)} → {sorted(SEG_LABS)}")
    print(f"NOSEG entities touching nloapcgp-1: {len(NOSEG_LABS)} → "
          f"{sorted(NOSEG_LABS)}", flush=True)

    seg_df = df[df["seg_lab"].isin(SEG_LABS)].copy()
    noseg_df = df[df["noseg_lab"].isin(NOSEG_LABS)].copy()
    print(f"\nSEG: {len(seg_df)} tx across {seg_df['seg_lab'].nunique()} entities")
    print(f"NOSEG: {len(noseg_df)} tx across {noseg_df['noseg_lab'].nunique()} entities")

    # Plot bounds: tight box around the union of SEG + NOSEG tx, snapped to G=2
    union = pd.concat([seg_df, noseg_df])
    pad = 1.0
    xmin = np.floor((union["x"].min() - pad) / 2.0) * 2.0
    xmax = np.ceil((union["x"].max() + pad) / 2.0) * 2.0
    ymin = np.floor((union["y"].min() - pad) / 2.0) * 2.0
    ymax = np.ceil((union["y"].max() + pad) / 2.0) * 2.0

    fig, axes = plt.subplots(2, 2, figsize=(20, 20), dpi=140)
    _draw_panel(axes[0, 0], seg_df, "seg_lab",
                  "SEG entities of nloapcgp-1",
                  (xmin, xmax), (ymin, ymax), G=2.0)
    _draw_panel(axes[0, 1], noseg_df, "noseg_lab",
                  "NOSEG entities touching nloapcgp-1",
                  (xmin, xmax), (ymin, ymax), G=2.0)
    _draw_panel(axes[1, 0], seg_df, "seg_lab",
                  "SEG entities of nloapcgp-1",
                  (xmin, xmax), (ymin, ymax), G=1.0)
    _draw_panel(axes[1, 1], noseg_df, "noseg_lab",
                  "NOSEG entities touching nloapcgp-1",
                  (xmin, xmax), (ymin, ymax), G=1.0)
    plt.suptitle(
        "Entities associated with cell nloapcgp-1 — final pipeline outputs\n"
        "left: SEG    right: NOSEG    top row: 2 µm tiles    bottom row: 1 µm tiles",
        fontsize=14, y=1.0,
    )
    plt.tight_layout()
    out = ZOOM_DIR / "zoom_nloapcgp_entities_only.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
