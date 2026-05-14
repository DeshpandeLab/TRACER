#!/usr/bin/env python3
"""SEG vs NOSEG pre-Stitch tile-and-outline visualization.

Two columns × two rows:
    column 1: SEG (Xenium + Phase 1c)
    column 2: NOSEG cascade (pre-Stitch)
    row 1:    G=2 µm tiles
    row 2:    G=1 µm tiles

No stitcher output here — just the two upstream partitions side-by-side
at two tile granularities.
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

REPO = Path(__file__).resolve().parents[1]
ZOOM_DIR = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg"
PDAC = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
ZOOM = ZOOM_DIR / "zoom_worst_tx.parquet"
SENT = {"-1", "DROP", "UNASSIGNED", "nan"}


def _draw_outline_panel(ax, df, label_col, title, xlim, ylim, G,
                         alpha_fill=0.10, dot_size=6, dot_alpha=0.45):
    labels = sorted(df[label_col].astype(str).unique())
    real_labels = [l for l in labels if l not in SENT]
    n = max(len(real_labels), 1)
    cmap = plt.get_cmap("tab20" if n <= 20 else "gist_ncar", n)
    color_for = {l: cmap(i % cmap.N) for i, l in enumerate(real_labels)}

    # Faint grid
    x0 = np.floor(xlim[0] / G) * G; x1 = np.ceil(xlim[1] / G) * G
    y0 = np.floor(ylim[0] / G) * G; y1 = np.ceil(ylim[1] / G) * G
    for x in np.arange(x0, x1 + G * 0.5, G):
        ax.axvline(x, color="#eeeeee", linewidth=0.3, zorder=0)
    for y in np.arange(y0, y1 + G * 0.5, G):
        ax.axhline(y, color="#eeeeee", linewidth=0.3, zorder=0)

    df_a = df[~df[label_col].astype(str).isin(SENT)].copy()
    df_a["xb"] = np.floor(df_a["x"].to_numpy() / G).astype(int)
    df_a["yb"] = np.floor(df_a["y"].to_numpy() / G).astype(int)

    for lab in real_labels:
        ent = df_a[df_a[label_col].astype(str) == lab]
        if len(ent) == 0:
            continue
        bins = set(zip(ent["xb"].tolist(), ent["yb"].tolist()))
        c = color_for[lab]
        # Faint fill
        for (xb, yb) in bins:
            ax.add_patch(Rectangle(
                (xb * G, yb * G), G, G,
                facecolor=c, alpha=alpha_fill, edgecolor="none", zorder=1,
            ))
        # Boundary outline
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
            lc = LineCollection(segs, colors=[c], linewidths=1.3,
                                  alpha=0.95, zorder=3)
            ax.add_collection(lc)

    # tx dots (very small)
    for lab in real_labels:
        ent = df_a[df_a[label_col].astype(str) == lab]
        if len(ent) == 0:
            continue
        ax.scatter(ent["x"], ent["y"], s=dot_size, c=[color_for[lab]],
                    alpha=dot_alpha, linewidths=0, zorder=2)
    un = df[df[label_col].astype(str).isin(SENT)]
    if len(un):
        ax.scatter(un["x"], un["y"], s=dot_size, c="#bbbbbb",
                    alpha=0.4, linewidths=0, zorder=2)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title(f"{title}  ({len(real_labels)} entities, G={G} µm)",
                  fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    zoom = pd.read_parquet(ZOOM)
    feats = pd.read_parquet(PDAC, columns=["transcript_id", "feature_name"])
    df = zoom.merge(feats, on="transcript_id", how="left")
    df["feature_name"] = df["feature_name"].astype(str)
    df["seg_lab"] = df["seg_lab"].astype(str)
    df["noseg_lab"] = df["noseg_lab"].astype(str)

    pad = 1.0
    xmin = np.floor((df["x"].min() - pad) / 2.0) * 2.0
    xmax = np.ceil((df["x"].max() + pad) / 2.0) * 2.0
    ymin = np.floor((df["y"].min() - pad) / 2.0) * 2.0
    ymax = np.ceil((df["y"].max() + pad) / 2.0) * 2.0

    fig, axes = plt.subplots(2, 2, figsize=(18, 18), dpi=140)

    _draw_outline_panel(axes[0, 0], df, "seg_lab",
                          "SEG final (cells + Phase 1c partials, post-pipeline)",
                          (xmin, xmax), (ymin, ymax), G=2.0)
    _draw_outline_panel(axes[0, 1], df, "noseg_lab",
                          "NOSEG final (cascade partials, post-pipeline)",
                          (xmin, xmax), (ymin, ymax), G=2.0)
    _draw_outline_panel(axes[1, 0], df, "seg_lab",
                          "SEG final (cells + Phase 1c partials, post-pipeline)",
                          (xmin, xmax), (ymin, ymax), G=1.0)
    _draw_outline_panel(axes[1, 1], df, "noseg_lab",
                          "NOSEG final (cascade partials, post-pipeline)",
                          (xmin, xmax), (ymin, ymax), G=1.0)

    plt.suptitle(
        "Final pipeline outputs after Stitch + Demote + Final Rescue\n"
        "left: SEG    right: NOSEG    top row: 2 µm tiles    bottom row: 1 µm tiles",
        fontsize=14, y=1.0,
    )
    plt.tight_layout()
    out = ZOOM_DIR / "zoom_seg_vs_noseg_tiles.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}", flush=True)

    # Headline numbers
    n_seg = df["seg_lab"][~df["seg_lab"].isin(SENT)].nunique()
    n_noseg = df["noseg_lab"][~df["noseg_lab"].isin(SENT)].nunique()
    print(f"  SEG: {n_seg} entities")
    print(f"  NOSEG-cascade: {n_noseg} entities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
