#!/usr/bin/env python3
"""Plot the 13 NOSEG cascade fragments of cell `nloapcgp-1` with the
production Stitch grid (G=2.0 µm xy) overlaid.

Three panels:
  1. tx colored by NOSEG fragment, G=2.0 grid drawn, fragment IDs labeled
  2. same tx colored by SEG entity (nloapcgp-1 main vs nloapcgp-1-1 partial)
     to visualize the underlying epi-vs-CAF biological split
  3. accepted/blocked candidate edges from the per-pair log overlaid as
     line segments between fragment centroids
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as _patheffects

REPO = Path(__file__).resolve().parents[1]
ZOOM_DIR = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg"
PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
G_XY = 2.0
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}


def _draw_grid(ax, xmin, xmax, ymin, ymax, G=G_XY, color="#bbbbbb", lw=0.4):
    # Snap to grid
    x0 = np.floor(xmin / G) * G
    x1 = np.ceil(xmax / G) * G
    y0 = np.floor(ymin / G) * G
    y1 = np.ceil(ymax / G) * G
    for x in np.arange(x0, x1 + G * 0.5, G):
        ax.axvline(x, color=color, linewidth=lw, zorder=0)
    for y in np.arange(y0, y1 + G * 0.5, G):
        ax.axhline(y, color=color, linewidth=lw, zorder=0)


def main() -> int:
    zoom = pd.read_parquet(ZOOM_DIR / "zoom_worst_tx.parquet")
    cell = zoom[zoom["cell_id"].astype(str) == "nloapcgp-1"].copy()
    print(f"nloapcgp-1: {len(cell)} tx in ROI", flush=True)

    # Subset to non-sentinel cascade entities
    cell["noseg_lab"] = cell["noseg_lab"].astype(str)
    cell["seg_lab"] = cell["seg_lab"].astype(str)
    cell_assigned = cell[~cell["noseg_lab"].isin(SENTINELS)].copy()

    # Load candidate-edge log
    log = pd.read_csv(ZOOM_DIR / "zoom_worst_stitch_candidates.csv")

    # Plot bounds: pad cell footprint then snap to grid
    pad = 2.0
    xmin, xmax = cell["x"].min() - pad, cell["x"].max() + pad
    ymin, ymax = cell["y"].min() - pad, cell["y"].max() + pad

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5), dpi=130)

    # ---------- Panel 1: NOSEG fragments + grid ----------
    ax = axes[0]
    _draw_grid(ax, xmin, xmax, ymin, ymax)

    frag_ids = sorted(cell_assigned["noseg_lab"].unique())
    cmap = plt.get_cmap("tab20", max(len(frag_ids), 1))
    frag_color = {fid: cmap(i % cmap.N) for i, fid in enumerate(frag_ids)}
    frag_centroid = {}
    for fid in frag_ids:
        pts = cell_assigned[cell_assigned["noseg_lab"] == fid][["x", "y"]].to_numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=32, c=[frag_color[fid]], alpha=0.92,
                    linewidths=0.4, edgecolor="white", zorder=3,
                    label=f"{fid.replace('cascade_', 'c_')} (n={len(pts)})")
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        frag_centroid[fid] = (cx, cy)
        ax.text(cx, cy, fid.replace("cascade_", "").replace("-1", ""),
                fontsize=6, color="black", ha="center", va="center",
                zorder=5,
                path_effects=[_patheffects.withStroke(linewidth=1.5,
                                                       foreground="white")])
    # Unassigned tx
    unassigned = cell[cell["noseg_lab"].isin(SENTINELS)]
    if len(unassigned):
        ax.scatter(unassigned["x"], unassigned["y"], s=14, c="#dddddd",
                    alpha=0.6, linewidths=0, zorder=2,
                    label=f"unassigned ({len(unassigned)})")

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title(f"NOSEG cascade fragments + G=2.0µm grid  "
                  f"({len(frag_ids)} entities)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=5.5, ncol=2, framealpha=0.9,
               markerscale=0.7)

    # ---------- Panel 2: SEG entities + grid ----------
    ax = axes[1]
    _draw_grid(ax, xmin, xmax, ymin, ymax)
    seg_colors = {"nloapcgp-1": "#1f77b4", "nloapcgp-1-1": "#d62728"}
    for seg_id, color in seg_colors.items():
        pts = cell[cell["seg_lab"] == seg_id][["x", "y"]].to_numpy()
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=32, c=color, alpha=0.85,
                        linewidths=0.4, edgecolor="white", zorder=3,
                        label=f"{seg_id} (n={len(pts)})")
    other = cell[~cell["seg_lab"].isin(list(seg_colors) + list(SENTINELS))]
    if len(other):
        ax.scatter(other["x"], other["y"], s=14, c="#dddddd", alpha=0.6,
                    linewidths=0, zorder=2,
                    label=f"other SEG / unassigned ({len(other)})")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title("SEG entities (epi main vs CAF partial) + grid",
                  fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    # ---------- Panel 3: candidate edges over NOSEG tx ----------
    ax = axes[2]
    _draw_grid(ax, xmin, xmax, ymin, ymax)
    # Faded tx as background
    for fid in frag_ids:
        pts = cell_assigned[cell_assigned["noseg_lab"] == fid][["x", "y"]].to_numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=24, c=[frag_color[fid]], alpha=0.35,
                    linewidths=0, zorder=2)
    # Edge segments
    for _, r in log.iterrows():
        a, b = r["A"], r["B"]
        if a not in frag_centroid or b not in frag_centroid:
            continue
        (xa, ya), (xb, yb) = frag_centroid[a], frag_centroid[b]
        if r["accepted"]:
            ax.plot([xa, xb], [ya, yb], color="#2ca02c", linewidth=2.5,
                     alpha=0.95, zorder=5,
                     label="accepted" if r.name == log[log["accepted"]].index[0] else None)
        elif r["pass_witness"] and not r["pass_deltaC"]:
            color = "#ff7f0e" if r["dC_pen"] >= 0 else "#d62728"
            ax.plot([xa, xb], [ya, yb], color=color, linewidth=1.2,
                     alpha=0.7, linestyle="--", zorder=4)
        elif r["pass_deltaC"] and not r["pass_witness"]:
            # ΔC passes but no spatial witness — usually means too far
            if r["co_bins"] > 0:
                ax.plot([xa, xb], [ya, yb], color="#9467bd", linewidth=1.0,
                         alpha=0.5, linestyle=":", zorder=3)

    # Labels at centroids
    for fid, (cx, cy) in frag_centroid.items():
        short = fid.replace("cascade_", "").replace("-1", "")
        ax.text(cx, cy, short, fontsize=6, color="black", ha="center",
                va="center", zorder=6,
                path_effects=[_patheffects.withStroke(linewidth=1.5,
                                                       foreground="white")])

    # Legend lines
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color="#2ca02c", linewidth=2.5,
                label=f"accepted (n={int(log['accepted'].sum())})"),
        Line2D([0], [0], color="#ff7f0e", linewidth=1.2, linestyle="--",
                label="adj + ΔC_pen ∈ [0, 0.03) (rejected)"),
        Line2D([0], [0], color="#d62728", linewidth=1.2, linestyle="--",
                label="adj + ΔC_pen < 0 (cross-type, rejected)"),
        Line2D([0], [0], color="#9467bd", linewidth=1.0, linestyle=":",
                label="adj but <3 witness (rejected)"),
    ]
    ax.legend(handles=legend_lines, loc="upper right", fontsize=7,
               framealpha=0.9)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title(f"Candidate edges (Stitch gates: deltaC_min=-0.01, "
                  f"min_local_tx=3)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"NOSEG fragments on grid — nloapcgp-1 (157 tx, 13 cascade entities)",
                  fontsize=13, y=1.0)
    plt.tight_layout()
    out = ZOOM_DIR / "zoom_worst_grid_overlay.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
