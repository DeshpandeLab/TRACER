#!/usr/bin/env python3
"""Plot SEG entities `nloapcgp-1` (main) and `nloapcgp-1-1` (sub-partial) separately.

Each on its own panel with convex hull, against a faded background of the
full cell_id footprint for context. Lets us see the spatial split that
Phase-1c imposed on this cell.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import patheffects as _patheffects
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[1]
ZOOM_DIR = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg"
CELL_ID = "nloapcgp-1"
TARGETS = ["nloapcgp-1", "nloapcgp-1-1"]


def _hull_polygon(pts):
    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            return pts[hull.vertices]
        except Exception:
            return None
    return None


def main() -> int:
    df = pd.read_parquet(ZOOM_DIR / "zoom_worst_tx.parquet")
    # Subset to all tx of the input cell_id (background context)
    cell_tx = df[df["cell_id"].astype(str) == CELL_ID].copy()
    print(f"{CELL_ID}: {len(cell_tx)} tx in ROI", flush=True)

    # Determine common axis from cell footprint with a small pad
    pad = 4.0
    xmin, xmax = cell_tx["x"].min() - pad, cell_tx["x"].max() + pad
    ymin, ymax = cell_tx["y"].min() - pad, cell_tx["y"].max() + pad

    # Full cell hull for context
    cell_pts = cell_tx[["x", "y"]].to_numpy()
    cell_hull = _hull_polygon(cell_pts)

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(14, 7), dpi=140)
    colors = {"nloapcgp-1": "#1f77b4", "nloapcgp-1-1": "#d62728"}

    for ax, target in zip(axes, TARGETS):
        # All tx of the cell as gray (context)
        other = cell_tx[cell_tx["seg_lab"].astype(str) != target]
        ax.scatter(other["x"], other["y"], s=14, c="#dddddd", alpha=0.7,
                    linewidths=0, label=f"other cell tx ({len(other)})",
                    zorder=1)
        # The cell's full hull, faded
        if cell_hull is not None:
            ax.add_patch(Polygon(cell_hull, closed=True, alpha=0.06,
                                  facecolor="#888888", edgecolor="#888888",
                                  linewidth=0.8, linestyle="--", zorder=0))

        # The target entity tx + hull
        ent_tx = cell_tx[cell_tx["seg_lab"].astype(str) == target]
        if len(ent_tx) == 0:
            ax.text(0.5, 0.5, f"no tx for {target}", transform=ax.transAxes,
                     ha="center", va="center")
        else:
            color = colors.get(target, "#1f77b4")
            pts = ent_tx[["x", "y"]].to_numpy()
            ax.scatter(pts[:, 0], pts[:, 1], s=28, c=color, alpha=0.92,
                        linewidths=0, label=f"{target} ({len(ent_tx)} tx)",
                        zorder=3)
            hull_pts = _hull_polygon(pts)
            if hull_pts is not None:
                ax.add_patch(Polygon(hull_pts, closed=True, alpha=0.25,
                                      facecolor=color, edgecolor=color,
                                      linewidth=2, zorder=2))
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            ax.text(cx, cy, target, fontsize=9, ha="center", va="center",
                     color="black", zorder=5,
                     path_effects=[_patheffects.withStroke(
                         linewidth=2, foreground="white"
                     )])

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
        ax.set_title(f"SEG entity: {target}", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    plt.suptitle(f"SEG split of cell_id {CELL_ID}  — main vs Phase-1c sub-partial",
                  fontsize=13, y=1.0)
    plt.tight_layout()
    out = ZOOM_DIR / "zoom_seg_main_vs_partial.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
