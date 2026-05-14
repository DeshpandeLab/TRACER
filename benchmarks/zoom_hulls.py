#!/usr/bin/env python3
"""Draw 2D convex hulls around each SEG / NOSEG entity in the worst-case zoom.

Uses scipy.spatial.ConvexHull on (x, y). Reuses the saved zoom_worst_tx.parquet
and zoom_best_tx.parquet from zoom_seg_vs_noseg_nontrivial.py.

Three side-by-side panels per zoom:
  1. tx colored by INPUT cell_id (ground truth)
  2. tx colored by SEG entity + SEG entity hulls
  3. tx colored by NOSEG-thr=4 entity + NOSEG hulls

Tiny entities (<3 tx) are drawn as dots only (can't form a polygon).
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
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}


def _draw_hulls(ax, df, label_col, title, draw_centroid=False):
    labels = df[label_col].astype(str)
    is_un = labels.isin(SENTINELS) | labels.str.endswith("_rejected", na=False)
    # First plot unassigned tx as light gray
    if is_un.any():
        ax.scatter(df.loc[is_un, "x"], df.loc[is_un, "y"],
                    s=10, c="#dddddd", alpha=0.6, linewidths=0,
                    label="unassigned" if is_un.sum() > 10 else None)

    # Per assigned entity, draw scatter + hull
    cats = sorted(labels[~is_un].unique())
    n_cats = len(cats)
    cmap = plt.get_cmap("tab20" if n_cats <= 20 else "gist_ncar", max(n_cats, 1))

    patches = []
    patch_colors = []
    for k, cat in enumerate(cats):
        sel = (labels == cat).to_numpy()
        pts = df.loc[sel, ["x", "y"]].to_numpy()
        color = cmap(k % cmap.N)
        # Scatter tx
        ax.scatter(pts[:, 0], pts[:, 1], s=20, c=[color], alpha=0.9,
                    linewidths=0, zorder=2)
        # Hull
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                patches.append(Polygon(hull_pts, closed=True))
                patch_colors.append(color)
            except Exception:
                pass
        # Optional centroid marker
        if draw_centroid:
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            ax.plot(cx, cy, marker="x", color=color, markersize=8,
                     markeredgewidth=2, zorder=4)
        # Label entity with short ID at centroid
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        ax.text(cx, cy, cat[:12], fontsize=5, color="black",
                 ha="center", va="center", zorder=5,
                 path_effects=[_patheffects.withStroke(
                     linewidth=1.5, foreground="white"
                 )])

    if patches:
        pc = PatchCollection(patches, facecolor=patch_colors, alpha=0.2,
                              edgecolor=patch_colors, linewidth=1.5, zorder=1)
        ax.add_collection(pc)

    ax.set_aspect("equal")
    ax.set_title(f"{title}  ({n_cats} entities)", fontsize=11)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.spines[["top", "right"]].set_visible(False)


def _plot_zoom(zoom_name: str) -> None:
    in_path = ZOOM_DIR / f"zoom_{zoom_name}_tx.parquet"
    if not in_path.exists():
        print(f"missing {in_path}", flush=True)
        return
    df = pd.read_parquet(in_path)
    print(f"{zoom_name}: {len(df):,} tx, "
          f"{df['cell_id'].astype(str).nunique()} unique input cell_ids", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=130)
    _draw_hulls(axes[0], df, "cell_id", "Input cell_id (Xenium)")
    _draw_hulls(axes[1], df, "seg_lab", "SEG entities")
    _draw_hulls(axes[2], df, "noseg_lab", "NOSEG (cascade thr=4) entities")
    plt.suptitle(f"Convex hulls — {zoom_name} case  (~50µm ROI)",
                  fontsize=13, y=1.0)
    plt.tight_layout()
    out = ZOOM_DIR / f"zoom_{zoom_name}_hulls.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}", flush=True)


def main() -> int:
    for name in ("best", "worst"):
        _plot_zoom(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
