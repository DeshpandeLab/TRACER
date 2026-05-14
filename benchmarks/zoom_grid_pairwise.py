#!/usr/bin/env python3
"""Pairwise NOSEG-fragment panels for cell `nloapcgp-1` over the G=2.0µm grid.

One subplot per spatially-relevant fragment pair (co_bins > 0). Each
component is assigned a unique (color, marker) identity that is reused
across panels so the eye can track a given fragment from panel to
panel. Each panel:
  - shows only the two component's tx (no faded background)
  - draws the G=2.0µm grid
  - title: pair id, ΔC_pen, witness counts, verdict
  - panel border colored by verdict (green=accept, red=reject)
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
G_XY = 2.0
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}

# Distinct markers — 13 needed for the 13 cascade fragments
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "p", "h", "<", ">", "H"]


def _draw_grid(ax, xmin, xmax, ymin, ymax, G=G_XY, color="#cccccc", lw=0.4):
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
    cell["noseg_lab"] = cell["noseg_lab"].astype(str)
    cell = cell[~cell["noseg_lab"].isin(SENTINELS)]

    log = pd.read_csv(ZOOM_DIR / "zoom_worst_stitch_candidates.csv")

    # Component identity table: stable (color, marker) per fragment
    frag_ids = sorted(cell["noseg_lab"].unique())
    cmap = plt.get_cmap("tab20", max(len(frag_ids), 1))
    ident = {fid: (cmap(i % cmap.N), MARKERS[i % len(MARKERS)])
             for i, fid in enumerate(frag_ids)}

    # Per-component tx
    by_ent = {fid: cell[cell["noseg_lab"] == fid][["x", "y"]].to_numpy()
              for fid in frag_ids}

    # Plot bounds (cell-wide, shared across panels for direct comparison)
    pad = 1.5
    xmin, xmax = cell["x"].min() - pad, cell["x"].max() + pad
    ymin, ymax = cell["y"].min() - pad, cell["y"].max() + pad

    # Restrict to spatially-relevant pairs (co_bins > 0); these are
    # the only pairs Stitch's grid-candidate enumeration considers.
    sub = log[log["co_bins"] > 0].copy()
    # Sort: accepted first, then by descending ΔC_pen
    sub["sort_key"] = sub["accepted"].astype(int) * 10 + sub["dC_pen"]
    sub = sub.sort_values("sort_key", ascending=False).reset_index(drop=True)
    n_pairs = len(sub)
    print(f"Pairs with co-occurring grid bins: {n_pairs}", flush=True)
    print(f"  accepted: {int(sub['accepted'].sum())}", flush=True)

    # Grid of panels: 6 columns
    ncols = 6
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows),
                              dpi=120)
    axes = np.atleast_2d(axes)

    for k in range(nrows * ncols):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        if k >= n_pairs:
            ax.axis("off")
            continue
        row = sub.iloc[k]
        a, b = row["A"], row["B"]
        pa = by_ent[a]; pb = by_ent[b]
        ca, ma = ident[a]
        cb, mb = ident[b]

        _draw_grid(ax, xmin, xmax, ymin, ymax)
        # Component A
        ax.scatter(pa[:, 0], pa[:, 1], s=46, marker=ma, c=[ca], alpha=0.92,
                    edgecolor="black", linewidths=0.5, zorder=3,
                    label=f"A: {a.replace('cascade_', 'c_')} (n={len(pa)})")
        # Component B
        ax.scatter(pb[:, 0], pb[:, 1], s=46, marker=mb, c=[cb], alpha=0.92,
                    edgecolor="black", linewidths=0.5, zorder=3,
                    label=f"B: {b.replace('cascade_', 'c_')} (n={len(pb)})")

        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

        # Border color by verdict
        if row["accepted"]:
            border = "#2ca02c"; verdict = "ACCEPT"
        elif row["pass_witness"] and not row["pass_deltaC"]:
            border = "#d62728" if row["dC_pen"] < 0 else "#ff7f0e"
            verdict = "REJECT (ΔC)"
        elif row["pass_deltaC"] and not row["pass_witness"]:
            border = "#9467bd"; verdict = "REJECT (witness)"
        else:
            border = "#888888"; verdict = "REJECT"
        for sp in ax.spines.values():
            sp.set_edgecolor(border)
            sp.set_linewidth(2.5)

        title = (f"{a.replace('cascade_', '').replace('-1','')} × "
                 f"{b.replace('cascade_', '').replace('-1','')}\n"
                 f"n=({int(row['n_A'])},{int(row['n_B'])})  "
                 f"wit=({int(row['witness_A'])},{int(row['witness_B'])})  "
                 f"co_bins={int(row['co_bins'])}\n"
                 f"C_uni={row['C_union']:.3f}  ΔC_pen={row['dC_pen']:+.4f}  "
                 f"{verdict}")
        ax.set_title(title, fontsize=7.5)
        ax.legend(loc="upper left", fontsize=6, framealpha=0.85,
                   markerscale=0.6)

    # Outer legend mapping each fragment to its (color, marker)
    from matplotlib.lines import Line2D
    fragment_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor=cc,
                markeredgecolor="black", markersize=8,
                label=fid.replace("cascade_", ""))
        for fid, (cc, m) in ident.items()
    ]
    plt.suptitle(
        "Pairwise NOSEG-fragment panels — nloapcgp-1, G=2.0µm grid, "
        "deltaC_min=-0.01, min_local_tx=3\n"
        "border: green=accept, red=ΔC<0 reject, orange=ΔC∈[0,0.03) "
        "reject, purple=witness reject",
        fontsize=11, y=1.0,
    )
    fig.legend(handles=fragment_handles, loc="lower center",
                ncol=min(len(fragment_handles), 7), fontsize=8,
                frameon=False, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    out = ZOOM_DIR / "zoom_worst_grid_pairwise.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
