#!/usr/bin/env python3
"""Single-pair zoom on cascade_103590-1 × cascade_55985-1.

Plots each tx with a distinct marker shape (+ for A, x for B), highlights
the bins where each entity contributes a witness, and overlays the
G=2.0 µm grid so each point's bin assignment is unambiguous. Annotates
each bin (xy) with the number of A-tx and B-tx inside it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patheffects as _patheffects

REPO = Path(__file__).resolve().parents[1]
ZOOM_DIR = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg"
PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
G_XY = 2.0
G_Z = 1.0
Z_DEPTH = 1
A_NAME = "cascade_103590-1"
B_NAME = "cascade_55985-1"
A_LABEL = "103590"
B_LABEL = "55985"
A_COLOR = "#1f77b4"  # blue
B_COLOR = "#d62728"  # red


def _draw_grid(ax, xmin, xmax, ymin, ymax, G=G_XY, color="#cccccc", lw=0.6):
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
    z = pd.read_parquet(
        PDAC_PARQUET, columns=["transcript_id", "z_location"]
    ).rename(columns={"z_location": "z"})
    cell = cell.merge(z, on="transcript_id", how="left")

    A = cell[cell["noseg_lab"] == A_NAME].reset_index(drop=True)
    B = cell[cell["noseg_lab"] == B_NAME].reset_index(drop=True)
    print(f"{A_NAME}: {len(A)} tx", flush=True)
    print(f"{B_NAME}: {len(B)} tx", flush=True)

    # Bin
    for d in (A, B):
        d["xb"] = np.floor(d["x"].to_numpy() / G_XY).astype(np.int64)
        d["yb"] = np.floor(d["y"].to_numpy() / G_XY).astype(np.int64)
        d["zb"] = np.floor(d["z"].to_numpy() / G_Z).astype(np.int64)

    A_bins = set(zip(A["xb"].tolist(), A["yb"].tolist(), A["zb"].tolist()))
    B_bins = set(zip(B["xb"].tolist(), B["yb"].tolist(), B["zb"].tolist()))

    nbrs = [(dx, dy, dz)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for dz in range(-Z_DEPTH, Z_DEPTH + 1)]

    # Identify witness tx (A-tx whose bin is within the 8-Moore+z window
    # of any B-bin, and symmetrically for B).
    A["is_witness"] = False
    for i, row in A.iterrows():
        for dx, dy, dz in nbrs:
            if (row["xb"] + dx, row["yb"] + dy, row["zb"] + dz) in B_bins:
                A.at[i, "is_witness"] = True
                break
    B["is_witness"] = False
    for i, row in B.iterrows():
        for dx, dy, dz in nbrs:
            if (row["xb"] + dx, row["yb"] + dy, row["zb"] + dz) in A_bins:
                B.at[i, "is_witness"] = True
                break

    print(f"A witnesses: {int(A['is_witness'].sum())} / {len(A)}", flush=True)
    print(f"B witnesses: {int(B['is_witness'].sum())} / {len(B)}", flush=True)

    # Plot bounds: snap to grid with small pad
    all_x = np.concatenate([A["x"].to_numpy(), B["x"].to_numpy()])
    all_y = np.concatenate([A["y"].to_numpy(), B["y"].to_numpy()])
    pad = 2.0
    xmin = np.floor((all_x.min() - pad) / G_XY) * G_XY
    xmax = np.ceil((all_x.max() + pad) / G_XY) * G_XY
    ymin = np.floor((all_y.min() - pad) / G_XY) * G_XY
    ymax = np.ceil((all_y.max() + pad) / G_XY) * G_XY

    fig, ax = plt.subplots(figsize=(11, 11), dpi=140)
    _draw_grid(ax, xmin, xmax, ymin, ymax)

    # Highlight witness bins: A-witness bins lightly blue tinted, B-witness
    # bins lightly red tinted (just the bin itself, not the 8-Moore reach).
    A_wit_bins = set(zip(A.loc[A["is_witness"], "xb"],
                          A.loc[A["is_witness"], "yb"]))
    B_wit_bins = set(zip(B.loc[B["is_witness"], "xb"],
                          B.loc[B["is_witness"], "yb"]))
    for (xb, yb) in A_wit_bins:
        ax.add_patch(Rectangle((xb * G_XY, yb * G_XY), G_XY, G_XY,
                                facecolor=A_COLOR, alpha=0.10, zorder=1,
                                edgecolor="none"))
    for (xb, yb) in B_wit_bins:
        ax.add_patch(Rectangle((xb * G_XY, yb * G_XY), G_XY, G_XY,
                                facecolor=B_COLOR, alpha=0.10, zorder=1,
                                edgecolor="none"))
    # If a bin is BOTH (rare here), it'll show as a darker blend.

    # Tx: + for A, x for B. Witness tx drawn with thicker stroke.
    for is_wit, lw, alpha in [(False, 1.6, 0.55), (True, 2.6, 1.0)]:
        Ai = A[A["is_witness"] == is_wit]
        ax.scatter(Ai["x"], Ai["y"], marker="+", s=320, c=A_COLOR,
                    linewidths=lw, alpha=alpha, zorder=4,
                    label=(f"A = {A_LABEL}  (witness, n={int(A['is_witness'].sum())})"
                           if is_wit else
                           f"A = {A_LABEL}  (non-witness, n={int((~A['is_witness']).sum())})"))
    for is_wit, lw, alpha in [(False, 1.6, 0.55), (True, 2.6, 1.0)]:
        Bi = B[B["is_witness"] == is_wit]
        ax.scatter(Bi["x"], Bi["y"], marker="x", s=240, c=B_COLOR,
                    linewidths=lw, alpha=alpha, zorder=4,
                    label=(f"B = {B_LABEL}  (witness, n={int(B['is_witness'].sum())})"
                           if is_wit else
                           f"B = {B_LABEL}  (non-witness, n={int((~B['is_witness']).sum())})"))

    # Annotate each bin (xy projection) with counts.
    # We pool over z for the visual annotation since the plot is 2D.
    bin_counts = {}
    for _, row in A.iterrows():
        key = (int(row["xb"]), int(row["yb"]))
        bin_counts.setdefault(key, [0, 0])
        bin_counts[key][0] += 1
    for _, row in B.iterrows():
        key = (int(row["xb"]), int(row["yb"]))
        bin_counts.setdefault(key, [0, 0])
        bin_counts[key][1] += 1
    for (xb, yb), (na, nb) in bin_counts.items():
        if na == 0 and nb == 0:
            continue
        txt = f"A:{na} B:{nb}" if (na and nb) else (f"A:{na}" if na else f"B:{nb}")
        ax.text(xb * G_XY + G_XY * 0.05, yb * G_XY + G_XY * 0.05, txt,
                 fontsize=6.5, color="#222222", ha="left", va="bottom",
                 zorder=6,
                 path_effects=[_patheffects.withStroke(linewidth=1.8,
                                                       foreground="white")])

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title(
        f"{A_NAME} × {B_NAME}\n"
        f"G={G_XY} µm grid, G_z={G_Z} µm, 8-Moore xy + ±{Z_DEPTH} z window\n"
        f"witness counts: A={int(A['is_witness'].sum())}/{len(A)}  "
        f"B={int(B['is_witness'].sum())}/{len(B)}    "
        f"shaded bins = host a witness tx of that side",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.93)
    plt.tight_layout()
    out = ZOOM_DIR / "zoom_pair_103590_55985.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
