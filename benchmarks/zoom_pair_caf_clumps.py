#!/usr/bin/env python3
"""Plot the two CAF Stitch components on the G=2.0 µm grid:
    cascade_103636-1-1  =  {103636, 5180}                 n=32
    cascade_135869-1-2  =  {135796, 135869-1-1, 77253-1-1} n=28

Both end up in SEG's nloapcgp-1-1 (CAF). Question: why didn't Stitch
merge them with each other? Plot their tx with + / x, overlay the
grid, annotate per-bin counts, and report the post-merge witness +
coherence values for this pair.
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
from matplotlib import patheffects as _patheffects

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tracer.stitching import apply_stitching_to_transcripts_memory_efficient

PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
ZOOM_DIR = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg"

PMI_THR = 0.2
G_XY = 2.0
G_Z = 1.0
Z_DEPTH = 1
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}

A_FINAL = "cascade_103636-1-1"
B_FINAL = "cascade_135869-1-2"
A_LABEL = "103636-1-1"
B_LABEL = "135869-1-2"
A_COLOR = "#e377c2"   # pink (CAF clump 1)
B_COLOR = "#8c564b"   # brown (CAF clump 2)


def _build_W(panel_path, all_genes):
    panel = pd.read_parquet(panel_path).rename(columns={"value": "NPMI"})
    panel["gene_i"] = panel["gene_i"].astype(str)
    panel["gene_j"] = panel["gene_j"].astype(str)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    G = len(all_genes)
    W = np.full((G, G), np.nan, dtype=np.float32)
    gi = panel["gene_i"].map(gene_to_idx)
    gj = panel["gene_j"].map(gene_to_idx)
    have = gi.notna() & gj.notna()
    gi = gi[have].to_numpy(dtype=np.int64)
    gj = gj[have].to_numpy(dtype=np.int64)
    v = panel.loc[have, "NPMI"].to_numpy(dtype=np.float32)
    W[gi, gj] = v
    W[gj, gi] = v
    np.fill_diagonal(W, np.nan)
    return W, gene_to_idx


def _coherence(gene_set, W, g2i, tau=PMI_THR):
    gids = [g2i[g] for g in gene_set if g in g2i]
    if len(gids) < 2:
        return float("nan"), 0
    gids = np.asarray(sorted(set(gids)), dtype=np.int64)
    k = len(gids)
    sub = W[np.ix_(gids, gids)]
    iu = np.triu_indices(k, k=1)
    w = sub[iu]
    w = w[~np.isnan(w)]
    if w.size == 0:
        return float("nan"), k
    purity = float((w > tau).mean())
    conflict = float((w < -tau).mean())
    return purity - conflict, k


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
    feats = pd.read_parquet(
        PDAC_PARQUET, columns=["transcript_id", "feature_name"]
    )
    z_col = pd.read_parquet(
        PDAC_PARQUET, columns=["transcript_id", "z_location"]
    ).rename(columns={"z_location": "z"})
    df = zoom.merge(feats, on="transcript_id", how="left").merge(
        z_col, on="transcript_id", how="left"
    )
    df["feature_name"] = df["feature_name"].astype(str)
    df["noseg_lab"] = df["noseg_lab"].astype(str)

    cell = df[df["cell_id"].astype(str) == "nloapcgp-1"].copy().reset_index(
        drop=True
    )
    cell["tracer_id"] = cell["noseg_lab"]
    cell.loc[cell["tracer_id"].isin(SENTINELS), "tracer_id"] = "-1"

    panel_raw = pd.read_parquet(PANEL)
    all_genes = sorted(
        set(panel_raw["gene_i"].astype(str))
        | set(panel_raw["gene_j"].astype(str))
        | set(cell["feature_name"].unique())
    )
    W, g2i = _build_W(PANEL, all_genes)
    aux = {"W": W, "gene_to_idx": g2i}

    # Re-run production Stitch (same settings as zoom_stitch_roi.py).
    df_stitched, _ = apply_stitching_to_transcripts_memory_efficient(
        df_final=cell, aux=aux,
        entity_col="tracer_id", gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        mode="count", threshold=PMI_THR, metric="pmi",
        penalize_simplicity=True, deltaC_min=0.03,
        c_union_bypass=0.9,
        dist_threshold=5.0, out_col="stitched", show_progress=False,
        candidate_source="grid", G=G_XY, stitch_neighborhood="8",
        G_z=G_Z, z_neighbor_depth=Z_DEPTH,
        min_local_tx_per_entity=3,
    )
    df_stitched["stitched"] = df_stitched["stitched"].astype(str)

    A = df_stitched[df_stitched["stitched"] == A_FINAL].copy()
    B = df_stitched[df_stitched["stitched"] == B_FINAL].copy()
    print(f"{A_FINAL}: {len(A)} tx", flush=True)
    print(f"{B_FINAL}: {len(B)} tx", flush=True)

    for d in (A, B):
        d["xb"] = np.floor(d["x"].to_numpy() / G_XY).astype(np.int64)
        d["yb"] = np.floor(d["y"].to_numpy() / G_XY).astype(np.int64)
        d["zb"] = np.floor(d["z"].to_numpy() / G_Z).astype(np.int64)

    A_bins_3d = set(zip(A["xb"], A["yb"], A["zb"]))
    B_bins_3d = set(zip(B["xb"], B["yb"], B["zb"]))

    nbrs = [(dx, dy, dz)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for dz in range(-Z_DEPTH, Z_DEPTH + 1)]

    A["is_witness"] = False
    for i, row in A.iterrows():
        for dx, dy, dz in nbrs:
            if (row["xb"] + dx, row["yb"] + dy, row["zb"] + dz) in B_bins_3d:
                A.at[i, "is_witness"] = True
                break
    B["is_witness"] = False
    for i, row in B.iterrows():
        for dx, dy, dz in nbrs:
            if (row["xb"] + dx, row["yb"] + dy, row["zb"] + dz) in A_bins_3d:
                B.at[i, "is_witness"] = True
                break

    n_wit_A = int(A["is_witness"].sum())
    n_wit_B = int(B["is_witness"].sum())
    print(f"witnesses: A={n_wit_A}/{len(A)}  B={n_wit_B}/{len(B)}", flush=True)

    # Coherence values for the merged components and their hypothetical union
    A_genes = set(A["feature_name"].unique())
    B_genes = set(B["feature_name"].unique())
    C_A, kA = _coherence(A_genes, W, g2i)
    C_B, kB = _coherence(B_genes, W, g2i)
    C_U, kU = _coherence(A_genes | B_genes, W, g2i)
    # ΔC under penalize_simplicity
    nA = max(kA, 1); nB = max(kB, 1); nU = max(kU, 1)
    dC_raw = C_U - max(C_A, C_B)
    C_sep_pen = max(C_A - 1.0 / nA, C_B - 1.0 / nB)
    dC_pen = (C_U - 1.0 / nU) - C_sep_pen
    print(f"\nCoherence:", flush=True)
    print(f"  {A_LABEL}:   C={C_A:.4f}  n_genes={kA}", flush=True)
    print(f"  {B_LABEL}:   C={C_B:.4f}  n_genes={kB}", flush=True)
    print(f"  union:           C={C_U:.4f}  n_genes={kU}", flush=True)
    print(f"  ΔC_raw = {dC_raw:+.4f}", flush=True)
    print(f"  ΔC_pen = {dC_pen:+.4f}", flush=True)
    print(f"  C(union) ≥ 0.9 bypass: "
          f"{'PASS' if C_U >= 0.9 else 'FAIL'}", flush=True)
    print(f"  ΔC ≥ 0.03 gate:        "
          f"{'PASS' if dC_pen >= 0.03 else 'FAIL'}", flush=True)
    print(f"  witness ≥ 3 gate:      "
          f"{'PASS' if (n_wit_A >= min(3, len(A)) and n_wit_B >= min(3, len(B))) else 'FAIL'}", flush=True)

    # Plot
    all_x = np.concatenate([A["x"].to_numpy(), B["x"].to_numpy()])
    all_y = np.concatenate([A["y"].to_numpy(), B["y"].to_numpy()])
    pad = 2.0
    xmin = np.floor((all_x.min() - pad) / G_XY) * G_XY
    xmax = np.ceil((all_x.max() + pad) / G_XY) * G_XY
    ymin = np.floor((all_y.min() - pad) / G_XY) * G_XY
    ymax = np.ceil((all_y.max() + pad) / G_XY) * G_XY

    fig, ax = plt.subplots(figsize=(11, 11), dpi=140)
    _draw_grid(ax, xmin, xmax, ymin, ymax)

    A_wit_bins = set(zip(A.loc[A["is_witness"], "xb"],
                          A.loc[A["is_witness"], "yb"]))
    B_wit_bins = set(zip(B.loc[B["is_witness"], "xb"],
                          B.loc[B["is_witness"], "yb"]))
    for (xb, yb) in A_wit_bins:
        ax.add_patch(Rectangle((xb * G_XY, yb * G_XY), G_XY, G_XY,
                                facecolor=A_COLOR, alpha=0.12, zorder=1,
                                edgecolor="none"))
    for (xb, yb) in B_wit_bins:
        ax.add_patch(Rectangle((xb * G_XY, yb * G_XY), G_XY, G_XY,
                                facecolor=B_COLOR, alpha=0.12, zorder=1,
                                edgecolor="none"))

    for is_wit, lw, alpha in [(False, 1.6, 0.55), (True, 2.6, 1.0)]:
        Ai = A[A["is_witness"] == is_wit]
        ax.scatter(Ai["x"], Ai["y"], marker="+", s=320, c=A_COLOR,
                    linewidths=lw, alpha=alpha, zorder=4,
                    label=(f"A = {A_LABEL}  (witness, n={n_wit_A})"
                           if is_wit else
                           f"A = {A_LABEL}  (non-witness, n={len(A)-n_wit_A})"))
    for is_wit, lw, alpha in [(False, 1.6, 0.55), (True, 2.6, 1.0)]:
        Bi = B[B["is_witness"] == is_wit]
        ax.scatter(Bi["x"], Bi["y"], marker="x", s=240, c=B_COLOR,
                    linewidths=lw, alpha=alpha, zorder=4,
                    label=(f"B = {B_LABEL}  (witness, n={n_wit_B})"
                           if is_wit else
                           f"B = {B_LABEL}  (non-witness, n={len(B)-n_wit_B})"))

    # Per-bin counts (2D projection)
    bin_counts = {}
    for _, row in A.iterrows():
        k = (int(row["xb"]), int(row["yb"]))
        bin_counts.setdefault(k, [0, 0])
        bin_counts[k][0] += 1
    for _, row in B.iterrows():
        k = (int(row["xb"]), int(row["yb"]))
        bin_counts.setdefault(k, [0, 0])
        bin_counts[k][1] += 1
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
    verdict = []
    if n_wit_A < min(3, len(A)) or n_wit_B < min(3, len(B)):
        verdict.append("WITNESS FAIL")
    if dC_pen < 0.03 and C_U < 0.9:
        verdict.append("ΔC FAIL + bypass FAIL")
    elif dC_pen < 0.03:
        verdict.append("ΔC FAIL (bypass would save)")
    if not verdict:
        verdict.append("would merge")
    ax.set_title(
        f"{A_FINAL} × {B_FINAL}\n"
        f"G={G_XY} µm grid, 8-Moore + ±{Z_DEPTH} z window\n"
        f"witnesses A={n_wit_A}/{len(A)}, B={n_wit_B}/{len(B)}    "
        f"C_uni={C_U:.3f}  ΔC_pen={dC_pen:+.4f}    "
        f"verdict: {'; '.join(verdict)}",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.93)
    plt.tight_layout()
    out = ZOOM_DIR / "zoom_pair_caf_clumps.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
