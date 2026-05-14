#!/usr/bin/env python3
"""Zoom into 50um ROIs where SEG and NOSEG-nontrivial diverge most / least.

For each input cell_id in the 2x2mm PDAC ROI:
  - Compute SEG modal cell label + n_tx mapped to it
  - Compute NOSEG entity labels its tx map to + per-entity tx count
  - "fragmentation_index" = n_tx_in_modal_NOSEG_entity / n_tx_total
    (1.0 = perfect match; 0 = max scatter)

Pick:
  - best-case: cell with high frag_idx, sized 10-25 tx (clean match)
  - worst-case: cell with low frag_idx, in dense region (severe scatter)

For each picked cell, define a 50um square ROI centered on it.
Within each ROI:
  - List all input cell_ids
  - For each, show how SEG and NOSEG-nontrivial label their tx
  - Side-by-side scatter plot colored by SEG entity / NOSEG entity
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
# Partitions live in the stoic-feynman-587f37 worktree (where the bench ran).
D = Path("/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
         "stoic-feynman-587f37/benchmarks/pdac_noseg_nontrivial_vs_seg")
PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
ROI_CENTER = (7255.0, 3023.7)
ROI_HALF_SIDE = 1000.0
ZOOM_HALF_SIDE = 25.0  # 50um square
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}


def main() -> int:
    print("loading partitions + tx coords ...", flush=True)
    seg = pd.read_parquet(D / "partition_seg.parquet").set_index("transcript_id")
    noseg = pd.read_parquet(D / "partition_noseg_1over9.parquet").set_index("transcript_id")
    # Note filename says "1over9" but contents are the non-trivial-fraction rule
    # (OUT_DIR was renamed but the partition filename kept the legacy stem)
    assert (seg.index == noseg.index).all()

    raw = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "x_location", "y_location"],
    ).rename(columns={"x_location": "x", "y_location": "y"})
    xc, yc = ROI_CENTER; h = ROI_HALF_SIDE
    mask = raw["x"].between(xc - h, xc + h) & raw["y"].between(yc - h, yc + h)
    raw = raw.loc[mask].reset_index(drop=True)
    assert len(raw) == len(seg)
    raw["cell_id"] = raw["cell_id"].astype(str)
    raw["seg_lab"] = seg.loc[raw["transcript_id"]]["label"].astype(str).to_numpy()
    raw["noseg_lab"] = noseg.loc[raw["transcript_id"]]["label"].astype(str).to_numpy()

    # Keep only tx where input cell_id is real
    df = raw.loc[~raw["cell_id"].isin(SENTINELS)].reset_index(drop=True)
    print(f"  total assigned-input tx: {len(df):,}  cells: {df['cell_id'].nunique():,}",
          flush=True)

    # Per-cell summary: count tx by NOSEG label, get modal entity + fragmentation index
    by_cell = df.groupby("cell_id")
    rows = []
    for cid, g in by_cell:
        n_tx = len(g)
        if n_tx < 8 or n_tx > 200:
            continue  # we want medium-sized cells in our zoom
        x_cent, y_cent = g["x"].mean(), g["y"].mean()
        noseg_counts = g["noseg_lab"].value_counts()
        seg_counts = g["seg_lab"].value_counts()
        # Modal NOSEG entity (counted only over non-sentinel)
        noseg_assigned = noseg_counts[~noseg_counts.index.isin(SENTINELS)]
        if len(noseg_assigned) == 0:
            continue
        modal_noseg = noseg_assigned.index[0]
        modal_count = int(noseg_assigned.iloc[0])
        frag_idx = modal_count / n_tx
        rows.append({
            "cell_id": cid, "n_tx": n_tx, "x_cent": x_cent, "y_cent": y_cent,
            "seg_n_entities": int(seg_counts[~seg_counts.index.isin(SENTINELS)].size),
            "noseg_n_entities": int(noseg_assigned.size),
            "frag_idx": frag_idx,
            "modal_noseg": modal_noseg,
        })
    cell_df = pd.DataFrame(rows)
    print(f"  candidate cells (8 <= n_tx <= 200): {len(cell_df):,}", flush=True)

    # Best case: high frag_idx (≥0.95), prefer larger tx counts
    best_pool = cell_df[(cell_df["frag_idx"] >= 0.95) & (cell_df["n_tx"] >= 30)]
    if len(best_pool) == 0:
        best_pool = cell_df[(cell_df["frag_idx"] >= 0.90)]
    best = best_pool.sort_values("n_tx", ascending=False).head(20).sample(
        1, random_state=0
    ).iloc[0]
    # Worst case: low frag_idx, lots of scatter
    worst_pool = cell_df[(cell_df["frag_idx"] <= 0.40) & (cell_df["noseg_n_entities"] >= 4)]
    if len(worst_pool) == 0:
        worst_pool = cell_df[cell_df["frag_idx"] <= 0.50]
    worst = worst_pool.sort_values("frag_idx").head(20).sample(
        1, random_state=0
    ).iloc[0]

    OUT_DIR = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nBEST  cell_id={best['cell_id']}  n_tx={int(best['n_tx'])}  "
          f"frag_idx={best['frag_idx']:.3f}  "
          f"noseg_n_entities={int(best['noseg_n_entities'])}  "
          f"@ ({best['x_cent']:.1f}, {best['y_cent']:.1f})", flush=True)
    print(f"WORST cell_id={worst['cell_id']}  n_tx={int(worst['n_tx'])}  "
          f"frag_idx={worst['frag_idx']:.3f}  "
          f"noseg_n_entities={int(worst['noseg_n_entities'])}  "
          f"@ ({worst['x_cent']:.1f}, {worst['y_cent']:.1f})", flush=True)

    # For each case, build the 50um ROI and produce a side-by-side plot + table.
    for label, row in [("best", best), ("worst", worst)]:
        xc_z, yc_z = row["x_cent"], row["y_cent"]
        zmask = (raw["x"].between(xc_z - ZOOM_HALF_SIDE, xc_z + ZOOM_HALF_SIDE)
                  & raw["y"].between(yc_z - ZOOM_HALF_SIDE, yc_z + ZOOM_HALF_SIDE))
        sub = raw.loc[zmask].copy()
        print(f"\n=== ZOOM ({label}) @ ({xc_z:.1f}, {yc_z:.1f}) ±{ZOOM_HALF_SIDE}µm ===",
              flush=True)
        print(f"  tx in ROI: {len(sub):,}", flush=True)
        sub_assigned = sub[~sub["cell_id"].isin(SENTINELS)]
        n_cells = sub_assigned["cell_id"].nunique()
        print(f"  unique input cell_ids: {n_cells:,}", flush=True)

        # Per-cell summary in this ROI
        per_cell = sub_assigned.groupby("cell_id").agg(
            n_tx=("transcript_id", "size"),
            n_seg_ents=("seg_lab", lambda s: s[~s.isin(SENTINELS)].nunique()),
            n_noseg_ents=("noseg_lab", lambda s: s[~s.isin(SENTINELS)].nunique()),
            modal_seg=("seg_lab", lambda s: s[~s.isin(SENTINELS)].mode().iloc[0] if (~s.isin(SENTINELS)).any() else "-1"),
            modal_noseg=("noseg_lab", lambda s: s[~s.isin(SENTINELS)].mode().iloc[0] if (~s.isin(SENTINELS)).any() else "-1"),
        ).reset_index()
        per_cell["fragmentation"] = per_cell["n_noseg_ents"]
        per_cell = per_cell.sort_values("fragmentation", ascending=False)
        print(f"\n  {'cell_id':>20s}  {'n_tx':>5s}  {'seg_ents':>8s}  "
              f"{'noseg_ents':>10s}  modal_seg → modal_noseg", flush=True)
        for _, r in per_cell.iterrows():
            print(f"  {r['cell_id']:>20s}  {int(r['n_tx']):>5d}  "
                  f"{int(r['n_seg_ents']):>8d}  {int(r['n_noseg_ents']):>10d}  "
                  f"{r['modal_seg']} → {r['modal_noseg']}", flush=True)

        # Plot: side by side scatter, colored by SEG entity vs NOSEG entity
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=130)
        # Pick a stable color per entity using factorize
        for ax, col, name in [
            (axes[0], "seg_lab", "SEG"),
            (axes[1], "noseg_lab", "NOSEG (non-trivial)"),
        ]:
            labels_series = sub[col].astype(str)
            assigned_mask = ~labels_series.isin(SENTINELS)
            # plot unassigned as light gray first
            unassigned = ~assigned_mask
            if unassigned.any():
                ax.scatter(sub.loc[unassigned, "x"], sub.loc[unassigned, "y"],
                            c="#cccccc", s=12, alpha=0.5, label="unassigned",
                            linewidths=0)
            cats, codes = np.unique(labels_series[assigned_mask], return_inverse=True)
            n_cats = len(cats)
            cmap = plt.get_cmap("tab20" if n_cats <= 20 else "gist_ncar", max(n_cats, 1))
            for k, cat in enumerate(cats):
                sel = assigned_mask & (labels_series == cat)
                ax.scatter(sub.loc[sel, "x"], sub.loc[sel, "y"],
                            c=[cmap(k % cmap.N)], s=18, alpha=0.8,
                            label=f"{cat[:18]} ({int(sel.sum())})", linewidths=0)
            ax.set_title(f"{name}  ({n_cats} entities)", fontsize=11)
            ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
            ax.set_xlim(xc_z - ZOOM_HALF_SIDE, xc_z + ZOOM_HALF_SIDE)
            ax.set_ylim(yc_z - ZOOM_HALF_SIDE, yc_z + ZOOM_HALF_SIDE)
            ax.set_aspect("equal")
            ax.spines[["top", "right"]].set_visible(False)
            if n_cats <= 15:
                ax.legend(loc="upper right", fontsize=6, markerscale=2,
                           framealpha=0.85)
        plt.suptitle(f"PDAC zoom — {label} case @ "
                      f"({xc_z:.1f}, {yc_z:.1f})  "
                      f"({len(sub):,} tx, {n_cells} input cells, "
                      f"frag_idx={row['frag_idx']:.2f})",
                      fontsize=12, y=1.0)
        plt.tight_layout()
        out = OUT_DIR / f"zoom_{label}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out}", flush=True)

        # Persist tx data for the ROI
        sub.to_parquet(OUT_DIR / f"zoom_{label}_tx.parquet", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
