#!/usr/bin/env python3
"""Geometry-only identification of boundary-affected cell PAIRS.

A pair (A, B) of input cells is "boundary-blocked" if their centroids
are within Stitch reach (Δ µm) but the orchestrator assigns them to
different tiles. These are exactly the candidate Stitch merges that
tile-parallel CANNOT make.

For each (A, B) in this set:
  - both cells (singly) get flagged as boundary-blocked
  - the pair gets flagged for the boundary-cleanup pass

Reports:
  - number of pairs at various Δ thresholds
  - number of unique cells involved
  - cross-reference: how many of these are actually merged together
    in the sequential reference?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
REPO = Path(__file__).resolve().parents[1]
SEQ_PART = REPO / "benchmarks" / "pdac_roi_tile_postprocess" / "partition_sequential.parquet"
NAIVE_PART = REPO / "benchmarks" / "pdac_roi_global_stitch" / "partition_naive.parquet"
ROI_CENTER = (7255.0, 3023.7)
ROI_HALF_SIDE = 1000.0
N_TILES_XY = (3, 3)
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}


def main() -> int:
    print("loading ROI tx ...", flush=True)
    df = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "x_location", "y_location"],
    ).rename(columns={"x_location": "x", "y_location": "y"})
    xc, yc = ROI_CENTER; h = ROI_HALF_SIDE
    mask = df["x"].between(xc - h, xc + h) & df["y"].between(yc - h, yc + h)
    df = df.loc[mask].reset_index(drop=True)
    df["cell_id"] = df["cell_id"].astype(str)
    df_a = df.loc[~df["cell_id"].isin(SENTINELS)].copy()

    # Centroids + tile assignment (cell-centroid based, matches orchestrator)
    cent = df_a.groupby("cell_id")[["x", "y"]].mean()
    n_x, n_y = N_TILES_XY
    x_edges = np.linspace(cent["x"].min(), cent["x"].max() + 1e-9, n_x + 1)
    y_edges = np.linspace(cent["y"].min(), cent["y"].max() + 1e-9, n_y + 1)
    ix = np.clip(np.searchsorted(x_edges, cent["x"], side="right") - 1, 0, n_x - 1)
    iy = np.clip(np.searchsorted(y_edges, cent["y"], side="right") - 1, 0, n_y - 1)
    cent["tile"] = ix * n_y + iy
    print(f"  {len(cent):,} cells across {N_TILES_XY[0]}×{N_TILES_XY[1]} tiles", flush=True)

    tree = cKDTree(cent[["x", "y"]].to_numpy())

    # Find all cell pairs (i, j) with centroid distance < Δ
    # Use cKDTree.query_pairs which yields unordered (i, j) with i < j.
    print(f"\nFlagging boundary-blocked pairs:", flush=True)
    print(f"  {'Δ (µm)':>7s}  {'all_pairs':>11s}  {'cross_tile':>12s}  "
          f"{'cross_tile %':>12s}  {'unique_cells_in_cross':>22s}  "
          f"{'% all cells':>11s}")
    cell_to_tile_arr = cent["tile"].to_numpy()
    cell_ids = cent.index.to_numpy()
    cross_pairs_at_10 = None
    for delta in [2, 5, 10, 15, 20, 30, 50]:
        pairs = tree.query_pairs(r=delta, output_type="ndarray")
        if len(pairs) == 0:
            continue
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        cross = cell_to_tile_arr[i_idx] != cell_to_tile_arr[j_idx]
        n_cross = int(cross.sum())
        cells_involved = np.unique(np.concatenate([i_idx[cross], j_idx[cross]]))
        print(f"  {delta:>7d}  {len(pairs):>11,}  {n_cross:>12,}  "
              f"{100*n_cross/max(len(pairs),1):>11.1f}%  "
              f"{len(cells_involved):>22,}  "
              f"{100*len(cells_involved)/len(cent):>10.1f}%")
        if delta == 10:
            cross_pairs_at_10 = (i_idx[cross], j_idx[cross])

    # Cross-reference at Δ=10µm: of the cross-tile pairs, how many
    # are merged together in sequential (= a real merge that
    # tile-parallel can't make)?
    if cross_pairs_at_10 is None:
        return 0
    seq = pd.read_parquet(SEQ_PART).set_index("transcript_id").reindex(df["transcript_id"]).reset_index()
    seq_lab = seq["label"].astype(str)
    naive = pd.read_parquet(NAIVE_PART).set_index("transcript_id").reindex(df["transcript_id"]).reset_index()
    naive_lab = naive["label"].astype(str)

    # Per-cell modal seq-label (assigned tx only)
    df_a2 = df_a.assign(
        seq_lab=seq_lab.loc[~df["cell_id"].isin(SENTINELS)].to_numpy(),
        naive_lab=naive_lab.loc[~df["cell_id"].isin(SENTINELS)].to_numpy(),
    )
    def _modal(s):
        return s.value_counts().index[0] if len(s) else "-1"
    cell_seq_lab = df_a2.groupby("cell_id")["seq_lab"].agg(_modal)
    cell_naive_lab = df_a2.groupby("cell_id")["naive_lab"].agg(_modal)
    cell_seq_lab_arr = cell_seq_lab.reindex(cell_ids).to_numpy()
    cell_naive_lab_arr = cell_naive_lab.reindex(cell_ids).to_numpy()

    i_idx, j_idx = cross_pairs_at_10
    same_seq = cell_seq_lab_arr[i_idx] == cell_seq_lab_arr[j_idx]
    same_naive = cell_naive_lab_arr[i_idx] == cell_naive_lab_arr[j_idx]
    n_pairs = len(i_idx)
    n_real_merge = int(same_seq.sum())
    n_blocked = int((same_seq & ~same_naive).sum())
    print(f"\nAt Δ=10µm cross-tile pairs (n={n_pairs:,}):")
    print(f"  Merged in seq (would have merged):        {n_real_merge:,} ({100*n_real_merge/n_pairs:.1f}%)")
    print(f"  Real merges that tile-parallel BLOCKED:   {n_blocked:,} ({100*n_blocked/n_pairs:.1f}%)")
    print(f"  Same label in tile-parallel:              "
          f"{int(same_naive.sum()):,} ({100*same_naive.sum()/n_pairs:.1f}%)")

    # Affected cells (unique) among blocked
    blocked_cell_set = np.unique(np.concatenate([
        i_idx[same_seq & ~same_naive],
        j_idx[same_seq & ~same_naive],
    ]))
    print(f"  Unique cells in blocked merges:           {len(blocked_cell_set):,}")
    print(f"  Of all 33,602 ROI cells, {100*len(blocked_cell_set)/len(cent):.1f}% are in a blocked-merge pair.")

    out_path = REPO / "benchmarks" / "pdac_roi_global_stitch" / "boundary_blocked_pairs.parquet"
    pd.DataFrame({
        "cell_a": cell_ids[i_idx],
        "cell_b": cell_ids[j_idx],
        "tile_a": cell_to_tile_arr[i_idx],
        "tile_b": cell_to_tile_arr[j_idx],
        "merged_in_seq": same_seq,
        "same_in_tiled": same_naive,
    }).to_parquet(out_path, index=False)
    print(f"\nsaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
