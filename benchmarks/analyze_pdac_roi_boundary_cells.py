#!/usr/bin/env python3
"""Geometry-only identification of boundary-at-risk cells in the PDAC ROI.

A Xenium cell_id is "boundary-at-risk" if its spatial extent gets near
a tile boundary, since the tile-parallel orchestrator's cell-centroid
assignment puts every cell wholly in one tile — denying Stitch any
cross-tile candidate pair. Cells far from any boundary cannot suffer
this loss.

This script flags cells purely from geometry (no SEG output involved):
  - per-cell centroid (mean x, y of its tx)
  - per-cell footprint bbox (min/max x, y of its tx)
  - per-cell footprint radius (max distance from centroid to any tx)
  - per-cell min distance to the nearest tile edge

Reports the distribution of these distances and counts at various Δ
thresholds, so we can compare against the actually-disagreeing cells
(measured separately) to validate.

Also cross-references against the saved sequential & tiled partitions
to report: of cells classified as at-risk (centroid Δ < 10µm), how
many do we actually see disagree in tile-parallel output?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
    assigned = ~df["cell_id"].isin(SENTINELS)
    df_a = df.loc[assigned].copy()
    print(f"  tx: {len(df):,} ({int(assigned.sum()):,} with assigned cell_id)", flush=True)
    print(f"  unique input cell_ids: {df_a['cell_id'].nunique():,}", flush=True)

    # 1. Cell-level geometry
    grp = df_a.groupby("cell_id")
    cell_geom = pd.DataFrame({
        "x_cent": grp["x"].mean(),
        "y_cent": grp["y"].mean(),
        "x_min": grp["x"].min(),
        "x_max": grp["x"].max(),
        "y_min": grp["y"].min(),
        "y_max": grp["y"].max(),
        "n_tx": grp.size(),
    })

    # 2. Tile edges (cell-centroid based assignment — same as orchestrator)
    x_lo, x_hi = float(cell_geom["x_cent"].min()), float(cell_geom["x_cent"].max())
    y_lo, y_hi = float(cell_geom["y_cent"].min()), float(cell_geom["y_cent"].max())
    n_x, n_y = N_TILES_XY
    x_edges = np.linspace(x_lo, x_hi + 1e-9, n_x + 1)
    y_edges = np.linspace(y_lo, y_hi + 1e-9, n_y + 1)
    print(f"\n  tile x_edges: {x_edges.round(1).tolist()}", flush=True)
    print(f"  tile y_edges: {y_edges.round(1).tolist()}", flush=True)

    # 3. Distance from centroid to nearest INTERIOR tile edge
    # (Interior edges only — outer ROI bounds aren't tile boundaries.)
    interior_x = x_edges[1:-1]
    interior_y = y_edges[1:-1]

    def _min_edge_dist(c, edges):
        if len(edges) == 0:
            return np.full(len(c), np.inf)
        return np.abs(c[:, None] - edges[None, :]).min(axis=1)

    cx = cell_geom["x_cent"].to_numpy()
    cy = cell_geom["y_cent"].to_numpy()
    dx_edge = _min_edge_dist(cx, interior_x)
    dy_edge = _min_edge_dist(cy, interior_y)
    cell_geom["centroid_dist_to_tile_edge"] = np.minimum(dx_edge, dy_edge)

    # 4. Cell footprint extent: how far does the cell's tx reach from its centroid?
    # max_tx_dist = max sqrt((x_tx - x_cent)**2 + (y_tx - y_cent)**2)
    # Cheaper proxy: half-bbox-diagonal
    half_diag = 0.5 * np.sqrt(
        (cell_geom["x_max"] - cell_geom["x_min"]) ** 2
        + (cell_geom["y_max"] - cell_geom["y_min"]) ** 2
    ).to_numpy()
    cell_geom["footprint_half_diag"] = half_diag

    # 5. Distance from CELL BBOX to nearest interior tile edge.
    # A cell's tx footprint can straddle a tile edge even if centroid doesn't.
    def _bbox_edge_dist(lo, hi, edges):
        # For each edge e, distance from interval [lo,hi] to e:
        # =0 if e in [lo,hi], else min(|lo-e|, |hi-e|)
        if len(edges) == 0:
            return np.full(len(lo), np.inf)
        elo = edges[None, :] - lo[:, None]  # +ve if edge to right of lo
        ehi = edges[None, :] - hi[:, None]
        # interval straddles e iff (lo <= e <= hi) iff (elo >= 0 and ehi <= 0)
        straddle = (elo >= 0) & (ehi <= 0)
        # else distance is min(|lo-e|, |hi-e|)
        dd = np.minimum(np.abs(elo), np.abs(ehi))
        dd[straddle] = 0.0
        return dd.min(axis=1)

    dx_bbox = _bbox_edge_dist(cell_geom["x_min"].to_numpy(), cell_geom["x_max"].to_numpy(), interior_x)
    dy_bbox = _bbox_edge_dist(cell_geom["y_min"].to_numpy(), cell_geom["y_max"].to_numpy(), interior_y)
    cell_geom["bbox_dist_to_tile_edge"] = np.minimum(dx_bbox, dy_bbox)

    # ----------------------------------------------------------------
    # Distribution summaries
    # ----------------------------------------------------------------
    print(f"\nDistance distributions (over {len(cell_geom):,} cells):")
    print(f"  {'metric':35s}  p10  p25  p50  p75  p90  p99   max")
    for col in ["centroid_dist_to_tile_edge", "bbox_dist_to_tile_edge",
                 "footprint_half_diag"]:
        v = cell_geom[col].to_numpy()
        q = np.percentile(v, [10, 25, 50, 75, 90, 99])
        print(f"  {col:35s}  " + "  ".join(f"{x:>4.1f}" for x in q) + f"   {v.max():>5.1f}")

    print(f"\nCounts at various boundary thresholds:")
    print(f"  {'Δ (µm)':>7s}  {'centroid<Δ':>14s}  {'bbox<Δ':>10s}  {'% of cells':>10s}")
    for delta in [2, 5, 10, 20, 50, 100]:
        n_cent = int((cell_geom["centroid_dist_to_tile_edge"] < delta).sum())
        n_bbox = int((cell_geom["bbox_dist_to_tile_edge"] < delta).sum())
        print(f"  {delta:>7d}  {n_cent:>14,}  {n_bbox:>10,}  "
              f"{100*n_bbox/len(cell_geom):>9.2f}%")

    # ----------------------------------------------------------------
    # Cross-reference: of cells flagged as at-risk, how many actually disagree?
    # ----------------------------------------------------------------
    seq = pd.read_parquet(SEQ_PART).set_index("transcript_id").reindex(df["transcript_id"]).reset_index()
    naive = pd.read_parquet(NAIVE_PART).set_index("transcript_id").reindex(df["transcript_id"]).reset_index()

    # For each cell_id, does any tx disagree between seq and naive?
    disagree_tx = (seq["label"].astype(str) != naive["label"].astype(str)).to_numpy()
    df_a["disagrees"] = disagree_tx[assigned.to_numpy()]
    cell_disagree = df_a.groupby("cell_id")["disagrees"].any()
    cell_geom["disagrees"] = cell_disagree.reindex(cell_geom.index).fillna(False)

    print(f"\nDisagreement vs at-risk geometry:")
    n_cells = len(cell_geom)
    n_disagree = int(cell_geom["disagrees"].sum())
    print(f"  Total cells: {n_cells:,}")
    print(f"  Cells with ≥1 disagreeing tx: {n_disagree:,} ({100*n_disagree/n_cells:.1f}%)")
    print()
    print(f"  {'Δ (µm)':>7s}  {'at-risk':>9s}  {'AR & disagree':>14s}  "
          f"{'sensitivity':>11s}  {'precision':>9s}")
    for delta in [2, 5, 10, 20, 50, 100, 200]:
        at_risk = cell_geom["bbox_dist_to_tile_edge"] < delta
        n_ar = int(at_risk.sum())
        n_ar_dis = int((at_risk & cell_geom["disagrees"]).sum())
        sens = n_ar_dis / max(n_disagree, 1)
        prec = n_ar_dis / max(n_ar, 1)
        print(f"  {delta:>7d}  {n_ar:>9,}  {n_ar_dis:>14,}  "
              f"{100*sens:>10.1f}%  {100*prec:>8.1f}%")

    out_path = REPO / "benchmarks" / "pdac_roi_global_stitch" / "boundary_cell_geometry.parquet"
    cell_geom.reset_index().to_parquet(out_path, index=False)
    print(f"\nsaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
