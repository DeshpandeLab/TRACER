#!/usr/bin/env python3
"""Post-hoc analysis of the tile-parallel vs sequential ARI on PDAC ROI.

Reads the partitions saved by ``bench_pdac_roi_tiled_ari.py`` and
reports three refined ARI metrics for each tiled config:

  1. ARI with each unassigned tx as its OWN cluster (singletons).
     The previous "ARI full" metric grouped all unassigned tx into
     one mega-class, dominating the score. With singletons, two
     unassigned tx are never co-clustered, so the metric reflects
     real structural agreement.

  2. ARI on the UNION of "assigned in seq" OR "assigned in tiled"
     (i.e., tx that were assigned in at least one partition). More
     inclusive than the previous "restricted" intersection metric.
     Tx unassigned in only one partition contribute to the score
     via the singleton encoding.

  3. ARI on tile INTERIORS only — tx whose cell-centroid sits at
     least ``BUFFER_UM`` (default 20 µm) from the nearest tile-edge.
     Isolates the structural agreement away from boundary effects;
     should approach 1.0 if the tile-parallel mechanism is sound and
     boundaries are the only source of disagreement.

Run from this worktree root:

    PYTHONPATH=src:. python benchmarks/analyze_pdac_roi_tiled_ari.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/"
    "data/outs/transcripts.parquet"
)
ROI_CENTER = (7255.0, 3023.7)
ROI_HALF_SIDE = 1000.0
BUFFER_UM = 20.0

REPO = Path(__file__).resolve().parents[1]
IN_DIR = REPO / "benchmarks" / "pdac_roi_tiled_ari"

SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}


def _is_unassigned(s_series: pd.Series) -> np.ndarray:
    return (
        s_series.isin(SENTINELS)
        | s_series.str.endswith("_rejected", na=False)
    ).to_numpy()


def _codes_with_singletons(labels: pd.Series) -> np.ndarray:
    """Encode labels as int codes. Each unassigned tx gets its own
    unique cluster id so no two unassigneds are co-clustered."""
    is_un = _is_unassigned(labels)
    codes, _ = pd.factorize(labels.to_numpy(), sort=False)
    codes = codes.astype(np.int64)
    # Sentinel codes get unique negative ids: -2, -3, ... (avoid the
    # default "-1" code that pandas can produce for NaN).
    n = labels.size
    sentinel_ids = -2 - np.arange(int(is_un.sum()), dtype=np.int64)
    codes[is_un] = sentinel_ids
    return codes


def _compute_tile_interior_mask(
    tx_df: pd.DataFrame,
    n_tiles_xy: tuple[int, int],
    buffer_um: float,
) -> np.ndarray:
    """For each tx, return True if its cell_id's centroid is at least
    `buffer_um` from any tile edge under the given tile grid."""
    cent = tx_df.groupby("cell_id")[["x", "y"]].mean()
    x_min, x_max = float(cent["x"].min()), float(cent["x"].max())
    y_min, y_max = float(cent["y"].min()), float(cent["y"].max())
    n_x, n_y = n_tiles_xy
    x_edges = np.linspace(x_min, x_max + 1e-9, n_x + 1)
    y_edges = np.linspace(y_min, y_max + 1e-9, n_y + 1)

    # For each cell_id, distance to nearest tile edge in x and y.
    cell_x = cent["x"].to_numpy()
    cell_y = cent["y"].to_numpy()
    # For each cell, find the two nearest x_edges (one below, one above)
    # via searchsorted; same for y.
    ix = np.clip(np.searchsorted(x_edges, cell_x, side="right") - 1, 0, n_x - 1)
    iy = np.clip(np.searchsorted(y_edges, cell_y, side="right") - 1, 0, n_y - 1)
    dx_lo = cell_x - x_edges[ix]
    dx_hi = x_edges[ix + 1] - cell_x
    dy_lo = cell_y - y_edges[iy]
    dy_hi = y_edges[iy + 1] - cell_y
    # Distance to nearest edge (min of all 4)
    nearest_edge_um = np.minimum.reduce([dx_lo, dx_hi, dy_lo, dy_hi])
    cell_interior = nearest_edge_um >= buffer_um  # cell-level mask
    cell_interior_s = pd.Series(cell_interior, index=cent.index, name="interior")
    # Map back to tx
    return tx_df["cell_id"].map(cell_interior_s).fillna(False).to_numpy()


def main() -> int:
    print("Loading partitions ...", flush=True)
    seq_part = pd.read_parquet(IN_DIR / "partition_sequential.parquet")
    t22_part = pd.read_parquet(IN_DIR / "partition_tiled_2x2.parquet")
    t33_part = pd.read_parquet(IN_DIR / "partition_tiled_3x3.parquet")
    print(f"  sequential: {len(seq_part):,} rows")
    print(f"  tiled 2x2:  {len(t22_part):,} rows")
    print(f"  tiled 3x3:  {len(t33_part):,} rows")

    # Align tiled partitions to seq order via transcript_id (already done in
    # the bench, but double-check).
    seq_part = seq_part.set_index("transcript_id")
    t22_part = t22_part.set_index("transcript_id").reindex(seq_part.index)
    t33_part = t33_part.set_index("transcript_id").reindex(seq_part.index)

    # Load coordinates for the same tx set, for the interior mask.
    print("\nLoading original tx coords for interior mask ...", flush=True)
    df_raw = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "x_location", "y_location"],
    ).rename(columns={"x_location": "x", "y_location": "y"})
    xc, yc = ROI_CENTER
    h = ROI_HALF_SIDE
    mask = df_raw["x"].between(xc - h, xc + h) & df_raw["y"].between(yc - h, yc + h)
    df_raw = df_raw.loc[mask].reset_index(drop=True)
    df_raw = df_raw.set_index("transcript_id").reindex(seq_part.index).reset_index()

    seq_labels = seq_part["label"].astype(str)
    seq_un = _is_unassigned(seq_labels)

    summary: dict = {
        "n_tx": int(len(seq_part)),
        "n_seq_assigned": int((~seq_un).sum()),
        "buffer_um": BUFFER_UM,
        "configs": {},
    }

    for label, tiled_part, n_xy in [
        ("tiled_2x2", t22_part, (2, 2)),
        ("tiled_3x3", t33_part, (3, 3)),
    ]:
        print(f"\n--- {label} (n_tiles_xy={n_xy}) ---", flush=True)
        tiled_labels = tiled_part["label"].astype(str)
        tiled_un = _is_unassigned(tiled_labels)

        # (1) ARI with each unassigned tx as a unique singleton cluster.
        seq_codes_singleton = _codes_with_singletons(seq_labels)
        tiled_codes_singleton = _codes_with_singletons(tiled_labels)
        ari_singletons = float(
            adjusted_rand_score(seq_codes_singleton, tiled_codes_singleton)
        )
        print(f"  ARI (unassigned=singletons, all tx):   {ari_singletons:.4f}")

        # (2) ARI on union(seq_assigned, tiled_assigned). Unassigned tx
        # inside the union are still singletons; tx unassigned in BOTH
        # are dropped (they would be co-clustered as singletons in both,
        # which doesn't really test anything).
        assigned_union = (~seq_un) | (~tiled_un)
        s_codes_u = seq_codes_singleton[assigned_union]
        t_codes_u = tiled_codes_singleton[assigned_union]
        ari_union = float(adjusted_rand_score(s_codes_u, t_codes_u))
        n_union = int(assigned_union.sum())
        print(f"  ARI on union(seq, tiled) assigned tx:  {ari_union:.4f}   "
              f"(n={n_union:,})")

        # (3) ARI on tile interiors (>= BUFFER_UM from any tile edge).
        interior_mask = _compute_tile_interior_mask(df_raw, n_xy, BUFFER_UM)
        # Combine with union-assigned mask so we only score tx that
        # contribute meaningful signal.
        interior_union = interior_mask & assigned_union
        s_codes_i = seq_codes_singleton[interior_union]
        t_codes_i = tiled_codes_singleton[interior_union]
        if int(interior_union.sum()) >= 2:
            ari_interior = float(adjusted_rand_score(s_codes_i, t_codes_i))
        else:
            ari_interior = float("nan")
        n_interior = int(interior_union.sum())
        print(f"  ARI on tile-interior (≥{BUFFER_UM:.0f} µm buffer): {ari_interior:.4f}   "
              f"(n={n_interior:,}, {100*n_interior/n_union:.2f}% of union)")

        summary["configs"][label] = {
            "ari_singletons": ari_singletons,
            "ari_union": ari_union,
            "ari_interior": ari_interior,
            "n_union_assigned": n_union,
            "n_interior_union": n_interior,
        }

    out_path = IN_DIR / "ari_refined_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsummary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
