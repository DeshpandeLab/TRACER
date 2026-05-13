#!/usr/bin/env python3
"""ARI agreement between sequential and tile-parallel SEG outputs on
the 2x2 mm PDAC ROI.

Runs three configs on the same ROI and panel:
  1. sequential
  2. tiled 2x2 (4 workers)
  3. tiled 3x3 (9 workers)

Then computes Adjusted Rand Index between each tiled output's per-tx
labels and the sequential reference. ARI ranges 0..1; 1 means
identical clustering (tx that share a label in tiled also share one
in sequential, and vice versa). Restricted ARI is reported on the
subset of tx that are assigned in BOTH partitions (excluding the
sentinel labels "-1" / DROP), so we measure the structural agreement
where both made a real assignment.

Run from the seg-tile-parallel worktree root:

    PYTHONPATH=src:. python benchmarks/bench_pdac_roi_tiled_ari.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/"
    "data/outs/transcripts.parquet"
)
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr10.parquet"
)
ROI_CENTER = (7255.0, 3023.7)
ROI_HALF_SIDE = 1000.0

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "benchmarks" / "pdac_roi_tiled_ari"
OUT_DIR.mkdir(parents=True, exist_ok=True)


SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}


def _labels_for_ari(df_out: pd.DataFrame, label_col: str) -> np.ndarray:
    """Encode labels as int codes for ARI. Sentinel labels are kept as
    a single -1 code so they count as "unassigned" cluster."""
    s_series = df_out[label_col].astype(str)
    sentinel_mask = (
        s_series.isin(SENTINELS)
        | s_series.str.endswith("_rejected", na=False)
    ).to_numpy()
    s = s_series.to_numpy()
    codes, _ = pd.factorize(s, sort=False)
    codes = codes.astype(np.int64)
    codes[sentinel_mask] = -1
    return codes


def main() -> int:
    t0 = time.time()
    df = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "overlaps_nucleus",
                 "feature_name", "x_location", "y_location", "z_location"],
    ).rename(columns={"x_location": "x", "y_location": "y", "z_location": "z"})
    xc, yc = ROI_CENTER
    h = ROI_HALF_SIDE
    mask = df["x"].between(xc - h, xc + h) & df["y"].between(yc - h, yc + h)
    df = df.loc[mask].reset_index(drop=True)
    panel = pd.read_parquet(PANEL).rename(columns={"value": "NPMI"})[["gene_i", "gene_j", "NPMI"]]
    print(f"loaded ROI: {len(df):,} tx / {df['cell_id'].nunique():,} cell_ids  "
          f"[{time.time()-t0:.1f}s]", flush=True)

    import tests._pipeline_runner as runner
    from tests._pipeline_runner import run_segmented_pipeline
    from tests._pipeline_runner_tiled import run_segmented_pipeline_tiled
    runner.PHASE1_RERANK_ENABLED = True
    runner.PHASE1_REASSIGN_AFTER_1C = True

    # ---------- Sequential reference ----------
    print("\n[1/3] sequential reference ...", flush=True)
    t = time.time()
    df_seq, _ = run_segmented_pipeline(df.copy(), panel)
    print(f"  wall: {time.time()-t:.1f}s", flush=True)
    col_seq = "stitched" if "stitched" in df_seq.columns else "tracer_id"
    seq_codes = _labels_for_ari(df_seq, col_seq)
    seq_assigned = (seq_codes >= 0)
    print(f"  assigned tx: {int(seq_assigned.sum()):,} / {len(seq_codes):,}", flush=True)
    df_seq[["transcript_id"]].assign(label=df_seq[col_seq].astype(str).to_numpy()).to_parquet(
        OUT_DIR / "partition_sequential.parquet", index=False,
    )

    summary = {"n_input_tx": int(len(df)), "n_input_cells": int(df["cell_id"].nunique())}
    summary["sequential_assigned_tx"] = int(seq_assigned.sum())

    # ---------- Tiled configs ----------
    for n_xy in [(2, 2), (3, 3)]:
        label = f"tiled_{n_xy[0]}x{n_xy[1]}"
        n_tiles = n_xy[0] * n_xy[1]
        print(f"\n[{2 if n_tiles == 4 else 3}/3] {label} ...", flush=True)
        t = time.time()
        result = run_segmented_pipeline_tiled(
            df.copy(), panel,
            n_tiles_xy=n_xy, n_workers=n_tiles,
            rerank=True, reassign=True,
            show_progress=False,
        )
        wall = time.time() - t
        df_tiled = result["df_out"]
        col_tiled = "stitched" if "stitched" in df_tiled.columns else "tracer_id"
        # Need to align tx ordering between sequential and tiled outputs
        # (the tiled output's row order is by tile, not original input order).
        df_tiled_sorted = (
            df_tiled.set_index("transcript_id")
            .reindex(df_seq["transcript_id"])
            .reset_index()
        )
        tiled_codes = _labels_for_ari(df_tiled_sorted, col_tiled)
        tiled_assigned = (tiled_codes >= 0)
        both_assigned = seq_assigned & tiled_assigned
        # Full ARI (includes sentinel "-1" cluster)
        ari_full = float(adjusted_rand_score(seq_codes, tiled_codes))
        # Restricted ARI (only on tx assigned in BOTH partitions)
        ari_restricted = float(
            adjusted_rand_score(seq_codes[both_assigned], tiled_codes[both_assigned])
        ) if int(both_assigned.sum()) >= 2 else float("nan")
        # Agreement on assignment (binary: assigned vs unassigned)
        n_seq_assigned = int(seq_assigned.sum())
        n_tiled_assigned = int(tiled_assigned.sum())
        n_both_assigned = int(both_assigned.sum())
        n_only_seq = int((seq_assigned & ~tiled_assigned).sum())
        n_only_tiled = int((~seq_assigned & tiled_assigned).sum())
        print(f"  wall: {wall:.1f}s")
        print(f"  ARI full       (incl. unassigned class): {ari_full:.4f}")
        print(f"  ARI restricted (assigned-in-both only):  {ari_restricted:.4f}")
        print(f"  agreement on assignment:")
        print(f"    assigned in sequential:   {n_seq_assigned:>8,}")
        print(f"    assigned in {label:<12s}  {n_tiled_assigned:>8,}")
        print(f"    assigned in BOTH:         {n_both_assigned:>8,}  "
              f"({100*n_both_assigned/n_seq_assigned:.2f}% of seq-assigned)")
        print(f"    only in sequential:       {n_only_seq:>8,}")
        print(f"    only in {label:<12s}    {n_only_tiled:>8,}")
        summary[label] = {
            "wall_seconds": round(wall, 2),
            "ari_full": ari_full,
            "ari_restricted": ari_restricted,
            "n_seq_assigned": n_seq_assigned,
            "n_tiled_assigned": n_tiled_assigned,
            "n_both_assigned": n_both_assigned,
            "n_only_seq": n_only_seq,
            "n_only_tiled": n_only_tiled,
        }
        df_tiled_sorted[["transcript_id"]].assign(
            label=df_tiled_sorted[col_tiled].astype(str).to_numpy()
        ).to_parquet(OUT_DIR / f"partition_{label}.parquet", index=False)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsummary: {OUT_DIR}/summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
