#!/usr/bin/env python3
"""Sequential SEG on the densest sub-tile of the 2x2mm PDAC ROI.

That sub-tile (tile 0 of the prior 3x3 split) had 7,375 cells in
~666x666 um and took 73.3s end-to-end with the Python Reassign-1c
implementation; Phase1-Reassign-1c alone was 39.3s of that.

This bench runs the SEG pipeline ONCE on that sub-tile only, with
stage-verbose printing, so we can directly measure the Cython
kernel's wall time on representative real data. No tile parallelism;
no comparison config — just one number.

Run from this worktree root:

    TRACER_STAGE_VERBOSE=1 PYTHONPATH=src:. python benchmarks/bench_pdac_densest_subtile.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/"
    "data/outs/transcripts.parquet"
)
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr10.parquet"
)
# Densest sub-tile bounds from the prior 3x3 split of the 2x2mm ROI.
# x ∈ [6255, 6921.67], y ∈ [2023.7, 2690.37]  (about 667 um × 667 um).
ROI_X = (6255.0, 6921.67)
ROI_Y = (2023.7, 2690.37)


def main() -> int:
    t0 = time.time()
    df = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "overlaps_nucleus",
                 "feature_name", "x_location", "y_location", "z_location"],
    )
    df = df.rename(columns={
        "x_location": "x", "y_location": "y", "z_location": "z",
    })
    mask = (
        df["x"].between(*ROI_X) & df["y"].between(*ROI_Y)
    )
    df = df.loc[mask].reset_index(drop=True)
    n_tx = len(df)
    n_cells = df["cell_id"].nunique()
    print(f"loaded densest sub-tile: {n_tx:,} tx / {n_cells:,} cell_ids "
          f"[{time.time()-t0:.1f}s]", flush=True)

    panel = pd.read_parquet(PANEL).rename(columns={"value": "NPMI"})
    panel = panel[["gene_i", "gene_j", "NPMI"]]
    print(f"panel: {len(panel):,} rows", flush=True)

    import tests._pipeline_runner as runner
    from tests._pipeline_runner import run_segmented_pipeline

    runner.PHASE1_RERANK_ENABLED = True
    runner.PHASE1_REASSIGN_AFTER_1C = True

    t = time.time()
    df_out, prog = run_segmented_pipeline(df, panel)
    wall = time.time() - t
    print(f"\n=== TOTAL WALL: {wall:.1f}s ===", flush=True)

    print("\nStage breakdown (sorted by stage_seconds desc):")
    rows = [(p.get("stage"), p.get("stage_seconds"))
            for p in prog if p.get("stage_seconds") is not None]
    for stage, secs in sorted(rows, key=lambda x: -(x[1] or 0)):
        print(f"  {stage:>22s}  {secs:>8.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
