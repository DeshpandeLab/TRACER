#!/usr/bin/env python3
"""Determinism / noise-floor check.

Runs the sequential SEG pipeline TWICE on the densest 666 µm sub-tile
and computes pair-agreement metrics between the two runs. The goal:
establish whether the pipeline is fully deterministic, or whether there
is some inherent run-to-run noise. Either result is informative:

  - If ARI == 1.0: SEG is deterministic. The 0.91 tile-vs-seq ARI is
    fully structural (caused by tile-parallel design), not noise.
  - If ARI < 1.0: there's a noise floor. The structural gap is smaller
    than the raw 0.09 suggests.

`_ensure_reproducibility_seed()` in the runner sets a fixed seed, so
we don't pass a seed argument — we run twice with the same seed to
test that the call chain genuinely respects determinism. (Most random
state in TRACER's pipeline shape is in KD-tree tie-breaks, dict
insertion order, and numpy reductions, not user-controllable seeds.)

Run from this worktree root:

    PYTHONPATH=src:. python benchmarks/bench_pdac_roi_seed_determinism.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score, rand_score,
    homogeneity_completeness_v_measure,
)

PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/"
    "data/outs/transcripts.parquet"
)
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr10.parquet"
)
# Densest 666 µm sub-tile (matches bench_pdac_densest_subtile.py)
ROI_X = (6255.0, 6921.67)
ROI_Y = (2023.7, 2690.37)

SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}


def _codes(labels: pd.Series, singletons: bool = True) -> np.ndarray:
    is_un = (
        labels.isin(SENTINELS)
        | labels.str.endswith("_rejected", na=False)
    ).to_numpy()
    codes, _ = pd.factorize(labels.to_numpy(), sort=False)
    codes = codes.astype(np.int64)
    if singletons:
        tx_idx = np.arange(labels.size, dtype=np.int64)
        codes[is_un] = -2 - tx_idx[is_un]
    else:
        codes[is_un] = -1
    return codes


def main() -> int:
    t0 = time.time()
    df = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "overlaps_nucleus",
                 "feature_name", "x_location", "y_location", "z_location"],
    ).rename(columns={"x_location": "x", "y_location": "y", "z_location": "z"})
    mask = df["x"].between(*ROI_X) & df["y"].between(*ROI_Y)
    df = df.loc[mask].reset_index(drop=True)
    panel = pd.read_parquet(PANEL).rename(columns={"value": "NPMI"})[["gene_i", "gene_j", "NPMI"]]
    print(f"sub-tile: {len(df):,} tx / {df['cell_id'].nunique():,} cell_ids  "
          f"[{time.time()-t0:.1f}s]", flush=True)

    import tests._pipeline_runner as runner
    from tests._pipeline_runner import run_segmented_pipeline
    runner.PHASE1_RERANK_ENABLED = True
    runner.PHASE1_REASSIGN_AFTER_1C = True

    # Run twice on the same input.
    runs = []
    for i in (1, 2):
        print(f"\nRun {i} ...", flush=True)
        t = time.time()
        df_out, _ = run_segmented_pipeline(df.copy(), panel)
        wall = time.time() - t
        col = "stitched" if "stitched" in df_out.columns else "tracer_id"
        labels = df_out[col].astype(str)
        labels_aligned = (
            df_out.set_index("transcript_id")[col].astype(str)
            .reindex(df["transcript_id"])
        )
        runs.append({
            "wall": wall,
            "labels": labels_aligned,
        })
        print(f"  wall: {wall:.1f}s")

    # Bit-exact comparison.
    a = runs[0]["labels"].to_numpy()
    b = runs[1]["labels"].to_numpy()
    n_diff = int((a != b).sum())
    print(f"\nBit-exact equality: {a.shape[0] - n_diff:,} / {a.shape[0]:,} "
          f"tx labels identical ({n_diff} differ)")

    a_mega = _codes(runs[0]["labels"], singletons=False)
    b_mega = _codes(runs[1]["labels"], singletons=False)
    a_sing = _codes(runs[0]["labels"], singletons=True)
    b_sing = _codes(runs[1]["labels"], singletons=True)

    print()
    print(f"{'encoding':24s}  {'ARI':>7s}  {'RI':>7s}  {'h':>7s}  {'c':>7s}  {'V':>7s}")
    for label, sl, tl in [
        ("mega-class",  a_mega, b_mega),
        ("singletons",  a_sing, b_sing),
    ]:
        ari = float(adjusted_rand_score(sl, tl))
        ri = float(rand_score(sl, tl))
        h, c, v = homogeneity_completeness_v_measure(sl, tl)
        print(f"{label:24s}  {ari:>7.4f}  {ri:>7.4f}  "
              f"{h:>7.4f}  {c:>7.4f}  {v:>7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
