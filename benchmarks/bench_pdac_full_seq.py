#!/usr/bin/env python3
"""Sequential SEG on the FULL PDAC tissue.

Runs ``run_segmented_pipeline`` once on the whole sample (no ROI crop)
and persists per-tx labels + entity-type for later comparison against
the tile-parallel reference. Designed to be launched in the background
while other work proceeds.

Outputs:
  - benchmarks/pdac_full_seq/partition_sequential.parquet
       columns: transcript_id, cell_id, x, y, z, label, _etype
  - benchmarks/pdac_full_seq/summary.json
       wall, n_input_tx, n_input_cells, entity counts, etype breakdown
  - benchmarks/pdac_full_seq/run.log (via tee from the launcher)

Run from this worktree root:

    PYTHONPATH=src:. python benchmarks/bench_pdac_full_seq.py
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _peak_rss_bytes() -> int:
    """Peak resident set size of this process, in bytes.

    macOS reports ru_maxrss in bytes; Linux reports it in kilobytes.
    """
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(r)
    return int(r) * 1024

PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/"
    "data/outs/transcripts.parquet"
)
PANEL = Path(os.environ.get(
    "PANEL_PARQUET",
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr10.parquet",
))

REPO = Path(__file__).resolve().parents[1]
OUT_TAG = os.environ.get("OUT_TAG", "")
OUT_DIR = REPO / "benchmarks" / (f"pdac_full_seq{OUT_TAG}" if OUT_TAG else "pdac_full_seq")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}


def main() -> int:
    t0 = time.time()
    df = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "overlaps_nucleus",
                 "feature_name", "x_location", "y_location", "z_location"],
    ).rename(columns={"x_location": "x", "y_location": "y", "z_location": "z"})
    panel = pd.read_parquet(PANEL).rename(columns={"value": "NPMI"})[["gene_i", "gene_j", "NPMI"]]
    n_in_tx = len(df)
    n_in_cells = int(df["cell_id"].nunique())
    print(f"loaded full PDAC: {n_in_tx:,} tx / {n_in_cells:,} cell_ids "
          f"[{time.time()-t0:.1f}s]", flush=True)
    print(f"panel: {PANEL.name}  ({len(panel):,} rows)", flush=True)
    print(f"out_dir: {OUT_DIR}", flush=True)

    import tests._pipeline_runner as runner
    from tests._pipeline_runner import run_segmented_pipeline
    runner.PHASE1_RERANK_ENABLED = True
    runner.PHASE1_REASSIGN_AFTER_1C = True

    # Optional threshold overrides via env vars
    pmi_thr_env = os.environ.get("PMI_THR")
    mean_admit_env = os.environ.get("RESCUE_MEAN_ADMIT")
    perc_env = os.environ.get("RESCUE_AGGREGATOR_PERCENTILE")
    if pmi_thr_env is not None:
        runner.PMI_THR = float(pmi_thr_env)
        runner.ANNOTATE_NEG_THR = -0.1 * (runner.PMI_THR / 0.05)
        runner.RESCUE_NEG_THR = -runner.PMI_THR
        print(f"OVERRIDE: PMI_THR = {runner.PMI_THR}  RESCUE_NEG_THR = {runner.RESCUE_NEG_THR}", flush=True)
    if mean_admit_env is not None:
        runner.RESCUE_MEAN_ADMIT = float(mean_admit_env)
        print(f"OVERRIDE: RESCUE_MEAN_ADMIT = {runner.RESCUE_MEAN_ADMIT}", flush=True)
    if perc_env is not None:
        runner.RESCUE_AGGREGATOR_PERCENTILE = float(perc_env)
        print(f"OVERRIDE: RESCUE_AGGREGATOR_PERCENTILE = {runner.RESCUE_AGGREGATOR_PERCENTILE}", flush=True)

    print("\nrunning sequential SEG ...", flush=True)
    t = time.time()
    df_out, info = run_segmented_pipeline(df.copy(), panel)
    wall = time.time() - t
    print(f"  wall: {wall:.1f}s", flush=True)

    col = "stitched" if "stitched" in df_out.columns else "tracer_id"
    labels = df_out[col].astype(str)
    is_un = (labels.isin(SENTINELS) | labels.str.endswith("_rejected", na=False)).to_numpy()

    # Entity-type breakdown
    if "_etype" in df_out.columns:
        etype = df_out["_etype"].astype(str)
    else:
        from tracer._etype import infer_etype_from_label
        etype = pd.Series(np.asarray(infer_etype_from_label(labels)).astype(str))
    et_assigned = etype.loc[~is_un]
    label_assigned = labels.loc[~is_un]
    per_lab = pd.DataFrame({"lab": label_assigned, "etype": et_assigned}).drop_duplicates("lab")
    et_counts = per_lab["etype"].value_counts().to_dict()
    n_cells = int(et_counts.get("cell", 0))
    n_partials = int(et_counts.get("partial", 0))
    n_components = int(et_counts.get("component", 0))
    n_unassigned_tx = int(is_un.sum())
    n_assigned_tx = int((~is_un).sum())
    coverage = round(100 * n_assigned_tx / max(len(labels), 1), 3)

    peak_gb = _peak_rss_bytes() / (1024 ** 3)
    print(f"  peak RSS: {peak_gb:.2f} GB", flush=True)

    summary = {
        "panel_path": str(PANEL),
        "panel_rows": int(len(panel)),
        "wall_seconds": round(wall, 2),
        "peak_rss_gb": round(peak_gb, 3),
        "n_input_tx": n_in_tx,
        "n_input_cells": n_in_cells,
        "n_cells_out": n_cells,
        "cells_lost": n_in_cells - n_cells,
        "retention_pct": round(100 * n_cells / max(n_in_cells, 1), 3),
        "n_partials": n_partials,
        "n_components": n_components,
        "n_assigned_tx": n_assigned_tx,
        "n_unassigned_tx": n_unassigned_tx,
        "coverage_pct": coverage,
        "label_column": col,
    }
    print(f"\nresult:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Persist per-tx partition aligned to the original input order.
    out = (
        df[["transcript_id", "cell_id", "x", "y", "z"]].copy()
    )
    out["label"] = (
        df_out.set_index("transcript_id")[col].astype(str)
        .reindex(df["transcript_id"]).to_numpy()
    )
    out["_etype"] = (
        df_out.set_index("transcript_id")["_etype"].astype(str)
        .reindex(df["transcript_id"]).to_numpy()
        if "_etype" in df_out.columns
        else np.asarray(
            __import__("tracer._etype", fromlist=["infer_etype_from_label"])
            .infer_etype_from_label(out["label"])
        ).astype(str)
    )
    part_path = OUT_DIR / "partition_sequential.parquet"
    out.to_parquet(part_path, index=False)
    print(f"\npartition -> {part_path}")

    sum_path = OUT_DIR / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"summary  -> {sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
