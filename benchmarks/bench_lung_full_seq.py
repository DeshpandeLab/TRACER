#!/usr/bin/env python3
"""Sequential SEG on the FULL lung_cancer tissue.

Uses the canonical lung NPMI panel (lung_cancer_npmi.csv). Outputs land
in benchmarks/lung_full_seq/ following the same schema as
bench_pdac_full_seq.py so downstream plots can reuse the same scripts.
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
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(r)
    return int(r) * 1024


LUNG_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/lung_cancer/"
    "data/lung_cancer_df.parquet"
)
PANEL_CSV = Path(os.environ.get(
    "PANEL_CSV",
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/lung_cancer/"
    "data/lung_cancer_npmi.csv",
))

REPO = Path(__file__).resolve().parents[1]
OUT_TAG = os.environ.get("OUT_TAG", "")
OUT_DIR = REPO / "benchmarks" / (f"lung_full_seq{OUT_TAG}" if OUT_TAG else "lung_full_seq")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}


def main() -> int:
    t0 = time.time()
    print("loading lung parquet ...", flush=True)
    df = pd.read_parquet(LUNG_PARQUET)
    # Schema check
    needed = ["transcript_id", "cell_id", "overlaps_nucleus",
              "feature_name", "x", "y", "z"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns in lung parquet: {missing}")
    df = df[needed].copy()
    n_in_tx = len(df)
    n_in_cells = int(df["cell_id"].nunique())
    print(f"  {n_in_tx:,} tx / {n_in_cells:,} cell_ids [{time.time()-t0:.1f}s]",
          flush=True)

    # Load panel (the canonical dense NPMI csv: gene_i,gene_j,...,NPMI)
    panel = pd.read_csv(PANEL_CSV)
    if "NPMI" not in panel.columns:
        raise SystemExit(f"panel missing NPMI column: {list(panel.columns)}")
    panel = panel[["gene_i", "gene_j", "NPMI"]].dropna(subset=["NPMI"])
    panel = panel[panel["gene_i"].astype(str) != panel["gene_j"].astype(str)]
    print(f"panel: {PANEL_CSV.name}  ({len(panel):,} rows after NaN drop)",
          flush=True)

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
    if "_etype" in df_out.columns:
        etype = df_out["_etype"].astype(str)
    else:
        from tracer._etype import infer_etype_from_label
        etype = pd.Series(np.asarray(infer_etype_from_label(labels)).astype(str))
    per_lab = (pd.DataFrame({"lab": labels[~is_un].to_numpy(),
                              "etype": etype[~is_un].to_numpy()})
                 .drop_duplicates("lab"))
    et_counts = per_lab["etype"].value_counts().to_dict()
    n_cells = int(et_counts.get("cell", 0))
    n_partials = int(et_counts.get("partial", 0))
    n_components = int(et_counts.get("component", 0))
    n_assigned = int((~is_un).sum())
    n_unassigned = int(is_un.sum())
    peak_gb = _peak_rss_bytes() / (1024 ** 3)

    summary = {
        "panel_path": str(PANEL_CSV),
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
        "n_assigned_tx": n_assigned,
        "n_unassigned_tx": n_unassigned,
        "coverage_pct": round(100 * n_assigned / max(len(labels), 1), 3),
        "label_column": col,
    }
    print("\nresult:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out = df[["transcript_id", "cell_id", "x", "y", "z"]].copy()
    out["label"] = (
        df_out.set_index("transcript_id")[col].astype(str)
        .reindex(df["transcript_id"]).to_numpy()
    )
    if "_etype" in df_out.columns:
        out["_etype"] = (
            df_out.set_index("transcript_id")["_etype"].astype(str)
            .reindex(df["transcript_id"]).to_numpy()
        )
    else:
        from tracer._etype import infer_etype_from_label
        out["_etype"] = np.asarray(infer_etype_from_label(out["label"])).astype(str)
    part_path = OUT_DIR / "partition_sequential.parquet"
    out.to_parquet(part_path, index=False)
    print(f"\npartition -> {part_path}")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"summary  -> {OUT_DIR/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
