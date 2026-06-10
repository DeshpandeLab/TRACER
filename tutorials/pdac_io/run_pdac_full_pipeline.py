#!/usr/bin/env python3
"""Sequential SEG on the FULL pdac_io tissue (no ROI crop).

Adapted from benchmarks/bench_pdac_full_seq.py but co-located with the
pdac_io data + bootstrap-PMI CSVs that the runner consumes.

Outputs (under tutorials/pdac_io/output/full_seq/):
  - partition_sequential.parquet   per-tx (transcript_id, cell_id, xyz,
                                   label, _etype)
  - summary.json                   wall-clock, RSS peak, entity counts,
                                   coverage %

Usage (from the worktree root):

    PYTHONPATH=src:. /opt/homebrew/Caskroom/miniconda/base/envs/genesis_env/bin/python \\
        tutorials/pdac_io/run_pdac_full_pipeline.py

Knobs (env vars):
  PANEL_CSV        override panel (default: pmi_bs_pdac_io_C_5_95.csv)
  PMI_THR          override the runner's PMI_THR
  RESCUE_MEAN_ADMIT, RESCUE_AGGREGATOR_PERCENTILE  override rescue knobs
  OUT_TAG          suffix on the output dir name (default empty)

Expected scale: ~3-5M transcripts on full pdac_io. Wall-clock likely
30 min - 2 h depending on hardware; peak RSS may exceed 20 GB. Consider
launching with `nohup ... &` or `caffeinate` on macOS.
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
    """Peak RSS in bytes. macOS reports bytes; Linux reports kilobytes."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if sys.platform == "darwin" else int(r) * 1024


TUT_DIR = Path(__file__).resolve().parent          # tutorials/pdac_io
REPO = TUT_DIR.parents[1]                          # worktree root

PDAC_PARQUET = TUT_DIR / "data" / "outs" / "transcripts.parquet"
PANEL_CSV = Path(os.environ.get(
    "PANEL_CSV",
    str(TUT_DIR / "data" / "pmi_bs_pdac_io_C_5_95.csv"),
))

OUT_TAG = os.environ.get("OUT_TAG", "")
OUT_DIR = TUT_DIR / "output" / (f"full_seq{OUT_TAG}" if OUT_TAG else "full_seq")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}


def main() -> int:
    print("=" * 78)
    print("FULL pdac_io SEG pipeline (sequential)")
    print("=" * 78)
    print(f"Parquet:  {PDAC_PARQUET}")
    print(f"Panel:    {PANEL_CSV.name}")
    print(f"Out dir:  {OUT_DIR}")
    print()

    t0 = time.time()
    df = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "overlaps_nucleus",
                 "feature_name", "x_location", "y_location", "z_location"],
    ).rename(columns={"x_location": "x", "y_location": "y", "z_location": "z"})
    n_in_tx = len(df)
    n_in_cells = int(df["cell_id"].nunique())
    print(f"loaded full pdac_io: {n_in_tx:,} tx / {n_in_cells:,} cell_ids "
          f"[{time.time()-t0:.1f}s]", flush=True)

    panel = pd.read_csv(PANEL_CSV)
    print(f"panel rows: {len(panel):,}", flush=True)

    # Load runner via importlib (tests/ has no __init__.py)
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("_pipeline_runner",
                                        REPO / "tests/_pipeline_runner.py")
    runner = _ilu.module_from_spec(spec)
    sys.modules["_pipeline_runner"] = runner
    spec.loader.exec_module(runner)

    runner.PHASE1_RERANK_ENABLED = True
    runner.PHASE1_REASSIGN_AFTER_1C = True

    # Optional threshold overrides via env vars
    for env_key, attr in [
        ("PMI_THR", "PMI_THR"),
        ("RESCUE_MEAN_ADMIT", "RESCUE_MEAN_ADMIT"),
        ("RESCUE_AGGREGATOR_PERCENTILE", "RESCUE_AGGREGATOR_PERCENTILE"),
    ]:
        if (val := os.environ.get(env_key)) is not None:
            setattr(runner, attr, float(val))
            print(f"OVERRIDE: {attr} = {getattr(runner, attr)}", flush=True)
            if env_key == "PMI_THR":
                runner.ANNOTATE_NEG_THR = -0.1 * (runner.PMI_THR / 0.05)
                runner.RESCUE_NEG_THR = -runner.PMI_THR
                print(f"OVERRIDE: derived ANNOTATE_NEG_THR={runner.ANNOTATE_NEG_THR}, "
                      f"RESCUE_NEG_THR={runner.RESCUE_NEG_THR}", flush=True)

    print()
    print("running sequential SEG ...", flush=True)
    t = time.time()
    df_out, info = runner.run_segmented_pipeline(df.copy(), panel)
    wall = time.time() - t
    print(f"  wall: {wall:.1f}s ({wall/60:.1f} min)", flush=True)

    col = "stitched" if "stitched" in df_out.columns else "tracer_id"
    labels = df_out[col].astype(str)
    is_un = (labels.isin(SENTINELS)
             | labels.str.endswith("_rejected", na=False)).to_numpy()

    if "_etype" in df_out.columns:
        etype = df_out["_etype"].astype(str)
    else:
        from tracer._etype import infer_etype_from_label
        etype = pd.Series(np.asarray(infer_etype_from_label(labels)).astype(str))
    per_lab = (pd.DataFrame({"lab": labels.loc[~is_un],
                             "etype": etype.loc[~is_un]})
               .drop_duplicates("lab"))
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
        "n_assigned_tx": n_assigned_tx,
        "n_unassigned_tx": n_unassigned_tx,
        "coverage_pct": coverage,
        "label_column": col,
    }
    print()
    print("result:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Persist per-tx partition aligned to original input order
    out = df[["transcript_id", "cell_id", "x", "y", "z"]].copy()
    out["label"] = (df_out.set_index("transcript_id")[col].astype(str)
                    .reindex(df["transcript_id"]).to_numpy())
    if "_etype" in df_out.columns:
        out["_etype"] = (df_out.set_index("transcript_id")["_etype"].astype(str)
                         .reindex(df["transcript_id"]).to_numpy())
    else:
        from tracer._etype import infer_etype_from_label
        out["_etype"] = np.asarray(infer_etype_from_label(out["label"])).astype(str)

    part_path = OUT_DIR / "partition_sequential.parquet"
    out.to_parquet(part_path, index=False)
    print(f"\npartition -> {part_path}")

    sum_path = OUT_DIR / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"summary   -> {sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
