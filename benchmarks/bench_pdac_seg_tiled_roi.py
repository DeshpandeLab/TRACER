#!/usr/bin/env python3
"""Tile-parallel SEG bench on a 2000x2000 um PDAC ROI.

Smaller scope than ``bench_pdac_seg_tiled.py``: a 2 mm x 2 mm ROI
around the PDAC sample centroid. Fast enough to iterate (~5-10 min
total for 3 runs) yet large enough that tiling provides meaningful
parallelism.

Runs three configs on the SAME ROI input + thr=10 PMI panel:
  1. sequential  (run_segmented_pipeline, single core)  -- reference
  2. tiled 2x2   (4 parallel workers)
  3. tiled 3x3   (9 parallel workers)

For each, reports wall, per-tile breakdown, and aggregate entity
counts so we can confirm:
  - Tiled output is internally consistent (within-tile cells +
    partials sum reasonably vs the sequential reference)
  - Speedup vs sequential is real and roughly tracks core count

Run from this worktree root:

    PYTHONPATH=src:. python benchmarks/bench_pdac_seg_tiled_roi.py
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
PANEL_DIR = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac"
)

ROI_CENTER = (7255.0, 3023.7)   # PDAC median (x, y)
ROI_HALF_SIDE = 1000.0          # 2000 um box

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "benchmarks" / "pdac_seg_tiled_roi"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "bench.log"

UNASSIGNED_TOKENS = {"-1", "DROP", "UNASSIGNED", "nan"}


def _panel_from_W(parquet_path: Path) -> pd.DataFrame:
    w = pd.read_parquet(parquet_path)
    return w.rename(columns={"value": "NPMI"})[["gene_i", "gene_j", "NPMI"]]


def _entity_counts(df_out: pd.DataFrame, label_col: str) -> dict:
    s = df_out[label_col].astype(str)
    is_un = s.isin(UNASSIGNED_TOKENS) | s.str.endswith("_rejected", na=False)
    if "_etype" in df_out.columns:
        etype = df_out["_etype"].astype(str)
    else:
        from tracer._etype import infer_etype_from_label
        etype = pd.Series(np.asarray(infer_etype_from_label(s)).astype(str))
    pairs = pd.DataFrame({"lab": s, "etype": etype}).loc[~is_un.to_numpy()]
    per = pairs.drop_duplicates("lab")["etype"].value_counts().to_dict()
    return {
        "n_cells": int(per.get("cell", 0)),
        "n_partials": int(per.get("partial", 0)),
        "n_components": int(per.get("component", 0)),
        "n_unassigned_tx": int(is_un.sum()),
        "n_assigned_tx": int((~is_un).sum()),
        "coverage_pct": round(100*(~is_un).sum()/max(len(s),1), 3),
    }


def main() -> int:
    log_lines: list[str] = []

    def log(s: str = ""):
        print(s, flush=True)
        log_lines.append(s)
        LOG_PATH.write_text("\n".join(log_lines) + "\n")

    log("=" * 100)
    log("PDAC 2x2 mm ROI — tile-parallel SEG bench")
    log("=" * 100)
    log(f"ROI center: {ROI_CENTER}, half-side: {ROI_HALF_SIDE} um")
    log()

    # Load + slice
    t0 = time.time()
    df_full = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "overlaps_nucleus",
                 "feature_name", "x_location", "y_location", "z_location"],
    )
    df_full = df_full.rename(columns={
        "x_location": "x", "y_location": "y", "z_location": "z",
    })
    xc, yc = ROI_CENTER
    h = ROI_HALF_SIDE
    mask = df_full["x"].between(xc-h, xc+h) & df_full["y"].between(yc-h, yc+h)
    df = df_full.loc[mask].reset_index(drop=True)
    del df_full
    n_in_tx = len(df)
    n_in_cells = df["cell_id"].nunique()
    log(f"  loaded full PDAC and sliced ROI: {n_in_tx:,} tx / "
        f"{n_in_cells:,} cell_ids  [{time.time()-t0:.1f}s]")
    log()

    panel = _panel_from_W(PANEL_DIR / "W_thr10.parquet")
    log(f"  panel thr=10: {len(panel):,} rows")
    log()

    summary: dict = {
        "roi_center": list(ROI_CENTER),
        "roi_half_side": ROI_HALF_SIDE,
        "n_input_tx": int(n_in_tx),
        "n_input_cells": int(n_in_cells),
        "panel_rows": int(len(panel)),
        "configs": {},
    }

    # ---------------------------------------------------------------------
    # 1. Sequential reference
    # ---------------------------------------------------------------------
    log("-" * 100)
    log("RUN 1: sequential reference (run_segmented_pipeline, single core)")
    log("-" * 100)
    import tests._pipeline_runner as runner
    from tests._pipeline_runner import run_segmented_pipeline

    orig_rerank = runner.PHASE1_RERANK_ENABLED
    orig_reassign = runner.PHASE1_REASSIGN_AFTER_1C
    runner.PHASE1_RERANK_ENABLED = True
    runner.PHASE1_REASSIGN_AFTER_1C = True
    try:
        t = time.time()
        df_seq, prog_seq = run_segmented_pipeline(df.copy(), panel)
        wall_seq = time.time() - t
    finally:
        runner.PHASE1_RERANK_ENABLED = orig_rerank
        runner.PHASE1_REASSIGN_AFTER_1C = orig_reassign

    col_seq = "stitched" if "stitched" in df_seq.columns else "tracer_id"
    ec_seq = _entity_counts(df_seq, col_seq)
    ec_seq["cells_lost"] = int(n_in_cells - ec_seq["n_cells"])
    ec_seq["retention_pct"] = round(100*ec_seq["n_cells"]/max(n_in_cells,1), 3)
    log(f"  wall: {wall_seq:.1f}s  stages: {len(prog_seq)}")
    log(f"  entity counts: {ec_seq}")
    # Per-stage timing breakdown (stage_seconds added by _record_stage).
    stage_timings = [
        (p.get("stage"), p.get("stage_seconds"))
        for p in prog_seq if p.get("stage_seconds") is not None
    ]
    if stage_timings:
        log(f"  per-stage wall (top stages, descending):")
        for stage, secs in sorted(stage_timings, key=lambda x: -(x[1] or 0))[:8]:
            log(f"    {stage:>22s}  {secs:>8.2f}s")
    summary["configs"]["sequential"] = {
        "wall_total_seconds": round(wall_seq, 2),
        "entity_counts": ec_seq,
        "stage_timings": [
            {"stage": s, "seconds": t} for s, t in stage_timings
        ],
    }
    log()

    # ---------------------------------------------------------------------
    # 2 + 3. Tiled configs
    # ---------------------------------------------------------------------
    from tests._pipeline_runner_tiled import run_segmented_pipeline_tiled

    for n_xy in [(2, 2), (3, 3)]:
        cfg_label = f"tiled_{n_xy[0]}x{n_xy[1]}"
        n_tiles = n_xy[0] * n_xy[1]
        log("-" * 100)
        log(f"RUN: {cfg_label} ({n_tiles} tiles, {n_tiles} workers)")
        log("-" * 100)

        t = time.time()
        result = run_segmented_pipeline_tiled(
            df.copy(), panel,
            n_tiles_xy=n_xy, n_workers=n_tiles,
            rerank=True, reassign=True,
            show_progress=True,
        )
        wall = time.time() - t
        col_tiled = ("stitched" if "stitched" in result["df_out"].columns
                     else "tracer_id")
        ec = _entity_counts(result["df_out"], col_tiled)
        ec["cells_lost"] = int(n_in_cells - ec["n_cells"])
        ec["retention_pct"] = round(100*ec["n_cells"]/max(n_in_cells, 1), 3)

        log(f"  wall_total: {wall:.1f}s "
            f"(max single tile: {result['wall_max_tile_seconds']:.1f}s)")
        log(f"  speedup vs serial-sum estimate: "
            f"{result['speedup_vs_serial_estimate']:.2f}x")
        log(f"  actual speedup vs RUN 1: {wall_seq / wall:.2f}x")
        log(f"  entity counts: {ec}")

        summary["configs"][cfg_label] = {
            "n_tiles_xy": list(n_xy),
            "wall_total_seconds": result["wall_total_seconds"],
            "wall_max_tile_seconds": result["wall_max_tile_seconds"],
            "speedup_vs_serial_sum_estimate": result["speedup_vs_serial_estimate"],
            "actual_speedup_vs_sequential": round(wall_seq / wall, 2),
            "per_tile_wall": {
                int(ti): float(per["wall_seconds"])
                for ti, per in result["per_tile"].items()
            },
            "entity_counts": ec,
        }
        log()

    # ---------------------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------------------
    log("=" * 100)
    log("Comparison")
    log("=" * 100)
    labels = ["sequential", "tiled_2x2", "tiled_3x3"]
    metrics = ["wall_total_seconds", "actual_speedup_vs_sequential"]
    ec_metrics = ["n_cells", "cells_lost", "retention_pct",
                  "n_partials", "n_components", "coverage_pct"]

    log(f"  {'metric':30s}  " + " ".join(f"{l:>14s}" for l in labels))
    for m in metrics:
        row = []
        for l in labels:
            v = summary["configs"][l].get(m)
            row.append(f"{v:>14}" if isinstance(v, (int, float))
                       else f"{'—':>14s}")
        log(f"  {m:30s}  " + " ".join(row))
    log()
    log("Entity counts:")
    for m in ec_metrics:
        row = []
        for l in labels:
            v = summary["configs"][l]["entity_counts"].get(m)
            row.append(f"{v:>14}" if isinstance(v, (int, float))
                       else f"{'—':>14s}")
        log(f"  {m:30s}  " + " ".join(row))

    log()
    log("Boundary loss vs sequential (cells / partials):")
    seq_ec = summary["configs"]["sequential"]["entity_counts"]
    for l in ["tiled_2x2", "tiled_3x3"]:
        e = summary["configs"][l]["entity_counts"]
        d_cells = e["n_cells"] - seq_ec["n_cells"]
        d_parts = e["n_partials"] - seq_ec["n_partials"]
        log(f"  {l:14s}  Δcells={d_cells:+d}  Δpartials={d_parts:+d}")

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log()
    log(f"summary: {summary_path}")
    log(f"log:     {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
