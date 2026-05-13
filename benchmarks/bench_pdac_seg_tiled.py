#!/usr/bin/env python3
"""Tile-parallel SEG bench on full PDAC sample.

Compares wall-time of:
  - sequential run_segmented_pipeline (single core)  -- NOT re-run here;
    we read the wall from the previous flavor bench's summary.json
  - run_segmented_pipeline_tiled with n_tiles_xy=(2,2) and (3,3)

Both tile configs use the thr=10 PMI panel (the higher-density variant)
so we measure the parallel speed-up under the harder load. Output entity
counts are reported alongside so we can quantify the boundary loss
(entities/tx that fall through the cracks vs the sequential reference).

Run from this worktree root:

    PYTHONPATH=src:. python benchmarks/bench_pdac_seg_tiled.py
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
# Reuse the panel saved by bench_bootstrap_thr_pdac.py
PANEL_DIR = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac"
)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "benchmarks" / "pdac_seg_tiled"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "bench.log"


def _panel_from_W(parquet_path: Path) -> pd.DataFrame:
    w = pd.read_parquet(parquet_path)
    return w.rename(columns={"value": "NPMI"})[["gene_i", "gene_j", "NPMI"]]


def _entity_counts(df_out: pd.DataFrame, label_col: str) -> dict:
    UN = {"-1", "DROP", "UNASSIGNED", "nan"}
    s = df_out[label_col].astype(str)
    is_un = s.isin(UN) | s.str.endswith("_rejected", na=False)
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
    log("PDAC SEG pipeline — tile-parallel orchestrator bench")
    log("=" * 100)

    t0 = time.time()
    df = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "overlaps_nucleus",
                 "feature_name", "x_location", "y_location", "z_location"],
    )
    df = df.rename(columns={
        "x_location": "x", "y_location": "y", "z_location": "z",
    })
    n_in_tx = len(df); n_in_cells = df["cell_id"].nunique()
    log(f"loaded full PDAC: {n_in_tx:,} tx / {n_in_cells:,} cell_ids "
        f"[{time.time()-t0:.1f}s]")
    log()

    panel = _panel_from_W(PANEL_DIR / "W_thr10.parquet")
    log(f"panel thr=10: {len(panel):,} rows")
    log()

    from tests._pipeline_runner_tiled import run_segmented_pipeline_tiled

    summary: dict = {
        "n_input_tx": int(n_in_tx),
        "n_input_cells": int(n_in_cells),
        "panel_path": str(PANEL_DIR / "W_thr10.parquet"),
        "panel_rows": int(len(panel)),
        "configs": {},
    }

    for n_xy in [(2, 2), (3, 3)]:
        cfg_label = f"{n_xy[0]}x{n_xy[1]}"
        n_tiles = n_xy[0] * n_xy[1]
        log("-" * 100)
        log(f"CONFIG: n_tiles_xy={n_xy} ({n_tiles} tiles, {n_tiles} workers)")
        log("-" * 100)

        t = time.time()
        result = run_segmented_pipeline_tiled(
            df.copy(), panel,
            n_tiles_xy=n_xy, n_workers=n_tiles,
            rerank=True, reassign=True,
            show_progress=True,
        )
        wall = time.time() - t
        log(f"  wall_total: {wall:.1f}s  (max single tile: "
            f"{result['wall_max_tile_seconds']:.1f}s)")
        log(f"  speedup-vs-serial estimate: "
            f"{result['speedup_vs_serial_estimate']:.2f}x")

        col = "stitched" if "stitched" in result["df_out"].columns else "tracer_id"
        ec = _entity_counts(result["df_out"], col)
        ec["cells_lost"] = int(n_in_cells - ec["n_cells"])
        ec["retention_pct"] = round(100*ec["n_cells"]/max(n_in_cells, 1), 3)
        log(f"  entity counts: {ec}")

        # Save concatenated partitions
        parts_path = OUT_DIR / f"partition_{cfg_label}.parquet"
        result["df_out"][[
            "transcript_id", "cell_id", "x", "y", "z",
            col, "_etype" if "_etype" in result["df_out"].columns else col,
        ]].rename(columns={col: "label_out"}).to_parquet(parts_path, index=False)
        log(f"  partitions -> {parts_path.name}")

        summary["configs"][cfg_label] = {
            "n_tiles_xy": list(n_xy),
            "wall_total_seconds": result["wall_total_seconds"],
            "wall_max_tile_seconds": result["wall_max_tile_seconds"],
            "speedup_vs_serial_estimate": result["speedup_vs_serial_estimate"],
            "per_tile": result["per_tile"],
            "tile_info": result["tile_info"],
            "entity_counts": ec,
        }
        log()

    # Side-by-side
    log("=" * 100)
    log("Comparison")
    log("=" * 100)
    labels = sorted(summary["configs"].keys())
    metrics = ["wall_total_seconds", "wall_max_tile_seconds",
               "speedup_vs_serial_estimate"]
    ec_metrics = ["n_cells", "cells_lost", "retention_pct",
                  "n_partials", "n_components", "coverage_pct"]
    log(f"  {'metric':30s}  " + " ".join(f"{l:>10s}" for l in labels))
    for m in metrics:
        row = []
        for l in labels:
            v = summary["configs"][l].get(m)
            row.append(f"{v:>10}" if isinstance(v, (int, float))
                       else f"{'—':>10s}")
        log(f"  {m:30s}  " + " ".join(row))
    log()
    log("Entity counts:")
    for m in ec_metrics:
        row = []
        for l in labels:
            v = summary["configs"][l]["entity_counts"].get(m)
            row.append(f"{v:>10}" if isinstance(v, (int, float))
                       else f"{'—':>10s}")
        log(f"  {m:30s}  " + " ".join(row))

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log()
    log(f"summary: {summary_path}")
    log(f"log:     {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
