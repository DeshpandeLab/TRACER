#!/usr/bin/env python3
"""Parameter sweep over PMI / rescue thresholds on PDAC densest sub-tile.

Uses W_thr0 (pure bootstrap-only) panel. Densest 666 µm sub-tile is the
test bed — ~750k tx, ~30s/config sequential SEG.

Grid:
  PMI_THR ∈ {0.05, 0.2, 0.5, 1.0}                    # natural-log PMI
  RESCUE_MEAN_ADMIT ∈ {-1.0 (off), 0.1, 0.5}
  RESCUE_AGGREGATOR_PERCENTILE ∈ {25, 50}

24 configs total. Reports per-config:
  wall, n_cells_out, n_partials, retention_pct, coverage_pct
Writes summary CSV and per-config partition parquet for offline
downstream analysis (clustering, lymphoid checks).
"""
from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
ROI_X = (6255.0, 6921.67)   # densest 666 µm sub-tile
ROI_Y = (2023.7, 2690.37)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "benchmarks" / "pdac_pmi_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}

GRID = list(product(
    [0.05, 0.2, 0.5, 1.0],           # PMI_THR
    [-1.0, 0.1, 0.5],                  # RESCUE_MEAN_ADMIT
    [25, 50],                          # RESCUE_AGGREGATOR_PERCENTILE
))


def _summarize_output(df_out: pd.DataFrame, label_col: str, n_in_cells: int):
    s = df_out[label_col].astype(str)
    is_un = (s.isin(SENTINELS) | s.str.endswith("_rejected", na=False)).to_numpy()
    if "_etype" in df_out.columns:
        etype = df_out["_etype"].astype(str)
    else:
        from tracer._etype import infer_etype_from_label
        etype = pd.Series(np.asarray(infer_etype_from_label(s)).astype(str))
    per = (pd.DataFrame({"lab": s[~is_un].to_numpy(),
                          "etype": etype[~is_un].to_numpy()})
             .drop_duplicates("lab"))
    counts = per["etype"].value_counts().to_dict()
    n_cells = int(counts.get("cell", 0))
    n_partials = int(counts.get("partial", 0))
    n_assigned = int((~is_un).sum())
    return {
        "n_cells_out": n_cells,
        "n_partials": n_partials,
        "n_assigned_tx": n_assigned,
        "n_unassigned_tx": int(is_un.sum()),
        "retention_pct": round(100 * n_cells / max(n_in_cells, 1), 3),
        "coverage_pct": round(100 * n_assigned / max(len(s), 1), 3),
    }


def main() -> int:
    t0 = time.time()
    print("loading ROI ...", flush=True)
    df = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "overlaps_nucleus",
                 "feature_name", "x_location", "y_location", "z_location"],
    ).rename(columns={"x_location": "x", "y_location": "y", "z_location": "z"})
    mask = df["x"].between(*ROI_X) & df["y"].between(*ROI_Y)
    df = df.loc[mask].reset_index(drop=True)
    n_in_cells = int(df["cell_id"].nunique())
    print(f"  {len(df):,} tx / {n_in_cells:,} cell_ids", flush=True)

    panel = pd.read_parquet(PANEL).rename(columns={"value": "NPMI"})[["gene_i", "gene_j", "NPMI"]]
    print(f"  panel: {PANEL.name}  ({len(panel):,} rows)  "
          f"value range=[{panel['NPMI'].min():.2f}, {panel['NPMI'].max():.2f}]",
          flush=True)

    import tests._pipeline_runner as runner
    from tests._pipeline_runner import run_segmented_pipeline
    runner.PHASE1_RERANK_ENABLED = True
    runner.PHASE1_REASSIGN_AFTER_1C = True

    rows = []
    for cfg_idx, (pmi_thr, mean_admit, perc) in enumerate(GRID, start=1):
        cfg_label = f"pmi{pmi_thr}_mean{mean_admit}_perc{perc}"
        cfg_path = OUT_DIR / f"partition_{cfg_label}.parquet"
        print(f"\n[{cfg_idx}/{len(GRID)}] {cfg_label}", flush=True)
        runner.PMI_THR = pmi_thr
        runner.RESCUE_MEAN_ADMIT = mean_admit
        runner.RESCUE_AGGREGATOR_PERCENTILE = perc
        # ANNOTATE_NEG_THR is derived from PMI_THR in module — recompute
        runner.ANNOTATE_NEG_THR = -0.1 * (pmi_thr / 0.05)
        # NEG_THR scales with PMI_THR too: keep at -PMI_THR
        runner.RESCUE_NEG_THR = -pmi_thr

        t = time.time()
        try:
            df_out, _ = run_segmented_pipeline(df.copy(), panel)
            wall = time.time() - t
            col = "stitched" if "stitched" in df_out.columns else "tracer_id"
            stats = _summarize_output(df_out, col, n_in_cells)
            stats.update({
                "config": cfg_label, "wall_seconds": round(wall, 2),
                "pmi_thr": pmi_thr, "mean_admit": mean_admit, "percentile": perc,
                "ok": True, "error": "",
            })
            print(f"  wall {wall:.1f}s  cells {stats['n_cells_out']:,}  "
                  f"partials {stats['n_partials']:,}  "
                  f"retention {stats['retention_pct']:.1f}%  "
                  f"coverage {stats['coverage_pct']:.1f}%", flush=True)
            # Save partition for offline analysis
            out_part = df[["transcript_id"]].copy()
            out_part["label"] = (
                df_out.set_index("transcript_id")[col].astype(str)
                .reindex(df["transcript_id"]).to_numpy()
            )
            out_part.to_parquet(cfg_path, index=False)
        except Exception as e:
            wall = time.time() - t
            stats = {"config": cfg_label, "wall_seconds": round(wall, 2),
                     "pmi_thr": pmi_thr, "mean_admit": mean_admit,
                     "percentile": perc, "ok": False, "error": str(e)[:200]}
            print(f"  FAILED: {e}", flush=True)
        rows.append(stats)
        # Persist incremental
        pd.DataFrame(rows).to_csv(OUT_DIR / "sweep_summary.csv", index=False)

    summary_path = OUT_DIR / "sweep_summary.csv"
    print(f"\nsummary -> {summary_path}", flush=True)
    print(f"total wall: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
