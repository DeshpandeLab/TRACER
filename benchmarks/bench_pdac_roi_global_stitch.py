#!/usr/bin/env python3
"""Validate global post-tile Stitch on PDAC 2x2mm ROI.

Compares the structural fidelity (ARI vs sequential reference) of:
  - tile-parallel 3x3 (naive concat, no global pass)
  - tile-parallel 3x3 + global post-tile Stitch + Final Rescue

Reuses the saved sequential partition from pdac_roi_tile_postprocess/
as the reference. Reports per-tx ARI / RI / h / c / V battery.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score, rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)

PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr10.parquet"
)
ROI_CENTER = (7255.0, 3023.7)
ROI_HALF_SIDE = 1000.0
N_TILES_XY = (3, 3)

REPO = Path(__file__).resolve().parents[1]
SEQ_PART = REPO / "benchmarks" / "pdac_roi_tile_postprocess" / "partition_sequential.parquet"
OUT_DIR = REPO / "benchmarks" / "pdac_roi_global_stitch"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}


def _is_un(s: pd.Series) -> np.ndarray:
    return (s.isin(SENTINELS) | s.str.endswith("_rejected", na=False)).to_numpy()


def _codes(labels: pd.Series, singletons: bool = True) -> np.ndarray:
    is_un = _is_un(labels)
    codes, _ = pd.factorize(labels.to_numpy(), sort=False)
    codes = codes.astype(np.int64)
    if singletons:
        tx_idx = np.arange(labels.size, dtype=np.int64)
        codes[is_un] = -2 - tx_idx[is_un]
    else:
        codes[is_un] = -1
    return codes


def _metrics_table(seq_labels, t_labels, name):
    seq_un = _is_un(seq_labels); t_un = _is_un(t_labels)
    sing_s = _codes(seq_labels, True); sing_t = _codes(t_labels, True)
    mega_s = _codes(seq_labels, False); mega_t = _codes(t_labels, False)
    both = (~seq_un) & (~t_un)
    print(f"\n=== {name} ===")
    print(f"  {'set':24s}  {'ARI':>7s}  {'RI':>7s}  {'NMI':>7s}  {'h':>7s}  {'c':>7s}  {'V':>7s}")
    for set_label, sl, tl in [
        ("singletons (all tx)", sing_s, sing_t),
        ("assigned-in-both",    mega_s[both], mega_t[both]),
    ]:
        ari = float(adjusted_rand_score(sl, tl))
        ri = float(rand_score(sl, tl))
        nmi = float(normalized_mutual_info_score(sl, tl))
        h, c, v = homogeneity_completeness_v_measure(sl, tl)
        print(f"  {set_label:24s}  {ari:>7.4f}  {ri:>7.4f}  {nmi:>7.4f}  "
              f"{h:>7.4f}  {c:>7.4f}  {v:>7.4f}")


def main() -> int:
    t0 = time.time()
    df = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "overlaps_nucleus",
                 "feature_name", "x_location", "y_location", "z_location"],
    ).rename(columns={"x_location": "x", "y_location": "y", "z_location": "z"})
    xc, yc = ROI_CENTER; h = ROI_HALF_SIDE
    mask = df["x"].between(xc - h, xc + h) & df["y"].between(yc - h, yc + h)
    df = df.loc[mask].reset_index(drop=True)
    panel = pd.read_parquet(PANEL).rename(columns={"value": "NPMI"})[["gene_i", "gene_j", "NPMI"]]
    print(f"loaded ROI: {len(df):,} tx [{time.time()-t0:.1f}s]", flush=True)

    import tests._pipeline_runner as runner
    from tests._pipeline_runner_tiled import run_segmented_pipeline_tiled
    runner.PHASE1_RERANK_ENABLED = True
    runner.PHASE1_REASSIGN_AFTER_1C = True

    # Load saved sequential partition (the reference)
    seq_part = pd.read_parquet(SEQ_PART)
    seq_part = seq_part.set_index("transcript_id").reindex(df["transcript_id"]).reset_index()
    seq_lab = seq_part["label"].astype(str)

    # Run tiled WITHOUT global stitch
    print(f"\nrunning tile-parallel {N_TILES_XY} (naive concat) ...", flush=True)
    t = time.time()
    result_naive = run_segmented_pipeline_tiled(
        df.copy(), panel,
        n_tiles_xy=N_TILES_XY, n_workers=N_TILES_XY[0] * N_TILES_XY[1],
        rerank=True, reassign=True, global_stitch=False, show_progress=True,
    )
    print(f"  wall_total: {result_naive['wall_total_seconds']:.1f}s", flush=True)
    df_naive = result_naive["df_out"]
    col = "stitched" if "stitched" in df_naive.columns else "tracer_id"
    naive_lab = (df_naive.set_index("transcript_id")[col].astype(str)
                  .reindex(df["transcript_id"]))

    # Run tiled WITH global stitch
    print(f"\nrunning tile-parallel {N_TILES_XY} + global Stitch ...", flush=True)
    t = time.time()
    result_gs = run_segmented_pipeline_tiled(
        df.copy(), panel,
        n_tiles_xy=N_TILES_XY, n_workers=N_TILES_XY[0] * N_TILES_XY[1],
        rerank=True, reassign=True, global_stitch=True, show_progress=True,
    )
    print(f"  wall_total: {result_gs['wall_total_seconds']:.1f}s "
          f"(global-stitch stats: {result_gs['global_stitch_stats']})", flush=True)
    df_gs = result_gs["df_out"]
    gs_lab = (df_gs.set_index("transcript_id")[col].astype(str)
              .reindex(df["transcript_id"]))

    # Metric tables
    _metrics_table(seq_lab, naive_lab, "tiled_3x3 naive vs sequential")
    _metrics_table(seq_lab, gs_lab, "tiled_3x3 + global Stitch vs sequential")

    # Persist partitions
    pd.DataFrame({"transcript_id": df["transcript_id"], "label": naive_lab.to_numpy()}
                ).to_parquet(OUT_DIR / "partition_naive.parquet", index=False)
    pd.DataFrame({"transcript_id": df["transcript_id"], "label": gs_lab.to_numpy()}
                ).to_parquet(OUT_DIR / "partition_global_stitch.parquet", index=False)
    summary = {
        "wall_naive_seconds": result_naive["wall_total_seconds"],
        "wall_gs_seconds": result_gs["wall_total_seconds"],
        "global_stitch_stats": result_gs["global_stitch_stats"],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsummary -> {OUT_DIR}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
