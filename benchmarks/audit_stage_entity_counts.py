#!/usr/bin/env python3
"""Per-stage entity-count audit on the 2 mm PDAC ROI (arm D config).

Runs the full pipeline once and prints n_cells / n_partials / n_components
(and their sum) at every stage in the progression record, so we can see
exactly where and how much entity consolidation happens around Stitch.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import pandas as pd

PDAC = Path("/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/"
            "data/outs/transcripts.parquet")
BOOT = Path("/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
            "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac")
OUT = Path("analysis/phase1_vs_stitch_maha/pdac_2mm")
ROI_CX, ROI_CY, ROI_HS = 7255.0, 3023.7, 1000.0


def main() -> int:
    import tests._pipeline_runner as runner
    from tests._pipeline_runner import run_segmented_pipeline
    from tracer.config import load_config

    df0 = pd.read_parquet(PDAC, columns=["transcript_id", "cell_id", "overlaps_nucleus",
        "feature_name", "x_location", "y_location", "z_location"]).rename(
        columns={"x_location": "x", "y_location": "y", "z_location": "z"})
    lo_x, hi_x = ROI_CX - ROI_HS, ROI_CX + ROI_HS
    lo_y, hi_y = ROI_CY - ROI_HS, ROI_CY + ROI_HS
    df0 = df0.loc[df0.x.between(lo_x, hi_x) & df0.y.between(lo_y, hi_y)].reset_index(drop=True)
    df0["cell_id"] = df0["cell_id"].astype(str)
    df0["feature_name"] = df0["feature_name"].astype(str)
    print(f"loaded {len(df0):,} tx in 2 mm ROI", flush=True)

    w = pd.read_parquet(BOOT / "W_thr10.parquet")[["gene_i", "gene_j", "value"]]
    boot = w.rename(columns={"value": "NPMI"})
    ci = pd.read_parquet(BOOT / "pair_ci_thr10.parquet")
    have = set(map(frozenset, zip(w.gene_i, w.gene_j)))
    extra = ci[ci.legacy_pmi.notna()]
    extra = extra[[fs not in have for fs in map(frozenset, zip(extra.gene_i, extra.gene_j))]]
    aug = pd.concat([boot, extra[["gene_i", "gene_j", "legacy_pmi"]].rename(
        columns={"legacy_pmi": "NPMI"})], ignore_index=True)

    cfg = load_config()
    cfg = replace(cfg,
                  phase1=replace(cfg.phase1, maha_remerge_d=1.0),
                  stitch=replace(cfg.stitch, mahalanobis_d_rescue=1.0))
    runner.PHASE1_RERANK_ENABLED = True
    runner.PHASE1_REASSIGN_AFTER_1C = True

    df_out, progression = run_segmented_pipeline(df0.copy(), aug, cfg=cfg)

    rows = []
    if progression:
        for i, s in enumerate(progression):
            stage = s.get("stage", f"stage_{i}")
            nc = int(s.get("n_cells", 0))
            np_ = int(s.get("n_partials", 0))
            ncomp = int(s.get("n_components", 0))
            sec_raw = s.get("stage_seconds", 0.0)
            sec = float(sec_raw) if sec_raw is not None else 0.0
            rows.append((i, stage, nc, np_, ncomp, nc + np_ + ncomp, sec))
    print(f"\n=== Per-stage entity counts (arm D, 2 mm PDAC ROI) ===")
    print(f"{'#':>2}  {'stage':32s}  {'cells':>7s}  {'partials':>9s}  {'comp':>7s}  {'total':>7s}  {'Δtotal':>8s}  {'sec':>6s}")
    print("-" * 100)
    prev = None
    for i, stage, nc, np_, ncomp, tot, sec in rows:
        delta = "" if prev is None else f"{tot - prev:+d}"
        print(f"{i:>2}  {stage:32s}  {nc:>7d}  {np_:>9d}  {ncomp:>7d}  {tot:>7d}  {delta:>8s}  {sec:>6.1f}")
        prev = tot
    # Save as CSV too.
    pd.DataFrame(rows, columns=["idx","stage","cells","partials","components","total","seconds"]).to_csv(
        OUT / "stage_entity_audit.csv", index=False)
    print(f"\nsaved -> {OUT/'stage_entity_audit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
