#!/usr/bin/env python3
"""Re-run arm D on the 2 mm PDAC ROI to capture the Stitch gate-stats funnel.

Dumps `_LAST_GATE_STATS` (init bypasses, total merges, mahalanobis_rescue_checks,
mahalanobis_rescues) plus an extra "candidate_pairs_total" counter we patch in.
"""
from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

PDAC = Path("/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/"
            "data/outs/transcripts.parquet")
BOOT_DIR = Path("/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
                "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac")
OUT = Path("analysis/phase1_vs_stitch_maha/pdac_2mm")
ROI_CX, ROI_CY, ROI_HS = 7255.0, 3023.7, 1000.0


def main() -> int:
    import tests._pipeline_runner as runner
    from tests._pipeline_runner import run_segmented_pipeline
    from tracer.config import load_config
    from tracer import stitching

    # Monkey-patch a candidate-counter into the heap init via callback.
    orig_init = getattr(stitching, "_stitch_eager_heap_init", None)
    candidate_counter = {"n": 0, "after_witness": 0}

    # Wrap the heap-push to count every (i,j) the Stitch driver pushed.
    import heapq
    _orig_heappush = heapq.heappush

    def counted_heappush(heap, item):
        candidate_counter["n"] += 1
        return _orig_heappush(heap, item)

    heapq.heappush = counted_heappush

    df0 = pd.read_parquet(PDAC, columns=["transcript_id", "cell_id", "overlaps_nucleus",
        "feature_name", "x_location", "y_location", "z_location"]).rename(
        columns={"x_location": "x", "y_location": "y", "z_location": "z"})
    lo_x, hi_x = ROI_CX - ROI_HS, ROI_CX + ROI_HS
    lo_y, hi_y = ROI_CY - ROI_HS, ROI_CY + ROI_HS
    df0 = df0.loc[df0.x.between(lo_x, hi_x) & df0.y.between(lo_y, hi_y)].reset_index(drop=True)
    df0["cell_id"] = df0["cell_id"].astype(str)
    df0["feature_name"] = df0["feature_name"].astype(str)
    print(f"loaded {len(df0):,} tx in 2 mm ROI", flush=True)

    w = pd.read_parquet(BOOT_DIR / "W_thr10.parquet")[["gene_i", "gene_j", "value"]]
    boot = w.rename(columns={"value": "NPMI"})
    ci = pd.read_parquet(BOOT_DIR / "pair_ci_thr10.parquet")
    have = set(map(frozenset, zip(w.gene_i, w.gene_j)))
    extra = ci[ci.legacy_pmi.notna()]
    extra = extra[[fs not in have for fs in map(frozenset, zip(extra.gene_i, extra.gene_j))]]
    aug = pd.concat([boot, extra[["gene_i", "gene_j", "legacy_pmi"]].rename(
        columns={"legacy_pmi": "NPMI"})], ignore_index=True)
    print(f"panel: {len(aug):,} pairs", flush=True)

    from dataclasses import replace
    cfg = load_config()
    cfg = replace(cfg,
                  phase1=replace(cfg.phase1, maha_remerge_d=1.0),
                  stitch=replace(cfg.stitch, mahalanobis_d_rescue=1.0))
    runner.PHASE1_RERANK_ENABLED = True
    runner.PHASE1_REASSIGN_AFTER_1C = True

    t0 = time.time()
    out = run_segmented_pipeline(df0.copy(), aug, cfg=cfg)
    print(f"pipeline done in {time.time()-t0:.1f}s", flush=True)

    gate = dict(getattr(stitching, "_LAST_GATE_STATS", {}))
    gate["heap_pushes_total_including_phase1"] = candidate_counter["n"]
    print("\n=== Stitch gate stats (arm D, 2 mm PDAC ROI) ===")
    print(json.dumps(gate, indent=2, sort_keys=True))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stitch_gate_stats.json").write_text(json.dumps(gate, indent=2, sort_keys=True))
    print(f"saved -> {OUT/'stitch_gate_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
