#!/usr/bin/env python3
"""PDAC 500 µm ROI: USE_ETYPE_COLUMN flag on/off comparison.

Two SEG runs on the same ROI, differing only in the
USE_ETYPE_COLUMN flag. Expected behavior:

  - flag=False (legacy regex rerank): silently no-ops on FFPE cell_ids;
    n_parents_reranked = 0; Phase1-Rerank produces no real moves.
  - flag=True (etype-aware rerank): correctly identifies depth-1
    entities via cell_id column; n_parents_reranked > 0; real moves
    happen.

Uses an etype-aware entity counter (reads _etype if present) so the
n_cells / n_partials breakdown is correct on FFPE labels in BOTH
runs (the bench's classifier is independent of the rerank reader).

Run from this worktree root:
    PYTHONPATH=src:. python benchmarks/bench_pdac_roi_etype.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROI_CENTER = (7255.0, 3023.7)   # PDAC median (x, y)
ROI_HALF_SIDE = 250.0           # 500 µm box

PARQUET = Path("/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/transcripts.parquet")
PANEL = Path("/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/stoic-feynman-587f37/benchmarks/pmi_bs_pdac_io_C_5_95_freshbench.csv")

REPO = Path(__file__).resolve().parents[1]
OUT_JSON = REPO / "benchmarks" / "pdac_roi_etype.json"
OUT_LOG = REPO / "benchmarks" / "pdac_roi_etype.log"


# ---------------------------------------------------------------------------
# Etype-aware entity counter: prefers _etype column; falls back to legacy.
# ---------------------------------------------------------------------------

UNASSIGNED_TOKENS = {"-1", "DROP", "UNASSIGNED", "nan"}


def _entity_counts_smart(df: pd.DataFrame, entity_col: str = "tracer_id",
                          cell_id_col: str = "cell_id") -> dict:
    """Classify entities using the _etype column when available;
    otherwise fall back to cell_id-prefix-stripping (which works on
    FFPE cell_ids unlike the dash-based legacy classifier)."""
    labels = df[entity_col].astype(str).to_numpy()
    if "_etype" in df.columns:
        etype = df["_etype"].astype(str).to_numpy()
        # Unique (label, etype) pairs → count by etype
        pairs = pd.DataFrame({"label": labels, "etype": etype}).drop_duplicates(
            "label"
        )
        per_kind = pairs["etype"].value_counts().to_dict()
        n_cell = int(per_kind.get("cell", 0))
        n_partial = int(per_kind.get("partial", 0))
        n_component = int(per_kind.get("component", 0))
        n_unknown_entities = int(per_kind.get("unknown", 0))
    else:
        # Fallback for older bench: classify via cell_id_prefix-strip
        # (NOT the legacy dash-only classifier).
        cell_ids = df[cell_id_col].astype(str).to_numpy()
        pairs = pd.DataFrame({"label": labels, "cid": cell_ids}).drop_duplicates(
            "label"
        )
        kinds = []
        for lab, cid in zip(pairs["label"], pairs["cid"]):
            if lab in UNASSIGNED_TOKENS or lab.endswith("_rejected"):
                kinds.append("unknown")
            elif lab.startswith("UNASSIGNED_"):
                kinds.append("component")
            elif cid in UNASSIGNED_TOKENS:
                kinds.append("unknown")
            elif lab == cid:
                kinds.append("cell")
            elif lab.startswith(cid + "-"):
                kinds.append("partial")
            elif lab.startswith("cascade_"):
                kinds.append("partial")  # cascade entities classified as partial
            else:
                kinds.append("unknown")
        per_kind = pd.Series(kinds).value_counts().to_dict()
        n_cell = int(per_kind.get("cell", 0))
        n_partial = int(per_kind.get("partial", 0))
        n_component = int(per_kind.get("component", 0))
        n_unknown_entities = int(per_kind.get("unknown", 0))

    # Tx-level
    s = pd.Series(labels)
    is_un = s.isin(UNASSIGNED_TOKENS) | s.str.endswith("_rejected", na=False)
    n_unassigned_tx = int(is_un.sum())
    n_assigned_tx = int(len(s) - n_unassigned_tx)

    return {
        "n_cells": n_cell,
        "n_partials": n_partial,
        "n_components": n_component,
        "n_unknown_entities": n_unknown_entities,
        "n_unassigned_tx": n_unassigned_tx,
        "n_assigned_tx": n_assigned_tx,
        "coverage_pct": round(100 * n_assigned_tx / max(len(s), 1), 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log_lines: list[str] = []

    def log(s: str = ""):
        print(s, flush=True)
        log_lines.append(s)
        OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
        OUT_LOG.write_text("\n".join(log_lines) + "\n")

    log("=" * 100)
    log("PDAC 500 µm ROI — USE_ETYPE_COLUMN flag on/off comparison")
    log("=" * 100)
    log(f"ROI center: {ROI_CENTER}, half-side: {ROI_HALF_SIDE} µm")
    log(f"Parquet:    {PARQUET}")
    log(f"Panel:      {PANEL.name}")
    log()

    if not PANEL.exists():
        log(f"ERROR: panel not found at {PANEL}")
        return 1

    t0 = time.time()
    df_full = pd.read_parquet(
        PARQUET,
        columns=["transcript_id", "cell_id", "overlaps_nucleus",
                 "feature_name", "x_location", "y_location", "z_location"],
    )
    df_full = df_full.rename(columns={
        "x_location": "x", "y_location": "y", "z_location": "z"
    })
    xc, yc = ROI_CENTER
    h = ROI_HALF_SIDE
    mask = (
        df_full["x"].between(xc - h, xc + h)
        & df_full["y"].between(yc - h, yc + h)
    )
    df = df_full.loc[mask].reset_index(drop=True)
    del df_full
    n_tx = len(df)
    n_cells_input = df["cell_id"].nunique()
    log(f"Loaded {n_tx:,} tx, {n_cells_input:,} unique input cell_ids  [{time.time()-t0:.1f}s]")
    log(f"Sample cell_ids: {list(df['cell_id'].astype(str).unique()[:3])}")
    log()

    panel = pd.read_csv(PANEL)
    log(f"Panel rows: {len(panel):,}")
    log()

    import tests._pipeline_runner as runner
    from tests._pipeline_runner import run_segmented_pipeline

    # Confirm production state before flipping any knobs.
    log("Runner constants at load time:")
    log(f"  PHASE1_REASSIGN_AFTER_1C  = {runner.PHASE1_REASSIGN_AFTER_1C}")
    log(f"  PHASE1_RERANK_ENABLED     = {runner.PHASE1_RERANK_ENABLED}")
    log(f"  USE_ETYPE_COLUMN          = {runner.USE_ETYPE_COLUMN}")
    log(f"  PMI_THR                   = {runner.PMI_THR}")
    log()

    # We want to isolate the Rerank-on-PDAC effect, so:
    #   PHASE1_REASSIGN_AFTER_1C = False (don't conflate with reassign moves)
    #   PHASE1_RERANK_ENABLED    = True
    #   USE_ETYPE_COLUMN         = False (legacy) then True (etype)
    orig_reassign = runner.PHASE1_REASSIGN_AFTER_1C
    orig_rerank = runner.PHASE1_RERANK_ENABLED
    orig_flag = runner.USE_ETYPE_COLUMN

    results: dict = {
        "roi_center": ROI_CENTER, "roi_half_side": ROI_HALF_SIDE,
        "n_input_tx": int(n_tx),
        "n_input_cells": int(n_cells_input),
        "panel": str(PANEL),
    }

    try:
        runner.PHASE1_REASSIGN_AFTER_1C = False
        runner.PHASE1_RERANK_ENABLED = True

        # ---- Legacy rerank reader (flag off) ----
        log("=" * 100)
        log("[1/2] USE_ETYPE_COLUMN=False (legacy regex rerank)")
        log("=" * 100)
        runner.USE_ETYPE_COLUMN = False
        t = time.time()
        df_legacy, prog_legacy = run_segmented_pipeline(df.copy(), panel)
        wall_legacy = time.time() - t
        stages_legacy = [p["stage"] for p in prog_legacy]
        log(f"  wall: {wall_legacy:.1f}s  stages: {len(stages_legacy)}")
        log(f"  Phase1-Rerank in stages: {'Phase1-Rerank' in stages_legacy}")
        counts_legacy = _entity_counts_smart(df_legacy)
        log(f"  entities: {counts_legacy}")
        results["legacy"] = {
            "wall_seconds": round(wall_legacy, 2),
            "rerank_stage_present": "Phase1-Rerank" in stages_legacy,
            "entity_counts": counts_legacy,
        }
        log()

        # ---- Etype-aware rerank reader (flag on) ----
        log("=" * 100)
        log("[2/2] USE_ETYPE_COLUMN=True (etype-aware rerank)")
        log("=" * 100)
        runner.USE_ETYPE_COLUMN = True
        t = time.time()
        df_etype, prog_etype = run_segmented_pipeline(df.copy(), panel)
        wall_etype = time.time() - t
        stages_etype = [p["stage"] for p in prog_etype]
        log(f"  wall: {wall_etype:.1f}s  stages: {len(stages_etype)}")
        log(f"  Phase1-Rerank in stages: {'Phase1-Rerank' in stages_etype}")
        counts_etype = _entity_counts_smart(df_etype)
        log(f"  entities: {counts_etype}")
        results["etype"] = {
            "wall_seconds": round(wall_etype, 2),
            "rerank_stage_present": "Phase1-Rerank" in stages_etype,
            "entity_counts": counts_etype,
        }
        log()
    finally:
        runner.PHASE1_REASSIGN_AFTER_1C = orig_reassign
        runner.PHASE1_RERANK_ENABLED = orig_rerank
        runner.USE_ETYPE_COLUMN = orig_flag

    # ---- Compare ----
    log("=" * 100)
    log("Comparison: legacy vs etype")
    log("=" * 100)
    leg = results["legacy"]["entity_counts"]
    etyp = results["etype"]["entity_counts"]
    log(f"  {'metric':30s}  {'legacy':>10s}  {'etype':>10s}  {'Δ':>8s}")
    for k in ["n_cells", "n_partials", "n_components",
              "n_unknown_entities", "n_unassigned_tx", "n_assigned_tx"]:
        d = etyp[k] - leg[k]
        log(f"  {k:30s}  {leg[k]:>10d}  {etyp[k]:>10d}  {d:>+8d}")
    log(f"  {'coverage_pct':30s}  {leg['coverage_pct']:>10.3f}  {etyp['coverage_pct']:>10.3f}")

    # Label-level diff
    diff = (
        df_legacy["tracer_id"].astype(str).to_numpy()
        != df_etype["tracer_id"].astype(str).to_numpy()
    )
    n_diff = int(diff.sum())
    log(f"  Tx with different final label: {n_diff:,} / {len(df_legacy):,} ({100*n_diff/len(df_legacy):.2f}%)")
    results["n_tx_label_diff"] = n_diff

    log()
    log("Verdict:")
    if n_diff > 0 and counts_etype["n_cells"] > counts_legacy["n_cells"]:
        log("  ✅ etype path corrects the FFPE classification bug — more cells recovered.")
    elif n_diff > 0:
        log(f"  ⚠️ etype changes output but not in the expected direction. Inspect counts.")
    else:
        log("  ❌ etype and legacy produce identical output — the flag isn't taking effect, OR")
        log("     PDAC ROI happens to have no rerank candidates regardless of reader.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nWrote: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
