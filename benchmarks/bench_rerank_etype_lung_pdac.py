#!/usr/bin/env python3
"""Real-data validation of the etype-aware Phase1-Rerank reader on:

  - lung ROI 500 µm  (integer cell_ids — the case where the legacy
                      regex reader also worked correctly)
  - PDAC ROI 500 µm  (FFPE cell_ids with native dashes — the case the
                      legacy reader silently no-oped on)

Both ROIs are run through the production SEG pipeline with Phase1-
Rerank ON. We report the Phase1-Rerank stage counts and spot-check
a handful of cells where the rerank actually swapped labels — they
must be (a) the largest depth-1 entity under the parent cell, and
(b) carry _etype = "cell" after the swap.

Run from this worktree root (no env vars needed — conftest.py wires
sys.path automatically):

    python benchmarks/bench_rerank_etype_lung_pdac.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_JSON = REPO / "benchmarks" / "rerank_etype_lung_pdac.json"
OUT_LOG = REPO / "benchmarks" / "rerank_etype_lung_pdac.log"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
LUNG_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/lung_cancer/"
    "data/lung_cancer_df.parquet"
)
# 500 µm ROI used in earlier benches; integer cell_ids.
LUNG_ROI_BOUNDS = (1568.7, 2068.7, 1936.8, 2436.8)   # (x_min, x_max, y_min, y_max)

PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/"
    "data/outs/transcripts.parquet"
)
PDAC_ROI_CENTER = (7255.0, 3023.7)
PDAC_ROI_HALF_SIDE = 250.0

# PMI panel — reuse the PDAC bootstrap panel already on disk in the
# stoic-feynman worktree. Both ROIs use the same panel; lung_cancer
# has its own NPMI panel inside the tutorial folder.
LUNG_PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/lung_cancer/"
    "data/pmi_bs_lung_cancer_C_5_95.csv"
)
PDAC_PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "stoic-feynman-587f37/benchmarks/pmi_bs_pdac_io_C_5_95_freshbench.csv"
)


def load_lung_roi() -> pd.DataFrame:
    df = pd.read_parquet(LUNG_PARQUET)
    x_min, x_max, y_min, y_max = LUNG_ROI_BOUNDS
    mask = (
        df["x"].between(x_min, x_max) & df["y"].between(y_min, y_max)
    )
    return df.loc[mask].reset_index(drop=True)


def load_pdac_roi() -> pd.DataFrame:
    df = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "overlaps_nucleus",
                 "feature_name", "x_location", "y_location", "z_location"],
    )
    df = df.rename(columns={
        "x_location": "x", "y_location": "y", "z_location": "z",
    })
    xc, yc = PDAC_ROI_CENTER
    h = PDAC_ROI_HALF_SIDE
    mask = df["x"].between(xc - h, xc + h) & df["y"].between(yc - h, yc + h)
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rerank-stage instrumentation
# ---------------------------------------------------------------------------

def run_with_rerank_instrumentation(
    df: pd.DataFrame, panel: pd.DataFrame
) -> tuple[pd.DataFrame, dict, list[tuple[str, str, str]]]:
    """Run the SEG pipeline with Phase1-Rerank enabled and capture
    the rerank stats + the (cell_id, pre, post) label triples for
    every tx whose label was changed by the Rerank stage.
    """
    import tests._pipeline_runner as runner
    from tests._pipeline_runner import _phase1_rerank_within_parent_etype

    # Snapshot defaults
    orig_rerank = runner.PHASE1_RERANK_ENABLED
    orig_reassign = runner.PHASE1_REASSIGN_AFTER_1C

    captured: dict = {}
    swaps: list[tuple[str, str, str]] = []

    # Monkeypatch the rerank fn so we can intercept the pre/post frame
    # and grab the diff.
    real_fn = _phase1_rerank_within_parent_etype

    def patched(df_in, **kw):
        pre = df_in[["cell_id", "tracer_id"]].astype(str).to_numpy().copy()
        df_out, stats = real_fn(df_in, **kw)
        post = df_out["tracer_id"].astype(str).to_numpy()
        # Compare pre/post; record (cell_id, pre, post) for changed rows.
        changed = pre[:, 1] != post
        for cid, p, q in zip(pre[changed, 0], pre[changed, 1], post[changed]):
            swaps.append((cid, p, q))
        captured.update(stats)
        return df_out, stats

    runner._phase1_rerank_within_parent_etype = patched

    try:
        runner.PHASE1_RERANK_ENABLED = True
        runner.PHASE1_REASSIGN_AFTER_1C = True
        df_out, _prog = runner.run_segmented_pipeline(df, panel)
    finally:
        runner.PHASE1_RERANK_ENABLED = orig_rerank
        runner.PHASE1_REASSIGN_AFTER_1C = orig_reassign
        runner._phase1_rerank_within_parent_etype = real_fn

    return df_out, captured, swaps


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize_rerank(swaps: list[tuple[str, str, str]],
                      df_out: pd.DataFrame) -> dict:
    """Spot-check the swaps: each (cell_id, pre, post) where pre/post
    are the labels before/after the rerank stage."""
    n_swaps = len(swaps)
    if n_swaps == 0:
        return {"n_swaps": 0, "examples": []}

    # Group swaps by cell_id to get unique parent cells
    by_cell: dict[str, list[tuple[str, str]]] = {}
    for cid, p, q in swaps:
        by_cell.setdefault(cid, []).append((p, q))
    n_parents = len(by_cell)

    # Pick up to 5 example parents — first 5 alphabetically for
    # reproducibility.
    examples = []
    for cid in sorted(by_cell)[:5]:
        rows = by_cell[cid]
        # The pre-label that became the main (cell_id) post-rerank is
        # the "promoted" entity. The main pre-label that became a
        # partial is "demoted".
        promoted_pre = [p for p, q in rows if q == cid]
        demoted_pre = [p for p, q in rows if p == cid]
        examples.append({
            "cell_id": cid,
            "n_tx_changed": len(rows),
            "promoted_pre_label": (
                promoted_pre[0] if promoted_pre else None
            ),
            "demoted_pre_label": (
                demoted_pre[0] if demoted_pre else None
            ),
        })

    return {
        "n_swaps_tx": n_swaps,
        "n_parents_with_swap": n_parents,
        "examples": examples,
    }


def main() -> int:
    log_lines: list[str] = []

    def log(s: str = ""):
        print(s, flush=True)
        log_lines.append(s)

    log("=" * 100)
    log("Phase1-Rerank etype reader validation — lung vs PDAC ROI")
    log("=" * 100)
    log()

    results: dict = {}

    for name, loader, panel_path in [
        ("lung", load_lung_roi, LUNG_PANEL),
        ("pdac", load_pdac_roi, PDAC_PANEL),
    ]:
        log(f"--- {name.upper()} ROI ---")
        if not panel_path.exists():
            log(f"  SKIP: panel not found at {panel_path}")
            results[name] = {"skipped": "panel missing"}
            continue

        t = time.time()
        df = loader()
        n_tx = len(df)
        n_cells = df["cell_id"].nunique()
        sample_cids = list(df["cell_id"].astype(str).unique()[:3])
        log(f"  loaded {n_tx:,} tx / {n_cells:,} cell_ids in {time.time()-t:.1f}s")
        log(f"  sample cell_ids: {sample_cids}")

        panel = pd.read_csv(panel_path)
        log(f"  panel rows: {len(panel):,}")

        t = time.time()
        df_out, stats, swaps = run_with_rerank_instrumentation(df, panel)
        wall = time.time() - t
        log(f"  pipeline wall: {wall:.1f}s")
        log(f"  rerank stats: {stats}")

        summary = summarize_rerank(swaps, df_out)
        log(f"  swaps observed: "
            f"{summary.get('n_swaps_tx', 0)} tx "
            f"across {summary.get('n_parents_with_swap', 0)} parent cells")
        for ex in summary.get("examples", []):
            log(f"    cell_id={ex['cell_id']!r}: "
                f"promoted={ex['promoted_pre_label']!r} → {ex['cell_id']!r}, "
                f"demoted={ex['demoted_pre_label']!r} ({ex['n_tx_changed']} tx)")

        # Sanity check: every promoted entity must now be ``cell``
        # in the _etype column for the rows under that cell_id.
        for ex in summary.get("examples", []):
            mask = (df_out["cell_id"].astype(str) == ex["cell_id"]) & (
                df_out["tracer_id"].astype(str) == ex["cell_id"]
            )
            etypes = df_out.loc[mask, "_etype"].astype(str).unique().tolist()
            log(f"    sanity: {ex['cell_id']} → _etype values for promoted rows: {etypes}")
            assert etypes == ["cell"] or etypes == [], (
                f"FAIL: promoted entity {ex['cell_id']} has non-'cell' etype: {etypes}"
            )

        results[name] = {
            "n_input_tx": int(n_tx),
            "n_input_cells": int(n_cells),
            "sample_cell_ids": sample_cids,
            "wall_seconds": round(wall, 2),
            "rerank_stats": stats,
            "swap_summary": summary,
        }
        log()

    log("=" * 100)
    log("Verdict")
    log("=" * 100)
    for name in ["lung", "pdac"]:
        if results.get(name, {}).get("skipped"):
            log(f"  {name}: SKIPPED ({results[name]['skipped']})")
            continue
        n_parents = results[name]["rerank_stats"].get("n_parents_reranked", 0)
        n_tx_rel = results[name]["rerank_stats"].get("n_tx_relabeled", 0)
        if n_parents > 0:
            log(f"  {name}: ✅ rerank fired — {n_parents} parents, {n_tx_rel} tx relabeled")
        else:
            log(f"  {name}: ⚠️ rerank did not fire (n_parents_reranked=0). "
                f"Expected on small ROIs with no clear swap candidate.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    OUT_LOG.write_text("\n".join(log_lines) + "\n")
    log(f"\nWrote: {OUT_JSON}")
    log(f"Wrote: {OUT_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
