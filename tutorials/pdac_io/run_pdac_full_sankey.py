#!/usr/bin/env python3
"""Full pdac_io SEG pipeline + transcript-flow Sankey rendering.

Same pipeline as `run_pdac_full_pipeline.py` but monkey-patches the
runner's `_record_stage` to also call `tracer.sankey_log.snapshot_phase`
at every phase boundary, then renders Tier A + Tier B Sankey artifacts
showing how transcripts move between assignment classes across the full
SEG pipeline.

Outputs (under tutorials/pdac_io/output/full_sankey/):
  - partition_sequential.parquet   per-tx (incl. all etype_at_<phase>
                                   snapshot columns)
  - summary.json                   wall-clock, RSS peak, entity counts
  - pdac_full_tier_a.{html,png}    collapsed view (4 columns)
  - pdac_full_tier_b.{html,png}    default view (8 columns)
  - phase_counts.csv               per-phase class breakdown

Usage (from the worktree root):

    PYTHONPATH=src:. /opt/homebrew/Caskroom/miniconda/base/envs/genesis_env/bin/python \\
        tutorials/pdac_io/run_pdac_full_sankey.py

Knobs (env vars):
  PANEL_CSV        override panel (default: pmi_bs_pdac_io_C_5_95.csv)
  PMI_THR, RESCUE_MEAN_ADMIT, RESCUE_AGGREGATOR_PERCENTILE   runner knobs
  OUT_TAG          suffix on the output dir name (default empty)

Expected scale: ~3-5M transcripts. Wall-clock 30 min - 2 h. Plus
rendering: matplotlib ~10-30 s; plotly ~30-60 s. Consider launching
with `nohup ... > run.log 2>&1 &`.
"""
from __future__ import annotations

import json
import os
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if sys.platform == "darwin" else int(r) * 1024


TUT_DIR = Path(__file__).resolve().parent
REPO = TUT_DIR.parents[1]

PDAC_PARQUET = TUT_DIR / "data" / "outs" / "transcripts.parquet"
PANEL_CSV = Path(os.environ.get(
    "PANEL_CSV",
    str(TUT_DIR / "data" / "pmi_bs_pdac_io_C_5_95.csv"),
))

OUT_TAG = os.environ.get("OUT_TAG", "")
OUT_DIR = TUT_DIR / "output" / (f"full_sankey{OUT_TAG}" if OUT_TAG else "full_sankey")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── load tracer (full install required: needs torch_geometric, etc.) ──
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from tracer import sankey_log as sl
from tracer import flow_plot as fp


# ─── stage → phase-key mapping (must mirror the runner's _record_stage names) ──
STAGE_TO_PHASE_KEY = {
    "input": "input",
    "Prune": "phase1",            # Tier B canonical key for Phase-1 output
    "Phase1-Reassign-1c": None,   # verbose-only sub-step; skip in default
    "Split-Phase1": None,
    "Phase1-Rerank": None,
    "Phase1-QC": None,
    "Phase1-Maha-Remerge": None,
    "Split": None,                # second split (post-Phase-1) — no plan key
    "Rescue": "rescue",
    "Group": "group",
    "Mid-QC": "mid_qc",
    "Post-Group Rescue": "post_group_rescue",
    "Stitch": "stitch",
    "Demote": "demote",
    "Final Rescue": "final_rescue",
    "Finalize": "finalize",       # snapshot it; default tier excludes it
}

POST_RESCUE_PHASES = {
    "rescue", "group", "mid_qc", "post_group_rescue",
    "stitch", "demote", "final_rescue", "finalize",
}

# ─── extended classification: distinguish neighbor-cell mains ──
CLASS_MAIN_NEIGHBOR = 5

sl.CLASS_NAMES.update({
    sl.CLASS_MAIN:        "original cell",
    sl.CLASS_PARTIAL:     "partial",
    sl.CLASS_UNASSIGNED:  "unassigned",
    sl.CLASS_DROPPED:     "dropped",
    sl.CLASS_COMPONENT:   "component",
    CLASS_MAIN_NEIGHBOR:  "neighboring cell",
})
sl.CLASS_COLLAPSE_3.update({
    CLASS_MAIN_NEIGHBOR: sl.CLASS_MAIN,
})

EXT_PALETTE = {
    sl.CLASS_MAIN:        "#1f77b4",  # blue
    sl.CLASS_PARTIAL:     "#2ca02c",  # green
    sl.CLASS_COMPONENT:   "#9467bd",  # purple
    sl.CLASS_UNASSIGNED:  "#7f7f7f",  # grey
    sl.CLASS_DROPPED:     "#d62728",  # red
    CLASS_MAIN_NEIGHBOR:  "#ff7f0e",  # orange
}

# Top-to-bottom visual order
ORDER = [
    CLASS_MAIN_NEIGHBOR,
    sl.CLASS_MAIN,
    sl.CLASS_PARTIAL,
    sl.CLASS_UNASSIGNED,
    sl.CLASS_COMPONENT,
    sl.CLASS_DROPPED,
]

_UNASSIGNED_TOKENS = {"-1", "DROP", "demote_rejected", "UNASSIGNED", "nan", ""}


def _is_original_match(current_id: str, orig_cell_id: str) -> bool:
    """True if the tx's CURRENT entity id is a partition of (or equals) its
    ORIGINAL segmentation cell_id. PDAC FFPE cell_ids look like
    'dafehkie-1'; partials look like 'dafehkie-1-N'."""
    if orig_cell_id in _UNASSIGNED_TOKENS or current_id in _UNASSIGNED_TOKENS:
        return False
    if current_id == orig_cell_id:
        return True
    return current_id.startswith(orig_cell_id + "-")


_neighboring_cols: list[str] = []


def main() -> int:
    print("=" * 78)
    print("FULL pdac_io SEG pipeline + transcript-flow Sankey")
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

    # ─── load runner via importlib (tests/ has no __init__.py) ─────────
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("_pipeline_runner",
                                        REPO / "tests/_pipeline_runner.py")
    runner = _ilu.module_from_spec(spec)
    sys.modules["_pipeline_runner"] = runner
    spec.loader.exec_module(runner)

    runner.PHASE1_RERANK_ENABLED = True
    runner.PHASE1_REASSIGN_AFTER_1C = True
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

    # ─── monkey-patch _record_stage to snapshot etype + neighbor flag ──
    _orig_record = runner._record_stage

    def _patched_record_stage(progression, stage_name, df_p, col):
        phase_key = STAGE_TO_PHASE_KEY.get(stage_name)
        if phase_key is not None:
            try:
                sl.snapshot_phase(df_p, phase_key, id_col=col)
            except Exception as e:
                print(f"  ⚠️  snapshot_phase failed for {stage_name!r}: {e}",
                      flush=True)
            if phase_key in POST_RESCUE_PHASES and "cell_id" in df_p.columns:
                orig = df_p["cell_id"].astype(str).to_numpy()
                curr = df_p[col].astype(str).to_numpy()
                matches = np.fromiter(
                    (_is_original_match(c, o) for c, o in zip(curr, orig)),
                    dtype=bool, count=len(curr),
                )
                is_real_assign = ~np.isin(curr, list(_UNASSIGNED_TOKENS))
                is_real_orig = ~np.isin(orig, list(_UNASSIGNED_TOKENS))
                df_p[f"neighboring_at_{phase_key}"] = (
                    is_real_assign & is_real_orig & (~matches)
                ).astype(bool)
                if phase_key not in _neighboring_cols:
                    _neighboring_cols.append(phase_key)
        return _orig_record(progression, stage_name, df_p, col)

    runner._record_stage = _patched_record_stage

    # ─── pre-pipeline input snapshot (runner's _record_stage("input") uses a
    #     temp df.assign(...) so a monkey-patched snapshot never reaches df) ──
    in_ids = df["cell_id"].astype(str).to_numpy()
    in_un = np.isin(in_ids, list(_UNASSIGNED_TOKENS))
    df["etype_at_input"] = np.where(
        in_un, sl.CLASS_UNASSIGNED, sl.CLASS_MAIN
    ).astype(np.int8)
    print(f"input snapshot: main={int((df['etype_at_input']==sl.CLASS_MAIN).sum()):,}, "
          f"unassigned={int((df['etype_at_input']==sl.CLASS_UNASSIGNED).sum()):,}",
          flush=True)

    # ─── run ───────────────────────────────────────────────────────────
    print()
    print("running sequential SEG ...", flush=True)
    t = time.time()
    df_out, info = runner.run_segmented_pipeline(df.copy(), panel)
    wall = time.time() - t
    print(f"  wall: {wall:.1f}s ({wall/60:.1f} min)", flush=True)
    print(f"  stages: {[p['stage'] for p in info]}", flush=True)

    etype_cols = sorted([c for c in df_out.columns if c.startswith("etype_at_")])
    print(f"  snapshot columns: {etype_cols}", flush=True)

    # ─── apply original-vs-neighbor distinction (mains only) ───────────
    n_neighbor_total = 0
    for phase_key in _neighboring_cols:
        col = f"etype_at_{phase_key}"
        if col not in df_out.columns:
            continue
        nflag = df_out[f"neighboring_at_{phase_key}"].to_numpy()
        codes = df_out[col].to_numpy()
        main_neighbor = (codes == sl.CLASS_MAIN) & nflag
        codes_ext = codes.copy()
        codes_ext[main_neighbor] = CLASS_MAIN_NEIGHBOR
        df_out[col] = codes_ext.astype("int8")
        n_neighbor_total += int(main_neighbor.sum())
    print(f"  marked {n_neighbor_total:,} (tx × phase) main cells as 'neighbor'",
          flush=True)

    # ─── render Sankey ─────────────────────────────────────────────────
    print()
    print("rendering plots ...", flush=True)
    common = dict(palette=EXT_PALETTE, class_grouping="five",
                  class_order=ORDER, pipeline="seg",
                  drop_unchanged=False, label_target="ribbons")

    out_a_html = OUT_DIR / "pdac_full_tier_a.html"
    out_a_png  = OUT_DIR / "pdac_full_tier_a.png"
    out_b_html = OUT_DIR / "pdac_full_tier_b.html"
    out_b_png  = OUT_DIR / "pdac_full_tier_b.png"

    title_a = f"pdac_io full sample — Tier A (n={len(df_out):,} tx)"
    title_b = f"pdac_io full sample — Tier B (n={len(df_out):,} tx)"

    fp.plot_transcript_flow(df_out, backend="plotly", view="collapsed",
                            output=str(out_a_html), title=title_a, **common)
    fp.plot_transcript_flow(df_out, backend="matplotlib", view="collapsed",
                            output=str(out_a_png), title=title_a, **common)
    fp.plot_transcript_flow(df_out, backend="plotly", view="default",
                            output=str(out_b_html), title=title_b, **common)
    fp.plot_transcript_flow(df_out, backend="matplotlib", view="default",
                            output=str(out_b_png), title=title_b, **common)
    for p in (out_a_html, out_a_png, out_b_html, out_b_png):
        print(f"  wrote {p}")

    # ─── persist partition (incl. snapshot cols) + summary + counts CSV ──
    print()
    out_cols = ["transcript_id", "cell_id", "x", "y", "z"]
    keep_extra = [c for c in df_out.columns
                  if c.startswith("etype_at_") or c.startswith("neighboring_at_")]
    label_col = "stitched" if "stitched" in df_out.columns else "tracer_id"
    df_persist = df_out[out_cols + [label_col] + keep_extra].copy()
    df_persist = df_persist.rename(columns={label_col: "label"})

    part_path = OUT_DIR / "partition_sequential.parquet"
    df_persist.to_parquet(part_path, index=False)
    print(f"partition -> {part_path}")

    # per-phase class counts
    counts_rows = []
    class_cols = [0, 1, 2, 3, 4, CLASS_MAIN_NEIGHBOR]
    for phase_key in ["input", "phase1", "rescue", "group", "mid_qc",
                       "post_group_rescue", "stitch", "demote",
                       "final_rescue", "finalize"]:
        col = f"etype_at_{phase_key}"
        if col not in df_out.columns:
            continue
        cnts = pd.Series(df_out[col]).value_counts().reindex(class_cols, fill_value=0)
        counts_rows.append({
            "phase": phase_key,
            **{sl.CLASS_NAMES[c]: int(cnts.loc[c]) for c in class_cols},
        })
    counts_path = OUT_DIR / "phase_counts.csv"
    pd.DataFrame(counts_rows).to_csv(counts_path, index=False)
    print(f"counts    -> {counts_path}")

    peak_gb = _peak_rss_bytes() / (1024 ** 3)
    summary = {
        "panel_path": str(PANEL_CSV),
        "panel_rows": int(len(panel)),
        "wall_seconds": round(wall, 2),
        "peak_rss_gb": round(peak_gb, 3),
        "n_input_tx": n_in_tx,
        "n_input_cells": n_in_cells,
        "n_neighbor_main_marks_total": int(n_neighbor_total),
        "stages": [p["stage"] for p in info],
        "snapshot_phases": etype_cols,
    }
    sum_path = OUT_DIR / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"summary   -> {sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
