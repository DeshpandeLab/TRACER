"""Tile-parallel orchestrator for the SEG pipeline.

Partitions a large transcript frame into NxM spatially-disjoint tiles
by cell_id centroid, then runs the full SEG pipeline on each tile in
a separate process. Concatenates results.

Cell-centroid assignment guarantees that every transcript of a given
cell_id ends up in the same tile. Since Stitch/Rescue's spatial reach
is small (~5-10 um) relative to a typical tile (~1-2 mm on a side),
the loss from skipping cross-tile resolution is small: only entities
within ~10 um of a tile edge can fail to stitch with a neighbor in an
adjacent tile.

This module deliberately keeps the boundary-resolution pass as a
documented gap rather than implementing it inline. Callers that need
perfect boundary handling should either (a) use larger tiles with a
margin pass, or (b) post-process the concatenated output with a
boundary-only Stitch.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Worker function (must be module-level so multiprocessing can pickle it)
# ---------------------------------------------------------------------------

def _tile_worker(args: dict) -> dict:
    """Run run_segmented_pipeline on a single tile.

    args dict (passed as a single dict so ProcessPoolExecutor.submit
    can serialize cleanly):
      - tile_idx: identifying integer
      - df:       pd.DataFrame (cell-complete tile slice)
      - panel:    pd.DataFrame (PMI panel; shared across tiles)
      - rerank:   bool (override PHASE1_RERANK_ENABLED)
      - reassign: bool (override PHASE1_REASSIGN_AFTER_1C)

    Returns dict with `tile_idx`, `df_out`, `progression`, `wall_seconds`.
    """
    import time as _t  # repeat-import for worker process
    import tests._pipeline_runner as runner
    from tests._pipeline_runner import run_segmented_pipeline

    tile_idx = int(args["tile_idx"])
    df = args["df"]
    panel = args["panel"]
    rerank = bool(args.get("rerank", True))
    reassign = bool(args.get("reassign", True))

    orig_rerank = runner.PHASE1_RERANK_ENABLED
    orig_reassign = runner.PHASE1_REASSIGN_AFTER_1C
    try:
        runner.PHASE1_RERANK_ENABLED = rerank
        runner.PHASE1_REASSIGN_AFTER_1C = reassign
        t0 = _t.time()
        df_out, prog = run_segmented_pipeline(df, panel)
        wall = _t.time() - t0
    finally:
        runner.PHASE1_RERANK_ENABLED = orig_rerank
        runner.PHASE1_REASSIGN_AFTER_1C = orig_reassign

    return {
        "tile_idx": tile_idx,
        "df_out": df_out,
        "progression": prog,
        "wall_seconds": round(wall, 2),
        "n_input_tx": int(len(df)),
        "n_input_cell_ids": int(df["cell_id"].nunique()),
    }


# ---------------------------------------------------------------------------
# Tiler
# ---------------------------------------------------------------------------

def _assign_cells_to_tiles(
    df: pd.DataFrame,
    *,
    n_tiles_xy: tuple[int, int],
    cell_id_col: str = "cell_id",
    coord_cols: tuple[str, str] = ("x", "y"),
) -> tuple[pd.Series, dict]:
    """Assign each cell_id to a tile by its centroid.

    Returns (cell_to_tile, tile_info) where:
      - cell_to_tile: pd.Series indexed by cell_id with integer tile_idx
        (flattened row-major index over the n_x * n_y grid)
      - tile_info: dict with bbox per tile_idx
    """
    n_x, n_y = n_tiles_xy
    x_col, y_col = coord_cols

    # Compute per-cell centroid
    cent = df.groupby(cell_id_col)[[x_col, y_col]].mean()

    # Bbox of the sample
    x_min, x_max = float(cent[x_col].min()), float(cent[x_col].max())
    y_min, y_max = float(cent[y_col].min()), float(cent[y_col].max())

    # Tile edges (clip the upper bound so a centroid AT x_max falls in the last bin)
    x_edges = np.linspace(x_min, x_max + 1e-9, n_x + 1)
    y_edges = np.linspace(y_min, y_max + 1e-9, n_y + 1)

    # Tile index per cell
    x_bin = np.clip(np.searchsorted(x_edges, cent[x_col].to_numpy(), side="right") - 1,
                    0, n_x - 1)
    y_bin = np.clip(np.searchsorted(y_edges, cent[y_col].to_numpy(), side="right") - 1,
                    0, n_y - 1)
    tile_idx = (x_bin * n_y + y_bin).astype(np.int32)

    # Build tile_info with bbox
    tile_info: dict[int, dict] = {}
    for ix in range(n_x):
        for iy in range(n_y):
            ti = ix * n_y + iy
            tile_info[ti] = {
                "x_min": float(x_edges[ix]), "x_max": float(x_edges[ix + 1]),
                "y_min": float(y_edges[iy]), "y_max": float(y_edges[iy + 1]),
                "ix": ix, "iy": iy,
            }

    cell_to_tile = pd.Series(tile_idx, index=cent.index, name="tile_idx")
    return cell_to_tile, tile_info


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def run_segmented_pipeline_tiled(
    df: pd.DataFrame,
    npmi_panel: pd.DataFrame,
    *,
    n_tiles_xy: tuple[int, int] = (2, 2),
    n_workers: int | None = None,
    cell_id_col: str = "cell_id",
    coord_cols: tuple[str, str] = ("x", "y"),
    rerank: bool = True,
    reassign: bool = True,
    show_progress: bool = False,
) -> dict:
    """Run run_segmented_pipeline across a spatial tile grid in parallel.

    Parameters
    ----------
    df : DataFrame
        Long-format transcript table with cell_id, x, y, z, feature_name,
        overlaps_nucleus, etc. — the same input run_segmented_pipeline
        expects.
    npmi_panel : DataFrame
        PMI panel (gene_i, gene_j, NPMI). Shared across all tiles.
    n_tiles_xy : (int, int)
        Tile grid: n_x * n_y tiles.
    n_workers : int or None
        Process pool size. Defaults to n_tiles_xy[0] * n_tiles_xy[1].
    cell_id_col, coord_cols : str, tuple
        Column names for cell_id and the xy coordinates used for tile
        assignment.
    rerank, reassign : bool
        Whether to enable Phase1-Rerank / Phase1-Reassign-1c in each
        tile's run (override the runner module's defaults).
    show_progress : bool
        Print tile-start/tile-finish lines.

    Returns
    -------
    result : dict with keys:
      - df_out: concatenated per-tx output (sorted by original index)
      - per_tile: dict[tile_idx, {wall_seconds, progression, n_input_tx,
                                  n_input_cell_ids, n_out_cells, n_out_partials}]
      - tile_info: bbox per tile
      - wall_total_seconds: end-to-end wall time
      - wall_max_tile_seconds: longest single-tile wall time
    """
    n_x, n_y = n_tiles_xy
    n_tiles = n_x * n_y
    if n_workers is None:
        n_workers = n_tiles

    # 1. Tile-assign each cell.
    cell_to_tile, tile_info = _assign_cells_to_tiles(
        df, n_tiles_xy=n_tiles_xy,
        cell_id_col=cell_id_col, coord_cols=coord_cols,
    )

    # 2. Slice df per tile (no copy of panel; that goes in via args).
    tile_dfs: dict[int, pd.DataFrame] = {}
    df_with_tile = df.assign(_tile_idx=df[cell_id_col].map(cell_to_tile).astype(np.int32))
    for ti, sub in df_with_tile.groupby("_tile_idx", sort=False):
        tile_dfs[int(ti)] = sub.drop(columns=["_tile_idx"]).reset_index(drop=True)

    if show_progress:
        print(f"[tiled] tile sizes: " + ", ".join(
            f"tile{ti}={len(td):,}tx/{td['cell_id'].nunique():,}cells"
            for ti, td in sorted(tile_dfs.items())
        ), flush=True)

    # 3. Dispatch worker jobs.
    args_list = [
        {"tile_idx": ti, "df": tile_dfs[ti], "panel": npmi_panel,
         "rerank": rerank, "reassign": reassign}
        for ti in sorted(tile_dfs)
    ]

    t_total = time.time()
    per_tile_results: dict[int, dict] = {}
    if n_workers <= 1:
        # Serial fallback: useful for tests / debugging.
        for a in args_list:
            r = _tile_worker(a)
            per_tile_results[r["tile_idx"]] = r
            if show_progress:
                print(f"[tiled] tile {r['tile_idx']} done in "
                      f"{r['wall_seconds']:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_tile_worker, a): a["tile_idx"] for a in args_list}
            for fut in as_completed(futs):
                r = fut.result()
                per_tile_results[r["tile_idx"]] = r
                if show_progress:
                    print(f"[tiled] tile {r['tile_idx']} done in "
                          f"{r['wall_seconds']:.1f}s "
                          f"({r['n_input_cell_ids']:,} input cells)", flush=True)
    wall_total = time.time() - t_total

    # 4. Concatenate outputs.
    parts = [per_tile_results[ti]["df_out"] for ti in sorted(per_tile_results)]
    df_out = pd.concat(parts, axis=0, ignore_index=True)

    # 5. Compute per-tile entity stats for the result summary.
    summary_per_tile: dict[int, dict] = {}
    for ti, r in per_tile_results.items():
        df_t = r["df_out"]
        col = "stitched" if "stitched" in df_t.columns else "tracer_id"
        s = df_t[col].astype(str)
        if "_etype" in df_t.columns:
            etype = df_t["_etype"].astype(str)
        else:
            from tracer._etype import infer_etype_from_label
            etype = pd.Series(np.asarray(infer_etype_from_label(s)).astype(str))
        unassigned_tokens = {"-1", "DROP", "UNASSIGNED", "nan"}
        is_un = s.isin(unassigned_tokens) | s.str.endswith("_rejected", na=False)
        pairs = pd.DataFrame({"lab": s, "etype": etype}).loc[~is_un.to_numpy()]
        per = pairs.drop_duplicates("lab")["etype"].value_counts().to_dict()
        summary_per_tile[ti] = {
            "wall_seconds": r["wall_seconds"],
            "progression": r["progression"],
            "n_input_tx": r["n_input_tx"],
            "n_input_cell_ids": r["n_input_cell_ids"],
            "n_out_cells": int(per.get("cell", 0)),
            "n_out_partials": int(per.get("partial", 0)),
            "n_out_components": int(per.get("component", 0)),
            "n_out_unassigned_tx": int(is_un.sum()),
        }

    wall_max_tile = max(r["wall_seconds"] for r in per_tile_results.values())
    speedup_vs_serial = (
        sum(r["wall_seconds"] for r in per_tile_results.values()) / wall_total
        if wall_total > 0 else float("nan")
    )

    return {
        "df_out": df_out,
        "per_tile": summary_per_tile,
        "tile_info": tile_info,
        "n_tiles_xy": n_tiles_xy,
        "n_workers": n_workers,
        "wall_total_seconds": round(wall_total, 2),
        "wall_max_tile_seconds": round(wall_max_tile, 2),
        "speedup_vs_serial_estimate": round(speedup_vs_serial, 2),
    }
