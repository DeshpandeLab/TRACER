#!/usr/bin/env python3
"""Tile-parallel post-processing experiments on the 2x2 mm PDAC ROI.

Two interventions on top of the existing tile-parallel orchestrator:

  (A) Tile-disambiguation of generic labels.  Cascade and UNASSIGNED_*
      labels are tile-local (each tile starts from index 0). When the
      orchestrator concatenates per-tile outputs, ``cascade_5-1`` from
      tile-0 and ``cascade_5-1`` from tile-2 get merged into one
      apparent entity in the concatenated frame. We prefix these
      generic labels with ``tile<idx>_`` before concat so each entity
      has a globally unique label.

  (B) Post-merge Final Rescue.  After concat (and disambiguation), run
      reassign_unassigned_grid_pool one more time so transcripts that
      were unassigned in their own tile because their best fit was an
      entity in a NEIGHBOURING tile can now be admitted globally.

For each variant we report the ARI / RI / h / c / V / purity battery
against the sequential reference.

Run from the seg-tile-parallel worktree root:

    PYTHONPATH=src:. python benchmarks/bench_pdac_roi_tile_postprocess.py
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
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/"
    "data/outs/transcripts.parquet"
)
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr10.parquet"
)
ROI_CENTER = (7255.0, 3023.7)
ROI_HALF_SIDE = 1000.0
N_TILES_XY = (3, 3)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "benchmarks" / "pdac_roi_tile_postprocess"
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


def _metric_table(seq_labels: pd.Series, t_labels: pd.Series, name: str) -> dict:
    seq_un = _is_un(seq_labels)
    t_un = _is_un(t_labels)
    sing_s = _codes(seq_labels, singletons=True)
    sing_t = _codes(t_labels, singletons=True)
    both = (~seq_un) & (~t_un)
    mega_s = _codes(seq_labels, singletons=False)
    mega_t = _codes(t_labels, singletons=False)
    # purity (seq → tiled)
    sub = pd.DataFrame({"seq": seq_labels[~seq_un].to_numpy(),
                         "til": t_labels[~seq_un].to_numpy()})
    by_seq = sub.groupby("seq")["til"]
    sizes = by_seq.size().to_numpy()
    mode_count = by_seq.apply(lambda x: x.value_counts().iloc[0]).to_numpy()
    purity = float(np.average(mode_count / sizes, weights=sizes))

    metrics = {}
    print(f"\n=== {name} ===", flush=True)
    print(f"  {'set':24s}  {'ARI':>7s}  {'RI':>7s}  "
          f"{'NMI':>7s}  {'h':>7s}  {'c':>7s}  {'V':>7s}")
    for set_label, sl, tl in [
        ("singletons (all tx)", sing_s, sing_t),
        ("assigned-in-both",    mega_s[both], mega_t[both]),
    ]:
        if sl.size < 2:
            continue
        ari = float(adjusted_rand_score(sl, tl))
        ri = float(rand_score(sl, tl))
        nmi = float(normalized_mutual_info_score(sl, tl))
        h, c, v = homogeneity_completeness_v_measure(sl, tl)
        print(f"  {set_label:24s}  {ari:>7.4f}  {ri:>7.4f}  "
              f"{nmi:>7.4f}  {h:>7.4f}  {c:>7.4f}  {v:>7.4f}")
        metrics[set_label] = {
            "ARI": ari, "RI": ri, "NMI": nmi,
            "h": float(h), "c": float(c), "V": float(v),
        }
    print(f"  purity (seq → tiled-mode):  {purity:.4f}")
    metrics["purity"] = purity
    return metrics


def _disambiguate_tile_labels(df_tile_local: pd.DataFrame, tile_idx: int,
                                 label_col: str) -> pd.Series:
    """Prefix tile-local generic labels with ``tile<idx>_`` so cross-tile
    concat does not merge unrelated entities."""
    labels = df_tile_local[label_col].astype(str)
    needs_prefix = (
        labels.str.startswith("cascade_")
        | labels.str.startswith("UNASSIGNED_")
    )
    out = labels.copy()
    out.loc[needs_prefix] = "tile{}_".format(tile_idx) + out.loc[needs_prefix]
    return out


def main() -> int:
    t0 = time.time()
    df = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "overlaps_nucleus",
                 "feature_name", "x_location", "y_location", "z_location"],
    ).rename(columns={"x_location": "x", "y_location": "y", "z_location": "z"})
    xc, yc = ROI_CENTER
    h = ROI_HALF_SIDE
    mask = df["x"].between(xc - h, xc + h) & df["y"].between(yc - h, yc + h)
    df = df.loc[mask].reset_index(drop=True)
    panel = pd.read_parquet(PANEL).rename(columns={"value": "NPMI"})[["gene_i", "gene_j", "NPMI"]]
    print(f"loaded ROI: {len(df):,} tx / {df['cell_id'].nunique():,} cell_ids "
          f"[{time.time()-t0:.1f}s]", flush=True)

    import tests._pipeline_runner as runner
    from tests._pipeline_runner import run_segmented_pipeline
    from tests._pipeline_runner_tiled import run_segmented_pipeline_tiled
    runner.PHASE1_RERANK_ENABLED = True
    runner.PHASE1_REASSIGN_AFTER_1C = True

    # ---------- Sequential reference ----------
    print("\nrunning sequential ...", flush=True)
    t = time.time()
    df_seq, _ = run_segmented_pipeline(df.copy(), panel)
    print(f"  wall: {time.time()-t:.1f}s")
    col_seq = "stitched" if "stitched" in df_seq.columns else "tracer_id"
    seq_part = (
        df_seq.set_index("transcript_id")[col_seq].astype(str)
        .reindex(df["transcript_id"]).rename("label")
    )

    # ---------- Tile-parallel run; capture per-tile dfs ----------
    print(f"\nrunning tile-parallel {N_TILES_XY} ...", flush=True)
    t = time.time()
    result = run_segmented_pipeline_tiled(
        df.copy(), panel,
        n_tiles_xy=N_TILES_XY, n_workers=N_TILES_XY[0] * N_TILES_XY[1],
        rerank=True, reassign=True, show_progress=False,
    )
    wall_tiled = time.time() - t
    print(f"  wall_total: {wall_tiled:.1f}s")

    # Variant 0 — naive concat (what existing bench measured).
    df_naive = result["df_out"]
    col_t = "stitched" if "stitched" in df_naive.columns else "tracer_id"
    df_naive_aligned = (
        df_naive.set_index("transcript_id")[col_t].astype(str)
        .reindex(df["transcript_id"]).rename("label")
    )
    _metric_table(seq_part, df_naive_aligned, "tiled_3x3 (naive concat)")

    # Variant A — re-run tiled with per-tile label disambiguation by
    # re-tiling the SAME tile assignment and disambiguating in-place.
    # We rebuild the concat from per_tile_results (saved in result), but
    # the orchestrator only exposes the merged frame. So we have to
    # re-run the orchestrator with a thin post-process. Easiest path:
    # do the prefix on the naive concat using the tile_info bbox to
    # determine each tx's tile, since cell-centroid-tile assignment is
    # deterministic.
    from tests._pipeline_runner_tiled import _assign_cells_to_tiles
    cell_to_tile, _tile_info = _assign_cells_to_tiles(
        df, n_tiles_xy=N_TILES_XY,
        cell_id_col="cell_id", coord_cols=("x", "y"),
    )
    tx_tile = df.set_index("transcript_id")["cell_id"].map(cell_to_tile)
    tx_tile_aligned = tx_tile.reindex(df["transcript_id"]).to_numpy()
    # Apply prefix only to cascade_* and UNASSIGNED_* labels.
    # NOTE: numpy U-dtype broadcasting silently truncates strings that
    # exceed the inferred max width, which would chop off the "-1"
    # suffix on cascade_N-1 labels. Use pandas object dtype throughout.
    naive_series = df_naive_aligned.astype(str)
    needs_series = (
        naive_series.str.startswith("cascade_")
        | naive_series.str.startswith("UNASSIGNED_")
    )
    tile_prefix = (
        "tile" + pd.Series(tx_tile_aligned, index=naive_series.index).astype(str)
        + "_"
    )
    df_disambig = naive_series.copy()
    df_disambig.loc[needs_series] = (
        tile_prefix.loc[needs_series] + naive_series.loc[needs_series]
    )
    df_disambig.name = "label"
    _metric_table(seq_part, df_disambig, "tiled_3x3 + tile-prefix disambiguation")

    # Variant B — disambiguation + post-merge Final Rescue.
    # Apply the prefixed labels back into a dataframe that we pass to
    # reassign_unassigned_grid_pool. Rebuild aux from the panel.
    from tests._pipeline_runner import PMI_THR  # noqa: F401
    # PMI panel → gene_to_idx + W matrix
    panel_str = panel.copy()
    panel_str["gene_i"] = panel_str["gene_i"].astype(str)
    panel_str["gene_j"] = panel_str["gene_j"].astype(str)
    all_genes = pd.unique(
        pd.concat([panel_str["gene_i"], panel_str["gene_j"]], ignore_index=True)
    )
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    G = len(all_genes)
    W = np.full((G, G), np.nan, dtype=np.float32)
    gi = panel_str["gene_i"].map(gene_to_idx).to_numpy()
    gj = panel_str["gene_j"].map(gene_to_idx).to_numpy()
    val = panel_str["NPMI"].to_numpy(dtype=np.float32)
    W[gi, gj] = val
    W[gj, gi] = val  # symmetric
    aux = {"gene_to_idx": gene_to_idx, "W": W}

    # Make a fresh df with the disambiguated labels in a "stitched" col
    df_for_rescue = df.copy()
    # The label series is in df["transcript_id"] order already.
    df_for_rescue["stitched"] = df_disambig.to_numpy()
    # _etype propagation: assume "cell" for cell_id labels, "partial"
    # for cell_id-N labels, "component" for cascade_/UNASSIGNED_ labels,
    # "unknown" otherwise. Easiest: re-derive via infer_etype_from_label
    # for the rescue call (it uses _etype if present).
    from tracer._etype import infer_etype_from_label
    df_for_rescue["_etype"] = np.asarray(
        infer_etype_from_label(df_for_rescue["stitched"])
    ).astype(str)

    from tracer.spatial import reassign_unassigned_grid_pool
    print("\nrunning post-merge Final Rescue ...", flush=True)
    t = time.time()
    df_resc, n_resc, stats = reassign_unassigned_grid_pool(
        df_for_rescue, aux=aux,
        entity_col="stitched", gene_col="feature_name",
        coord_cols=("x", "y", "z"), out_col="stitched",
        G=2.0, neg_npmi_threshold=-0.05,
        only_partial_component=False,
        veto_mode="hybrid",
        mean_threshold=0.1,
        small_entity_guard_n=0,
        min_admit_threshold=0.0,
        real_signal_threshold=0.0,
        aggregator_percentile=50.0,
    )
    print(f"  wall: {time.time()-t:.1f}s  n_rescued: {n_resc:,}")

    rescue_part = pd.Series(
        df_resc["stitched"].astype(str).to_numpy(),
        index=df_disambig.index, name="label",
    )
    _metric_table(seq_part, rescue_part,
                   "tiled_3x3 + disambig + post-merge Final Rescue")

    # Persist partitions for offline re-analysis.
    pd.DataFrame({"transcript_id": df["transcript_id"], "label": seq_part.to_numpy()}).to_parquet(
        OUT_DIR / "partition_sequential.parquet", index=False,
    )
    pd.DataFrame({"transcript_id": df["transcript_id"], "label": df_naive_aligned.to_numpy()}).to_parquet(
        OUT_DIR / "partition_tiled_naive.parquet", index=False,
    )
    pd.DataFrame({"transcript_id": df["transcript_id"], "label": df_disambig.to_numpy()}).to_parquet(
        OUT_DIR / "partition_tiled_disambig.parquet", index=False,
    )
    pd.DataFrame({"transcript_id": df["transcript_id"], "label": rescue_part.to_numpy()}).to_parquet(
        OUT_DIR / "partition_tiled_postrescue.parquet", index=False,
    )
    print(f"\npartitions saved under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
