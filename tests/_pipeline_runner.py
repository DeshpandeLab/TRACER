"""Wrap the production segmented + noseg TRACER pipelines into thin
runners suitable for tests.

Both runners return ``(df_final, stage_progression)`` where
``stage_progression`` is a list of dicts ``{stage, n_cells, n_partials,
n_components, n_unassigned_tx}`` recording the pipeline state after
each stage. The final state is the last entry.

Mirrors the configuration used by the segmented_workflow.ipynb and
noseg_workflow.ipynb notebooks: PMI metric, mean-PMI rescue veto,
grid_3d Stage 4, same-bin Stage 2 at G=8, post-S5 G=2.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from tracer.graph import build_grid_graph_xy, build_grid_graph_xyz
from tracer.pruning import prune_transcripts_fast, prune_genes_by_npmi_greedy
from tracer.spatial import (
    annotate_unassigned_components_fast,
    enforce_spatial_coherence_fast,
    pre_stage2_rescue,
    reassign_unassigned_grid_pool,
    demote_small_entities,
    finalize_unassigned,
)
from tracer.pruning import prune_transcripts_nuclear_seed
from tracer.stitching import (
    apply_stitching_to_transcripts_memory_efficient,
    estimate_within_cell_dz_threshold,
)


# Modern config — matches segmented_workflow.ipynb / noseg_workflow.ipynb.
# PMI_THR relaxed to 1e-5 ("essentially zero positive PMI") on the
# strength of the cell-37742 EMT analysis: log(1.5) ≈ 0.405 sat ABOVE
# the in-cell max NPMI for that cell, blowing up its prune. With NaN→0
# fill in nuclear-seed Prune, threshold ≈ 0 admits any non-negative
# evidence to the seed, which gave +29 % ARI(vs Xenium cell_id) on the
# 50×50 µm validation crop (0.442 → 0.573).
PMI_THR = 1e-5
RESCUE_NEG_THR = -0.05
ANNOTATE_NEG_THR = -0.1 * (PMI_THR / 0.05)
# Iterative Rescue caps: 3 passes captures ≥98 % of asymptotic gain at
# any scale (per /tmp/iterative_rescue_*.png diagnostic). Early-stop
# fires when a pass adds zero tx — covers tight crops in 1–2 passes.
RESCUE_MAX_PASSES = 3


def _classify(label: str) -> str:
    s = str(label)
    # All unassigned-class labels (fixed sentinels + stage-rejected
    # diagnostics like "prune_rejected"/"group_rejected"/"demote_rejected")
    # collapse to "unassigned" for stage-snapshot accounting.
    if s in ("DROP", "-1", "nan", "UNASSIGNED") or s.endswith("_rejected"):
        return "unassigned"
    if s.startswith("UNASSIGNED_"):
        return "component"
    if "-" in s:
        return "partial"
    return "cell"


def _state_dict(df: pd.DataFrame, col: str) -> dict[str, int]:
    s = df[col].astype(str)
    types = s.map(_classify)
    n_ent = s.groupby(types).nunique().to_dict()
    n_tx = types.value_counts().to_dict()
    return {
        "n_cells": int(n_ent.get("cell", 0)),
        "n_partials": int(n_ent.get("partial", 0)),
        "n_components": int(n_ent.get("component", 0)),
        "n_unassigned_tx": int(n_tx.get("unassigned", 0)),
    }


def _record_stage(progression: list, stage_name: str, df: pd.DataFrame, col: str):
    progression.append({"stage": stage_name, **_state_dict(df, col)})


def _grid_3d_graph_fn(df_in, *, k=None, dist_threshold=None,
                      coord_cols=("x", "y", "z"),
                      G_z=2.0, z_neighborhood_depth=1):
    return build_grid_graph_xyz(
        df_in, k=k, dist_threshold=dist_threshold, coord_cols=coord_cols,
        G_xy=2.0, G_z=G_z, xy_neighborhood="8",
        z_neighborhood_depth=z_neighborhood_depth,
        exact_distance_filter=False,
    )


def _grid_self_graph_fn(df_in, *, k=None, dist_threshold=None,
                        coord_cols=("x", "y", "z")):
    return build_grid_graph_xy(
        df_in, k=k, dist_threshold=dist_threshold or 1.5, coord_cols=coord_cols,
        G=8.0, neighborhood="self", exact_distance_filter=False,
    )


def run_segmented_pipeline(df: pd.DataFrame,
                           npmi_panel: pd.DataFrame
                           ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run the segmented workflow on ``df`` (must have ``cell_id`` set).

    Returns
    -------
    df_final : DataFrame with ``stitched`` column carrying the final per-tx label.
    stage_progression : list of state dicts, one per stage.
    """
    progression: list[dict[str, Any]] = []
    _record_stage(progression, "input", df.assign(_lbl=df["cell_id"].astype(str)), "_lbl")

    # Auto-derive within-cell |Δz| threshold + recommended G_z. The
    # recommended_G_z is bimodality-aware:
    #   - bimodal data (z-stratified, Cohen's d ≥ 3): G_z = floor(thr),
    #     small enough to leave an empty-bin moat between the within-
    #     cell mode and the cross-stratum mode.
    #   - unimodal data: G_z = ceil(thr), wide enough to admit cell-
    #     spanning merges at depth=1.
    # The same G_z drives both Split's grid (where depth=1 is fixed)
    # and Stitch's grid + Δz guard.
    dz_stats = estimate_within_cell_dz_threshold(df, entity_col="cell_id")
    auto_dz = dz_stats["threshold"]
    auto_Gz = dz_stats.get("recommended_G_z", float("nan"))
    if not np.isfinite(auto_dz):
        auto_dz, auto_n = None, 0
        auto_Gz = 1.0
    if not np.isfinite(auto_Gz):
        auto_Gz = 1.0

    # Stage 1 — Prune (nuclear-seed when overlaps_nucleus is available).
    # Use PMI column when available; nuclear-seed identity prune
    # anchors each cell on its compact nucleus, then admits cytoplasmic
    # tx via mean PMI to the seed. Recursive Phase 1c surfaces
    # secondary modules as partials. Fall back to NPMI/whole-cell prune
    # if the panel lacks a PMI column or the input has no nuclear flag
    # (legacy synthetic panels).
    metric_col = "PMI" if "PMI" in npmi_panel.columns else "NPMI"
    if "overlaps_nucleus" in df.columns:
        df_pruned, aux = prune_transcripts_nuclear_seed(
            df, npmi_panel,
            cell_id_col="cell_id", gene_col="feature_name",
            nuclear_col="overlaps_nucleus",
            threshold=PMI_THR, unassigned_id="-1",
            metric_col=metric_col, nan_fill=0.0,
            min_nuclear_genes=3,
            n_jobs=-1, show_progress=False,
        )
    else:
        df_pruned, aux = prune_transcripts_fast(
            df, npmi_panel,
            cell_id_col="cell_id", gene_col="feature_name",
            threshold=PMI_THR, unassigned_id="-1",
            metric_col=metric_col, nan_fill=0.0,
            n_jobs=-1, show_progress=False,
        )
    _record_stage(progression, "Prune", df_pruned, "tracer_id")

    # Split stage REMOVED. The nuclear-seed prune emits spatially
    # compact entities by construction (anchored on the nucleus), so
    # there are no spatially-disconnected gene clusters for Split to
    # fragment. For legacy whole-cell prune (no overlaps_nucleus), we
    # still run Split as a safety net.
    if "overlaps_nucleus" not in df.columns:
        def _split_graph_fn(df_in, *, k=None, dist_threshold=None,
                            coord_cols=("x", "y", "z")):
            return _grid_3d_graph_fn(df_in, k=k, dist_threshold=dist_threshold,
                                     coord_cols=coord_cols,
                                     G_z=auto_Gz, z_neighborhood_depth=1)
        df_pruned = enforce_spatial_coherence_fast(
            df_stitched=df_pruned, build_graph_fn=_split_graph_fn,
            entity_col="tracer_id", coord_cols=("x", "y", "z"),
            k=5, dist_threshold=5.0,
            out_col="tracer_id", show_progress=False,
        )
        _record_stage(progression, "Split", df_pruned, "tracer_id")

    # Group BEFORE Rescue (variant C / yield-optimal default).
    #
    # Empirically (lung 500 µm ROI, post-DROP-retirement sweep):
    #   A baseline (Prune → Rescue → Group → Stitch → ...): ARI 0.798, unas 6,106
    #   B skip-IR  (Prune → Group → Stitch → ...):           ARI 0.813, unas 6,738
    #   C swap     (Prune → Group → Rescue → Stitch → ...):  ARI 0.793, unas 5,557 ← yield-optimal
    #
    # C wins on yield (~−500 to −1,200 unas vs A/B): Group assembles
    # spatially-coherent unassigned tx into comp pseudo-entities, then
    # Rescue picks up the leftovers (incl. group_rejected DROP-pile tx
    # whose individual gene happens to fit a nearby cell). Two passes at
    # the unassigned pool — one entity-level (Group), one tx-level
    # (Rescue) — recover more than either alone.
    df_grouped = annotate_unassigned_components_fast(
        df_pruned=df_pruned, aux=aux,
        build_graph_fn=_grid_self_graph_fn, prune_fn=prune_genes_by_npmi_greedy,
        coord_cols=("x", "y", "z"),
        k=8, dist_threshold=1.5, min_comp_size=4,
        npmi_threshold=ANNOTATE_NEG_THR,
        entity_col="tracer_id", out_col="tracer_id",
        cell_id_col="cell_id", gene_col="feature_name",
        transcript_id_col="transcript_id", show_progress=False,
    )
    _record_stage(progression, "Group", df_grouped, "tracer_id")

    # Rescue — iterate up to RESCUE_MAX_PASSES with early-stop. Each pass
    # reseeds gene sets from newly-rescued tx, letting tendrils of
    # irregular cells extend by one neighbour per pass. Convergence
    # diagnostic: 3 passes captures ≥98 % of asymptotic gain.
    n_rescued = 0
    for _pass in range(RESCUE_MAX_PASSES):
        df_grouped, n_pass_rescued, _, _ = pre_stage2_rescue(
            df_grouped, aux=aux,
            entity_col="tracer_id", gene_col="feature_name",
            coord_cols=("x", "y", "z"), out_col="tracer_id",
            G=2.0, pos_npmi_threshold=PMI_THR, neg_npmi_threshold=RESCUE_NEG_THR,
            cluster_guard_n=3, veto_mode="mean", mean_threshold=0.0,
            small_entity_guard_n=0,
        )
        n_rescued += n_pass_rescued
        if n_pass_rescued == 0:
            break
    _record_stage(progression, "Rescue", df_grouped, "tracer_id")

    # Stitch — uses the same dz_stats computed before Split.
    df_grouped["post_stage4"] = df_grouped["tracer_id"]
    df_stitched, _ = apply_stitching_to_transcripts_memory_efficient(
        df_final=df_grouped, aux=aux,
        entity_col="post_stage4", gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        mode="count", threshold=PMI_THR, metric="pmi",
        penalize_simplicity=True, deltaC_min=0.0,
        dist_threshold=5.0, out_col="stitched", show_progress=False,
        candidate_source="grid", G=2.0, stitch_neighborhood="8",
        G_z=auto_Gz, z_neighbor_depth=1,
        min_close_edges_dz=auto_dz,
        min_close_edges_n=5 if auto_dz is not None else 0,
    )
    _record_stage(progression, "Stitch", df_stitched, "stitched")

    # Demote
    df_stitched, n_demoted = demote_small_entities(
        df_stitched, entity_col="stitched", out_col="stitched",
        min_size=5, unassigned_label="-1",
    )
    _record_stage(progression, "Demote", df_stitched, "stitched")

    # Final Rescue
    df_stitched, n_reassigned, _ = reassign_unassigned_grid_pool(
        df_stitched, aux=aux,
        entity_col="stitched", gene_col="feature_name",
        coord_cols=("x", "y", "z"), out_col="stitched",
        G=2.0, pos_npmi_threshold=PMI_THR, neg_npmi_threshold=RESCUE_NEG_THR,
        only_partial_component=False,
        veto_mode="mean", mean_threshold=0.0, small_entity_guard_n=0,
    )
    _record_stage(progression, "Final Rescue", df_stitched, "stitched")

    # Finalize: collapse all stage-rejected / sentinel labels in the
    # entity column to a single canonical "DROP". Mid-pipeline labels
    # like "group_rejected", "demote_rejected", "-1", "UNASSIGNED",
    # "nan" all become "DROP" — published output has exactly two label
    # categories: real entity IDs (cell/partial/component) and "DROP".
    # Diagnostic info (which stage rejected each tx) is recoverable
    # via the per-stage progression snapshots and the
    # `unassigned_qc_status` column emitted by Group.
    finalize_unassigned(df_stitched, col="stitched")
    _record_stage(progression, "Finalize", df_stitched, "stitched")

    return df_stitched, progression


def run_noseg_pipeline(df: pd.DataFrame, npmi_panel: pd.DataFrame
                       ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run the no-segmentation workflow on ``df``.

    The input ``cell_id`` column is overwritten to ``"-1"`` everywhere
    (so any prior segmentation is discarded). Stages: Group → Stitch →
    Demote → Final Rescue.

    Accepts xy-grid-only input (Visium HD or any 2D imaging modality):
    when the ``z`` column is absent it is synthesised as ``0`` so the
    3D-aware stages (which expect ``coord_cols=("x", "y", "z")``) run
    unchanged. With z constant, ``z_neighborhood_depth`` is a no-op —
    every transcript lands in the same z-bin and any depth ≥ 0 admits
    every candidate pair.
    """
    df = df.copy()
    if "z" not in df.columns:
        df["z"] = 0.0
    df["cell_id"] = "-1"

    progression: list[dict[str, Any]] = []
    _record_stage(progression, "input (cell_id all -1)",
                  df.assign(_lbl=df["cell_id"].astype(str)), "_lbl")

    # Init aux via Stage 1 prune at -inf threshold (no actual pruning).
    df, aux = prune_transcripts_fast(
        df, npmi_panel,
        cell_id_col="cell_id", gene_col="feature_name",
        threshold=-1e9, unassigned_id="-1",
        n_jobs=-1, show_progress=False,
    )
    df["tracer_id"] = "-1"
    _record_stage(progression, "after init", df, "tracer_id")

    # Group
    df_grouped = annotate_unassigned_components_fast(
        df_pruned=df, aux=aux,
        build_graph_fn=_grid_self_graph_fn, prune_fn=prune_genes_by_npmi_greedy,
        coord_cols=("x", "y", "z"),
        k=8, dist_threshold=1.5, min_comp_size=5,
        npmi_threshold=ANNOTATE_NEG_THR,
        entity_col="tracer_id", out_col="tracer_id",
        cell_id_col="cell_id", gene_col="feature_name",
        transcript_id_col="transcript_id", show_progress=False,
    )
    _record_stage(progression, "Group", df_grouped, "tracer_id")

    # Stitch
    df_grouped["post_stage4"] = df_grouped["tracer_id"]
    df_stitched, _ = apply_stitching_to_transcripts_memory_efficient(
        df_final=df_grouped, aux=aux,
        entity_col="post_stage4", gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        mode="count", threshold=PMI_THR, metric="pmi",
        penalize_simplicity=True, deltaC_min=0.0,
        dist_threshold=5.0, out_col="stitched", show_progress=False,
        candidate_source="grid", G=2.0, stitch_neighborhood="8",
        G_z=1.0, z_neighbor_depth=1,
    )
    _record_stage(progression, "Stitch", df_stitched, "stitched")

    # Demote
    df_stitched, n_demoted = demote_small_entities(
        df_stitched, entity_col="stitched", out_col="stitched",
        min_size=5, unassigned_label="-1",
    )
    _record_stage(progression, "Demote", df_stitched, "stitched")

    # Final Rescue
    df_stitched, n_reassigned, _ = reassign_unassigned_grid_pool(
        df_stitched, aux=aux,
        entity_col="stitched", gene_col="feature_name",
        coord_cols=("x", "y", "z"), out_col="stitched",
        G=2.0, pos_npmi_threshold=PMI_THR, neg_npmi_threshold=RESCUE_NEG_THR,
        only_partial_component=False,
        veto_mode="mean", mean_threshold=0.0, small_entity_guard_n=0,
    )
    _record_stage(progression, "Final Rescue", df_stitched, "stitched")

    # Finalize unassigned-class labels → "DROP" (see segmented runner
    # for full rationale).
    finalize_unassigned(df_stitched, col="stitched")
    _record_stage(progression, "Finalize", df_stitched, "stitched")

    return df_stitched, progression
