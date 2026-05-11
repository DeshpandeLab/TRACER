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
    coherence,
    estimate_within_cell_dz_threshold,
)
from tracer.density_cascade import cascade_as_residual_handler


# Modern config — matches segmented_workflow.ipynb / noseg_workflow.ipynb.
# PMI_THR relaxed to 1e-5 ("essentially zero positive PMI") on the
# strength of the cell-37742 EMT analysis: log(1.5) ≈ 0.405 sat ABOVE
# the in-cell max NPMI for that cell, blowing up its prune. With NaN→0
# fill in nuclear-seed Prune, threshold ≈ 0 admits any non-negative
# evidence to the seed, which gave +29 % ARI(vs Xenium cell_id) on the
# 50×50 µm validation crop (0.442 → 0.573).
PMI_THR = 0.05
SEED_COHERENCE_FLOOR = 0.10
TX_WEIGHTED_PRUNE = True   # tx-weighted greedy bad-edge prune (1a/1c)
SPLIT_PHASE1_DZ = 2.0      # µm; if consecutive z-sorted tx gap > this,
                            #     split entity at that point.
SPLIT_PHASE1_MIN_TX = 1     # min tx per sub-group during the split itself
                            #     (1 = keep singletons; QC pass below
                            #     handles the actual demotion).
SPLIT_PHASE1_MIN_ENTITY = 2 # minimum entity size to consider (2 = bare
                            #     minimum to compute a diff).
PHASE1_QC_MIN_TX = 3        # post-split QC: any Phase 1 entity (main or
                            #     partial) with < this tx → unassigned.
                            #     Prevents 1- and 2-tx degenerate seeds
                            #     from anchoring Rescue admissions.


def _qc_demote_small_phase1_entities(df_in: pd.DataFrame, *,
                                       entity_col: str,
                                       min_size: int = PHASE1_QC_MIN_TX,
                                       unassigned_id: str = "-1"
                                       ) -> tuple[pd.DataFrame, dict]:
    """Demote any Phase 1 entity (main `{cell}` or partial) with < min_size tx
    to unassigned. Skips already-unassigned labels (`-1`, `UNASSIGNED`,
    `UNASSIGNED_*`) — these are not seeded entities."""
    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    counts = df_out[entity_col].value_counts()
    bad = []
    for ent, n in counts.items():
        if n >= min_size:
            continue
        if ent == unassigned_id or ent == "UNASSIGNED" or ent.startswith("UNASSIGNED_"):
            continue
        bad.append(ent)
    n_demoted = 0
    if bad:
        mask = df_out[entity_col].isin(bad)
        df_out.loc[mask, entity_col] = unassigned_id
        n_demoted = int(mask.sum())
        # Mirror the demotion in _etype if the column exists.
        if "_etype" in df_out.columns:
            df_out.loc[mask, "_etype"] = "unknown"
    return df_out, {
        "entities_demoted": len(bad),
        "tx_demoted": n_demoted,
    }


def _phase1_rerank_within_parent(df_in: pd.DataFrame, *,
                                   entity_col: str,
                                   nuclear_col: str = "overlaps_nucleus",
                                   margin_tx: int = 1,
                                   ) -> tuple[pd.DataFrame, dict]:
    """Within each parent cell, re-rank depth-1 entities by nuclear-tx
    count and promote the largest to the main `{cell_id}` label.

    Sub-partials follow their depth-1 ancestor's renaming. Naming
    collisions (deposed main vs renumbered sub-partials of new main)
    are resolved by reserving sub-suffix slots for rank-0's sub-partials
    first and bumping deposed depth-1 entities past the reserved range.

    Parent identity is derived from the label regex; no cell_id column
    is read.

    Spec: docs/superpowers/specs/2026-05-11-phase1-rerank-design.md
    """
    import re as _re

    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    labels = df_out[entity_col].to_numpy(dtype=object).copy()
    is_nuclear = df_out[nuclear_col].to_numpy(dtype=bool)

    _re_label = _re.compile(r"^(\d+)(?:-(\d+)(?:-(\d+))?)?$")

    parent_to_depth1_rows: dict[str, dict[str, list[int]]] = {}
    for i, lab in enumerate(labels):
        m = _re_label.match(str(lab))
        if not m:
            continue
        parent = m.group(1)
        d1 = m.group(2)
        depth1 = parent if d1 is None else f"{parent}-{d1}"
        parent_to_depth1_rows.setdefault(parent, {}).setdefault(
            depth1, []
        ).append(i)

    stats = {"n_parents_reranked": 0, "n_tx_relabeled": 0}

    for parent, depth1_map in parent_to_depth1_rows.items():
        if len(depth1_map) < 2:
            continue

        sizes: list[tuple[str, int]] = []
        current_main = parent if parent in depth1_map else None
        for d1, rows in depth1_map.items():
            n_nuc = int(sum(1 for r in rows if is_nuclear[r]))
            sizes.append((d1, n_nuc))

        def _sort_key(d1_size: tuple[str, int], _cm=current_main) -> tuple:
            d1, n = d1_size
            return (-n, -(d1 == _cm), d1)
        sizes.sort(key=_sort_key)

        n_largest = sizes[0][1]
        n_runner_up = sizes[1][1]
        if (n_largest - n_runner_up) < margin_tx:
            continue
        if sizes[0][0] == current_main:
            continue

        rank0_old_d1 = sizes[0][0]
        rank0_subs: set[str] = set()
        for r in depth1_map[rank0_old_d1]:
            m = _re_label.match(str(labels[r]))
            assert m is not None
            if m.group(3) is not None:
                rank0_subs.add(m.group(3))
        n_rank0_subs = len(rank0_subs)

        new_depth1: dict[str, str] = {}
        for k, (d1, _) in enumerate(sizes):
            if k == 0:
                new_depth1[d1] = parent
            else:
                new_depth1[d1] = f"{parent}-{k + n_rank0_subs}"

        # Renumber sub-suffixes for EVERY old depth-1 (not just rank-0) so the
        # output suffix set is always contiguous starting at 1. Spec says
        # "preserved verbatim"; in practice Split-Phase1 emits contiguous
        # suffixes so renumbering is a no-op for non-rank-0 entities. Keeping
        # the uniform rule simplifies the code without observable behavior
        # change on the production data.
        sub_rename: dict[tuple[str, str], str] = {}
        for d1, _ in sizes:
            old_d2js: list[str] = []
            for r in depth1_map[d1]:
                m = _re_label.match(str(labels[r]))
                assert m is not None
                if m.group(3) is not None and m.group(3) not in old_d2js:
                    old_d2js.append(m.group(3))
            old_d2js.sort(key=int)
            for new_idx, old_d2j in enumerate(old_d2js, start=1):
                sub_rename[(d1, old_d2j)] = str(new_idx)

        all_rows = [r for rows in depth1_map.values() for r in rows]
        # Track which rows transition between roles for the _etype update
        # below. Sub-partials (d2j is not None) stay "partial" regardless;
        # only the depth-1 entities flip cell↔partial based on the new
        # rank.
        rows_to_cell: list[int] = []
        rows_to_partial: list[int] = []
        for r in all_rows:
            m = _re_label.match(str(labels[r]))
            assert m is not None
            d1k = m.group(2)
            d2j = m.group(3)
            old_d1 = parent if d1k is None else f"{parent}-{d1k}"
            new_d1 = new_depth1[old_d1]
            if d2j is None:
                labels[r] = new_d1
                if new_d1 == parent:
                    rows_to_cell.append(r)
                else:
                    rows_to_partial.append(r)
            else:
                labels[r] = f"{new_d1}-{sub_rename[(old_d1, d2j)]}"
                # sub-partial keeps its existing "partial" etype

        # Apply _etype updates for this parent if the column exists.
        if "_etype" in df_out.columns:
            if rows_to_cell:
                mask = np.zeros(len(df_out), dtype=bool)
                mask[rows_to_cell] = True
                df_out.loc[mask, "_etype"] = "cell"
            if rows_to_partial:
                mask = np.zeros(len(df_out), dtype=bool)
                mask[rows_to_partial] = True
                df_out.loc[mask, "_etype"] = "partial"

        stats["n_parents_reranked"] += 1
        stats["n_tx_relabeled"] += len(all_rows)

    df_out[entity_col] = labels
    return df_out, stats


def _phase1_rerank_within_parent_etype(df_in: pd.DataFrame, *,
                                         entity_col: str,
                                         cell_id_col: str = "cell_id",
                                         nuclear_col: str = "overlaps_nucleus",
                                         margin_tx: int = 1,
                                         ) -> tuple[pd.DataFrame, dict]:
    """Sibling of `_phase1_rerank_within_parent` that uses the input
    `cell_id_col` for parent identity (works regardless of cell_id
    format — integer or FFPE-style dash-containing) instead of regex-
    parsing the label string.

    Depth is determined by parsing the suffix that follows the cell_id
    prefix. A tx with label ``L`` and cell_id ``C`` is:

      - main (depth 0):       ``L == C``
      - depth-1 partial:      ``L == C + "-{k}"`` with k an integer
      - sub-partial (depth 2):``L == C + "-{k}-{j}"``

    Cell_ids that natively contain dashes (e.g. PDAC's ``adohnpem-1``)
    are handled correctly: the suffix-after-cell_id (``""`` /
    ``"-1"`` / ``"-1-1"``) is parsed, not the cell_id itself.

    Output semantics identical to the legacy version: sub-partials
    follow their depth-1 ancestor's renaming, bump-on-collision for
    deposed mains, strict `>` margin gate.

    Reads from the `_etype` column when present to filter rerank
    candidates (only ``cell`` / ``partial`` types are considered;
    cascade ``component`` entities and ``unknown`` sentinels are
    skipped). On dataframes without `_etype`, falls back to a label-
    structure check that treats any pattern ``{cell_id}(-\\d+){0,2}``
    as a rerank candidate.
    """
    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    labels = df_out[entity_col].to_numpy(dtype=object).copy()
    is_nuclear = df_out[nuclear_col].to_numpy(dtype=bool)
    cell_ids = df_out[cell_id_col].astype(str).to_numpy()
    has_etype = "_etype" in df_out.columns
    etype_arr = (
        df_out["_etype"].astype(str).to_numpy() if has_etype else None
    )

    # Set of labels classified as TRACER-managed entities (cell or
    # partial). On dataframes with `_etype` we read directly; otherwise
    # fall back to checking that the suffix-after-cell_id matches the
    # `{cell_id}(-\\d+){0,2}` shape.
    UNASSIGNED_SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}

    def _suffix_indices(lab: str, cid: str) -> list[int] | None:
        """Return the integer suffix-after-cell_id components, or None
        if the label doesn't follow the expected form for parent ``cid``.

        Examples:
          ('42',         '42')        → []
          ('42-1',       '42')        → [1]
          ('42-1-1',     '42')        → [1, 1]
          ('adohnpem-1', 'adohnpem-1')→ []
          ('adohnpem-1-1','adohnpem-1')→ [1]
          ('cascade_3-1','42')        → None (different parent)
        """
        if lab == cid:
            return []
        if not lab.startswith(cid + "-"):
            return None
        suffix = lab[len(cid) + 1:]
        parts = suffix.split("-")
        try:
            return [int(p) for p in parts]
        except ValueError:
            return None

    # Bucket tx by (parent_cell_id, depth_1_label). Depth-1 label is
    # `cid` for mains, `cid-{k}` for depth-1 partials, and for sub-
    # partials we use their depth-1 ancestor (`cid-{k}`).
    parent_to_depth1_rows: dict[str, dict[str, list[int]]] = {}
    for i, (lab, cid) in enumerate(zip(labels, cell_ids)):
        if cid in UNASSIGNED_SENTINELS:
            continue
        if has_etype and etype_arr[i] not in ("cell", "partial"):
            continue
        suffix_idx = _suffix_indices(str(lab), str(cid))
        if suffix_idx is None or len(suffix_idx) > 2:
            continue  # not a Phase 1-style entity (e.g., cascade)
        depth1 = str(cid) if not suffix_idx else f"{cid}-{suffix_idx[0]}"
        parent_to_depth1_rows.setdefault(str(cid), {}).setdefault(
            depth1, []
        ).append(i)

    stats = {"n_parents_reranked": 0, "n_tx_relabeled": 0}

    for parent, depth1_map in parent_to_depth1_rows.items():
        if len(depth1_map) < 2:
            continue

        sizes: list[tuple[str, int]] = []
        current_main = parent if parent in depth1_map else None
        for d1, rows in depth1_map.items():
            n_nuc = int(sum(1 for r in rows if is_nuclear[r]))
            sizes.append((d1, n_nuc))

        def _sort_key(d1_size: tuple[str, int], _cm=current_main) -> tuple:
            d1, n = d1_size
            return (-n, -(d1 == _cm), d1)
        sizes.sort(key=_sort_key)

        n_largest = sizes[0][1]
        n_runner_up = sizes[1][1]
        if (n_largest - n_runner_up) < margin_tx:
            continue
        if sizes[0][0] == current_main:
            continue

        # Count rank-0's sub-partials to reserve their suffix slots.
        rank0_old_d1 = sizes[0][0]
        rank0_subs: set[int] = set()
        for r in depth1_map[rank0_old_d1]:
            suffix_idx = _suffix_indices(str(labels[r]), parent)
            assert suffix_idx is not None
            if len(suffix_idx) == 2:
                rank0_subs.add(suffix_idx[1])
        n_rank0_subs = len(rank0_subs)

        # Build the depth-1 rename map.
        new_depth1: dict[str, str] = {}
        for k, (d1, _) in enumerate(sizes):
            new_depth1[d1] = parent if k == 0 else f"{parent}-{k + n_rank0_subs}"

        # Sub-suffix renumber: uniform per old depth-1, starting at 1.
        sub_rename: dict[tuple[str, int], int] = {}
        for d1, _ in sizes:
            old_d2js: list[int] = []
            for r in depth1_map[d1]:
                suffix_idx = _suffix_indices(str(labels[r]), parent)
                assert suffix_idx is not None
                if len(suffix_idx) == 2 and suffix_idx[1] not in old_d2js:
                    old_d2js.append(suffix_idx[1])
            old_d2js.sort()
            for new_idx, old_d2j in enumerate(old_d2js, start=1):
                sub_rename[(d1, old_d2j)] = new_idx

        all_rows = [r for rows in depth1_map.values() for r in rows]
        rows_to_cell: list[int] = []
        rows_to_partial: list[int] = []
        for r in all_rows:
            suffix_idx = _suffix_indices(str(labels[r]), parent)
            assert suffix_idx is not None
            old_d1 = parent if not suffix_idx else f"{parent}-{suffix_idx[0]}"
            new_d1 = new_depth1[old_d1]
            if len(suffix_idx) < 2:
                labels[r] = new_d1
                if new_d1 == parent:
                    rows_to_cell.append(r)
                else:
                    rows_to_partial.append(r)
            else:
                new_d2 = sub_rename[(old_d1, suffix_idx[1])]
                labels[r] = f"{new_d1}-{new_d2}"
                # sub-partial keeps its existing "partial" etype

        if has_etype:
            if rows_to_cell:
                mask = np.zeros(len(df_out), dtype=bool)
                mask[rows_to_cell] = True
                df_out.loc[mask, "_etype"] = "cell"
            if rows_to_partial:
                mask = np.zeros(len(df_out), dtype=bool)
                mask[rows_to_partial] = True
                df_out.loc[mask, "_etype"] = "partial"

        stats["n_parents_reranked"] += 1
        stats["n_tx_relabeled"] += len(all_rows)

    df_out[entity_col] = labels
    return df_out, stats


def _spatial_split_phase1_entities(df_in: pd.DataFrame, *,
                                     entity_col: str,
                                     coord_cols=("x", "y", "z"),
                                     dz_threshold: float = SPLIT_PHASE1_DZ,
                                     min_size: int = SPLIT_PHASE1_MIN_TX,
                                     min_entity_size: int = SPLIT_PHASE1_MIN_ENTITY,
                                     unassigned_id: str = "-1") -> tuple[pd.DataFrame, dict]:
    """Sort tx by z within each entity; split where Δz > dz_threshold.

    Applies to all main cells and partials (≥ min_entity_size tx).
    Largest sub-group keeps the original label; smaller groups get a
    fresh appended suffix; groups < min_size demoted to unassigned_id.

    Returns (df_out, stats) where stats reports counts of entities
    examined / split / demoted etc.
    """
    import re as _re_inner
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    import re as _re

    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    coords_arr = df_out[list(coord_cols)].to_numpy(dtype=np.float64)

    # Pre-compute next-suffix-counter per (cell, depth1) namespace so
    # split-emitted labels don't collide with existing partials.
    #   "37962"     in_partials → next_suffix["37962"] = 1 + max(d1)
    #   "37962-1"   exists      → next_subsuffix["37962-1"] = 1 + max(d2)
    next_suffix: dict[str, int] = {}
    next_subsuffix: dict[str, int] = {}
    for lab in df_out[entity_col].unique():
        m = _re.match(r"^(\d+)-(\d+)(?:-(\d+))?$", str(lab))
        if not m:
            continue
        cell, d1 = m.group(1), int(m.group(2))
        d2 = int(m.group(3)) if m.group(3) else 0
        next_suffix[cell] = max(next_suffix.get(cell, 0), d1) + 1 if cell in next_suffix or d1 >= next_suffix.get(cell, 0) else next_suffix.get(cell, 1)
        # simpler: just track max d1, max d2 per (cell, d1)
        next_suffix[cell] = max(next_suffix.get(cell, 0), d1)
        if d2:
            key = f"{cell}-{d1}"
            next_subsuffix[key] = max(next_subsuffix.get(key, 0), d2)

    out_labels = df_out[entity_col].to_numpy().copy()
    z_arr = df_out[coord_cols[2]].to_numpy(dtype=np.float64)

    stats = {
        "entities_examined": 0,
        "entities_split": 0,
        "subgroups_minted": 0,
        "tx_demoted_singletons": 0,
        "tx_total_relabelled": 0,
    }

    for ent, group in df_out.groupby(entity_col, sort=False):
        if ent == unassigned_id or ent == "UNASSIGNED" or ent.startswith("UNASSIGNED_"):
            continue
        if not _re.match(r"^\d+(-\d+){0,2}$", ent):
            continue  # not a known cell/partial label
        rows = group.index.to_numpy()
        if len(rows) < min_entity_size:
            continue
        stats["entities_examined"] += 1

        # Sort tx by z; find gaps > dz_threshold in the sorted sequence.
        z_vals = z_arr[rows]
        sort_order = np.argsort(z_vals, kind="stable")
        rows_sorted = rows[sort_order]
        z_sorted = z_vals[sort_order]
        diffs = np.diff(z_sorted)
        split_positions = np.where(diffs > dz_threshold)[0]
        if len(split_positions) == 0:
            continue

        # Slice rows_sorted into contiguous z-groups at split positions.
        # split_positions[i] is the last index of group i (group boundaries
        # are between i and i+1 in the sorted array).
        groups_rows = []
        prev = 0
        for sp in split_positions:
            groups_rows.append(rows_sorted[prev : sp + 1])
            prev = sp + 1
        groups_rows.append(rows_sorted[prev:])

        # Rank groups by size descending — largest keeps original label.
        groups_rows.sort(key=lambda a: -len(a))
        stats["entities_split"] += 1

        # Pre-parse the parent label for relabel logic
        m_main = _re.match(r"^(\d+)$", ent)
        m_part = _re.match(r"^(\d+)-(\d+)$", ent)
        m_sub = _re.match(r"^(\d+)-(\d+)-(\d+)$", ent)

        # Collect (rows, new_etype) updates per parent for batched _etype write.
        rows_to_unknown_local: list[np.ndarray] = []
        rows_to_partial_local: list[np.ndarray] = []

        for k, gr in enumerate(groups_rows):
            sz = len(gr)
            if sz < min_size:
                out_labels[gr] = unassigned_id
                stats["tx_demoted_singletons"] += sz
                rows_to_unknown_local.append(gr)
                continue
            if k == 0:
                continue  # largest keeps original label (and its existing _etype)

            # Mint fresh label without colliding with existing partials
            if m_main is not None:
                cell = m_main.group(1)
                next_suffix[cell] = next_suffix.get(cell, 0) + 1
                new_label = f"{cell}-{next_suffix[cell]}"
            elif m_part is not None:
                cell, d1 = m_part.group(1), m_part.group(2)
                key = f"{cell}-{d1}"
                next_subsuffix[key] = next_subsuffix.get(key, 0) + 1
                new_label = f"{cell}-{d1}-{next_subsuffix[key]}"
            elif m_sub is not None:
                cell, d1, d2 = m_sub.group(1), m_sub.group(2), m_sub.group(3)
                key = f"{cell}-{d1}-{d2}"
                next_subsuffix[key] = next_subsuffix.get(key, 0) + 1
                new_label = f"{key}-{next_subsuffix[key]}"
            else:
                continue

            out_labels[gr] = new_label
            stats["subgroups_minted"] += 1
            stats["tx_total_relabelled"] += sz
            # New sub-group inherits "partial" etype regardless of whether
            # the parent was a main (depth-1 partial emitted) or a partial
            # (sub-partial emitted) — per the design, sub-partials are
            # "partial" flat.
            rows_to_partial_local.append(gr)

        # Apply _etype updates for this parent
        if "_etype" in df_out.columns:
            if rows_to_unknown_local:
                rows_concat = np.concatenate(rows_to_unknown_local)
                mask = np.zeros(len(df_out), dtype=bool)
                mask[rows_concat] = True
                df_out.loc[mask, "_etype"] = "unknown"
            if rows_to_partial_local:
                rows_concat = np.concatenate(rows_to_partial_local)
                mask = np.zeros(len(df_out), dtype=bool)
                mask[rows_concat] = True
                df_out.loc[mask, "_etype"] = "partial"

    df_out[entity_col] = out_labels
    return df_out, stats


def _vectorized_mean_pmi_excl_self(W: np.ndarray,
                                     query_gene_idx: np.ndarray,
                                     seed_gene_idx: np.ndarray
                                     ) -> np.ndarray:
    """For each gene index in `query_gene_idx`, compute the mean PMI
    against `seed_gene_idx`, excluding any seed entry equal to the
    query gene itself (matches the legacy `_mean_pmi(g, S)` semantics
    where `others = [x for x in S if x != g]`).

    Parameters
    ----------
    W : (G, G) float32 PMI matrix.
    query_gene_idx : (N,) int array, values in [0, G). All entries
        must be ≥ 0.
    seed_gene_idx : (M,) int array, the seed gene-set indices. Order
        is irrelevant; duplicates are allowed but treated as one slot.

    Returns
    -------
    (N,) float64 array. NaN where the seed has no finite-PMI entries
    after self-exclusion.
    """
    if query_gene_idx.size == 0 or seed_gene_idx.size == 0:
        return np.full(query_gene_idx.shape, np.nan, dtype=np.float64)
    # (N, M) PMI slice
    pmi = W[np.ix_(query_gene_idx, seed_gene_idx)].astype(np.float64, copy=False)
    # (N, M) self-exclusion mask: seed_j == query_i
    self_mask = seed_gene_idx[None, :] == query_gene_idx[:, None]
    valid = np.isfinite(pmi) & ~self_mask
    # safe mean: sum / count
    pmi_clean = np.where(valid, pmi, 0.0)
    count = valid.sum(axis=1)
    total = pmi_clean.sum(axis=1)
    out = np.full(query_gene_idx.shape, np.nan, dtype=np.float64)
    nonzero = count > 0
    out[nonzero] = total[nonzero] / count[nonzero]
    return out


def _reassign_nuclear_post_1c(df_in: pd.DataFrame, *,
                                entity_col: str,
                                aux: dict,
                                cell_id_col: str = "cell_id",
                                gene_col: str = "feature_name",
                                nuclear_col: str = "overlaps_nucleus",
                                margin: float = 0.05,
                                threshold: float = 0.05,
                                ) -> tuple[pd.DataFrame, dict]:
    """Re-evaluate nuclear tx admitted to main entities against any
    sibling partials emitted by Phase 1c. If a partial sibling has a
    strictly higher mean-PMI fit than the main entity (by at least
    `margin`), MOVE the tx to that partial.

    Vectorized rewrite of the per-parent inner loop:
      - Build the (N_main_nuc, |S_seed|) PMI slice once per (parent, seed).
      - Apply the sequential margin-walk vectorized across N
        (preserving legacy semantics: `mp > best_mean + margin`).

    Operates only on NUCLEAR tx (consistent with Phase 1's nuclear-only
    admission policy). Cyto tx are left alone for downstream Rescue.

    Closes Gap B: a nuclear tx weakly admitted to the main seed via
    mean-PMI test (its gene NOT a true seed member) might fit a
    1c-emitted partial's sub-seed strictly better. Currently it's
    locked in the main entity; this stage frees it to the partial.

    Returns (df_out, stats).
    """
    import re as _re

    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    label_arr = df_out[entity_col].to_numpy(dtype=object).copy()

    gene_to_idx = aux["gene_to_idx"]
    W = aux["W"]
    if not isinstance(W, np.ndarray):
        W = np.asarray(W.todense() if hasattr(W, "todense") else W,
                       dtype=np.float32)
    if W.dtype != np.float32:
        W = W.astype(np.float32)

    is_nuclear = df_out[nuclear_col].to_numpy(dtype=bool)
    nuc_genes = df_out[gene_col].astype(str).to_numpy()
    cell_id_arr = df_out[cell_id_col].astype(str).to_numpy()
    nuc_idx = np.where(is_nuclear)[0]

    # Build per-entity nuclear-gene set via pandas groupby — much faster
    # than the per-tx Python loop on large frames.
    _UN = {"-1", "DROP", "UNASSIGNED", "nan"}
    label_s = pd.Series(label_arr)
    is_un_label = label_s.isin(_UN) | label_s.str.startswith("UNASSIGNED_", na=False)
    nuc_with_entity_mask = is_nuclear & (~is_un_label.to_numpy())
    if nuc_with_entity_mask.any():
        gi_arr_all = pd.Series(nuc_genes).map(gene_to_idx).fillna(-1).astype(np.int64).to_numpy()
        valid_for_build = nuc_with_entity_mask & (gi_arr_all >= 0)
        if valid_for_build.any():
            build_df = pd.DataFrame({
                "entity": label_arr[valid_for_build].astype(str),
                "gi":     gi_arr_all[valid_for_build],
            }).drop_duplicates()
            entity_to_genes_lists = build_df.groupby("entity", sort=False)["gi"].apply(np.array).to_dict()
            entity_to_genes: dict[str, set[int]] = {
                e: set(int(x) for x in arr) for e, arr in entity_to_genes_lists.items()
            }
        else:
            entity_to_genes = {}
    else:
        entity_to_genes = {}

    # Group by parent cell_id (first numeric token of the label)
    parent_to_entities: dict[str, list[str]] = {}
    for ent in entity_to_genes:
        m = _re.match(r"^(\d+)(?:-\d+){0,2}$", ent)
        if not m:
            continue
        parent_to_entities.setdefault(m.group(1), []).append(ent)

    n_moves = 0
    n_parents_with_partials = 0
    # Accumulate tx indices that get moved from main → partial across all
    # parents. Their _etype flips cell → partial in a single batched
    # write below.
    moved_tx_indices: list[int] = []

    # Precompute candidate tx indices per parent in ONE pass: for each
    # nuclear tx whose entity label is purely numeric (a main label),
    # record (parent, tx_idx). Avoids the per-parent boolean mask scan
    # that was the dominant cost in the legacy implementation.
    cand_by_parent: dict[str, np.ndarray] = {}
    if nuc_idx.size > 0:
        nuc_label_arr = label_arr[nuc_idx]
        nuc_cell_id   = cell_id_arr[nuc_idx]
        # Main labels are pure digits (no '-'); fast string filter
        nuc_label_str = nuc_label_arr.astype(str)
        is_main_label = np.fromiter(
            (s.isdigit() for s in nuc_label_str),
            dtype=bool, count=nuc_label_str.size,
        )
        # Also require label == cell_id (matches legacy's `cell_id_arr == parent` guard)
        is_owned = nuc_label_str == nuc_cell_id
        is_cand = is_main_label & is_owned
        if is_cand.any():
            cand_tx_idx = nuc_idx[is_cand]
            cand_parent = nuc_label_str[is_cand]
            order = np.argsort(cand_parent, kind="stable")
            cand_tx_idx_sorted = cand_tx_idx[order]
            cand_parent_sorted = cand_parent[order]
            # Group boundaries
            change = np.r_[True, cand_parent_sorted[1:] != cand_parent_sorted[:-1]]
            starts = np.where(change)[0]
            ends = np.r_[starts[1:], cand_parent_sorted.size]
            for s, e in zip(starts, ends):
                cand_by_parent[str(cand_parent_sorted[s])] = cand_tx_idx_sorted[s:e]

    for parent, ent_list in parent_to_entities.items():
        if len(ent_list) < 2:
            continue
        main = parent
        if main not in entity_to_genes:
            continue
        partials = [e for e in ent_list if e != main]
        if not partials:
            continue
        n_parents_with_partials += 1

        cand_idx = cand_by_parent.get(parent)
        if cand_idx is None or cand_idx.size == 0:
            continue

        S_main = np.fromiter(entity_to_genes[main], dtype=np.int64)
        S_partials_arrays = [np.fromiter(entity_to_genes[p], dtype=np.int64) for p in partials]

        # Lazy gene-index lookup, only for this parent's candidates.
        cand_gene_idx = np.fromiter(
            (gene_to_idx.get(nuc_genes[i], -1) for i in cand_idx),
            dtype=np.int64, count=cand_idx.size,
        )
        valid_gene = cand_gene_idx >= 0
        if not valid_gene.any():
            continue
        cand_idx_v = cand_idx[valid_gene]
        cand_g_v = cand_gene_idx[valid_gene]

        # mean PMI vs main (vectorized)
        mean_main_v = _vectorized_mean_pmi_excl_self(W, cand_g_v, S_main)
        finite_main_v = np.isfinite(mean_main_v)
        if not finite_main_v.any():
            continue
        # Only consider tx with finite mean_main
        cand_idx_f = cand_idx_v[finite_main_v]
        cand_g_f = cand_g_v[finite_main_v]
        best_mean = mean_main_v[finite_main_v].copy()
        best_p_idx = np.full(best_mean.shape, -1, dtype=np.int32)

        # Sequential margin walk, vectorized across tx for each partial.
        # Iteration order over partials must match the legacy
        # `for p, S_p in S_partials.items()`: dict insertion order ==
        # insertion order of `partials` list.
        for k, S_p in enumerate(S_partials_arrays):
            mp = _vectorized_mean_pmi_excl_self(W, cand_g_f, S_p)
            accept = np.isfinite(mp) & (mp > best_mean + margin)
            if accept.any():
                best_mean[accept] = mp[accept]
                best_p_idx[accept] = k

        moves = best_p_idx >= 0
        if moves.any():
            for n_local in np.where(moves)[0]:
                k = int(best_p_idx[n_local])
                label_arr[cand_idx_f[n_local]] = partials[k]
                moved_tx_indices.append(int(cand_idx_f[n_local]))
            n_moves += int(moves.sum())

    df_out[entity_col] = label_arr
    if "_etype" in df_out.columns and moved_tx_indices:
        mask = np.zeros(len(df_out), dtype=bool)
        mask[moved_tx_indices] = True
        df_out.loc[mask, "_etype"] = "partial"
    return df_out, {
        "n_tx_moved": n_moves,
        "n_parents_with_partials": n_parents_with_partials,
    }


def _reassign_nuclear_post_1c_legacy(df_in: pd.DataFrame, *,
                                entity_col: str,
                                aux: dict,
                                cell_id_col: str = "cell_id",
                                gene_col: str = "feature_name",
                                nuclear_col: str = "overlaps_nucleus",
                                margin: float = 0.05,
                                threshold: float = 0.05,
                                ) -> tuple[pd.DataFrame, dict]:
    """Legacy reference implementation of post-1c reassignment.

    Kept for byte-identicality testing against the vectorized
    `_reassign_nuclear_post_1c`. NOT called by the production pipeline.
    See `_reassign_nuclear_post_1c` for the active implementation and
    full docstring."""
    import re as _re

    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    label_arr = df_out[entity_col].to_numpy(dtype=object).copy()

    gene_to_idx = aux["gene_to_idx"]
    W = aux["W"]
    if hasattr(W, "dtype") and W.dtype != np.float32:
        W = W.astype(np.float32)
    if not isinstance(W, np.ndarray):
        # sparse matrices: densify for the per-tx mean-PMI lookups
        W = np.asarray(W.todense() if hasattr(W, "todense") else W,
                       dtype=np.float32)

    # Build per-entity nuclear-gene set (using nuclear tx only)
    is_nuclear = df_out[nuclear_col].to_numpy(dtype=bool)
    nuc_idx = np.where(is_nuclear)[0]
    nuc_genes = df_out[gene_col].astype(str).to_numpy()
    nuc_labels = df_out[entity_col].to_numpy(dtype=object)

    entity_to_genes: dict[str, set[int]] = {}
    for i in nuc_idx:
        e = nuc_labels[i]
        if e in ("-1", "DROP", "UNASSIGNED", "nan") or str(e).startswith("UNASSIGNED_"):
            continue
        g = nuc_genes[i]
        gi = gene_to_idx.get(g, -1)
        if gi < 0:
            continue
        entity_to_genes.setdefault(str(e), set()).add(int(gi))

    # Group by parent cell_id (first numeric token of the label)
    parent_to_entities: dict[str, list[str]] = {}
    for ent in entity_to_genes:
        m = _re.match(r"^(\d+)(?:-\d+){0,2}$", ent)
        if not m:
            continue
        parent_to_entities.setdefault(m.group(1), []).append(ent)

    def _mean_pmi(g_idx: int, gene_set: set[int]) -> float:
        if not gene_set:
            return float("nan")
        others = [x for x in gene_set if x != g_idx]
        if not others:
            return float("nan")
        vals = W[g_idx, np.asarray(list(others), dtype=np.int64)]
        finite = np.isfinite(vals)
        if not finite.any():
            return float("nan")
        return float(vals[finite].mean())

    n_moves = 0
    moved_tx_indices: list[int] = []
    cell_id_arr = df_out[cell_id_col].astype(str).to_numpy()

    for parent, ent_list in parent_to_entities.items():
        if len(ent_list) < 2:
            continue
        main = parent
        if main not in entity_to_genes:
            continue
        partials = [e for e in ent_list if e != main]
        if not partials:
            continue

        S_main = entity_to_genes[main]
        S_partials = {p: entity_to_genes[p] for p in partials}

        # Examine each NUCLEAR tx currently in the main entity
        main_mask = (label_arr == main) & is_nuclear & (cell_id_arr == parent)
        for tx_idx in np.where(main_mask)[0]:
            g = nuc_genes[tx_idx]
            gi = gene_to_idx.get(g, -1)
            if gi < 0:
                continue
            mean_main = _mean_pmi(gi, S_main)
            if not np.isfinite(mean_main):
                continue
            best_p = None
            best_mean = mean_main
            for p, S_p in S_partials.items():
                mp = _mean_pmi(gi, S_p)
                if np.isfinite(mp) and mp > best_mean + margin:
                    best_mean = mp
                    best_p = p
            if best_p is not None:
                label_arr[tx_idx] = best_p
                moved_tx_indices.append(int(tx_idx))
                n_moves += 1

    df_out[entity_col] = label_arr
    if "_etype" in df_out.columns and moved_tx_indices:
        mask = np.zeros(len(df_out), dtype=bool)
        mask[moved_tx_indices] = True
        df_out.loc[mask, "_etype"] = "partial"
    return df_out, {
        "n_tx_moved": n_moves,
        "n_parents_with_partials": int(sum(
            1 for ents in parent_to_entities.values() if len(ents) >= 2
        )),
    }


def _split_unassigned_components(df_in: pd.DataFrame, *,
                                   entity_col: str,
                                   coord_cols=("x", "y", "z"),
                                   dz_threshold: float,
                                   min_size: int = 1,
                                   min_entity_size: int = 2,
                                   unassigned_id: str = "-1"
                                   ) -> tuple[pd.DataFrame, dict]:
    """Sort tx by z within each UNASSIGNED_* component; split where
    consecutive Δz > dz_threshold.

    Mirrors `_spatial_split_phase1_entities` but operates on
    UNASSIGNED_<base> labels emitted by Group. Group's spatial graph
    is essentially 2D (xy bins + d ≤ 1.5 µm in xy), so its components
    can span large z gaps; this stage catches that.

    Largest sub-cluster keeps the original label; smaller ones get a
    fresh suffix `UNASSIGNED_<base>-<k>` for k = 1, 2, ...
    Sub-clusters smaller than ``min_size`` demote to ``unassigned_id``.
    """
    import re as _re

    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    out_labels = df_out[entity_col].to_numpy().copy()
    z_arr = df_out[coord_cols[2]].to_numpy(dtype=np.float64)

    stats = {
        "components_examined": 0,
        "components_split": 0,
        "subcomps_minted": 0,
        "tx_demoted_singletons": 0,
        "tx_total_relabelled": 0,
    }

    # Track existing UNASSIGNED_<base>-<k> suffixes to avoid collisions
    next_subidx: dict[str, int] = {}
    for lab in df_out[entity_col].unique():
        m = _re.match(r"^(UNASSIGNED_\d+)-(\d+)$", str(lab))
        if m:
            base, k = m.group(1), int(m.group(2))
            next_subidx[base] = max(next_subidx.get(base, 0), k)

    for ent, group in df_out.groupby(entity_col, sort=False):
        if not isinstance(ent, str) or not ent.startswith("UNASSIGNED_"):
            continue
        # Only base labels — already-suffixed labels (e.g. UNASSIGNED_42-1)
        # were emitted by a prior split; skip to avoid recursion.
        if "-" in ent.replace("UNASSIGNED_", "", 1):
            continue
        rows = group.index.to_numpy()
        if len(rows) < min_entity_size:
            continue
        stats["components_examined"] += 1

        z_vals = z_arr[rows]
        sort_order = np.argsort(z_vals, kind="stable")
        rows_sorted = rows[sort_order]
        z_sorted = z_vals[sort_order]
        diffs = np.diff(z_sorted)
        split_positions = np.where(diffs > dz_threshold)[0]
        if split_positions.size == 0:
            continue

        groups_rows = []
        prev = 0
        for sp in split_positions:
            groups_rows.append(rows_sorted[prev: sp + 1])
            prev = sp + 1
        groups_rows.append(rows_sorted[prev:])

        # Sort by size desc — largest keeps the original label
        groups_rows.sort(key=lambda a: -len(a))
        stats["components_split"] += 1

        for k, gr in enumerate(groups_rows):
            sz = len(gr)
            if sz < min_size:
                out_labels[gr] = unassigned_id
                stats["tx_demoted_singletons"] += sz
                continue
            if k == 0:
                continue  # largest keeps original label
            next_subidx[ent] = next_subidx.get(ent, 0) + 1
            new_label = f"{ent}-{next_subidx[ent]}"
            out_labels[gr] = new_label
            stats["subcomps_minted"] += 1
            stats["tx_total_relabelled"] += sz

    df_out[entity_col] = out_labels
    return df_out, stats


def _qc_demote_low_coherence(df_in: pd.DataFrame, *,
                               entity_col: str,
                               aux: dict,
                               min_C: float,
                               min_n_genes: int = 2,
                               threshold: float = 0.05,
                               metric: str = "pmi",
                               unassigned_id: str = "-1",
                               real_signal_threshold: float = 0.0,
                               ) -> tuple[pd.DataFrame, dict]:
    """Demote any entity (cell, partial, or component) whose internal
    coherence is below ``min_C``, OR whose distinct-gene count is
    below ``min_n_genes``. The latter forces single-gene entities
    to fail (coherence is undefined for n_genes < 2).

    Uses the Cython batch coherence kernel
    (`tracer._cy_prune.coherence_count_per_entity_batch`) for the
    per-entity coherence computation. ~50–100x faster than a Python
    groupby + per-entity coherence call on large entity sets.

    Returns (df_out, stats). When ``min_C <= 0`` AND
    ``min_n_genes <= 1``, the function is a no-op.
    """
    if min_C <= 0 and min_n_genes <= 1:
        return df_in.copy(), {
            "entities_examined": 0, "entities_demoted_low_C": 0,
            "entities_demoted_few_genes": 0, "tx_demoted": 0,
        }

    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    gene_to_idx = aux["gene_to_idx"]
    W = aux["W"]

    # Densify W to float32 contiguous if needed (Cython expects this)
    if not isinstance(W, np.ndarray):
        # sparse → dense
        W = np.asarray(W.todense() if hasattr(W, "todense") else W,
                       dtype=np.float32)
    if W.dtype != np.float32:
        W = W.astype(np.float32)
    if not W.flags.c_contiguous:
        W = np.ascontiguousarray(W)

    # Map each tx's gene to its W-index in one vectorized pass.
    gene_arr = df_out["feature_name"].astype(str).to_numpy()
    gene_idx_arr = np.array(
        [gene_to_idx.get(g, -1) for g in gene_arr], dtype=np.int32,
    )
    label_arr = df_out[entity_col].to_numpy()
    label_str = label_arr.astype(str)

    # Filter: drop unassigned-class sentinels and tx with unknown gene.
    # NOTE: UNASSIGNED_<n> components ARE valid entities — they're kept.
    # Only the literal sentinels are excluded.
    drop_set = {str(unassigned_id), "UNASSIGNED", "DROP", "nan"}
    is_unassigned_sentinel = np.isin(label_str, list(drop_set))
    keep = (~is_unassigned_sentinel) & (gene_idx_arr >= 0)

    sub_labels = label_str[keep]
    sub_genes = gene_idx_arr[keep]
    if sub_labels.size == 0:
        return df_out, {
            "entities_examined": 0, "entities_demoted_low_C": 0,
            "entities_demoted_few_genes": 0, "tx_demoted": 0,
        }

    # Build CSR (entity → unique gene indices). Pandas sort + drop_duplicates.
    tmp = pd.DataFrame({"label": sub_labels, "gene": sub_genes})
    tmp = tmp.drop_duplicates(subset=["label", "gene"], keep="first")
    tmp = tmp.sort_values(["label", "gene"], kind="stable")
    counts = tmp.groupby("label", sort=False).size().to_numpy(dtype=np.int32)
    entity_ids = tmp.groupby("label", sort=False).size().index.to_numpy()
    offsets = np.empty(counts.size + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    # `to_numpy(dtype=...)` returns a read-only view in some
    # pandas/numpy combinations (notably pandas + numpy<2). The
    # Cython kernel requests a writable typed memoryview, so we
    # need an explicit writable copy here.
    flat_genes = np.ascontiguousarray(
        tmp["gene"].to_numpy(dtype=np.int32, copy=True)
    )

    # Cython batch call. When real_signal_threshold > 0, the kernel
    # uses the "real players" denominator (only pairs with |W| above
    # the noise floor count) — making C panel-shape-agnostic across
    # dense (legacy) and sparse (bootstrap, Visium HD) W matrices.
    from tracer._cy_prune import coherence_count_per_entity_batch
    C_arr, _P_arr, _N_arr = coherence_count_per_entity_batch(
        offsets, flat_genes, W, float(threshold),
        float(real_signal_threshold),
    )

    # Decide demotion per entity
    n_genes_per_ent = counts
    bad_few = (n_genes_per_ent < min_n_genes)
    # n_finite == 0 returns C=0 from kernel; that's caught by "C ≤ min_C"
    # for any positive min_C.
    bad_low = (~bad_few) & (C_arr <= min_C)

    bad_set = set(entity_ids[bad_few | bad_low].tolist())
    n_demoted = 0
    if bad_set:
        mask = df_out[entity_col].isin(bad_set)
        df_out.loc[mask, entity_col] = unassigned_id
        n_demoted = int(mask.sum())

    return df_out, {
        "entities_examined": int(entity_ids.size),
        "entities_demoted_low_C": int(bad_low.sum()),
        "entities_demoted_few_genes": int(bad_few.sum()),
        "tx_demoted": n_demoted,
    }


NUCLEAR_ONLY_ADMIT = True   # restrict 1b/1c to nuclear tx; cyto via Rescue
RESCUE_NEG_THR = -0.05
ANNOTATE_NEG_THR = -0.1 * (PMI_THR / 0.05)
# Iterative Rescue caps: 3 passes captures ≥98 % of asymptotic gain at
# any scale (per /tmp/iterative_rescue_*.png diagnostic). Early-stop
# fires when a pass adds zero tx — covers tight crops in 1–2 passes.
RESCUE_MAX_PASSES = 3
# Rescue veto: hybrid two-stage admission for tx of gene g into entity E:
#   1. If g ∈ E.genes  → admit (no test; E was deemed compatible w/ g earlier).
#   2. Else if min PMI(g, E.genes) > RESCUE_MIN_ADMIT → admit (unanimous-pos).
#   3. Else if mean PMI(g, E.genes, finite) > RESCUE_MEAN_ADMIT → admit.
#   4. Else → reject. Tx remains "-1".
RESCUE_VETO_MODE = "hybrid"
RESCUE_MIN_ADMIT = 0.0      # any negative pair drops to mean-stage
RESCUE_MEAN_ADMIT = 0.1     # aggregate must be solidly positive

# Stitch spatial-gate tightening — opt-in knobs. Defaults preserve
# current production behavior; raise values to tighten.
#   STITCH_MIN_LOCAL_TX     0 = off (current). >=1 requires that many
#                              UNIQUE tx of EACH candidate entity in
#                              the shared xy 8-Moore + z-window bins.
#                              Catches single-bridging-tx pairs where
#                              two spatially-separated entities are
#                              glued by one diagonal-Moore pair.
#   STITCH_GZ_UM            None = use auto (current). 1.0 → max
#                              Δz reach in candidate enumeration =
#                              2·G_z = 2 µm, matching SPLIT_PHASE1_DZ.
STITCH_MIN_LOCAL_TX: int = 3
STITCH_GZ_UM: float | None = 1.0

# Mid-pipeline QC (after Group, before Stitch). Two opt-in controls;
# both default off so current production behavior is unchanged.
#
#   MID_SPLIT_UNASSIGNED_DZ  None = off (current). Float (e.g. 2.0)
#                            applies sorted-Δz fragmentation to
#                            UNASSIGNED_<base> components from Group,
#                            mirroring Split-Phase1's logic. Group's
#                            spatial graph is essentially 2D, so its
#                            components routinely span > 2 µm in z.
#                            ROI data: 57 % of components have max-gap
#                            > 2 µm without this stage.
#
#   MID_QC_C_FLOOR           0.0 = off (current). Float (e.g. 0.05)
#                            demotes any entity (cell / partial /
#                            component) with coherence ≤ floor OR
#                            n_genes < 2. Catches incoherent Phase-1
#                            entities that survived size QC and any
#                            low-C Group components.
MID_SPLIT_UNASSIGNED_DZ: float | None = 2.0
MID_QC_C_FLOOR: float = 0.05

# "Real players" gate for Mid-QC coherence + Rescue veto.
#   0.0 = legacy n_finite-denominator (every non-NaN pair counts —
#         this conflates explicit zeros, tight_nulls, and dead_zones
#         with informative pairs, biasing C toward 0 on sparse W).
#   >0  = panel-shape-agnostic gate. Pairs with |W[i,j]| ≤ floor are
#         excluded from BOTH numerator and denominator; coherence
#         and rescue-veto reflect the (signed) strength of pairs
#         that actually carry information. Required for sparse
#         bootstrap / Visium HD panels where most off-diagonal cells
#         are implicit zero rather than NaN.
REAL_SIGNAL_THRESHOLD: float = 0.05
# Percentile of real-signal PMIs used in the Rescue mean/hybrid veto.
# 50 = median. <50 = stricter (more pairs must clear mean_threshold).
# >50 = liberal (tolerates a long left tail of weak/negative pairs).
RESCUE_AGGREGATOR_PERCENTILE: float = 50.0

# Post-Group Rescue (between Group and Stitch). Closes the gap where
# Group's UNASSIGNED_* components — freshly created — cannot serve as
# Rescue targets in the main 3-pass Rescue (which runs BEFORE Group).
# Without this stage, tx that "belong" to a Group component sit as
# "-1" through Stitch (which then sees an incomplete component) and
# Demote (which may cull on size before the component is fully grown),
# only getting a chance at Final Rescue.
#
#   0 = off (current production behavior)
#  >0 = number of post-Group rescue passes; admits "-1" tx to BOTH
#        Phase-1 entities AND Group components via the same hybrid
#        veto used elsewhere. Same compute profile as the main Rescue.
RESCUE_POST_GROUP_PASSES: int = 3

# Phase-1 post-1c nuclear reassignment (opt-in). Gap B: a nuclear tx
# weakly admitted to the main seed via mean-PMI test, whose gene later
# turns out to be a STRONG fit to a Phase-1c-emitted partial's
# sub-seed, is currently locked in the main entity. This stage moves
# such tx to the partial sibling that gives strictly higher mean PMI.
#
#   False = off (legacy).
#   True  = enable post-1c reassignment.
# Promoted to default-on 2026-05-11 after numpy vectorization eliminated
# the wall-cost objection (3.1x full-pipeline speedup, byte-identical;
# see benchmarks/reassign_full_tissue_speedup.json). The cells-only
# coherence evidence (mean Δ +0.00316, win/loss 2070/954 on full tissue)
# already supported promotion; only the +250s wall cost was the blocker.
PHASE1_REASSIGN_AFTER_1C: bool = True

PHASE1_RERANK_ENABLED: bool = False    # opt-in: rerank depth-1 entities
                                        # under each parent by nuclear-tx
                                        # count. See
                                        # docs/superpowers/specs/2026-05-11-phase1-rerank-design.md
PHASE1_RERANK_MARGIN_TX: int = 1

# Opt-in: use the etype-aware rerank/classifier helpers that derive
# parent identity from the cell_id column instead of regex-parsing the
# label string. Required for correctness on Xenium FFPE / IO data
# where cell_ids natively contain dashes (e.g. `adohnpem-1`); leaving
# this off preserves byte-identical behavior on integer-cell_id data
# (lung cancer). See docs/superpowers/specs/2026-05-11-etype-column-design.md.
USE_ETYPE_COLUMN: bool = False


# Opt-in: replace Group's `annotate_unassigned_components_fast` (G=8 self,
# spatial-only connected components) with the density-cascade Phase 1 on
# the same post-Rescue residual. Cascade emits `cascade_<n>` synthetic
# anchors, classified as 'cell' by `_classify`. Floor is selected at runtime
# from runtime tx-coverage in the residual pool — no hand-tuned thresholds.
#
# On 500 µm Xenium ROI, head-to-head:
#   default Group (G=8 self):     1,826 components,  ARI vs raw +0.6808
#   cascade-residual ('auto'):     ~602 components,  ARI vs raw +0.6877
#       (3x fewer components, ~99 % of default's assignment, equal-or-better ARI)
#
# See density-cascade-handoff.md for full design and bench results.
#
#   False = off (legacy: spatial-only annotate_unassigned_components_fast).
#   True  = on (default 2026-05-07 onward): cascade as Group replacement.
#
# Default flip rationale (full-tissue Xenium, /tmp/bench_cascade_step6.py):
#   default Group  : V_β=2 +0.9410, ARI +0.6896, 62,980 entities
#   cascade partial: V_β=2 +0.9435, ARI +0.6987, 59,207 entities (closest
#                    to input cell_id's 58,405).
# Cascade emits `cascade_<n>-1` labels; downstream Stitch merges fragments
# of the same biological cell via the existing two-dash partial-merger logic.
PHASE1_SEG_RESIDUAL_CASCADE: bool = True
# Cascade auto-floor target: fraction of residual tx mass to capture in
# the R=1 Moore-dilated anchor mask. 0.65 was reverse-engineered from the
# empirical NOSEG winner (66.5 % cov at floor=4 on full Xenium tissue) and
# also lands on floor=2 on the SEG-residual ROI (which can't reach 65 %
# coverage at any floor, so falls back to hard_min=2).
PHASE1_SEG_RESIDUAL_CASCADE_TARGET_COV: float = 0.65
PHASE1_SEG_RESIDUAL_CASCADE_HARD_MIN: int = 2


# Replace the NOSEG path's Group call (de-facto cell-finder, not
# residual handler) with the density-cascade. NOSEG runs at higher density
# than SEG-residual since the entire pool is the input — auto-floor will
# typically pick floor=4 (~66 % tx coverage) on Xenium full pool.
#
# See density-cascade-handoff.md for ROI ARI improvements (+0.085 over
# baseline NOSEG) and full-tissue homogeneity bench results.
#
#   False = off (legacy: G=8 self spatial-only annotate_unassigned_
#           components_fast; bin restriction acts as a crude
#           anchoring substitute).
#   True  = on (default 2026-05-09 onward): cascade as Phase-1
#           replacement in the NOSEG path. Promotes cascade to default
#           symmetrically with PHASE1_SEG_RESIDUAL_CASCADE (flipped
#           2026-05-07). Test references that pinned legacy NOSEG
#           entity counts will need refresh.
PHASE1_NOSEG_CASCADE: bool = True
PHASE1_NOSEG_CASCADE_TARGET_COV: float = 0.65
PHASE1_NOSEG_CASCADE_HARD_MIN: int = 2


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
            seed_coherence_floor=SEED_COHERENCE_FLOOR,
            nuclear_only_admit=NUCLEAR_ONLY_ADMIT,
            tx_weighted=TX_WEIGHTED_PRUNE,
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

    # Phase-1 post-1c nuclear reassignment (opt-in). Closes Gap B:
    # nuclear tx weakly admitted to the main seed, whose gene fits a
    # 1c partial's sub-seed strictly better, get moved to the partial.
    if PHASE1_REASSIGN_AFTER_1C and "overlaps_nucleus" in df_pruned.columns:
        df_pruned, _reassign_stats = _reassign_nuclear_post_1c(
            df_pruned, entity_col="tracer_id", aux=aux,
            cell_id_col="cell_id", gene_col="feature_name",
            nuclear_col="overlaps_nucleus",
            margin=0.05, threshold=PMI_THR,
        )
        _record_stage(progression, "Phase1-Reassign-1c", df_pruned, "tracer_id")

    # Spatial-split Phase 1 entities. Phase 1c is purely gene-based —
    # if a cell has TWO contamination sources contributing similar
    # gene programs (e.g. lymphoid tx from cells above AND below the
    # target cell in z), Phase 1c emits one merged partial. Split it
    # into spatially-distinct sub-partials so downstream Stitch sees
    # them as separate entities.
    df_pruned, _split_stats = _spatial_split_phase1_entities(
        df_pruned, entity_col="tracer_id",
        coord_cols=("x", "y", "z"),
        dz_threshold=SPLIT_PHASE1_DZ,
        min_size=SPLIT_PHASE1_MIN_TX,
        min_entity_size=SPLIT_PHASE1_MIN_ENTITY,
        unassigned_id="-1",
    )
    _record_stage(progression, "Split-Phase1", df_pruned, "tracer_id")

    # Phase1-Rerank (opt-in): within each parent cell_id, promote the
    # depth-1 entity with the most nuclear tx to the main `{cell_id}`
    # label. Defuses Phase 1's greedy 1a->1b->1c privilege.
    # Only runs when overlaps_nucleus is present (nuclear-seed prune path).
    if PHASE1_RERANK_ENABLED and "overlaps_nucleus" in df_pruned.columns:
        if USE_ETYPE_COLUMN:
            df_pruned, _rerank_stats = _phase1_rerank_within_parent_etype(
                df_pruned, entity_col="tracer_id",
                cell_id_col="cell_id",
                nuclear_col="overlaps_nucleus",
                margin_tx=PHASE1_RERANK_MARGIN_TX,
            )
        else:
            df_pruned, _rerank_stats = _phase1_rerank_within_parent(
                df_pruned, entity_col="tracer_id",
                nuclear_col="overlaps_nucleus",
                margin_tx=PHASE1_RERANK_MARGIN_TX,
            )
        _record_stage(progression, "Phase1-Rerank", df_pruned, "tracer_id")

    # Post-split QC: demote tiny Phase 1 entities (1-2 tx) so they
    # don't act as degenerate routing anchors in Rescue.
    df_pruned, _qc_stats = _qc_demote_small_phase1_entities(
        df_pruned, entity_col="tracer_id",
        min_size=PHASE1_QC_MIN_TX,
        unassigned_id="-1",
    )
    _record_stage(progression, "Phase1-QC", df_pruned, "tracer_id")

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

    # Prune → Rescue → Group → Stitch (re-ordered for nuclear-only
    # Phase 1). Rationale: under nuclear-only admission, ~50% of tx
    # exit Phase 1 unassigned — most are cyto tx of cells that already
    # have a nuclear-anchored seed. Running Rescue first lets those tx
    # attach to nearby seeded entities before Group clusters the
    # residual into orphan UNASSIGNED_* components.
    df_rescued = df_pruned
    n_rescued = 0
    for _pass in range(RESCUE_MAX_PASSES):
        df_rescued, n_pass_rescued, _, _ = pre_stage2_rescue(
            df_rescued, aux=aux,
            entity_col="tracer_id", gene_col="feature_name",
            coord_cols=("x", "y", "z"), out_col="tracer_id",
            G=2.0, pos_npmi_threshold=PMI_THR, neg_npmi_threshold=RESCUE_NEG_THR,
            cluster_guard_n=3,
            veto_mode=RESCUE_VETO_MODE,
            mean_threshold=RESCUE_MEAN_ADMIT,
            min_admit_threshold=RESCUE_MIN_ADMIT,
            small_entity_guard_n=0,
            real_signal_threshold=REAL_SIGNAL_THRESHOLD,
            aggregator_percentile=RESCUE_AGGREGATOR_PERCENTILE,
        )
        n_rescued += n_pass_rescued
        if n_pass_rescued == 0:
            break
    _record_stage(progression, "Rescue", df_rescued, "tracer_id")

    if PHASE1_SEG_RESIDUAL_CASCADE:
        df_grouped = cascade_as_residual_handler(
            df_pruned=df_rescued, aux=aux,
            entity_col="tracer_id",
            G=2.0, thresholds="auto",
            territory_radius_bins=1,
            pmi_threshold=PMI_THR,
            min_anchor_tx=3,
            auto_target_cov=PHASE1_SEG_RESIDUAL_CASCADE_TARGET_COV,
            auto_hard_min=PHASE1_SEG_RESIDUAL_CASCADE_HARD_MIN,
        )
    else:
        df_grouped = annotate_unassigned_components_fast(
            df_pruned=df_rescued, aux=aux,
            build_graph_fn=_grid_self_graph_fn, prune_fn=prune_genes_by_npmi_greedy,
            coord_cols=("x", "y", "z"),
            k=8, dist_threshold=1.5, min_comp_size=4,
            npmi_threshold=ANNOTATE_NEG_THR,
            entity_col="tracer_id", out_col="tracer_id",
            cell_id_col="cell_id", gene_col="feature_name",
            transcript_id_col="transcript_id", show_progress=False,
        )
    _record_stage(progression, "Group", df_grouped, "tracer_id")

    # Mid-pipeline QC (after Group, before Stitch). Both stages are
    # opt-in via constants at the top of this module.
    mid_did_anything = False
    if MID_SPLIT_UNASSIGNED_DZ is not None:
        df_grouped, _ = _split_unassigned_components(
            df_grouped, entity_col="tracer_id", coord_cols=("x", "y", "z"),
            dz_threshold=float(MID_SPLIT_UNASSIGNED_DZ),
            min_size=1, min_entity_size=2, unassigned_id="-1",
        )
        mid_did_anything = True
    if MID_QC_C_FLOOR > 0:
        df_grouped, _ = _qc_demote_low_coherence(
            df_grouped, entity_col="tracer_id", aux=aux,
            min_C=float(MID_QC_C_FLOOR), min_n_genes=2,
            threshold=PMI_THR, metric="pmi", unassigned_id="-1",
            real_signal_threshold=REAL_SIGNAL_THRESHOLD,
        )
        mid_did_anything = True
    if mid_did_anything:
        _record_stage(progression, "Mid-QC", df_grouped, "tracer_id")

    # Post-Group Rescue (opt-in). Admits any remaining "-1" tx to
    # Phase-1 entities AND Group components — closing the gap where
    # Group's UNASSIGNED_* couldn't be Rescue targets in the main pass.
    if RESCUE_POST_GROUP_PASSES > 0:
        for _pass in range(RESCUE_POST_GROUP_PASSES):
            df_grouped, n_pass_rescued, _, _ = pre_stage2_rescue(
                df_grouped, aux=aux,
                entity_col="tracer_id", gene_col="feature_name",
                coord_cols=("x", "y", "z"), out_col="tracer_id",
                G=2.0, pos_npmi_threshold=PMI_THR,
                neg_npmi_threshold=RESCUE_NEG_THR,
                cluster_guard_n=3,
                veto_mode=RESCUE_VETO_MODE,
                mean_threshold=RESCUE_MEAN_ADMIT,
                min_admit_threshold=RESCUE_MIN_ADMIT,
                small_entity_guard_n=0,
                real_signal_threshold=REAL_SIGNAL_THRESHOLD,
                aggregator_percentile=RESCUE_AGGREGATOR_PERCENTILE,
            )
            if n_pass_rescued == 0:
                break
        _record_stage(progression, "Post-Group Rescue", df_grouped, "tracer_id")

    # Stitch — symmetric kwargs with NOSEG path (close-edges guard removed
    # 2026-05-09: empirical bench on 500 µm Xenium ROI showed the guard is
    # dormant under current production constants — flipping it on/off in
    # SEG / NOSEG produced byte-identical Stitch output. Removing for
    # symmetry; auto_dz is still computed but no longer fed in here.
    # entity_col reads `tracer_id` directly (the active partition column
    # after Post-Group Rescue); previous code aliased this as
    # "post_stage4", a stale name from the numbered-stages era.
    df_stitched, _ = apply_stitching_to_transcripts_memory_efficient(
        df_final=df_grouped, aux=aux,
        entity_col="tracer_id", gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        mode="count", threshold=PMI_THR, metric="pmi",
        penalize_simplicity=True, deltaC_min=0.03,
        dist_threshold=5.0, out_col="stitched", show_progress=False,
        candidate_source="grid", G=2.0, stitch_neighborhood="8",
        G_z=(STITCH_GZ_UM if STITCH_GZ_UM is not None else auto_Gz),
        z_neighbor_depth=1,
        min_local_tx_per_entity=STITCH_MIN_LOCAL_TX,
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
        veto_mode=RESCUE_VETO_MODE,
        mean_threshold=RESCUE_MEAN_ADMIT,
        min_admit_threshold=RESCUE_MIN_ADMIT,
        small_entity_guard_n=0,
        real_signal_threshold=REAL_SIGNAL_THRESHOLD,
        aggregator_percentile=RESCUE_AGGREGATOR_PERCENTILE,
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

    # Group (or cascade replacement for cell-finding in NOSEG)
    if PHASE1_NOSEG_CASCADE:
        df_grouped = cascade_as_residual_handler(
            df_pruned=df, aux=aux,
            entity_col="tracer_id",
            G=2.0, thresholds="auto",
            territory_radius_bins=1,
            pmi_threshold=PMI_THR,
            min_anchor_tx=3,
            auto_target_cov=PHASE1_NOSEG_CASCADE_TARGET_COV,
            auto_hard_min=PHASE1_NOSEG_CASCADE_HARD_MIN,
        )
    else:
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

    # Mid-pipeline QC (after Group, before Stitch). Same opt-in
    # knobs as the segmented path.
    mid_did_anything = False
    if MID_SPLIT_UNASSIGNED_DZ is not None:
        df_grouped, _ = _split_unassigned_components(
            df_grouped, entity_col="tracer_id", coord_cols=("x", "y", "z"),
            dz_threshold=float(MID_SPLIT_UNASSIGNED_DZ),
            min_size=1, min_entity_size=2, unassigned_id="-1",
        )
        mid_did_anything = True
    if MID_QC_C_FLOOR > 0:
        df_grouped, _ = _qc_demote_low_coherence(
            df_grouped, entity_col="tracer_id", aux=aux,
            min_C=float(MID_QC_C_FLOOR), min_n_genes=2,
            threshold=PMI_THR, metric="pmi", unassigned_id="-1",
            real_signal_threshold=REAL_SIGNAL_THRESHOLD,
        )
        mid_did_anything = True
    if mid_did_anything:
        _record_stage(progression, "Mid-QC", df_grouped, "tracer_id")

    # Post-Group Rescue (opt-in) — see segmented runner for rationale.
    if RESCUE_POST_GROUP_PASSES > 0:
        for _pass in range(RESCUE_POST_GROUP_PASSES):
            df_grouped, n_pass_rescued, _, _ = pre_stage2_rescue(
                df_grouped, aux=aux,
                entity_col="tracer_id", gene_col="feature_name",
                coord_cols=("x", "y", "z"), out_col="tracer_id",
                G=2.0, pos_npmi_threshold=PMI_THR,
                neg_npmi_threshold=RESCUE_NEG_THR,
                cluster_guard_n=3,
                veto_mode=RESCUE_VETO_MODE,
                mean_threshold=RESCUE_MEAN_ADMIT,
                min_admit_threshold=RESCUE_MIN_ADMIT,
                small_entity_guard_n=0,
                real_signal_threshold=REAL_SIGNAL_THRESHOLD,
                aggregator_percentile=RESCUE_AGGREGATOR_PERCENTILE,
            )
            if n_pass_rescued == 0:
                break
        _record_stage(progression, "Post-Group Rescue", df_grouped, "tracer_id")

    # Stitch — entity_col reads `tracer_id` directly (was aliased as
    # "post_stage4" — a stale name from the numbered-stages era).
    df_stitched, _ = apply_stitching_to_transcripts_memory_efficient(
        df_final=df_grouped, aux=aux,
        entity_col="tracer_id", gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        mode="count", threshold=PMI_THR, metric="pmi",
        penalize_simplicity=True, deltaC_min=0.03,
        dist_threshold=5.0, out_col="stitched", show_progress=False,
        candidate_source="grid", G=2.0, stitch_neighborhood="8",
        G_z=(STITCH_GZ_UM if STITCH_GZ_UM is not None else 1.0),
        z_neighbor_depth=1,
        min_local_tx_per_entity=STITCH_MIN_LOCAL_TX,
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
        veto_mode=RESCUE_VETO_MODE,
        mean_threshold=RESCUE_MEAN_ADMIT,
        min_admit_threshold=RESCUE_MIN_ADMIT,
        small_entity_guard_n=0,
        real_signal_threshold=REAL_SIGNAL_THRESHOLD,
        aggregator_percentile=RESCUE_AGGREGATOR_PERCENTILE,
    )
    _record_stage(progression, "Final Rescue", df_stitched, "stitched")

    # Finalize unassigned-class labels → "DROP" (see segmented runner
    # for full rationale).
    finalize_unassigned(df_stitched, col="stitched")
    _record_stage(progression, "Finalize", df_stitched, "stitched")

    return df_stitched, progression
