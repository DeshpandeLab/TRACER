"""Phase 1/2: Conservative NPMI pruning.

Denoise cell and create partial cell IDs based on NPMI gene coherence (Phase 1),
then further denoise partial cells (Phase 2).
"""

import concurrent.futures  # noqa: F401 — retained for API compatibility

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm.auto import tqdm  # noqa: F401 — used by prune_transcripts_fast

from . import _cy_prune
from ._etype import etype_from_codes
from ._repro import _ensure_reproducibility_seed
from ._utils import prepare_transcript_df


# ---------- Phase 1/2: Conservative NPMI pruning ----------
# Denoise cell and create partial cell IDs based on NPMI gene coherence (Phase 1)
# Then further denoise partial cells (Phase 2)
def build_dense_npmi_matrix(
    npmi_df,
    gene_i_col="gene_i",
    gene_j_col="gene_j",
    npmi_col="NPMI",
):
    """
    Build dense symmetric NPMI matrix.
    Missing pairs remain NaN (conservative).
    """
    npmi_df = npmi_df.copy()
    npmi_df[gene_i_col] = npmi_df[gene_i_col].astype(str).str.strip()
    npmi_df[gene_j_col] = npmi_df[gene_j_col].astype(str).str.strip()
    npmi_df[npmi_col] = pd.to_numeric(npmi_df[npmi_col], errors="coerce")

    genes = pd.Index(
        np.unique(
            np.concatenate(
                [npmi_df[gene_i_col].values, npmi_df[gene_j_col].values]
            )
        )
    ).astype(str)

    gene_to_idx = {g: i for i, g in enumerate(genes)}
    G = len(genes)

    W = np.full((G, G), np.nan, dtype=np.float32)
    np.fill_diagonal(W, np.nan)

    ai = npmi_df[gene_i_col].map(gene_to_idx).to_numpy()
    bi = npmi_df[gene_j_col].map(gene_to_idx).to_numpy()
    vv = npmi_df[npmi_col].to_numpy(dtype=np.float32)

    W[ai, bi] = vv
    W[bi, ai] = vv

    return np.asarray(genes), gene_to_idx, W


def build_sparse_npmi_matrix(result):
    """Adapt a :class:`tracer.metrics.NpmiBootstrapResult` to the
    ``(genes, gene_to_idx, W)`` triple consumed by
    :func:`prune_transcripts_fast`.

    The CSR upper-triangle ``W_sparse`` is returned as-is; downstream
    code (notably :func:`tracer.stitching.coherence`) detects sparse
    inputs and densifies the per-entity submatrix on the fly. Pairs
    encoded as absent in CSR are treated as zero by the coherence
    kernel — by design (see :func:`compute_npmi_bootstrap`).
    """
    genes = np.asarray(result.genes, dtype=str)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    W = result.W_sparse
    if not sp.isspmatrix_csr(W):
        W = W.tocsr()
    return genes, gene_to_idx, W


def _is_bootstrap_result(obj) -> bool:
    """True for a :class:`tracer.metrics.NpmiBootstrapResult` (or any
    duck-typed object carrying ``W_sparse`` + ``genes``)."""
    return hasattr(obj, "W_sparse") and hasattr(obj, "genes")


def _pairs_df_to_bootstrap_result(
    npmi_df,
    *,
    metric_col: str = "PMI",
    gene_i_col: str = "gene_i",
    gene_j_col: str = "gene_j",
):
    """Adapt a pairs DataFrame ``(gene_i, gene_j, <metric>)`` into a
    :class:`tracer.metrics.NpmiBootstrapResult`-shaped object by building a
    sparse upper-triangle CSR directly from the rows — no dense ``(G, G)``
    is ever materialized.

    Lets the prune / reassign pipelines stay on the sparse backend even when
    the caller supplies a pre-computed pairs table instead of a real
    bootstrap result. Identical end-to-end semantics to the bootstrap path,
    because everything downstream consumes only ``(genes, W_sparse)``.

    Pairs are normalized to the upper triangle (``min(i, j), max(i, j)``);
    non-finite values are dropped (those would be NaN in the legacy dense
    build); explicit zeros are KEPT (an observed PMI of 0.0 must remain a
    counted entry — `eliminate_zeros()` is never called here).
    """
    from .metrics import NpmiBootstrapResult

    df = npmi_df[[gene_i_col, gene_j_col, metric_col]].copy()
    df[gene_i_col] = df[gene_i_col].astype(str).str.strip()
    df[gene_j_col] = df[gene_j_col].astype(str).str.strip()
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df[np.isfinite(df[metric_col].to_numpy(dtype=np.float64))]

    genes = pd.Index(
        np.unique(np.concatenate([
            df[gene_i_col].to_numpy(),
            df[gene_j_col].to_numpy(),
        ]))
    ).astype(str).to_numpy()
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    G = int(genes.size)

    i = df[gene_i_col].map(gene_to_idx).to_numpy(dtype=np.int64)
    j = df[gene_j_col].map(gene_to_idx).to_numpy(dtype=np.int64)
    v = df[metric_col].to_numpy(dtype=np.float32)
    # Upper-triangle normalization, drop self-pairs.
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    keep = lo != hi
    W = sp.coo_matrix(
        (v[keep], (lo[keep], hi[keep])), shape=(G, G), dtype=np.float32
    ).tocsr()
    # Multiple rows for the same (i,j) get summed by coo→csr; downstream
    # callers assume unique pairs, but `pd.DataFrame.drop_duplicates` is
    # the caller's responsibility — this matches `build_dense_npmi_matrix`
    # which overwrites instead.
    return NpmiBootstrapResult(W_sparse=W, genes=genes)


def _bootstrap_result_to_pairs_df(result, *, metric_col: str = "PMI"):
    """Inverse of :func:`_pairs_df_to_bootstrap_result`. Used only by the
    nuclear-seed wrapper's no-nucleus fallback to feed the still-dense
    :func:`prune_transcripts_fast` (its kernels are not yet sparse). Will
    go away when the whole-cell path is sparsified."""
    W = result.W_sparse.tocoo()
    genes = np.asarray(result.genes, dtype=str)
    return pd.DataFrame({
        "gene_i": genes[W.row],
        "gene_j": genes[W.col],
        metric_col: W.data.astype(np.float32),
    })


def _symmetric_csr_arrays(W_upper):
    """Symmetrize an upper-triangle CSR PMI panel for the sparse prune /
    reassign kernels and return ``(indptr, indices, data)`` as int32/
    float32 with columns sorted within each row.

    Uses COO-stacking, NOT ``W + W.T``: scipy's sparse add eliminates
    explicit zeros, which would silently drop an observed PMI of exactly
    0.0. ``eliminate_zeros()`` is likewise never called — an observed 0.0
    must remain a stored, counted entry. Structurally-absent pairs stay
    absent so the kernel skips them (the gene-fit / seed-coherence
    convention; the opposite of coherence's absent-as-0).
    """
    Wu = W_upper.tocoo()
    Wsym = sp.csr_matrix(
        (
            np.concatenate([Wu.data, Wu.data]).astype(np.float32),
            (
                np.concatenate([Wu.row, Wu.col]),
                np.concatenate([Wu.col, Wu.row]),
            ),
        ),
        shape=W_upper.shape,
        dtype=np.float32,
    )
    Wsym.sort_indices()
    return (
        Wsym.indptr.astype(np.int32),
        Wsym.indices.astype(np.int32),
        Wsym.data.astype(np.float32),
    )


def prune_genes_by_npmi_greedy(
    gene_ids: np.ndarray,
    W: np.ndarray,
    threshold: float = -0.1,
):
    """
    Iteratively remove gene with the largest number of
    observed NPMI < threshold edges.
    Missing (NaN) pairs are ignored.
    """
    k = gene_ids.size
    if k <= 1:
        return np.ones(k, dtype=bool)

    subW = W[np.ix_(gene_ids, gene_ids)]
    bad = (subW < threshold)
    bad &= np.isfinite(subW)  # only penalize observed pairs
    np.fill_diagonal(bad, False)

    active = np.ones(k, dtype=bool)
    bad_counts = bad.sum(axis=1).astype(int)

    while active.sum() > 1:
        act = np.flatnonzero(active)
        if bad_counts[act].max() == 0:
            break

        rm = act[np.argmax(bad_counts[act])]
        active[rm] = False

        neighbors = np.flatnonzero(active & bad[rm])
        bad_counts[neighbors] -= 1
        bad_counts[rm] = 0

    return active


# NOTE: There was previously an attempt here to rebind
# `prune_genes_by_npmi_greedy` to `_cy_prune.prune_genes_by_npmi_greedy`.
# That attribute does not exist on the compiled extension — _cy_prune
# exposes `prune_cells` (batch over many cells) and `prune_single`, which
# the `_fast` entry points call directly. The rebind was silently failing
# under a try/except, so the pure-Python `prune_genes_by_npmi_greedy`
# above has always been the one running here. Keeping it as the single
# reference implementation is intentional; the hot path in production
# uses `_cy_prune.prune_cells` via `prune_transcripts_fast` etc.

#
def prune_transcripts(
    df,
    npmi_df,
    cell_id_col="cell_id",
    gene_col="feature_name",
    threshold=-0.1,
    unassigned_id="-1",
):
    """
    Two-pass conservative NPMI pruning.
    Partial cell IDs are string-based: cellID-1
    """
    _ensure_reproducibility_seed()
    df = df.copy()
    df["_cell_str"] = df[cell_id_col].astype(str)
    df[gene_col] = df[gene_col].astype(str).str.strip()

    genes, gene_to_idx, W = build_dense_npmi_matrix(npmi_df)
    df["_gene_idx"] = df[gene_col].map(gene_to_idx)

    # ---------- PASS 1 ----------
    df["cell_id_npmi_cons_p1"] = df["_cell_str"]
    df["npmi_cons_p1_status"] = np.where(
        df["_cell_str"] == unassigned_id,
        "unassigned_input",
        "core",
    )

    partial_map = {}

    for cid, sub in df[df["_cell_str"] != unassigned_id].groupby("_cell_str", sort=False):
        g_local = np.sort(sub["_gene_idx"].dropna().astype(int).unique())
        if g_local.size <= 1:
            continue

        keep_mask = prune_genes_by_npmi_greedy(g_local, W, threshold)
        removed = g_local[~keep_mask]
        if removed.size == 0:
            continue

        pid = f"{cid}-1"
        partial_map[cid] = pid
        rem_set = set(removed.tolist())

        mask = (df["_cell_str"] == cid) & (df["_gene_idx"].isin(rem_set))
        df.loc[mask, "cell_id_npmi_cons_p1"] = pid
        df.loc[mask, "npmi_cons_p1_status"] = "partial_p1"

    # ---------- PASS 2 ----------
    df["cell_id_npmi_cons_p2"] = df["cell_id_npmi_cons_p1"]
    df["npmi_cons_p2_status"] = "unchanged"

    for pid in sorted(set(partial_map.values())):
        sub = df[df["cell_id_npmi_cons_p1"] == pid]
        g_local = np.sort(sub["_gene_idx"].dropna().astype(int).unique())
        if g_local.size <= 1:
            df.loc[sub.index, "npmi_cons_p2_status"] = "partial_p2"
            continue

        keep_mask = prune_genes_by_npmi_greedy(g_local, W, threshold)
        removed = g_local[~keep_mask]

        if removed.size == 0:
            df.loc[sub.index, "npmi_cons_p2_status"] = "partial_p2"
            continue

        rem_set = set(removed.tolist())
        mask = (df["cell_id_npmi_cons_p1"] == pid) & (df["_gene_idx"].isin(rem_set))

        df.loc[~mask & (df["cell_id_npmi_cons_p1"] == pid), "npmi_cons_p2_status"] = "partial_p2"
        df.loc[mask, "cell_id_npmi_cons_p2"] = unassigned_id
        df.loc[mask, "npmi_cons_p2_status"] = "unassigned_from_partial"

    df.drop(columns=["_cell_str", "_gene_idx"], inplace=True)

    from .stitching import compute_housekeeping_mask

    aux = {
        "genes": genes,
        "gene_to_idx": gene_to_idx,
        "W": W,
        "partial_map": partial_map,
        "threshold": threshold,
        "housekeeping_mask": compute_housekeeping_mask(
            W,
            pos_thresh=housekeeping_pos_thresh,
            neg_thresh=housekeeping_neg_thresh,
            min_strong_count=housekeeping_min_strong_count,
        ),
    }
    return df, aux


def prune_transcripts_fast(
    df,
    npmi_df,
    *,
    cell_id_col: str = "cell_id",
    out_col: str = "tracer_id",
    gene_col: str = "feature_name",
    threshold: float = -0.1,
    unassigned_id: str = "-1",
    debug_stages: bool = False,
    n_jobs: int = 1,
    show_progress: bool = True,
    in_place: bool = False,
    housekeeping_pos_thresh: float = 0.05,
    housekeeping_neg_thresh: float = -0.05,
    housekeeping_min_strong_count: int = 5,
    metric_col: str = "NPMI",
    nan_fill: float | None = None,
):
    """
    Two-pass NPMI pruning. Writes pass-2 result to `out_col` in place;
    pass 1's intermediate state is internal-only unless `debug_stages` is
    True. Status columns are also gated behind `debug_stages`.

    Parameters
    ----------
    out_col : str
        Canonical output column name (default `"tracer_id"`). Each
        pipeline stage writes its current best assignment here, in place,
        so the column always reflects the latest stage's output.
    debug_stages : bool
        When True, additionally writes legacy snapshot columns for
        inspection: `cell_id_npmi_cons_p1` (post pass 1), and
        `cell_id_npmi_cons_p2` (post pass 2 = mirrors `out_col`), plus
        the status columns `npmi_cons_p1_status` / `npmi_cons_p2_status`.
        Default False keeps the output minimal for production use.
    n_jobs : int
        Accepted for back-compat; no-op (Cython batch).
    in_place : bool
        Skip the defensive `df.copy()` when caller hands ownership of
        the df to this stage.
    """
    _ensure_reproducibility_seed()
    if not in_place:
        df = df.copy()

    prepare_transcript_df(df, gene_col=gene_col)

    # `_cell_str`: a string view of cell_id used for the pid partial label
    # concatenation (`f"{cid}-1"`) below. If cell_id is already
    # string/object, reuse the reference (no copy); otherwise cast once.
    if df[cell_id_col].dtype != "object":
        df["_cell_str"] = df[cell_id_col].astype(str)
    else:
        df["_cell_str"] = df[cell_id_col]

    if _is_bootstrap_result(npmi_df) or sp.issparse(npmi_df):
        # The two-pass whole-cell prune (prune_cells / prune_single) is
        # still dense-only. The sparse CSR backend currently lives on the
        # nuclear-seed path (prune_transcripts_nuclear_seed); sparsifying
        # this path is tracked as a follow-up.
        raise NotImplementedError(
            "prune_transcripts_fast does not yet accept a sparse / "
            "NpmiBootstrapResult panel; use prune_transcripts_nuclear_seed "
            "for whole-transcriptome sparse panels."
        )
    genes, gene_to_idx, W = build_dense_npmi_matrix(npmi_df, npmi_col=metric_col)
    if nan_fill is not None:
        np.nan_to_num(W, copy=False, nan=float(nan_fill))
    df["_gene_idx"] = df[gene_col].map(gene_to_idx)

    # ---------- PASS 1 ----------
    # Initialise the canonical output column with the raw cell_id labels;
    # pass 1 will overwrite some entries with "{cid}-1" partials, and
    # pass 2 may further demote some partials to `unassigned_id` ("-1").
    # Working column = `out_col` itself (no separate intermediate).
    df[out_col] = df["_cell_str"]
    p1_status = None
    if debug_stages:
        # Pre-declare the full category vocabulary so `.loc[…, col] = "partial_p1"`
        # below doesn't trigger a "not in categories" error. Categorical
        # storage drops this column from ~75 MiB to ~1.5 MiB at 1.4M rows.
        _STATUS_P1_CATS = ["unassigned_input", "core", "partial_p1"]
        p1_status = pd.Categorical(
            np.where(df["_cell_str"] == unassigned_id, "unassigned_input", "core"),
            categories=_STATUS_P1_CATS,
        )

    partial_map = {}

    # Prepare per-cell unique gene lists (only cells that are not unassigned)
    grp = df[df["_cell_str"] != unassigned_id].groupby("_cell_str")["_gene_idx"].apply(
        lambda s: np.asarray(np.sort(pd.Index(s.dropna().astype(int)).unique()), dtype=np.int32)
    )

    cell_items = list(grp.items())
    total_cells = len(cell_items)

    # `n_jobs` is accepted for API compatibility but no longer used: the
    # per-cell pruning now runs as one C-level batch inside the compiled
    # Cython kernel (_cy_prune.prune_cells), so there is no Python
    # ThreadPoolExecutor to parallelize.
    _ = n_jobs

    results = []

    # Batch prune all cells through the compiled Cython kernel. The Python
    # fallback was removed — it was 100–1000× slower and silently ran for
    # hours when the .so wasn't built. If _cy_prune.prune_cells raises
    # (e.g. corrupted gene lists), surface it instead of papering over.
    cell_ids = [cid for cid, _ in cell_items]
    g_arrays = [gl if (gl is not None and gl.size > 0) else None for _, gl in cell_items]
    removed_lists = _cy_prune.prune_cells(g_arrays, W, float(threshold))
    for cid, removed in zip(cell_ids, removed_lists):
        if removed:
            results.append((cid, removed))

    if show_progress:
        pbar = tqdm(total=total_cells, desc="prune_pass1")
        pbar.update(total_cells)
        pbar.close()

    # Deterministic application order (stable across thread completion)
    if results:
        results.sort(key=lambda x: str(x[0]))
        for cid, _ in results:
            partial_map[cid] = f"{cid}-1"

    # Apply pass1 removals — write `pid` partial labels into `out_col`
    # for matching (cell, gene) rows. Status (when debug) marked as
    # `partial_p1` for those rows.
    if results:
        rows = []
        for cid, removed in results:
            pid = partial_map[cid]
            for g in removed:
                rows.append((cid, int(g), pid))

        if rows:
            map_df = pd.DataFrame(rows, columns=["_cell_str_map", "_gene_idx_map", "_pid"])

            # prepare indexed view of original df for efficient merging
            df_idx = df.reset_index().rename(columns={"index": "_orig_index"})[["_orig_index", "_cell_str", "_gene_idx"]]
            df_idx["_gene_idx"] = df_idx["_gene_idx"].astype("Int64")
            map_df["_gene_idx_map"] = map_df["_gene_idx_map"].astype("Int64")

            merged = pd.merge(
                df_idx, map_df,
                left_on=["_cell_str", "_gene_idx"],
                right_on=["_cell_str_map", "_gene_idx_map"],
                how="inner",
            )

            if not merged.empty:
                df.loc[merged["_orig_index"], out_col] = merged["_pid"].values
                if debug_stages and p1_status is not None:
                    p1_status[merged["_orig_index"].to_numpy()] = "partial_p1"

    # Snapshot pass-1 state if requested. Has to happen *before* pass 2
    # mutates `out_col`.
    if debug_stages:
        df["cell_id_npmi_cons_p1"] = df[out_col].copy()
        df["npmi_cons_p1_status"] = p1_status

    if show_progress:
        tqdm(desc="apply_pass1", total=1).update(1)

    # ---------- PASS 2 ----------
    # Pass 2 reads the partial labels currently in `out_col`, prunes their
    # gene sets, and demotes failing rows to `unassigned_id` — in place
    # on `out_col`. No separate `cell_id_npmi_cons_p2` column needed
    # outside debug.
    p2_status = None
    if debug_stages:
        _STATUS_P2_CATS = ["unchanged", "partial_p2", "unassigned_from_partial"]
        p2_status = pd.Categorical(
            ["unchanged"] * len(df), categories=_STATUS_P2_CATS,
        )

    pids = sorted(set(partial_map.values()))
    if pids:
        grp_p = df[df[out_col].isin(pids)].groupby(out_col)["_gene_idx"].apply(
            lambda s: np.asarray(np.sort(pd.Index(s.dropna().astype(int)).unique()), dtype=np.int32)
        )

        partial_items = list(grp_p.items())
        total_partials = len(partial_items)

        if show_progress:
            pbar2 = tqdm(total=total_partials, desc="prune_pass2")
        else:
            pbar2 = None

        results2 = []

        pids = [pid for pid, _ in partial_items]
        g_arrays = [gl if (gl is not None and gl.size > 0) else None for _, gl in partial_items]
        removed_lists = _cy_prune.prune_cells(g_arrays, W, float(threshold))
        for pid, removed in zip(pids, removed_lists):
            if removed:
                results2.append((pid, removed))
            if pbar2 is not None:
                pbar2.update(1)
        if pbar2 is not None:
            pbar2.close()

        if results2:
            results2.sort(key=lambda x: str(x[0]))

        rows2 = []
        removed_pids = set()
        for pid, removed in results2:
            removed_pids.add(pid)
            for g in removed:
                rows2.append((pid, int(g)))

        # df index view: keys = (current partial label in out_col, _gene_idx).
        df_idx2 = df.reset_index().rename(columns={"index": "_orig_index"})[["_orig_index", out_col, "_gene_idx"]]
        df_idx2["_gene_idx"] = df_idx2["_gene_idx"].astype("Int64")

        if rows2:
            map2 = pd.DataFrame(rows2, columns=["_pid_map", "_gene_idx_map"]).astype({"_gene_idx_map": "Int64"})
            merged2 = pd.merge(
                df_idx2, map2,
                left_on=[out_col, "_gene_idx"],
                right_on=["_pid_map", "_gene_idx_map"],
                how="inner",
            )

            if not merged2.empty:
                # demote: out_col → unassigned sentinel for these rows
                df.loc[merged2["_orig_index"], out_col] = unassigned_id
                if debug_stages and p2_status is not None:
                    p2_status[merged2["_orig_index"].to_numpy()] = "unassigned_from_partial"

        # Mark "still-partial" rows in pass 2 status (debug only). A row
        # has pass-2 status = "partial_p2" iff its post-pass-1 label was
        # a partial pid AND pass 2 did not demote it. The pre-pass-2
        # label snapshot is `df["cell_id_npmi_cons_p1"]` (set above
        # when debug_stages was True).
        if debug_stages and p2_status is not None and pids:
            pids_set = set(pids)
            pre_p2 = df["cell_id_npmi_cons_p1"].astype(str)
            still_partial = pre_p2.isin(pids_set) & (df[out_col].astype(str) != unassigned_id)
            kept_partial_idx = np.where(still_partial.to_numpy())[0]
            if kept_partial_idx.size:
                p2_status[kept_partial_idx] = "partial_p2"

    if debug_stages:
        # Snapshot pass-2 state under the legacy name. Mirrors `out_col`.
        df["cell_id_npmi_cons_p2"] = df[out_col].copy()
        if p2_status is not None:
            df["npmi_cons_p2_status"] = p2_status

    df.drop(columns=["_cell_str", "_gene_idx"], inplace=True)

    # Populate the _etype categorical column. The legacy whole-cell
    # prune path doesn't produce per-tx kernel codes (the Cython kernel
    # path does), so we classify from the final label string via
    # `infer_etype_from_label`. This is acceptable because this path
    # is taken ONLY when `overlaps_nucleus` is missing — i.e. legacy /
    # synthetic data with integer cell_ids, where label-string parsing
    # is correct. Production Xenium FFPE / IO data takes the
    # `prune_transcripts_nuclear_seed` path which writes `_etype`
    # directly from kernel codes (bug-free regardless of cell_id format).
    from ._etype import infer_etype_from_label
    df["_etype"] = infer_etype_from_label(df[out_col])

    from .stitching import compute_housekeeping_mask

    aux = {
        "genes": genes,
        "gene_to_idx": gene_to_idx,
        "W": W,
        "partial_map": partial_map,
        "threshold": threshold,
        "housekeeping_mask": compute_housekeeping_mask(W),
    }
    return df, aux

def prune_transcripts_nuclear_seed(
    df,
    npmi=None,
    *,
    cell_id_col: str = "cell_id",
    out_col: str = "tracer_id",
    gene_col: str = "feature_name",
    nuclear_col: str = "overlaps_nucleus",
    threshold: float = 1e-5,
    unassigned_id: str = "-1",
    metric_col: str = "PMI",
    nan_fill: float | None = None,  # noqa: ARG001 — accepted for back-compat
    min_nuclear_genes: int = 3,
    show_progress: bool = False,
    n_jobs: int = -1,
    debug_stages: bool = False,
    in_place: bool = False,
    housekeeping_pos_thresh: float = 0.05,
    housekeeping_neg_thresh: float = -0.05,
    housekeeping_min_strong_count: int = 5,
    skip_phase_1c: bool = False,
    seed_coherence_floor: float = -1e30,
    nuclear_only_admit: bool = False,
    tx_weighted: bool = True,
):
    """Nuclear-seed Prune: anchor cell identity on the spatially-compact
    nucleus, then admit cytoplasmic tx whose gene fits the seed by PMI.

    PMI panel input (``npmi``):
      * ``None`` (default) — compute the panel inline via
        :func:`tracer.metrics.compute_npmi_bootstrap` using ``df`` and the
        ``cell_id_col`` / ``gene_col`` / ``nuclear_col`` already passed
        here. Convenient for one-shot calls; for pipelines that prune more
        than once, run the (expensive) bootstrap externally and pass the
        result in to avoid re-computation.
      * :class:`tracer.metrics.NpmiBootstrapResult` — used directly,
        carries the sparse ``W_sparse`` CSR and gene vocabulary. The
        intended fast path.
      * ``pd.DataFrame`` (deprecated) — a pairs table ``(gene_i, gene_j,
        <metric>)`` is converted to a sparse bootstrap-shaped result by
        :func:`_pairs_df_to_bootstrap_result` (no dense ``(G, G)`` is
        materialized). Emits ``DeprecationWarning``.

    Either way, the gene-fit runs against a symmetric sparse CSR via
    :func:`tracer._cy_prune.prune_cells_nuclear_seed_sparse` and
    structurally-absent pairs are SKIPPED (not counted as 0). The legacy
    ``nan_fill`` knob is accepted for back-compat but ignored — the
    sparse backend is the only path now and its convention is skip.

    If ``skip_phase_1c=True``, the recursive sub-seed carve-out (Phase
    1c) is skipped — rest-pile tx (those that fail Phase 1b admission)
    go straight to ``unassigned_id``. Group + Stitch handle them
    downstream. Faster Prune; loses partial-label identity for EMT-
    style sub-modules.

    Per-cell algorithm:
      Phase 1a: Run the greedy bad-edge pruner on the cell's NUCLEAR-tx
        gene set only. The retained set is the cell's "seed" — its
        primary identity, anchored on the spatially compact nucleus.
        If the cell has fewer than ``min_nuclear_genes`` unique nuclear
        genes, fall back to the standard whole-cell prune.

      Phase 1b: For every tx in the cell (nuclear + cytoplasmic), test
        whether its gene fits the seed by mean PMI. Admit to the main
        cell if mean PMI ≥ ``threshold``; otherwise rejected to the
        rest-pile.

      Phase 1c: Run greedy prune recursively on the rest-pile's nuclear
        genes. The retained sub-seed becomes a partial entity; its
        cytoplasmic-gene-fit tx admit to the partial. Tx with neither
        seed nor sub-seed support get demoted to ``unassigned_id``,
        leaving them for downstream Rescue.

    Returns ``(df_out, aux)`` — ``aux`` carries the sparse upper-triangle
    ``W`` (CSR), ``gene_to_idx``, ``partial_map``, ``housekeeping_mask``
    and ``seed_per_cell``.
    """
    import warnings

    _ensure_reproducibility_seed()

    # ----- resolve the PMI panel into a NpmiBootstrapResult -----
    if npmi is None:
        from .metrics import compute_npmi_bootstrap
        npmi = compute_npmi_bootstrap(
            df,
            group_key=cell_id_col,
            feature_col=gene_col,
            nucleus_col=nuclear_col if nuclear_col in df.columns else None,
            metric=metric_col.lower(),
            show_progress=show_progress,
        )
    elif isinstance(npmi, pd.DataFrame):
        warnings.warn(
            "Passing a pairs DataFrame to prune_transcripts_nuclear_seed is "
            "deprecated and will be removed once all callers thread through "
            "a NpmiBootstrapResult; use tracer.metrics.compute_npmi_bootstrap "
            "or pass NpmiBootstrapResult directly. The DataFrame is being "
            "converted to a sparse CSR internally (no dense (G,G) is built).",
            DeprecationWarning,
            stacklevel=2,
        )
        npmi = _pairs_df_to_bootstrap_result(npmi, metric_col=metric_col)
    elif sp.issparse(npmi):
        raise TypeError(
            "Pass a NpmiBootstrapResult (carries gene names) for the sparse "
            "prune, not a bare scipy matrix."
        )
    elif not _is_bootstrap_result(npmi):
        raise TypeError(
            f"prune_transcripts_nuclear_seed: unsupported npmi type "
            f"{type(npmi).__name__}; expected NpmiBootstrapResult, "
            f"pd.DataFrame (deprecated), or None (auto-compute)."
        )

    if nuclear_col not in df.columns:
        # Fallback: no nucleus information → run standard whole-cell prune.
        # The fast path still wants a pairs DataFrame (its kernels are
        # dense-only); convert the bootstrap result back into one rather
        # than silently downgrading to NotImplementedError.
        npmi_df = _bootstrap_result_to_pairs_df(npmi, metric_col=metric_col)
        return prune_transcripts_fast(
            df, npmi_df,
            cell_id_col=cell_id_col, out_col=out_col, gene_col=gene_col,
            threshold=threshold, unassigned_id=unassigned_id,
            metric_col=metric_col,
            n_jobs=n_jobs, show_progress=show_progress,
            in_place=in_place, debug_stages=debug_stages,
            housekeeping_pos_thresh=housekeeping_pos_thresh,
            housekeeping_neg_thresh=housekeeping_neg_thresh,
            housekeeping_min_strong_count=housekeeping_min_strong_count,
        )

    if not in_place:
        df = df.copy()
    prepare_transcript_df(df, gene_col=gene_col)

    df["_cell_str"] = df[cell_id_col].astype(str)
    df["_is_nuc"] = df[nuclear_col].astype(bool)

    # Single sparse PMI backend: structurally-absent pairs are SKIPPED
    # (NOT counted as 0; that was the legacy `nan_fill=0.0` foot-gun);
    # observed PMI of exactly 0.0 is preserved as a stored, counted entry.
    # Never materializes the dense (G, G).
    genes, gene_to_idx, W = build_sparse_npmi_matrix(npmi)
    W_sp_indptr, W_sp_indices, W_sp_data = _symmetric_csr_arrays(W)
    df["_gene_idx"] = df[gene_col].map(gene_to_idx)

    df[out_col] = df["_cell_str"].copy()

    partial_map: dict[str, str] = {}

    # ---- Cython batch: per-cell Phase 1a/1b/1c in a single C-level call ----
    # The Python per-cell loop with DataFrame masking is O(n_cells × n_tx)
    # — for 58k cells × 1.4M tx that's ~80B mask ops. _cy_prune.prune_cells_nuclear_seed
    # processes all cells in one C pass with no per-cell DataFrame work.
    #
    # Inputs: per-cell row-index lists + flat tx arrays. Returned codes
    # per tx: 0 = main, 1 = partial, 2 = unassigned.
    #
    # We need integer row positions (not pandas index labels) so the
    # Cython kernel can index directly into the arrays. df is already
    # `.copy()` so its index is a fresh RangeIndex unless the caller
    # passed an unusual one — we use np.arange(len(df)) explicitly to
    # be safe.
    n_tx = len(df)
    df["_row_pos"] = np.arange(n_tx, dtype=np.int32)

    # Build per-cell row-position lists via groupby (one O(N) pass,
    # replaces the n_cells × n_tx masking loop).
    grouped_positions = df[df["_cell_str"] != unassigned_id].groupby(
        "_cell_str", sort=False
    )["_row_pos"].apply(lambda s: s.to_numpy(dtype=np.int32))
    cell_ids = list(grouped_positions.index)
    cell_tx_idx_lists = [grouped_positions.iloc[i] for i in range(len(cell_ids))]

    # Flat per-tx arrays. -1 marks "no gene-idx" (NaN in the map).
    gene_idx_int = (df["_gene_idx"]
                    .where(df["_gene_idx"].notna(), -1)
                    .astype(np.int32).to_numpy())
    is_nuc_int = df["_is_nuc"].to_numpy().astype(np.uint8)

    codes = _cy_prune.prune_cells_nuclear_seed_sparse(
        cell_tx_idx_lists,
        gene_idx_int,
        is_nuc_int,
        W_sp_indptr,
        W_sp_indices,
        W_sp_data,
        float(threshold),
        int(min_nuclear_genes),
        1 if skip_phase_1c else 0,
        float(seed_coherence_floor),
        1 if nuclear_only_admit else 0,
        1 if tx_weighted else 0,
    )

    # Apply codes to out_col. Default state of out_col is the cell_id
    # string already (set above as df[out_col] = df["_cell_str"].copy()),
    # which is what code 0 (main) wants. So we only need to overwrite
    # codes 1 and 2.
    code_arr = np.asarray(codes)
    cell_str_arr = df["_cell_str"].to_numpy()

    # Populate the _etype categorical column from the Cython codes.
    # This is the canonical source of entity-kind classification for
    # the etype-column refactor (see src/tracer/_etype.py); it sidesteps
    # the legacy label-string parsing rule, which misclassifies Xenium
    # FFPE / IO cell_ids that natively contain dashes (e.g. 'adohnpem-1').
    # Stage-1 emission only — readers still default to label parsing
    # until step 4 of the refactor flips the flag.
    df["_etype"] = etype_from_codes(code_arr)

    # Build partial labels (code == 1): "{cid}-1"
    partial_mask = (code_arr == 1)
    if partial_mask.any():
        partial_labels = np.array([f"{c}-1" for c in cell_str_arr[partial_mask]])
        # Update partial_map for any cell that has at least one partial tx
        for cid_with_partial in np.unique(cell_str_arr[partial_mask]):
            partial_map[str(cid_with_partial)] = f"{cid_with_partial}-1"
        df.loc[partial_mask, out_col] = partial_labels

    # Unassigned (code == 2)
    unas_mask = (code_arr == 2)
    if unas_mask.any():
        df.loc[unas_mask, out_col] = unassigned_id

    # seed_per_cell is a diagnostic; not returned by the batch (would
    # cost extra memory). Set to empty dict — caller's API still works,
    # just lacks the per-cell seed introspection. If a caller needs it,
    # they can recompute by re-running the Python ref impl.
    seed_per_cell: dict[str, list[int]] = {}

    df.drop(columns=["_cell_str", "_is_nuc", "_gene_idx", "_row_pos"], inplace=True)

    from .stitching import compute_housekeeping_mask
    aux = {
        "genes": genes,
        "gene_to_idx": gene_to_idx,
        "W": W,
        "partial_map": partial_map,
        "threshold": threshold,
        "housekeeping_mask": compute_housekeeping_mask(W),
        "seed_per_cell": seed_per_cell,
    }
    return df, aux


def pairwise_npmi_stats(gene_ids, W):
    if gene_ids.size <= 1:
        return dict(
            sum_npmi=np.nan,
            min_npmi=np.nan,
            p25=np.nan,
            p50=np.nan,
            p75=np.nan,
            n_pairs=0,
        )

    subW = W[np.ix_(gene_ids, gene_ids)]
    iu = np.triu_indices(len(gene_ids), k=1)
    vals = subW[iu]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return dict(
            sum_npmi=np.nan,
            min_npmi=np.nan,
            p25=np.nan,
            p50=np.nan,
            p75=np.nan,
            n_pairs=0,
        )

    return dict(
        sum_npmi=float(vals.sum()),
        min_npmi=float(vals.min()),
        p25=float(np.percentile(vals, 25)),
        p50=float(np.percentile(vals, 50)),
        p75=float(np.percentile(vals, 75)),
        n_pairs=int(vals.size),
    )

def diagnostic_npmi_report(df, aux, cell_id):
    W = aux["W"]
    gene_to_idx = aux["gene_to_idx"]
    cid = str(cell_id)
    pid = aux["partial_map"].get(cid)

    rows = []

    def summarize(name, sub):
        genes = np.sort(sub["feature_name"].astype(str).unique())
        gids = np.sort(pd.Index(genes).map(gene_to_idx).dropna().astype(int).unique())
        stats = pairwise_npmi_stats(gids, W)
        return {
            "stage": name,
            "n_transcripts": len(sub),
            "n_unique_genes": len(genes),
            **stats,
        }

    rows.append(summarize("original", df[df["cell_id"] == cell_id]))
    rows.append(summarize("core_pass1", df[(df["cell_id"] == cell_id) & (df["npmi_cons_p1_status"] == "core")]))

    if pid:
        rows.append(summarize("partial_pass1", df[df["cell_id_npmi_cons_p1"] == pid]))
        rows.append(summarize("partial_pass2", df[(df["cell_id_npmi_cons_p2"] == pid) &
                                                  (df["npmi_cons_p2_status"] == "partial_p2")]))
        rows.append(summarize("unassigned_from_partial",
                              df[(df["cell_id_npmi_cons_p1"] == pid) &
                                 (df["npmi_cons_p2_status"] == "unassigned_from_partial")]))

    return pd.DataFrame(rows)
