# cython: boundscheck=False, wraparound=False, nonecheck=False
# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free


def prune_cells(list g_lists, cnp.ndarray[cnp.float32_t, ndim=2] W, double threshold):
    """
    Bulk prune helper callable from Python.

    Parameters
    ----------
    g_lists : list
        List of 1D integer numpy arrays (gene indices) or None entries.
    W : ndarray[float32, 2D]
        Full NPMI matrix.
    threshold : float
        NPMI threshold.

    Returns
    -------
    list
        List of removed gene index lists (python lists) or []/None.
    """
    cdef Py_ssize_t n
    cdef list out
    cdef Py_ssize_t idx
    cdef object arr

    n = len(g_lists)
    out = [None] * n
    for idx in range(n):
        g = g_lists[idx]
        if g is None:
            out[idx] = None
            continue
        arr = np.asarray(g, dtype=np.int32)
        if arr.size <= 1:
            out[idx] = []
            continue
        out[idx] = prune_single(arr, W, threshold)

    return out


def prune_single(cnp.ndarray[cnp.int32_t, ndim=1] g_local, cnp.ndarray[cnp.float32_t, ndim=2] W, double threshold):
    """Prune a single gene list. Returns removed gene indices as Python list."""
    cdef int k
    cdef int i, j
    cdef int gi, gj
    cdef int active_count
    cdef int maxc, argmax
    cdef float val

    cdef object active
    cdef object bad
    cdef object bad_counts

    cdef cnp.int32_t[:] gids
    cdef cnp.float32_t[:, :] Wv
    cdef cnp.uint8_t[:, :] bad_mv
    cdef cnp.uint8_t[:] active_mv
    cdef cnp.int32_t[:] badc_mv

    k = g_local.shape[0]
    # create local numpy arrays for masks/counts (fast with memoryviews)
    active = np.ones(k, dtype=np.uint8)
    bad = np.zeros((k, k), dtype=np.uint8)
    bad_counts = np.zeros(k, dtype=np.int32)

    gids = g_local
    Wv = W
    bad_mv = bad
    active_mv = active
    badc_mv = bad_counts

    # compute bad matrix and counts
    for i in range(k):
        gi = int(gids[i])
        for j in range(k):
            if i == j:
                continue
            gj = int(gids[j])
            val = Wv[gi, gj]
            # NaN check
            if val != val:
                continue
            if val < threshold:
                bad_mv[i, j] = 1
                badc_mv[i] += 1

    active_count = k
    while active_count > 1:
        # find active index with max bad_counts
        maxc = -1
        argmax = -1
        for i in range(k):
            if active_mv[i]:
                if badc_mv[i] > maxc:
                    maxc = badc_mv[i]
                    argmax = i

        if maxc <= 0:
            break

        # remove argmax
        active_mv[argmax] = 0
        active_count -= 1

        # decrement neighbors' counts
        for j in range(k):
            if active_mv[j] and bad_mv[argmax, j]:
                badc_mv[j] -= 1
        badc_mv[argmax] = 0

    # collect removed genes (those inactive)
    cdef list removed = []
    for i in range(k):
        if not bool(active_mv[i]):
            removed.append(int(gids[i]))

    return removed


cdef inline int _mean_pmi_test(
    int gene_idx,
    cnp.int32_t[:] seed,
    int seed_len,
    cnp.float32_t[:, :] W,
    double threshold,
) nogil:
    """Returns 1 if mean PMI(gene_idx, seed) >= threshold (NaN-skipped),
    else 0. Returns 0 if all NaN or empty seed."""
    cdef int j
    cdef double total = 0.0
    cdef int count = 0
    cdef float v
    if seed_len == 0:
        return 0
    for j in range(seed_len):
        v = W[gene_idx, seed[j]]
        if v == v:  # NaN check (NaN != NaN)
            total += v
            count += 1
    if count == 0:
        return 0
    if (total / count) >= threshold:
        return 1
    return 0


cdef cnp.ndarray _greedy_prune_to_retained(
    cnp.int32_t[:] g_local,
    cnp.float32_t[:, :] W,
    double threshold,
):
    """One pass of greedy bad-edge prune; returns retained gene indices
    (the seed) as int32 ndarray. Mirrors prune_single's algorithm but
    returns retained instead of removed."""
    cdef int k = g_local.shape[0]
    if k <= 1:
        return np.asarray(g_local).copy()
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] active = np.ones(k, dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] bad = np.zeros((k, k), dtype=np.uint8)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] bad_counts = np.zeros(k, dtype=np.int32)
    cdef cnp.uint8_t[:] active_mv = active
    cdef cnp.uint8_t[:, :] bad_mv = bad
    cdef cnp.int32_t[:] badc_mv = bad_counts
    cdef int i, j, gi, gj, active_count, maxc, argmax
    cdef float val
    for i in range(k):
        gi = int(g_local[i])
        for j in range(k):
            if i == j:
                continue
            gj = int(g_local[j])
            val = W[gi, gj]
            if val != val:
                continue
            if val < threshold:
                bad_mv[i, j] = 1
                badc_mv[i] += 1
    active_count = k
    while active_count > 1:
        maxc = -1
        argmax = -1
        for i in range(k):
            if active_mv[i]:
                if badc_mv[i] > maxc:
                    maxc = badc_mv[i]
                    argmax = i
        if maxc <= 0:
            break
        active_mv[argmax] = 0
        active_count -= 1
        for j in range(k):
            if active_mv[j] and bad_mv[argmax, j]:
                badc_mv[j] -= 1
        badc_mv[argmax] = 0
    # collect retained (active) gene indices
    cdef int n_kept = 0
    for i in range(k):
        if active_mv[i]:
            n_kept += 1
    cdef cnp.ndarray[cnp.int32_t, ndim=1] kept = np.empty(n_kept, dtype=np.int32)
    cdef cnp.int32_t[:] kept_mv = kept
    cdef int idx = 0
    for i in range(k):
        if active_mv[i]:
            kept_mv[idx] = g_local[i]
            idx += 1
    return kept


def prune_cells_nuclear_seed(
    list cell_tx_idx_lists,
    cnp.ndarray[cnp.int32_t, ndim=1] tx_gene_idx,
    cnp.ndarray[cnp.uint8_t, ndim=1] tx_is_nuclear,
    cnp.ndarray[cnp.float32_t, ndim=2] W,
    double threshold,
    int min_nuclear_genes,
    int skip_phase_1c,
):
    """Batch nuclear-seed prune over many cells.

    Parameters
    ----------
    cell_tx_idx_lists : list of np.int32 1D arrays
        Per-cell row-index arrays (positions in tx_gene_idx / tx_is_nuclear).
    tx_gene_idx : int32 ndarray
        Per-tx gene-vocabulary index. -1 indicates "no gene index" (skip).
    tx_is_nuclear : uint8 ndarray
        Per-tx nuclear flag (0/1).
    W : float32 2D ndarray
        PMI matrix.
    threshold : float
        Admission threshold for mean PMI to seed.
    min_nuclear_genes : int
        Skip Phase 1a / 1c if a cell has fewer than this many unique nuclear genes.
    skip_phase_1c : int
        If non-zero, skip Phase 1c (rejected tx → unassigned directly).

    Returns
    -------
    np.int8 ndarray
        Per-tx assignment code: 0 = main (parent cell), 1 = partial,
        2 = unassigned, 3 = fallback-needed (cell had insufficient
        nuclear genes; caller should fall back to whole-cell prune).
    """
    cdef int n_tx = tx_gene_idx.shape[0]
    cdef int n_cells = len(cell_tx_idx_lists)
    cdef cnp.ndarray[cnp.int8_t, ndim=1] out = np.full(n_tx, 2, dtype=np.int8)
    cdef cnp.int8_t[:] out_mv = out

    cdef cnp.int32_t[:] tx_gene_mv = tx_gene_idx
    cdef cnp.uint8_t[:] tx_nuc_mv = tx_is_nuclear
    cdef cnp.float32_t[:, :] W_mv = W

    cdef Py_ssize_t cidx
    cdef cnp.ndarray[cnp.int32_t, ndim=1] tx_inds_arr
    cdef cnp.int32_t[:] tx_inds_mv
    cdef int n_cell_tx, ti, tx_row, g, n_unique_nuc, fitted, n_unique_all

    # Reusable scratch buffers
    cdef cnp.ndarray[cnp.int32_t, ndim=1] uniq_nuc
    cdef cnp.ndarray[cnp.int32_t, ndim=1] uniq_all
    cdef cnp.ndarray[cnp.int32_t, ndim=1] seed
    cdef cnp.int32_t[:] seed_mv
    cdef int seed_len
    cdef cnp.ndarray[cnp.int32_t, ndim=1] uniq_rej_nuc
    cdef cnp.ndarray[cnp.int32_t, ndim=1] sub_seed
    cdef cnp.int32_t[:] sub_seed_mv
    cdef int sub_seed_len

    # Seed-membership lookup via a compact "in" check using sorted seed.
    # For typical seed sizes (<50 genes), linear scan is faster than a
    # set; use a small inline helper.

    for cidx in range(n_cells):
        tx_inds_arr = cell_tx_idx_lists[cidx]
        tx_inds_mv = tx_inds_arr
        n_cell_tx = tx_inds_arr.shape[0]
        if n_cell_tx == 0:
            continue

        # Collect unique nuclear gene indices for this cell.
        # (Python-level numpy ops here are fine: per-cell, not per-tx.)
        nuc_genes = []
        for ti in range(n_cell_tx):
            tx_row = tx_inds_mv[ti]
            if tx_nuc_mv[tx_row]:
                g = tx_gene_mv[tx_row]
                if g >= 0:
                    nuc_genes.append(g)
        uniq_nuc = np.unique(np.asarray(nuc_genes, dtype=np.int32))
        n_unique_nuc = uniq_nuc.shape[0]

        if n_unique_nuc < min_nuclear_genes:
            # Fallback: prune on whole-cell gene set (matches the Python
            # reference impl). Phase 1b/1c still run on the resulting
            # seed.
            all_genes = []
            for ti in range(n_cell_tx):
                tx_row = tx_inds_mv[ti]
                g = tx_gene_mv[tx_row]
                if g >= 0:
                    all_genes.append(g)
            uniq_all = np.unique(np.asarray(all_genes, dtype=np.int32))
            n_unique_all = uniq_all.shape[0]
            if n_unique_all == 0:
                continue
            seed = _greedy_prune_to_retained(uniq_all, W_mv, threshold)
        else:
            # ---- Phase 1a: nuclear seed ----
            seed = _greedy_prune_to_retained(uniq_nuc, W_mv, threshold)
        seed_mv = seed
        seed_len = seed.shape[0]
        if seed_len == 0:
            # No seed found; all tx → unassigned (code 2 already default)
            continue

        # ---- Phase 1b: per-tx admit by mean PMI to seed ----
        # rejected accumulator (Python list, per-cell, OK overhead)
        rejected_rows = []
        for ti in range(n_cell_tx):
            tx_row = tx_inds_mv[ti]
            g = tx_gene_mv[tx_row]
            if g < 0:
                # No gene index — leave as unassigned (already default 2)
                rejected_rows.append(tx_row)
                continue
            # Check if g is in seed (linear scan; seed_len typically tiny)
            fitted = 0
            for j in range(seed_len):
                if seed_mv[j] == g:
                    fitted = 1
                    break
            if fitted == 0:
                fitted = _mean_pmi_test(g, seed_mv, seed_len, W_mv, threshold)
            if fitted:
                out_mv[tx_row] = 0  # main
            else:
                rejected_rows.append(tx_row)

        if skip_phase_1c or len(rejected_rows) == 0:
            continue

        # ---- Phase 1c: recursive prune on rejected tx's nuclear genes ----
        rej_nuc = []
        for tx_row in rejected_rows:
            if tx_nuc_mv[tx_row]:
                g = tx_gene_mv[tx_row]
                if g >= 0:
                    rej_nuc.append(g)
        if len(rej_nuc) == 0:
            continue
        uniq_rej_nuc = np.unique(np.asarray(rej_nuc, dtype=np.int32))
        if uniq_rej_nuc.shape[0] < min_nuclear_genes:
            continue

        sub_seed = _greedy_prune_to_retained(uniq_rej_nuc, W_mv, threshold)
        sub_seed_mv = sub_seed
        sub_seed_len = sub_seed.shape[0]
        if sub_seed_len == 0:
            continue

        # Re-test rejected tx against sub-seed
        for tx_row in rejected_rows:
            g = tx_gene_mv[tx_row]
            if g < 0:
                continue
            fitted = 0
            for j in range(sub_seed_len):
                if sub_seed_mv[j] == g:
                    fitted = 1
                    break
            if fitted == 0:
                fitted = _mean_pmi_test(g, sub_seed_mv, sub_seed_len, W_mv, threshold)
            if fitted:
                out_mv[tx_row] = 1  # partial
            # else: stays 2 (unassigned) — already default

    return out

