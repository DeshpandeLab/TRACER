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


cdef inline double _mean_internal_pmi(
    cnp.int32_t[:] genes,
    int n_genes,
    cnp.float32_t[:, :] W,
) nogil:
    """Mean PMI over off-diagonal pairs within a gene set. Returns
    +inf when the set has <2 genes (caller treats as "trivially
    coherent": no pairs means no internal incoherence). NaN entries
    in W are skipped."""
    cdef int i, j
    cdef double total = 0.0
    cdef int count = 0
    cdef float v
    if n_genes < 2:
        return 1e30
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            v = W[genes[i], genes[j]]
            if v == v:  # NaN check
                total += v
                count += 1
    if count == 0:
        return 0.0
    return total / count


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
    cnp.int32_t[:] tx_counts,
    cnp.float32_t[:, :] W,
    double threshold,
):
    """Greedy bad-edge prune with TX-WEIGHTED conflict scoring.

    For each gene i, compute score[i] = sum_j (bad[i,j] * tx_counts[j])
    — i.e., total tx of conflicting other-genes. Strip the gene with
    max score. Safeguard: a gene with own tx count > score is
    PROTECTED from being stripped (its own evidence majority-outweighs
    its conflicts). Returns retained gene indices as int32 ndarray.
    """
    cdef int k = g_local.shape[0]
    if k <= 1:
        return np.asarray(g_local).copy()
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] active = np.ones(k, dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] bad = np.zeros((k, k), dtype=np.uint8)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] bad_score = np.zeros(k, dtype=np.int32)
    cdef cnp.uint8_t[:] active_mv = active
    cdef cnp.uint8_t[:, :] bad_mv = bad
    cdef cnp.int32_t[:] badscore_mv = bad_score
    cdef int i, j, gi, gj, active_count, maxc, argmax
    cdef float val
    # Build bad-edge matrix and tx-weighted score per gene.
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
                badscore_mv[i] += tx_counts[j]
    active_count = k
    while active_count > 1:
        # Find gene with max tx-weighted bad-score among ACTIVE genes
        # NOT protected by the own-tx safeguard.
        maxc = -1
        argmax = -1
        for i in range(k):
            if not active_mv[i]:
                continue
            # Safeguard: own evidence > conflict-tx → protected, skip.
            if tx_counts[i] > badscore_mv[i]:
                continue
            if badscore_mv[i] > maxc:
                maxc = badscore_mv[i]
                argmax = i
        if maxc <= 0 or argmax < 0:
            break
        active_mv[argmax] = 0
        active_count -= 1
        # On strip: every neighbour j with bad[argmax,j]=1 loses
        # tx_counts[argmax] worth of conflict from their score.
        for j in range(k):
            if active_mv[j] and bad_mv[argmax, j]:
                badscore_mv[j] -= tx_counts[argmax]
        badscore_mv[argmax] = 0
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
    double seed_coherence_floor=-1e30,
    int nuclear_only_admit=0,
    int tx_weighted=1,
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
        Admission threshold for mean PMI to seed (also bad-edge threshold
        for the greedy prune).
    min_nuclear_genes : int
        Skip Phase 1a / 1c if a cell has fewer than this many unique nuclear genes.
    skip_phase_1c : int
        If non-zero, skip Phase 1c (rejected tx → unassigned directly).
    seed_coherence_floor : float
        Minimum mean internal PMI required for a seed (1a) or sub-seed
        (1c) to be accepted. Seeds below this floor are rejected:
        for the primary seed, all that cell's tx → unassigned; for the
        sub-seed, all rest-pile tx → unassigned (no partial formed).
        Default -1e30 disables the check (back-compat).
    nuclear_only_admit : int
        If non-zero, restrict 1b admission and 1c re-test to NUCLEAR
        tx only — cytoplasmic tx leave Phase 1 as unassigned (code 2)
        regardless of gene-fit. Group/Rescue/Stitch then route them
        downstream by spatial+gene proximity. Establishes per-cell
        identity exclusively from spatially-trusted nuclear tx.
        Default 0 disables (back-compat: cyto admitted by gene-fit).

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
            uniq_all_arr = np.asarray(all_genes, dtype=np.int32)
            uniq_all = np.unique(uniq_all_arr)
            n_unique_all = uniq_all.shape[0]
            if n_unique_all == 0:
                continue
            # Whole-cell tx counts: all tx contribute (nuc + cyto).
            tx_counts_all = np.zeros(n_unique_all, dtype=np.int32)
            for ti in range(n_unique_all):
                tx_counts_all[ti] = int(np.sum(uniq_all_arr == uniq_all[ti]))
            if not tx_weighted:
                tx_counts_all = np.ones(n_unique_all, dtype=np.int32)
            seed = _greedy_prune_to_retained(uniq_all, tx_counts_all, W_mv, threshold)
        else:
            # ---- Phase 1a: nuclear seed (tx-weighted) ----
            # Per-gene nuclear tx counts for the tx-weighted greedy.
            nuc_arr = np.asarray(nuc_genes, dtype=np.int32)
            tx_counts_nuc = np.zeros(n_unique_nuc, dtype=np.int32)
            for ti in range(n_unique_nuc):
                tx_counts_nuc[ti] = int(np.sum(nuc_arr == uniq_nuc[ti]))
            if not tx_weighted:
                tx_counts_nuc = np.ones(n_unique_nuc, dtype=np.int32)
            seed = _greedy_prune_to_retained(uniq_nuc, tx_counts_nuc, W_mv, threshold)
        seed_mv = seed
        seed_len = seed.shape[0]
        if seed_len == 0:
            # No seed found; all tx → unassigned (code 2 already default)
            continue

        # Coherence floor on primary seed: reject seeds whose internal
        # mean PMI is below the floor (avoids accepting a clique that
        # is merely "non-conflicting" but biologically degenerate).
        if _mean_internal_pmi(seed_mv, seed_len, W_mv) < seed_coherence_floor:
            continue

        # ---- Phase 1b: per-tx admit by mean PMI to seed ----
        # rejected accumulator (Python list, per-cell, OK overhead).
        # When nuclear_only_admit is set, cytoplasmic tx are not eligible
        # for admission to main here — they stay unassigned (code 2)
        # for downstream Group/Rescue/Stitch to route by spatial+gene
        # proximity. Identity is established exclusively from nuclear tx.
        rejected_rows = []
        for ti in range(n_cell_tx):
            tx_row = tx_inds_mv[ti]
            g = tx_gene_mv[tx_row]
            if g < 0:
                # No gene index — leave as unassigned (already default 2)
                rejected_rows.append(tx_row)
                continue
            if nuclear_only_admit and not tx_nuc_mv[tx_row]:
                # Cytoplasmic tx skipped — stays unassigned for Rescue.
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
        rej_nuc_arr = np.asarray(rej_nuc, dtype=np.int32)
        uniq_rej_nuc = np.unique(rej_nuc_arr)
        if uniq_rej_nuc.shape[0] < min_nuclear_genes:
            continue

        # Tx-weighted prune for sub-seed too.
        n_uniq_rej = uniq_rej_nuc.shape[0]
        tx_counts_rej = np.zeros(n_uniq_rej, dtype=np.int32)
        for ti in range(n_uniq_rej):
            tx_counts_rej[ti] = int(np.sum(rej_nuc_arr == uniq_rej_nuc[ti]))
        if not tx_weighted:
            tx_counts_rej = np.ones(n_uniq_rej, dtype=np.int32)
        sub_seed = _greedy_prune_to_retained(uniq_rej_nuc, tx_counts_rej, W_mv, threshold)
        sub_seed_mv = sub_seed
        sub_seed_len = sub_seed.shape[0]
        if sub_seed_len == 0:
            continue

        # Coherence floor on sub-seed: reject if internal mean PMI is
        # below the floor — prevents degenerate "any non-conflicting
        # clique" sub-seeds from forming spurious partials. Rest-pile
        # tx stay unassigned (code 2, already default) for downstream
        # Rescue to route based on neighbour gene composition.
        if _mean_internal_pmi(sub_seed_mv, sub_seed_len, W_mv) < seed_coherence_floor:
            continue

        # Re-test rejected tx against sub-seed. When nuclear_only_admit
        # is set, cytoplasmic rejected tx are not eligible for partial
        # admission either — they stay unassigned for Rescue.
        for tx_row in rejected_rows:
            g = tx_gene_mv[tx_row]
            if g < 0:
                continue
            if nuclear_only_admit and not tx_nuc_mv[tx_row]:
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


def top_k_positive_clique_per_entity(
    list gene_id_lists,
    cnp.ndarray[cnp.float32_t, ndim=2] W,
    int K,
    double pos_threshold,
):
    """For each entity, identify its top-K signature genes — the K genes
    with highest sum of positive PMI to other genes in the same entity.

    Returns int32 ndarray of shape (n_entities, K). Unused slots are -1
    (e.g., when an entity has fewer than K genes).

    Used as a cheap pre-filter signature: two cells whose top-clique
    cross-PMI block contains a strong-negative entry are very unlikely
    to have positive ΔC at the full-pair level.
    """
    cdef int n_ent = len(gene_id_lists)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] out = np.full((n_ent, K), -1, dtype=np.int32)
    cdef cnp.int32_t[:, :] out_mv = out
    cdef cnp.float32_t[:, :] W_mv = W
    cdef int e, n_genes, i, j, gi, gj, k
    cdef float v
    cdef cnp.ndarray[cnp.int32_t, ndim=1] g_arr
    cdef cnp.int32_t[:] g_mv
    cdef cnp.ndarray[cnp.float64_t, ndim=1] scores
    cdef cnp.float64_t[:] scores_mv
    cdef cnp.ndarray[cnp.int64_t, ndim=1] order

    for e in range(n_ent):
        obj = gene_id_lists[e]
        if obj is None:
            continue
        g_arr = np.asarray(obj, dtype=np.int32)
        n_genes = g_arr.shape[0]
        if n_genes == 0:
            continue
        if n_genes <= K:
            g_mv = g_arr
            for i in range(n_genes):
                out_mv[e, i] = g_mv[i]
            continue
        scores = np.zeros(n_genes, dtype=np.float64)
        scores_mv = scores
        g_mv = g_arr
        for i in range(n_genes):
            gi = g_mv[i]
            for j in range(n_genes):
                if i == j:
                    continue
                gj = g_mv[j]
                v = W_mv[gi, gj]
                if v != v:
                    continue
                if v > pos_threshold:
                    scores_mv[i] += v
        # Top K by score (np.argsort returns ascending; take last K)
        order = np.argsort(scores)
        for k in range(K):
            out_mv[e, k] = g_mv[order[n_genes - 1 - k]]
    return out


def fast_gate_pairs(
    cnp.ndarray[cnp.int32_t, ndim=2] top_cliques,        # [n_ent, K]
    cnp.ndarray[cnp.int32_t, ndim=2] pair_indices,        # [n_pairs, 2]
    cnp.ndarray[cnp.float32_t, ndim=2] W,
    double mean_threshold,
):
    """Vectorised batch gate using MEAN cross-PMI of the top-clique
    block. Returns uint8 array of length n_pairs: 1 if the pair
    SURVIVES (mean cross-PMI in K×K block is ≥ mean_threshold).

    For each pair (i, j): compute mean(W[top_cliques[i, *],
    top_cliques[j, *]]) over finite entries (skipping -1 placeholders
    and self-pairs g_a == g_b). Reject if mean < mean_threshold.

    A single strongly-negative entry no longer kills the pair (which
    over-rejected legitimate Phase-1c partial reattachments where
    one or two markers diverge but the broader signature is still
    coherent). Default `mean_threshold = 0.0` rejects only when the
    cross block is net-negative on average.
    """
    cdef int n_pairs = pair_indices.shape[0]
    cdef int K = top_cliques.shape[1]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] keep = np.ones(n_pairs, dtype=np.uint8)
    cdef cnp.uint8_t[:] keep_mv = keep
    cdef cnp.int32_t[:, :] tc_mv = top_cliques
    cdef cnp.int32_t[:, :] pi_mv = pair_indices
    cdef cnp.float32_t[:, :] W_mv = W
    cdef int p, i, j, a, b, gi, gj
    cdef int n_finite
    cdef double total
    cdef float v
    for p in range(n_pairs):
        i = pi_mv[p, 0]
        j = pi_mv[p, 1]
        total = 0.0
        n_finite = 0
        for a in range(K):
            gi = tc_mv[i, a]
            if gi < 0:
                continue
            for b in range(K):
                gj = tc_mv[j, b]
                if gj < 0:
                    continue
                if gi == gj:
                    continue
                v = W_mv[gi, gj]
                if v != v:
                    continue
                total += v
                n_finite += 1
        # No observations → keep (no evidence of incompatibility)
        if n_finite == 0:
            continue
        if (total / n_finite) < mean_threshold:
            keep_mv[p] = 0
    return keep


def coherence_count_kernel(
    cnp.ndarray[cnp.int32_t, ndim=1] gene_ids,
    cnp.ndarray[cnp.float32_t, ndim=2] W,
    double threshold,
):
    """Count-mode coherence in pure C. Returns (C, purity, conflict).

    Equivalent to coherence(gene_ids, W, mode='count', threshold=...)
    in stitching.py — single C loop over the upper-triangular gene-pair
    submatrix, no numpy intermediates. ~5-10× faster per call.
    """
    cdef int k = gene_ids.shape[0]
    if k < 2:
        return 0.0, 0.0, 0.0
    cdef cnp.int32_t[:] g_mv = gene_ids
    cdef cnp.float32_t[:, :] W_mv = W
    cdef int i, j, gi, gj
    cdef int n_above = 0
    cdef int n_below = 0
    cdef int n_finite = 0
    cdef float v
    cdef double neg_thr = -threshold
    cdef double purity, conflict
    for i in range(k):
        gi = g_mv[i]
        for j in range(i + 1, k):
            gj = g_mv[j]
            v = W_mv[gi, gj]
            if v != v:  # NaN
                continue
            n_finite += 1
            if v > threshold:
                n_above += 1
            elif v < neg_thr:
                n_below += 1
    if n_finite == 0:
        return 0.0, 0.0, 0.0
    purity = <double> n_above / n_finite
    conflict = <double> n_below / n_finite
    return (purity - conflict), purity, conflict


def coherence_count_primitives(
    cnp.ndarray[cnp.int32_t, ndim=1] gene_ids,
    cnp.ndarray[cnp.float32_t, ndim=2] W,
    double threshold,
):
    """Like `coherence_count_kernel` but returns the raw counts
    (n_above, n_below, n_finite) instead of (C, purity, conflict).
    Used by the decomposable-coherence Stitch path for primitive-sum
    arithmetic across merges."""
    cdef int k = gene_ids.shape[0]
    if k < 2:
        return 0, 0, 0
    cdef cnp.int32_t[:] g_mv = gene_ids
    cdef cnp.float32_t[:, :] W_mv = W
    cdef int i, j, gi, gj
    cdef int n_above = 0
    cdef int n_below = 0
    cdef int n_finite = 0
    cdef float v
    cdef double neg_thr = -threshold
    for i in range(k):
        gi = g_mv[i]
        for j in range(i + 1, k):
            gj = g_mv[j]
            v = W_mv[gi, gj]
            if v != v:  # NaN
                continue
            n_finite += 1
            if v > threshold:
                n_above += 1
            elif v < neg_thr:
                n_below += 1
    return n_above, n_below, n_finite


def coherence_cross_primitives(
    cnp.ndarray[cnp.int32_t, ndim=1] gene_ids_a,
    cnp.ndarray[cnp.int32_t, ndim=1] gene_ids_b,
    cnp.ndarray[cnp.float32_t, ndim=2] W,
    double threshold,
):
    """Cross-set primitives: count of (g_a, g_b) pairs with g_a in
    gene_ids_a, g_b in gene_ids_b, g_a != g_b, where W[g_a, g_b] is
    above/below threshold. Returns (n_above, n_below, n_finite).

    Used to compute coh(union) from primitives:
      coh(P ∪ Q) = (a + b + a×b - common_internal_double_count) / ...
    where common_internal subtracts overlap to avoid double-count.

    Caller must pass gene_ids_a and gene_ids_b as DISJOINT arrays for
    the simple-sum semantics. For overlap-aware union, decompose
    P ∪ Q = (P\Q) ∪ (Q\P) ∪ (P∩Q) into 3 disjoint segments and call
    this kernel pairwise.
    """
    cdef int ka = gene_ids_a.shape[0]
    cdef int kb = gene_ids_b.shape[0]
    if ka == 0 or kb == 0:
        return 0, 0, 0
    cdef cnp.int32_t[:] ga_mv = gene_ids_a
    cdef cnp.int32_t[:] gb_mv = gene_ids_b
    cdef cnp.float32_t[:, :] W_mv = W
    cdef int i, j, gi, gj
    cdef int n_above = 0
    cdef int n_below = 0
    cdef int n_finite = 0
    cdef float v
    cdef double neg_thr = -threshold
    for i in range(ka):
        gi = ga_mv[i]
        for j in range(kb):
            gj = gb_mv[j]
            if gi == gj:
                continue
            v = W_mv[gi, gj]
            if v != v:
                continue
            n_finite += 1
            if v > threshold:
                n_above += 1
            elif v < neg_thr:
                n_below += 1
    return n_above, n_below, n_finite


def rescue_per_tx_batch(
    cnp.ndarray[cnp.float32_t, ndim=2] una_coords,    # [n_una, 3] (x,y,z)
    cnp.ndarray[cnp.int64_t, ndim=1] una_g_idx,       # [n_una], gene-vocab idx, -1 = skip
    cnp.ndarray[cnp.int64_t, ndim=2] nb_bins,         # [n_una, 9] bin-keys (-1 = skip)
    cnp.ndarray[cnp.float32_t, ndim=2] assigned_coords,  # [n_assigned, 3]
    cnp.ndarray[cnp.int32_t, ndim=1] assigned_ent_id,    # [n_assigned] entity codes
    cnp.ndarray[cnp.int64_t, ndim=1] bin_offsets,        # [max_bin_key+2] CSR offsets
    cnp.ndarray[cnp.int64_t, ndim=1] bin_data,           # CSR data (assigned local idx)
    cnp.ndarray[cnp.int32_t, ndim=1] ent_gene_offsets,   # [n_entities+1] CSR
    cnp.ndarray[cnp.int32_t, ndim=1] ent_gene_idx,       # CSR data (sorted gene idxs)
    cnp.ndarray[cnp.float32_t, ndim=2] W,                # PMI matrix (NaN-filled = 0 ok)
    double z_bound,                                       # 0.0 ⇒ no z-bound
    int veto_mode,                                        # 0=min, 1=mean, 2=hybrid
    double mean_threshold,
    int small_entity_guard_n,
    double neg_npmi_threshold,
    double min_admit_threshold = 0.0,                     # hybrid: min-PMI fast-pass cutoff
):
    """Per-unassigned-tx Rescue batch.

    Mirrors the Python loop in `reassign_unassigned_grid_pool` (in
    `spatial.py`) — one C-level pass over all unassigned tx, bin-gated
    candidate gathering, veto check (min OR mean), nearest-candidate-tx
    distance pick.

    Returns
    -------
    best_ent_id : int32[n_una]
        Entity-id code of the best matching entity, or -1 if no rescue.
    matched_dist : float32[n_una]
        Distance to the best candidate (sqrt). NaN if no rescue.
    reason : int32[n_una]
        0 = rescued, 1 = no_candidates, 2 = blocked_by_veto.
    n_small_entity_fallback : int32[n_una]
        Per-tx count of entities that fell back to min-veto due to
        small-entity-guard. Sum across tx for the stat.
    """
    cdef int n_una = una_coords.shape[0]
    cdef int n_ent = ent_gene_offsets.shape[0] - 1
    cdef int max_bin_key_plus_one = bin_offsets.shape[0] - 1
    cdef int has_z = (una_coords.shape[1] >= 3) and (z_bound > 0.0)

    cdef cnp.ndarray[cnp.int32_t, ndim=1] best_ent_arr = np.full(n_una, -1, dtype=np.int32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] best_dist_arr = np.full(n_una, np.nan, dtype=np.float32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] reason_arr = np.zeros(n_una, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] sef_arr = np.zeros(n_una, dtype=np.int32)

    cdef cnp.float32_t[:, :] una_c_mv = una_coords
    cdef cnp.int64_t[:]     una_g_mv = una_g_idx
    cdef cnp.int64_t[:, :]  nb_bins_mv = nb_bins
    cdef cnp.float32_t[:, :] ass_c_mv = assigned_coords
    cdef cnp.int32_t[:]     ass_ent_mv = assigned_ent_id
    cdef cnp.int64_t[:]     bin_off_mv = bin_offsets
    cdef cnp.int64_t[:]     bin_data_mv = bin_data
    cdef cnp.int32_t[:]     ent_off_mv = ent_gene_offsets
    cdef cnp.int32_t[:]     ent_g_mv = ent_gene_idx
    cdef cnp.float32_t[:, :] W_mv = W
    cdef cnp.int32_t[:]     best_ent_mv = best_ent_arr
    cdef cnp.float32_t[:]   best_dist_mv = best_dist_arr
    cdef cnp.int32_t[:]     reason_mv = reason_arr
    cdef cnp.int32_t[:]     sef_mv = sef_arr

    # Per-entity decision cache, invalidated per tx via generation counter.
    # cache_state: 0=unset, 1=vetoed, 2=ok.
    cdef cnp.ndarray[cnp.int8_t, ndim=1] ent_cache = np.zeros(n_ent, dtype=np.int8)
    cdef cnp.int8_t[:] cache_mv = ent_cache
    cdef cnp.ndarray[cnp.int32_t, ndim=1] cache_gen = np.zeros(n_ent, dtype=np.int32)
    cdef cnp.int32_t[:] gen_mv = cache_gen
    cdef int current_gen = 0

    cdef int i, j, b, off_lo, off_hi, ass_li, ent
    cdef int e_off_lo, e_off_hi, n_ent_genes, ig, eg
    cdef int g_idx, vetoed, any_vetoed, found_neg, n_finite, used_fallback
    cdef int g_in_E
    cdef double dx, dy, dz, d, best_d, pmi_sum, pmi_val, mean_p, min_pmi

    for i in range(n_una):
        current_gen += 1
        g_idx = <int> una_g_mv[i]
        if g_idx < 0:
            reason_mv[i] = 1
            continue

        best_d = 1e300
        best_ent_mv[i] = -1
        any_vetoed = 0
        used_fallback = 0

        for j in range(9):
            b = <int> nb_bins_mv[i, j]
            if b < 0 or b >= max_bin_key_plus_one:
                continue
            off_lo = <int> bin_off_mv[b]
            off_hi = <int> bin_off_mv[b + 1]
            for k in range(off_lo, off_hi):
                ass_li = <int> bin_data_mv[k]
                # z-bound filter
                if has_z:
                    dz = ass_c_mv[ass_li, 2] - una_c_mv[i, 2]
                    if dz < 0: dz = -dz
                    if dz > z_bound:
                        continue

                ent = ass_ent_mv[ass_li]
                if ent < 0 or ent >= n_ent:
                    continue

                # Cache check
                if gen_mv[ent] == current_gen:
                    if cache_mv[ent] == 1:
                        continue
                    # else cache_mv[ent] == 2 → OK; fall through
                else:
                    # Compute veto for this entity (this tx).
                    e_off_lo = ent_off_mv[ent]
                    e_off_hi = ent_off_mv[ent + 1]
                    n_ent_genes = e_off_hi - e_off_lo
                    vetoed = 0

                    if n_ent_genes == 0:
                        vetoed = 0
                    elif veto_mode == 0:
                        # min-mode: any entity gene with W[g, eg] < neg_thr → veto
                        for ig in range(e_off_lo, e_off_hi):
                            eg = ent_g_mv[ig]
                            if eg == g_idx:
                                continue
                            if W_mv[g_idx, eg] < neg_npmi_threshold:
                                vetoed = 1
                                break
                    elif veto_mode == 1:
                        # mean-mode
                        pmi_sum = 0.0
                        n_finite = 0
                        found_neg = 0
                        for ig in range(e_off_lo, e_off_hi):
                            eg = ent_g_mv[ig]
                            if eg == g_idx:
                                continue
                            pmi_val = W_mv[g_idx, eg]
                            if pmi_val == pmi_val:  # not NaN
                                pmi_sum += pmi_val
                                n_finite += 1
                                if pmi_val < neg_npmi_threshold:
                                    found_neg = 1
                        if n_finite < small_entity_guard_n:
                            # fall back to min-mode
                            vetoed = found_neg
                            if not found_neg and n_finite > 0:
                                used_fallback += 1
                        elif n_finite == 0:
                            vetoed = 1
                        else:
                            mean_p = pmi_sum / n_finite
                            vetoed = 1 if mean_p <= mean_threshold else 0
                    else:
                        # hybrid mode (veto_mode == 2):
                        #   if g ∈ E.genes → admit (no test).
                        #   else if min PMI(g, E\{g}) > min_admit_threshold → admit.
                        #   else if mean PMI(g, E\{g}, finite) > mean_threshold → admit.
                        #   else → veto.
                        g_in_E = 0
                        for ig in range(e_off_lo, e_off_hi):
                            if ent_g_mv[ig] == g_idx:
                                g_in_E = 1
                                break
                        if g_in_E:
                            vetoed = 0
                        else:
                            pmi_sum = 0.0
                            n_finite = 0
                            min_pmi = 1e300
                            for ig in range(e_off_lo, e_off_hi):
                                eg = ent_g_mv[ig]
                                pmi_val = W_mv[g_idx, eg]
                                if pmi_val == pmi_val:  # not NaN
                                    pmi_sum += pmi_val
                                    n_finite += 1
                                    if pmi_val < min_pmi:
                                        min_pmi = pmi_val
                            if n_finite == 0:
                                vetoed = 1
                            elif min_pmi > min_admit_threshold:
                                vetoed = 0  # unanimous-positive fast-pass
                            elif pmi_sum / n_finite > mean_threshold:
                                vetoed = 0  # aggregate-positive slow-pass
                            else:
                                vetoed = 1

                    cache_mv[ent] = 1 if vetoed else 2
                    gen_mv[ent] = current_gen

                if cache_mv[ent] == 1:
                    any_vetoed = 1
                    continue

                # Distance (squared; final sqrt at write-out)
                dx = ass_c_mv[ass_li, 0] - una_c_mv[i, 0]
                dy = ass_c_mv[ass_li, 1] - una_c_mv[i, 1]
                d = dx * dx + dy * dy
                if has_z:
                    dz = ass_c_mv[ass_li, 2] - una_c_mv[i, 2]
                    d += dz * dz
                if d < best_d:
                    best_d = d
                    best_ent_mv[i] = ent

        if best_ent_mv[i] == -1:
            reason_mv[i] = 2 if any_vetoed else 1
        else:
            best_dist_mv[i] = <cnp.float32_t> (best_d ** 0.5)
        sef_mv[i] = used_fallback

    return best_ent_arr, best_dist_arr, reason_arr, sef_arr

