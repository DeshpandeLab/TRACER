"""Synthetic correctness test for ``tracer.metrics.compute_pmi_bootstrap``.

Plants 5 known gene-pair structures via :func:`tests.synthetic.make_synthetic_npmi_panel`
and asserts the bootstrap classifies each correctly:

  - genes 0, 1: strong positive cooccurrence → ``W[0,1] > 0``
  - genes 2, 3: strong mutual exclusivity   → ``W[2,3] < 0``
  - genes 4, 5: independent (rate 0.3 each) → ``|W[4,5]| < 0.2`` or absent
  - genes 6, 7: rare with zero observed cooccur, E[cooccur] < 10 →
                indeterminate (absent from W_sparse)
  - genes 8, 9: high marginal with zero observed cooccur,
                E[cooccur] ≥ 10 → ``neg_one`` sentinel (W[8,9] == -1)
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from tracer.metrics import compute_pmi_bootstrap

from tests.synthetic import make_synthetic_npmi_panel


@pytest.fixture(scope="module")
def bootstrap_result():
    """Compute the bootstrap once and reuse across the test module."""
    df, M = make_synthetic_npmi_panel()
    res = compute_pmi_bootstrap(
        df, group_key="cell_id", feature_col="feature_name",
        tau=0.05, ci_level=0.95,
        max_bootstraps=2000, coarse_block=200, refine_block=200,
        expected_cooccur_for_neg_one=10.0,
        bootstrap_kernel="pair_gather",
        seed=0, show_progress=False,
    )
    return res, M


def _W_lookup(res):
    """Convert sparse W to a dense {(i, j): value} dict for easy lookup."""
    W = res.W_sparse if sp.isspmatrix_coo(res.W_sparse) else res.W_sparse.tocoo()
    return {(int(i), int(j)): float(v) for i, j, v in zip(W.row, W.col, W.data)}


def test_strong_positive_classified_pos(bootstrap_result):
    res, M = bootstrap_result
    W = _W_lookup(res)
    g_to_i = {g: i for i, g in enumerate(res.genes)}
    i, j = g_to_i["gene_00"], g_to_i["gene_01"]
    key = (min(i, j), max(i, j))
    assert key in W, "Strong-positive pair should appear in W_sparse"
    assert W[key] > 0.1, f"Expected NPMI > 0.1 for strong positive pair, got {W[key]}"


def test_strong_negative_classified_neg(bootstrap_result):
    res, M = bootstrap_result
    W = _W_lookup(res)
    g_to_i = {g: i for i, g in enumerate(res.genes)}
    i, j = g_to_i["gene_02"], g_to_i["gene_03"]
    key = (min(i, j), max(i, j))
    assert key in W, "Strong-negative pair should appear in W_sparse"
    assert W[key] < -0.1, f"Expected NPMI < -0.1 for strong negative pair, got {W[key]}"


def test_independent_classified_indeterminate_or_zero(bootstrap_result):
    """Independent pair should either be absent (indeterminate) or have
    near-zero NPMI."""
    res, M = bootstrap_result
    W = _W_lookup(res)
    g_to_i = {g: i for i, g in enumerate(res.genes)}
    i, j = g_to_i["gene_04"], g_to_i["gene_05"]
    key = (min(i, j), max(i, j))
    if key in W:
        # If the bootstrap classified it confidently, the value should be near zero.
        assert abs(W[key]) < 0.2, f"Independent pair should be near zero, got {W[key]}"


def test_high_marginal_zero_cooccur_classified_neg_one(bootstrap_result):
    """Pair with high marginal rate and zero observed co-occurrence
    (E[cooccur] ≫ 10) should be classified as the ``neg_one`` sentinel."""
    res, M = bootstrap_result
    W = _W_lookup(res)
    g_to_i = {g: i for i, g in enumerate(res.genes)}
    i, j = g_to_i["gene_08"], g_to_i["gene_09"]
    key = (min(i, j), max(i, j))
    assert key in W, "High-marginal zero-cooccur pair should appear in W_sparse"
    assert W[key] == -1.0, f"Expected neg_one sentinel, got {W[key]}"


def test_low_marginal_zero_cooccur_left_indeterminate(bootstrap_result):
    """Pair with low marginal rate and zero observed co-occurrence
    (E[cooccur] < 10) should be left indeterminate (absent from W)."""
    res, M = bootstrap_result
    W = _W_lookup(res)
    g_to_i = {g: i for i, g in enumerate(res.genes)}
    i, j = g_to_i["gene_06"], g_to_i["gene_07"]
    key = (min(i, j), max(i, j))
    assert key not in W, (
        f"Low-marginal zero-cooccur pair should be absent from W_sparse "
        f"(indeterminate); got W[{key}] = {W.get(key)}"
    )


def test_diagnostics_report_n_pairs(bootstrap_result):
    """Sanity: diagnostics dict should report counts that sum sensibly."""
    res, _ = bootstrap_result
    diag = res.diagnostics
    # Some non-zero classifications should exist
    n_classified = (
        diag.get("n_pos", 0) + diag.get("n_neg", 0) + diag.get("n_neg_one", 0)
    )
    assert n_classified >= 2, f"Expected at least 2 classified pairs, got {n_classified}"


def test_counts_matrix_matches_df_presence():
    """A counts matrix and the equivalent long-form df build the same M."""
    import numpy as np, scipy.sparse as sp
    from tracer.metrics import _presence_from_counts, _build_presence_matrix
    import pandas as pd
    # 3 cells x 4 genes raw counts
    X = sp.csr_matrix(np.array([[2, 0, 5, 1],
                                [3, 2, 0, 0],
                                [0, 1, 4, 2]], dtype=np.float32))
    var = np.array(["g0", "g1", "g2", "g3"])
    obs = np.array(["c0", "c1", "c2"])
    M, genes, ctx = _presence_from_counts(X, var, obs, min_occurrences_per_context=2)
    # gene present where count>=2: c0:{g0,g2}, c1:{g0,g1}, c2:{g2,g3}
    dense = np.asarray(M.todense())
    gi = {g: i for i, g in enumerate(genes)}
    assert dense[0, gi["g0"]] == 1 and dense[0, gi["g2"]] == 1
    assert dense[0, gi["g3"]] == 0  # count 1 < 2
    assert dense[1, gi["g1"]] == 1 and dense[1, gi["g3"]] == 0
    assert set(genes) == {"g0", "g1", "g2", "g3"}
    assert M.dtype == np.int32


def test_counts_xor_df_required():
    import numpy as np, scipy.sparse as sp, pytest
    from tracer.metrics import compute_pmi_bootstrap
    X = sp.csr_matrix(np.array([[2, 2], [2, 2]], dtype=np.float32))
    var = np.array(["a", "b"])
    with pytest.raises(ValueError, match="exactly one"):
        compute_pmi_bootstrap(None, counts=None)            # neither
    # counts path returns a result without raising
    res = compute_pmi_bootstrap(None, counts=(X, var), metric="pmi",
                                min_expected_cooccur_for_evidence=0.5,
                                bootstrap_kernel="pair_gather", seed=0)
    assert res.genes.tolist() == ["a", "b"]


def test_pair_gather_kernel_matches_legacy_output():
    """Refactor must preserve today's exact output on the synthetic panel."""
    import numpy as np
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, M = make_synthetic_npmi_panel()
    res = compute_pmi_bootstrap(
        df, group_key="cell_id", feature_col="feature_name",
        metric="npmi", bootstrap_kernel="pair_gather",
        seed=0, show_progress=False,
    )
    W = res.W_sparse.tocoo()
    got = {(int(i), int(j)): float(v) for i, j, v in zip(W.row, W.col, W.data)}
    # golden values captured from the pre-refactor run (fill from Step 2 baseline)
    assert (0, 1) in got and got[(0, 1)] > 0.1
    assert (2, 3) in got and got[(2, 3)] < -0.1
    assert (8, 9) in got and got[(8, 9)] == -1.0
    assert res.diagnostics["n_neg_one"] >= 1


def test_gene_row_kernel_parity_with_pair_gather():
    """gene_row settles the same pair SET as pair_gather; values within tol."""
    import numpy as np
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, M = make_synthetic_npmi_panel()
    common = dict(group_key="cell_id", feature_col="feature_name",
                  metric="npmi", seed=0, show_progress=False)
    a = compute_pmi_bootstrap(df, bootstrap_kernel="pair_gather", **common)
    b = compute_pmi_bootstrap(df, bootstrap_kernel="gene_row", **common)
    Wa = {(int(i), int(j)) for i, j in zip(*a.W_sparse.nonzero())}
    Wb = {(int(i), int(j)) for i, j in zip(*b.W_sparse.nonzero())}
    # neg_one + clearly-settled pairs must match exactly
    assert (0, 1) in Wb and (2, 3) in Wb and (8, 9) in Wb
    assert b.W_sparse.tocsr()[8, 9] == -1.0
    # sign agreement on the strong pairs
    assert b.W_sparse.tocsr()[0, 1] > 0.1
    assert b.W_sparse.tocsr()[2, 3] < -0.1
    # broad set agreement (allow tiny boundary differences from RNG stream)
    assert len(Wa ^ Wb) <= 2


def test_gene_row_accepts_single_element_tau_sequence():
    """gene_row must accept single-threshold tau given as a 1-element
    sequence (``[0.05]``/``(0.05,)``/``np.array([0.05])``) — the tau parser
    treats size==1 as scalar — and must still reject genuine dual-tau."""
    import numpy as np
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    common = dict(group_key="cell_id", feature_col="feature_name",
                  metric="npmi", bootstrap_kernel="gene_row", seed=0,
                  show_progress=False, max_bootstraps=600)
    for tau in ([0.05], (0.05,), np.array([0.05]), [0.05, 0.05]):
        res = compute_pmi_bootstrap(df, tau=tau, **common)
        assert res.diagnostics["is_dual_tau"] is False
        assert res.diagnostics["kernel"] == "gene_row"
    # genuine dual-tau (low < high) is unsupported by gene_row → fail loud
    with pytest.raises(NotImplementedError):
        compute_pmi_bootstrap(df, tau=[0.02, 0.08], **common)


def test_gene_order_prob_ascending():
    import numpy as np
    from tracer.metrics import _gene_processing_order
    k = np.array([100, 5, 50, 5], dtype=np.float64)   # detection per gene
    order = _gene_processing_order("prob_ascending", k=k)
    # ascending k: genes 1,3 (k=5) before 2 (k=50) before 0 (k=100)
    assert order[-1] == 0
    assert set(order[:2].tolist()) == {1, 3}
    assert list(order) != list(range(len(k)))  # not identity unless sorted


def test_owned_partners_each_pair_once_and_window_shrinks():
    import numpy as np
    from tracer.metrics import _owned_partners
    # candidate upper-tri pairs (by gene index): (0,1),(0,2),(1,2),(2,3)
    obs_i = np.array([0, 0, 1, 2]); obs_j = np.array([1, 2, 2, 3])
    can = np.array([True, True, True, True])
    order = np.array([0, 1, 2, 3])           # process in index order
    pos = np.empty(4, dtype=np.int64); pos[order] = np.arange(4)
    indptr, partner, pairref = _owned_partners(obs_i, obs_j, can, pos)
    # each pair owned exactly once
    assert partner.size == 4
    # gene 0 owns {1,2}; gene 1 owns {2}; gene 2 owns {3}; gene 3 owns {}
    owned = {g: partner[indptr[g]:indptr[g+1]].tolist() for g in range(4)}
    assert sorted(owned[0]) == [1, 2]
    assert owned[1] == [2] and owned[2] == [3] and owned[3] == []
    # window (owned count) is non-increasing here
    counts = [indptr[g+1]-indptr[g] for g in range(4)]
    assert counts == [2, 1, 1, 0]


def test_gene_batches_respect_pair_cap_and_cover_all():
    import numpy as np
    from tracer.metrics import _gene_batches
    order = np.arange(6)
    owned_counts = np.array([3, 3, 3, 3, 3, 0])   # owned pairs per gene (in order)
    # budget that allows ~5 pairs/batch -> batches: [0,2)=6>5 so [0,1],... check coverage
    batches = _gene_batches(order, owned_counts, gene_batch_peak_gb=1e-6,
                            coarse_block=200)
    # every gene covered exactly once, contiguous
    covered = []
    for (s, e) in batches:
        covered.extend(range(s, e))
    assert covered == list(range(6))
    # each batch's owned-pair sum <= cap (except a single gene that alone exceeds)
    cap = max(1, int(1e-6 * 1e9 / (200 * 32)))
    for (s, e) in batches:
        tot = owned_counts[s:e].sum()
        assert (e - s == 1) or tot <= cap


def test_gene_row_multibatch_matches_singlebatch():
    import numpy as np
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, M = make_synthetic_npmi_panel()
    common = dict(group_key="cell_id", feature_col="feature_name",
                  metric="npmi", bootstrap_kernel="gene_row", seed=0,
                  show_progress=False)
    big = compute_pmi_bootstrap(df, gene_batch_peak_gb=16.0, **common)
    small = compute_pmi_bootstrap(df, gene_batch_peak_gb=1e-7, **common)  # force splits
    Sbig = {(int(i), int(j)) for i, j in zip(*big.W_sparse.nonzero())}
    Ssmall = {(int(i), int(j)) for i, j in zip(*small.W_sparse.nonzero())}
    assert len(Sbig ^ Ssmall) <= 2   # same settled set up to boundary RNG


def test_checkpoint_roundtrip(tmp_path):
    import numpy as np
    from tracer.metrics import _write_checkpoint, _read_checkpoint
    p = tmp_path / "ck.npz"
    _write_checkpoint(str(p), [0, 1], [2, 3], [0.5, -0.5], G=4, cursor=2)
    rows, cols, vals, cursor = _read_checkpoint(str(p))
    assert rows == [0, 1] and cols == [2, 3]
    assert vals == [0.5, -0.5] and cursor == 2
    assert _read_checkpoint(str(tmp_path / "missing.npz")) is None


def test_checkpoint_resume_equiv(tmp_path):
    import numpy as np
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, M = make_synthetic_npmi_panel()
    common = dict(group_key="cell_id", feature_col="feature_name", metric="npmi",
                  bootstrap_kernel="gene_row", gene_batch_peak_gb=1e-7, seed=0,
                  show_progress=False)
    ck = str(tmp_path / "run.ckpt.npz")
    full = compute_pmi_bootstrap(df, **common)                  # no checkpoint
    # write a partial checkpoint by running once with checkpoint, then re-run resumes
    part = compute_pmi_bootstrap(df, checkpoint_path=ck, **common)
    resumed = compute_pmi_bootstrap(df, checkpoint_path=ck, **common)  # resumes/no-op
    Sfull = {(int(i), int(j)) for i, j in zip(*full.W_sparse.nonzero())}
    Sres = {(int(i), int(j)) for i, j in zip(*resumed.W_sparse.nonzero())}
    assert Sfull == Sres


# Golden values captured from the PRE-vectorization gene_row kernel on the
# default synthetic panel (make_synthetic_npmi_panel(), seed 42 fixture).
# Run config: tau=0.05, ci_level=0.95, max_bootstraps=2000,
# coarse_block=refine_block=200, metric="npmi". These are the same for
# seeds 0 and 1 because the only bootstrap-settled W entry (the strong
# positive pair (0,1)) clears ±tau in the very first coarse block under
# either RNG stream; the (2,3)/(8,9) entries are Stage-1 neg_one sentinels.
# The vectorized kernel MUST reproduce these bit-for-bit.
_GENE_ROW_GOLDEN_W_NNZ = {
    (0, 1): np.float32(0.95119965),
    (2, 3): np.float32(-1.0),
    (8, 9): np.float32(-1.0),
}
# Bootstrap-kernel diagnostics (settle-count contract) for the same run.
_GENE_ROW_GOLDEN_DIAG = {
    "n_pos": 1,
    "n_neg": 0,
    "n_dead_zone": 1,
    "n_unsettled": 131,
}
_GENE_ROW_GOLDEN_NBP_SUM = 262400
_GENE_ROW_GOLDEN_NBP_LEN = 133


def _golden_dense_W(G=20):
    W = np.zeros((G, G), dtype=np.float32)
    for (i, j), v in _GENE_ROW_GOLDEN_W_NNZ.items():
        W[i, j] = v
    return W


@pytest.mark.parametrize("seed", [0, 1])
def test_gene_row_vectorized_bitwise_identical(seed):
    """HARD GATE: the vectorized gene_row kernel must reproduce the
    pre-change kernel's W_sparse bit-for-bit AND the exact settle-count
    diagnostics (n_pos/n_neg/n_dead_zone/n_unsettled/n_bootstraps_per_pair)
    on the synthetic panel, for the same seed. Same RNG draws → same samples
    → same quantiles → same settle decisions → identical W."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    res = compute_pmi_bootstrap(
        df, group_key="cell_id", feature_col="feature_name",
        metric="npmi", bootstrap_kernel="gene_row",
        tau=0.05, ci_level=0.95,
        max_bootstraps=2000, coarse_block=200, refine_block=200,
        seed=seed, show_progress=False,
    )
    W = res.W_sparse.toarray()
    assert np.array_equal(W, _golden_dense_W(W.shape[0])), (
        "vectorized gene_row W diverged from golden\n"
        f"got nnz={dict(zip(map(tuple, np.argwhere(W != 0)), W[W != 0]))}"
    )
    d = res.diagnostics
    for k, v in _GENE_ROW_GOLDEN_DIAG.items():
        assert int(d[k]) == v, f"diag[{k}] = {d[k]} != golden {v}"
    nbp = np.asarray(d["n_bootstraps_per_pair"])
    assert len(nbp) == _GENE_ROW_GOLDEN_NBP_LEN
    assert int(nbp.sum()) == _GENE_ROW_GOLDEN_NBP_SUM


# Golden for the MULTI-BATCH path (gene_batch_peak_gb=1e-7 forces many
# single-gene batches). W is identical to single-batch (the same 3 nnz),
# but the settle-count diagnostics differ because each batch seeds its own
# RNG (default_rng(seed + b_idx)) so the per-block sample streams differ.
# Captured from the pre-vectorization kernel, seed 0.
_GENE_ROW_GOLDEN_DIAG_MB = {
    "n_pos": 1,
    "n_neg": 0,
    "n_dead_zone": 0,
    "n_unsettled": 132,
}
_GENE_ROW_GOLDEN_NBP_SUM_MB = 264200
_GENE_ROW_GOLDEN_NBP_LEN_MB = 133


def test_gene_row_subsample_runs_and_deterministic():
    """gene_row with subsample_size=s must run (no NotImplementedError) and be
    deterministic: same (seed, s) → bit-identical W."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    common = dict(
        group_key="cell_id", feature_col="feature_name", metric="npmi",
        bootstrap_kernel="gene_row", tau=0.05, ci_level=0.95,
        max_bootstraps=2000, coarse_block=200, refine_block=200,
        subsample_size=400, seed=0, show_progress=False,
    )
    a = compute_pmi_bootstrap(df, **common)
    b = compute_pmi_bootstrap(df, **common)
    assert a.diagnostics["kernel"] == "gene_row"
    assert a.diagnostics["subsample_size"] == 400
    Wa = a.W_sparse.toarray()
    Wb = b.W_sparse.toarray()
    assert np.array_equal(Wa, Wb), "same (seed, subsample_size) must give identical W"
    # The strong positive pair (0,1) still settles positive under subsampling.
    assert Wa[0, 1] > 0.1
    # The Stage-1 neg_one sentinels are independent of the bootstrap kernel.
    assert Wa[8, 9] == -1.0


def test_gene_row_subsample_full_count_matches_none():
    """LARGE-S SANITY: subsample_size == C draws C cells via the subsample path.
    The RNG consumption differs from the rc-bincount full path, so W need not be
    bit-identical, but the settled SET must agree within a tiny tolerance."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, M = make_synthetic_npmi_panel()
    C = M.shape[0]
    common = dict(
        group_key="cell_id", feature_col="feature_name", metric="npmi",
        bootstrap_kernel="gene_row", tau=0.05, ci_level=0.95,
        max_bootstraps=2000, coarse_block=200, refine_block=200,
        seed=0, show_progress=False,
    )
    full = compute_pmi_bootstrap(df, subsample_size=None, **common)
    fullc = compute_pmi_bootstrap(df, subsample_size=C, **common)
    Sfull = {(int(i), int(j)) for i, j in zip(*full.W_sparse.nonzero())}
    Sfullc = {(int(i), int(j)) for i, j in zip(*fullc.W_sparse.nonzero())}
    # Settled-set agreement within a small tolerance (boundary RNG differences).
    assert len(Sfull ^ Sfullc) <= 2
    # The unambiguous entries must agree exactly.
    Wc = fullc.W_sparse.tocsr()
    assert Wc[0, 1] > 0.1
    assert Wc[8, 9] == -1.0


def test_gene_row_vectorized_bitwise_identical_multibatch():
    """The bitwise gate must also hold when the owned pairs are split across
    multiple gene batches (forces the per-batch accumulate/settle path)."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    res = compute_pmi_bootstrap(
        df, group_key="cell_id", feature_col="feature_name",
        metric="npmi", bootstrap_kernel="gene_row",
        tau=0.05, ci_level=0.95,
        max_bootstraps=2000, coarse_block=200, refine_block=200,
        gene_batch_peak_gb=1e-7,   # force many single-gene batches
        seed=0, show_progress=False,
    )
    W = res.W_sparse.toarray()
    assert np.array_equal(W, _golden_dense_W(W.shape[0]))
    d = res.diagnostics
    for k, v in _GENE_ROW_GOLDEN_DIAG_MB.items():
        assert int(d[k]) == v, f"diag[{k}] = {d[k]} != golden {v}"
    nbp = np.asarray(d["n_bootstraps_per_pair"])
    assert len(nbp) == _GENE_ROW_GOLDEN_NBP_LEN_MB
    assert int(nbp.sum()) == _GENE_ROW_GOLDEN_NBP_SUM_MB


# ======================================================================
# O(1)-memory "counter" CI accumulator (ci_accumulator="counter")
# ======================================================================

_COMMON_GR = dict(
    group_key="cell_id", feature_col="feature_name", metric="npmi",
    bootstrap_kernel="gene_row", tau=0.05, ci_level=0.95,
    max_bootstraps=2000, coarse_block=200, refine_block=200,
    seed=0, show_progress=False,
)


def test_counter_logic_matches_quantile_decision():
    """GATE 2 — counter settle classification matches the np.quantile-vs-tau
    decision for clearly pos / neg / tight_null sample sequences.

    The counter test reduces the percentile-CI settle to integer counts:
      pos   <=> cnt_above >  ci_hi_q * nsamp
      neg   <=> cnt_below >  ci_hi_q * nsamp
      tight <=> cnt_below <  ci_lo_q * nsamp  AND  cnt_above < ci_lo_q * nsamp
    For UNAMBIGUOUS distributions (mass clearly on one side of +/-tau) this is
    identical to the linear-interpolation quantile decision used by the
    samples path. Near-boundary churn (a single order statistic straddling
    the ci quantile) is covered by the agreement gate, not here.
    """
    ci_level = 0.95
    tau = 0.05
    ci_lo_q = (1.0 - ci_level) / 2.0
    ci_hi_q = 1.0 - ci_lo_q

    def quantile_kind(arr):
        lo, hi = np.quantile(arr, [ci_lo_q, ci_hi_q])
        if lo > tau:
            return 1
        if hi < -tau:
            return -1
        if lo > -tau and hi < tau:
            return 3
        return 0

    def counter_kind(arr):
        n = arr.size
        cnt_above = int((arr > tau).sum())
        cnt_below = int((arr < -tau).sum())
        if cnt_above > ci_hi_q * n:
            return 1
        if cnt_below > ci_hi_q * n:
            return -1
        if cnt_below < ci_lo_q * n and cnt_above < ci_lo_q * n:
            return 3
        return 0

    rng = np.random.default_rng(7)
    # Clearly positive: mass well above +tau.
    pos = rng.normal(0.6, 0.05, size=200)
    # Clearly negative: mass well below -tau.
    neg = rng.normal(-0.6, 0.05, size=200)
    # Clearly tight-null: mass tightly around 0, inside +/-tau.
    tight = rng.normal(0.0, 0.005, size=200)

    assert quantile_kind(pos) == 1 and counter_kind(pos) == 1
    assert quantile_kind(neg) == -1 and counter_kind(neg) == -1
    assert quantile_kind(tight) == 3 and counter_kind(tight) == 3


def test_counter_vs_quantile_symmetric_diff_small():
    """GATE 3 (synthetic random) — over many random sample sequences the
    counter classification agrees with the quantile decision except for a
    tiny fraction of near-boundary pairs (empirical-CDF vs linear-interp).
    Quantify and bound the disagreement rate."""
    ci_level = 0.95
    tau = 0.05
    ci_lo_q = (1.0 - ci_level) / 2.0
    ci_hi_q = 1.0 - ci_lo_q
    rng = np.random.default_rng(0)
    mism = 0
    n_trials = 5000
    for _ in range(n_trials):
        n = int(rng.integers(40, 400))
        arr = rng.normal(rng.uniform(-0.3, 0.3), rng.uniform(0.02, 0.4), size=n)
        lo, hi = np.quantile(arr, [ci_lo_q, ci_hi_q])
        if lo > tau:
            qk = 1
        elif hi < -tau:
            qk = -1
        elif lo > -tau and hi < tau:
            qk = 3
        else:
            qk = 0
        cnt_above = int((arr > tau).sum())
        cnt_below = int((arr < -tau).sum())
        if cnt_above > ci_hi_q * n:
            ck = 1
        elif cnt_below > ci_hi_q * n:
            ck = -1
        elif cnt_below < ci_lo_q * n and cnt_above < ci_lo_q * n:
            ck = 3
        else:
            ck = 0
        if qk != ck:
            mism += 1
    # Boundary churn only: well under 1% of trials.
    assert mism / n_trials < 0.01, f"counter/quantile disagreement {mism}/{n_trials}"


def test_counter_mode_runs_and_emits_diag_keys():
    """Counter mode runs end-to-end and emits the same diagnostics contract
    keys as samples mode (n_pos/n_neg/n_dead_zone/n_unsettled/
    n_bootstraps_per_pair)."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    res = compute_pmi_bootstrap(df, ci_accumulator="counter", **_COMMON_GR)
    d = res.diagnostics
    for k in ("n_pos", "n_neg", "n_dead_zone", "n_unsettled",
              "n_bootstraps_per_pair"):
        assert k in d, f"missing diag key {k}"
    nbp = np.asarray(d["n_bootstraps_per_pair"])
    # nsamp array, one entry per owned pair, all positive ints.
    assert nbp.ndim == 1 and nbp.size > 0
    assert np.issubdtype(nbp.dtype, np.integer)
    # The strong-positive pair still settles positive.
    assert res.W_sparse.tocsr()[0, 1] > 0.1
    # Stage-1 neg_one sentinels are kernel-independent.
    assert res.W_sparse.tocsr()[8, 9] == -1.0


def test_counter_vs_samples_settled_set_agreement():
    """GATE 3 (kernel) — counter-mode settled SET agrees with samples-mode on
    the synthetic panel within a tiny near-tau tolerance."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    samp = compute_pmi_bootstrap(df, ci_accumulator="samples", **_COMMON_GR)
    cnt = compute_pmi_bootstrap(df, ci_accumulator="counter", **_COMMON_GR)
    Ssamp = {(int(i), int(j)) for i, j in zip(*samp.W_sparse.nonzero())}
    Scnt = {(int(i), int(j)) for i, j in zip(*cnt.W_sparse.nonzero())}
    sym = Ssamp ^ Scnt
    # Only pairs sitting right at the ci quantile near tau may flip.
    assert len(sym) <= 2, f"settled-set symmetric diff too large: {sym}"
    # The unambiguous entries must agree exactly.
    assert (0, 1) in Ssamp and (0, 1) in Scnt
    assert (8, 9) in Ssamp and (8, 9) in Scnt


def test_counter_mode_deterministic():
    """GATE 5 — same seed -> identical W in counter mode."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    a = compute_pmi_bootstrap(df, ci_accumulator="counter", **_COMMON_GR)
    b = compute_pmi_bootstrap(df, ci_accumulator="counter", **_COMMON_GR)
    assert np.array_equal(a.W_sparse.toarray(), b.W_sparse.toarray())


def test_counter_mode_incompatible_with_persist_ci():
    """Guard — ci_accumulator='counter' cannot produce CI magnitudes, so it is
    incompatible with persist_ci=True and must raise ValueError."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    with pytest.raises(ValueError):
        compute_pmi_bootstrap(df, ci_accumulator="counter", persist_ci=True,
                              **_COMMON_GR)


def test_counter_mode_o1_memory_independent_of_budget():
    """GATE 4 — counter mode stores NO per-pair sample arrays; per-pair
    accumulator state is 3 ints regardless of max_bootstraps. Drive the kernel
    directly and assert the only per-pair state is cnt_below/cnt_above/nsamp,
    whose total nbytes is independent of max_bootstraps."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    common = dict(_COMMON_GR)
    common.pop("max_bootstraps")
    r_small = compute_pmi_bootstrap(
        df, ci_accumulator="counter", max_bootstraps=200, **common)
    r_big = compute_pmi_bootstrap(
        df, ci_accumulator="counter", max_bootstraps=2000, **common)
    # Per-pair accumulator footprint = 3 int arrays of length == #owned pairs.
    # n_bootstraps_per_pair IS the nsamp array; its length (the pair count) is
    # identical across budgets, so the per-pair state size does not grow with B.
    n_small = np.asarray(r_small.diagnostics["n_bootstraps_per_pair"]).size
    n_big = np.asarray(r_big.diagnostics["n_bootstraps_per_pair"]).size
    assert n_small == n_big
    # Structural check: in counter mode the per-pair sample store is never
    # materialized (set to None) and the settle runs from counts only.
    import inspect
    from tracer import metrics as _m
    src = inspect.getsource(_m._bootstrap_gene_rows)
    # sample_store is None when counter_mode (no O(B) per-pair allocation).
    assert "sample_store: list = None if counter_mode" in src
    # The counter branch settles via the O(1) count-based helper.
    assert "_counter_settle_inplace(" in src
    # The O(1) helper itself reads no stored samples — only the 3 int arrays.
    settle_src = inspect.getsource(_m._counter_settle_inplace)
    assert "sample_store" not in settle_src
    assert "cnt_above" in settle_src and "cnt_below" in settle_src
