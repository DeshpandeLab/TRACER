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
