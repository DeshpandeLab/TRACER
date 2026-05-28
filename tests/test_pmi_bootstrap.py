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
