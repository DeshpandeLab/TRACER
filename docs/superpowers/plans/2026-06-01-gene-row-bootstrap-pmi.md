# Gene-row Bootstrap PMI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `compute_pmi_bootstrap` scale to whole-transcriptome scRNA references by replacing the per-pair column-gather kernel with a gene-row (resample-multiplicity matvec) kernel, plus matrix/h5ad input, dedup ordering, memory-budgeted batching, and checkpointing.

**Architecture:** Refactor `compute_pmi_bootstrap` so its core consumes a contexts×genes presence matrix `M`. Two front-ends build `M` (long-form `df` via groupby; counts matrix via binarize). The Stage-4 bootstrap becomes a pluggable kernel: `pair_gather` (existing, kept for parity on small panels) and `gene_row` (new default). The gene-row kernel computes each gene's co-occurrence row via a single sparse matvec against the per-iteration cell-resample multiplicity vector `rc`, so peak memory is bounded by the per-batch sample accumulators (no giant column slices).

**Tech Stack:** Python 3.12, numpy<2-compatible code (pure numpy/scipy/pandas — no Cython here), scipy.sparse (CSR/CSC), pytest. Module: `src/tracer/metrics.py`. Tests: `tests/test_pmi_bootstrap.py`.

**Spec:** `docs/superpowers/specs/2026-06-01-blocked-pmi-bootstrap-design.md`

**Refinement vs spec §5/§6:** With the gene-row kernel the per-iteration column-slice term *vanishes* (co-occurrence is a matvec against `rc`, and the shared marginal `rc @ M` is length-G). So the batch memory budget is **accumulator-dominated**: `max_pairs_per_batch ≈ gene_batch_peak_gb·1e9 / (coarse_block · 32 B)` (~2.5M owned pairs/batch at 16 GB, coarse_block=200 → ~10 batches for 24.9M). The 70/30 slice/accum split in the spec is therefore replaced by a single accumulator cap; note this in the docstring.

---

## File Structure

- `src/tracer/metrics.py` — all kernel/input/ordering/batching/checkpoint logic. Existing file; add helpers, refactor `compute_pmi_bootstrap`. Keep helpers private (`_`-prefixed) next to the existing `_build_presence_matrix` / `_bootstrap_npmi_for_pairs`.
- `tests/test_pmi_bootstrap.py` — extend with parity, matrix-input, ordering, dedup, batching, checkpoint, determinism tests. Reuse `tests/synthetic.make_synthetic_npmi_panel`.
- `tutorials/kidney/compute_pmi_bootstrap.py` — add `--mode {legacy,bootstrap}`; bootstrap mode passes the counts matrix directly (drop `build_long_df`).

Helper inventory (names are contractual across tasks):
- `_presence_from_counts(X, var_names, obs_names=None, *, min_occurrences_per_context) -> (M, genes, contexts)`
- `_classify_ci(arr, tau_low, tau_high, ci_lo_q, ci_hi_q) -> (kind, lo, hi, median)`
- `_bootstrap_from_presence(M, genes, *, kernel, gene_order, gene_batch_peak_gb, checkpoint_path, **bootstrap_kwargs) -> PmiBootstrapResult`
- `_gene_processing_order(key, *, k, legacy_l1=None, stopping_mass=None) -> np.ndarray`
- `_owned_partners(obs_i, obs_j, can_bootstrap, pos) -> (own_indptr, own_partner, own_pairref)`
- `_gene_batches(order, owned_counts, *, gene_batch_peak_gb, coarse_block) -> list[tuple[int,int]]`
- `_bootstrap_gene_rows(M, order, own_indptr, own_partner, own_pairref, legacy_for_W, *, batches, checkpoint_path, G, **bootstrap_kwargs) -> (rows, cols, vals)`
- `_write_checkpoint(path, rows, cols, vals, G, cursor) / _read_checkpoint(path) -> (rows, cols, vals, cursor)`

---

## Task 1: Matrix / h5ad input path

**Files:**
- Modify: `src/tracer/metrics.py` (add `_presence_from_counts`; add `counts=` param + validation to `compute_pmi_bootstrap`)
- Test: `tests/test_pmi_bootstrap.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pmi_bootstrap.py::test_counts_matrix_matches_df_presence -v`
Expected: FAIL — `ImportError: cannot import name '_presence_from_counts'`

- [ ] **Step 3: Implement `_presence_from_counts`**

Add near `_build_presence_matrix` in `src/tracer/metrics.py`:

```python
def _presence_from_counts(X, var_names, obs_names=None, *, min_occurrences_per_context):
    """Build the contexts x genes binary presence CSR directly from a
    cells x genes count matrix (e.g. an h5ad layer), skipping the long-form
    groupby round-trip. "Present" = count >= min_occurrences_per_context.
    """
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr()
    M = (X >= min_occurrences_per_context)          # bool CSR, drops zeros
    M = M.astype(np.int32)
    M.eliminate_zeros()
    genes = np.asarray(var_names, dtype=str)
    n = X.shape[0]
    contexts = (np.asarray(obs_names, dtype=str)
                if obs_names is not None
                else np.arange(n).astype(str))
    if M.shape[1] != genes.size:
        raise ValueError(f"var_names ({genes.size}) != X columns ({M.shape[1]})")
    if M.shape[0] != contexts.size:
        raise ValueError(f"obs_names ({contexts.size}) != X rows ({M.shape[0]})")
    return M, genes, contexts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pmi_bootstrap.py::test_counts_matrix_matches_df_presence -v`
Expected: PASS

- [ ] **Step 5: Add `counts=` validation test (failing)**

```python
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
```

- [ ] **Step 6: Run to verify it fails**

Run: `python -m pytest tests/test_pmi_bootstrap.py::test_counts_xor_df_required -v`
Expected: FAIL — `TypeError`/`ValueError` (no `counts` kwarg yet)

- [ ] **Step 7: Add `counts=` param + dispatch to `compute_pmi_bootstrap`**

Change the signature: make `df_subset` default `None`, add keyword-only params. At the very top of the body (before the existing pre-filter pipeline), insert the input dispatch:

```python
def compute_pmi_bootstrap(
    df_subset=None,
    *,
    counts=None,                      # (X, var_names[, obs_names]) alternative to df
    bootstrap_kernel: str = "gene_row",
    gene_order: str = "prob_ascending",
    gene_batch_peak_gb: float = 16.0,
    checkpoint_path=None,
    group_key: str = "cell_id",
    feature_col: str = "feature_name",
    min_occurrences_per_context: int = 2,
    count_col: str | None = None,
    # ... (all existing kwargs unchanged below) ...
) -> PmiBootstrapResult:
    if (df_subset is None) == (counts is None):
        raise ValueError("Provide exactly one of `df_subset` or `counts`.")
    if counts is not None:
        X = counts[0]; var_names = counts[1]
        obs_names = counts[2] if len(counts) > 2 else None
        for bad, name in [(nuclear_only, "nuclear_only"),
                          (percentile_filter, "percentile_filter"),
                          (per_gene_percentile_filter, "per_gene_percentile_filter")]:
            if bad:
                import warnings
                warnings.warn(f"{name} ignored for matrix `counts=` input")
        M, genes, contexts = _presence_from_counts(
            X, var_names, obs_names,
            min_occurrences_per_context=min_occurrences_per_context,
        )
        return _bootstrap_from_presence(
            M, genes,
            kernel=bootstrap_kernel, gene_order=gene_order,
            gene_batch_peak_gb=gene_batch_peak_gb, checkpoint_path=checkpoint_path,
            tau=tau, ci_level=ci_level, max_bootstraps=max_bootstraps,
            coarse_block=coarse_block, refine_block=refine_block,
            min_expected_cooccur_for_evidence=min_expected_cooccur_for_evidence,
            min_expected_cooccur_for_bootstrap=min_expected_cooccur_for_bootstrap,
            min_samples_for_ci=min_samples_for_ci, alpha=alpha, metric=metric,
            set_neg_one=set_neg_one, seed=seed, show_progress=show_progress,
            persist_ci=persist_ci,
        )
    # else: existing df path continues below (Task 2 routes it through the core too)
```

> NOTE: `_bootstrap_from_presence` is created in Task 2. For Task 1, temporarily implement it as a thin wrapper that raises `NotImplementedError("kernel")` for `gene_row` and, for `pair_gather`, calls a stub that runs the *existing* Stage-1..4 body extracted in Task 2. To keep Task 1 green, gate the `test_counts_xor_df_required` assertion to `bootstrap_kernel="pair_gather"` (already done in Step 5) and land Task 2 immediately after.

- [ ] **Step 8: Run both Task-1 tests**

Run: `python -m pytest tests/test_pmi_bootstrap.py -k "counts" -v`
Expected: PASS (after Task 2 lands `_bootstrap_from_presence`; if running Task 1 alone, expect the xor-raise half to pass and mark the second half xfail until Task 2)

- [ ] **Step 9: Commit**

```bash
git add src/tracer/metrics.py tests/test_pmi_bootstrap.py
git commit -m "feat(metrics): accept counts matrix input for compute_pmi_bootstrap"
```

---

## Task 2: Extract `_bootstrap_from_presence` core + `_classify_ci`, kernel param (pair_gather parity)

**Files:**
- Modify: `src/tracer/metrics.py` (move Stages 1–4 into `_bootstrap_from_presence`; extract `_classify_ci`; add `kernel="pair_gather"` path = current behavior)
- Test: `tests/test_pmi_bootstrap.py`

- [ ] **Step 1: Write the parity regression test (failing)**

```python
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
```

- [ ] **Step 2: Capture the golden baseline BEFORE refactoring**

Run on the current code: `python -m pytest tests/test_pmi_bootstrap.py -v` (all existing tests pass). Record current `res.W_sparse` values for the synthetic panel by adding a temporary print, to confirm the Step-1 asserts match. Expected: existing tests PASS; Step-1 test PASS on current code (it only asserts stable structural facts).

- [ ] **Step 3: Extract `_classify_ci`**

Add to `src/tracer/metrics.py`:

```python
def _classify_ci(arr, tau_low, tau_high, ci_lo_q, ci_hi_q):
    """CI-based classification of one pair's bootstrap sample list.
    Returns (kind, ci_lo, ci_hi, median). kind: 1 pos_strong, -1 neg_strong,
    3 tight_null, 0 not-yet-settled (caller keeps iterating)."""
    lo, hi = np.quantile(arr, [ci_lo_q, ci_hi_q])
    med = float(np.median(arr))
    if lo > tau_high:
        return 1, lo, hi, med
    if hi < -tau_high:
        return -1, lo, hi, med
    if lo > -tau_low and hi < tau_low:
        return 3, lo, hi, med
    return 0, lo, hi, med
```

- [ ] **Step 4: Move Stages 1–4 into `_bootstrap_from_presence`**

Cut the body of `compute_pmi_bootstrap` from the point after `M, genes, contexts = _build_presence_matrix(...)` (Stage 1 onward) into a new function `_bootstrap_from_presence(M, genes, *, kernel, gene_order, gene_batch_peak_gb, checkpoint_path, tau, ci_level, max_bootstraps, coarse_block, refine_block, min_expected_cooccur_for_evidence, min_expected_cooccur_for_bootstrap, min_samples_for_ci, alpha, metric, set_neg_one, seed, show_progress, persist_ci)`. Inside it:
- Keep Stages 1–3 verbatim (neg_one, legacy estimates, evidence classification, `legacy_only` writes).
- Replace the inline Stage-4 `while` loop with a dispatch:

```python
    if n_can_bootstrap == 0:
        # ... existing early-return block, unchanged ...
        return PmiBootstrapResult(W_sparse=..., genes=genes, diagnostics=..., pair_ci=...)

    boot_idx = np.flatnonzero(can_bootstrap)
    if kernel == "pair_gather":
        rows4, cols4, vals4, settled_kind, n_boot = _bootstrap_pairs_gather(
            M, obs_i[boot_idx], obs_j[boot_idx], legacy_for_W[boot_idx],
            tau=tau, ci_level=ci_level, max_bootstraps=max_bootstraps,
            coarse_block=coarse_block, refine_block=refine_block,
            min_samples_for_ci=min_samples_for_ci, alpha=alpha, metric=metric,
            seed=seed, show_progress=show_progress,
        )
    elif kernel == "gene_row":
        rows4, cols4, vals4 = _bootstrap_gene_rows(...)   # Task 4 wires args
    else:
        raise ValueError(f"unknown bootstrap_kernel {kernel!r}")
    out_rows.extend(rows4); out_cols.extend(cols4); out_vals.extend(vals4)
    # ... existing diagnostics aggregation + W assembly, unchanged ...
```

Move the existing `while`-loop body verbatim into `_bootstrap_pairs_gather(M_or_presence, pairs_i, pairs_j, legacy_for_W, *, ...)` returning `(rows, cols, vals, settled_kind, n_bootstraps)`; it builds `M_sample` per iteration exactly as today (it already resamples internally). Have `compute_pmi_bootstrap`'s df path call `_bootstrap_from_presence(M, genes, kernel=bootstrap_kernel, ...)` after building `M`.

- [ ] **Step 5: Run parity + existing tests**

Run: `python -m pytest tests/test_pmi_bootstrap.py -v`
Expected: PASS — all existing tests + `test_pair_gather_kernel_matches_legacy_output` (refactor is behavior-preserving for `pair_gather`).

- [ ] **Step 6: Commit**

```bash
git add src/tracer/metrics.py tests/test_pmi_bootstrap.py
git commit -m "refactor(metrics): extract _bootstrap_from_presence core + _classify_ci, kernel dispatch"
```

---

## Task 3: Gene processing order + pair ownership (dedup)

**Files:**
- Modify: `src/tracer/metrics.py` (add `_gene_processing_order`, `_owned_partners`)
- Test: `tests/test_pmi_bootstrap.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_pmi_bootstrap.py -k "gene_order or owned_partners" -v`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement both helpers**

```python
def _gene_processing_order(key, *, k, legacy_l1=None, stopping_mass=None):
    """Return gene index order for the dedup/streaming pass.
    'prob_ascending'/'prob_descending' sort by detection k (= p*C).
    'l1_pmi' sorts by sum|PMI| row (magnitude proxy; not stopping-time).
    'stopping_mass' sorts by the near-tau/low-support mass (slow-last).
    'index' keeps native order."""
    if key == "index":
        return np.arange(k.size)
    if key == "prob_ascending":
        return np.argsort(k, kind="stable")
    if key == "prob_descending":
        return np.argsort(k, kind="stable")[::-1].copy()
    if key == "l1_pmi":
        if legacy_l1 is None:
            raise ValueError("l1_pmi order needs legacy_l1")
        return np.argsort(legacy_l1, kind="stable")
    if key == "stopping_mass":
        if stopping_mass is None:
            raise ValueError("stopping_mass order needs stopping_mass")
        return np.argsort(stopping_mass, kind="stable")
    raise ValueError(f"unknown gene_order {key!r}")


def _owned_partners(obs_i, obs_j, can_bootstrap, pos):
    """Assign each can_bootstrap pair to its earlier-in-order endpoint.
    Returns CSR-like (indptr, partner, pairref) over genes 0..G-1, where for
    gene g, partner[indptr[g]:indptr[g+1]] are the genes it owns (later in
    `pos`) and pairref the index back into obs_* for legacy lookup."""
    G = pos.size
    bi = obs_i[can_bootstrap]; bj = obs_j[can_bootstrap]
    ref = np.flatnonzero(can_bootstrap)
    earlier_is_i = pos[bi] < pos[bj]
    owner = np.where(earlier_is_i, bi, bj)
    partner = np.where(earlier_is_i, bj, bi)
    order_by_owner = np.argsort(owner, kind="stable")
    owner_s = owner[order_by_owner]
    partner_s = partner[order_by_owner].astype(np.int32)
    pairref_s = ref[order_by_owner].astype(np.int64)
    indptr = np.zeros(G + 1, dtype=np.int64)
    np.add.at(indptr, owner_s + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, partner_s, pairref_s
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_pmi_bootstrap.py -k "gene_order or owned_partners" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tracer/metrics.py tests/test_pmi_bootstrap.py
git commit -m "feat(metrics): gene processing order + pair-ownership dedup helpers"
```

---

## Task 4: Gene-row bootstrap kernel (default)

**Files:**
- Modify: `src/tracer/metrics.py` (add `_bootstrap_gene_rows`; wire `kernel="gene_row"` in `_bootstrap_from_presence`)
- Test: `tests/test_pmi_bootstrap.py`

- [ ] **Step 1: Write the parity test (failing)**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_pmi_bootstrap.py::test_gene_row_kernel_parity_with_pair_gather -v`
Expected: FAIL — `gene_row` not implemented (NotImplementedError / unknown kernel).

- [ ] **Step 3: Implement `_bootstrap_gene_rows`**

Add to `src/tracer/metrics.py`. Uses the resample-multiplicity identity: with `rc = bincount(resampled cell indices)`, bootstrap co-count of (g,h) = Σ_c rc[c]·M[c,g]·M[c,h], and marginals = `rc @ M`. Gene g's whole row = `(rc * presence_col_g) @ M`.

```python
def _bootstrap_gene_rows(
    M, order, own_indptr, own_partner, own_pairref, legacy_for_W,
    *, batches, checkpoint_path, G,
    tau, ci_level, max_bootstraps, coarse_block, refine_block,
    min_samples_for_ci, alpha, metric, seed, show_progress,
):
    """Gene-row (resample-multiplicity matvec) bootstrap over owned pairs.
    `batches` is a list of (start,end) index ranges into `order`. Returns
    (rows, cols, vals) for settled pairs (upper-triangle gene indices)."""
    tau_low = tau_high = float(tau) if np.isscalar(tau) else None
    if tau_low is None:
        tau_low, tau_high = float(tau[0]), float(tau[1])
    ci_lo_q = (1.0 - ci_level) / 2.0
    ci_hi_q = 1.0 - ci_lo_q
    Mcsc = M.tocsc()
    Mcsr = M.tocsr()
    C = M.shape[0]
    rows, cols, vals = [], [], []
    if checkpoint_path is not None:
        ck = _read_checkpoint(checkpoint_path)
        if ck is not None:
            rows, cols, vals, done_batches = ck
        else:
            done_batches = 0
    else:
        done_batches = 0

    for b_idx, (bs, be) in enumerate(batches):
        if b_idx < done_batches:
            continue
        rng = np.random.default_rng(None if seed is None else seed + b_idx)
        batch_genes = order[bs:be]
        # per-(owned pair) accumulators for this batch
        # flatten owned pairs of the batch into contiguous arrays
        seg = [(g, own_partner[own_indptr[g]:own_indptr[g+1]],
                   own_pairref[own_indptr[g]:own_indptr[g+1]]) for g in batch_genes]
        n_owned = sum(len(p) for _, p, _ in seg)
        if n_owned == 0:
            if checkpoint_path is not None:
                _write_checkpoint(checkpoint_path, rows, cols, vals, G, b_idx + 1)
            continue
        sample_lists = [[] for _ in range(n_owned)]
        nsamp = np.zeros(n_owned, dtype=np.int32)
        unsettled = np.ones(n_owned, dtype=bool)
        kind = np.zeros(n_owned, dtype=np.int8)
        # flat index map: pair t -> (gene g, partner h, legacy value)
        flat_g = np.empty(n_owned, dtype=np.int64)
        flat_h = np.empty(n_owned, dtype=np.int64)
        flat_legacy = np.empty(n_owned, dtype=np.float64)
        t = 0
        seg_slices = []
        for (g, partners, refs) in seg:
            s = slice(t, t + len(partners))
            seg_slices.append((g, partners, s))
            flat_g[s] = g
            flat_h[s] = partners
            flat_legacy[s] = legacy_for_W[refs]
            t += len(partners)

        n_done = 0
        while n_done < max_bootstraps and unsettled.any():
            block = coarse_block if n_done == 0 else refine_block
            block = min(block, max_bootstraps - n_done)
            for _ in range(block):
                draw = rng.integers(0, C, size=C)
                rc = np.bincount(draw, minlength=C).astype(np.float64)
                marg = rc @ M                       # length-G resampled marginals
                marg = np.asarray(marg).ravel()
                N_b = float(C)
                for (g, partners, s) in seg_slices:
                    if not unsettled[s].any():
                        continue
                    col = Mcsc.getcol(g)
                    gc = col.indices                # cells where g present
                    w = np.zeros(C); w[gc] = rc[gc]
                    co_row = np.asarray(w @ M).ravel()   # co(g, :) under resample
                    co = co_row[partners]
                    Pij = (co + alpha) / (N_b + 2.0 * alpha)
                    Pi = (marg[g] + alpha) / (N_b + 2.0 * alpha)
                    Pj = (marg[partners] + alpha) / (N_b + 2.0 * alpha)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        pmi = np.log(Pij / (Pi * Pj))
                        val = pmi if metric == "pmi" else pmi / (-np.log(Pij))
                    loc = np.arange(s.start, s.stop)
                    for li, vv in zip(loc, val):
                        if unsettled[li] and np.isfinite(vv):
                            sample_lists[li].append(float(vv)); nsamp[li] += 1
            n_done += block
            for li in np.flatnonzero(unsettled):
                if nsamp[li] < min_samples_for_ci:
                    continue
                kd, lo, hi, med = _classify_ci(sample_lists[li], tau_low, tau_high,
                                               ci_lo_q, ci_hi_q)
                if kd != 0:
                    unsettled[li] = False; kind[li] = kd
                    sample_lists[li] = []
        # post-budget classification for the rest
        for li in np.flatnonzero(unsettled):
            if nsamp[li] >= min_samples_for_ci:
                kd, lo, hi, med = _classify_ci(sample_lists[li], tau_low, tau_high,
                                               ci_lo_q, ci_hi_q)
                kind[li] = kd
        # emit settled pairs (kind != 0) with their LEGACY point estimate value
        sett = np.flatnonzero(kind != 0)
        for li in sett:
            gi, hi_ = int(flat_g[li]), int(flat_h[li])
            a, c = (gi, hi_) if gi < hi_ else (hi_, gi)
            rows.append(a); cols.append(c); vals.append(float(flat_legacy[li]))
        if show_progress:
            print(f"[gene_row] batch {b_idx+1}/{len(batches)} genes={be-bs} "
                  f"owned={n_owned} settled={sett.size}")
        if checkpoint_path is not None:
            _write_checkpoint(checkpoint_path, rows, cols, vals, G, b_idx + 1)
    return rows, cols, vals
```

> Value stored in `W` is the legacy point estimate (`legacy_for_W`), matching `pair_gather`'s settled-pair value semantics; the CI is used only for the settle decision. (If `persist_ci` support is later wanted here, return per-pair CI too — out of scope for v1.)

- [ ] **Step 4: Wire `gene_row` into `_bootstrap_from_presence`**

In the kernel dispatch (Task 2 Step 4), replace the `gene_row` branch:

```python
    elif kernel == "gene_row":
        k = np.asarray(M.sum(0)).ravel().astype(np.float64)
        legacy_l1 = None; stopping = None
        if gene_order == "l1_pmi":
            legacy_l1 = np.zeros(M.shape[1]); 
            np.add.at(legacy_l1, obs_i[can_bootstrap], np.abs(legacy_for_W[can_bootstrap]))
            np.add.at(legacy_l1, obs_j[can_bootstrap], np.abs(legacy_for_W[can_bootstrap]))
        if gene_order == "stopping_mass":
            tl = float(tau) if np.isscalar(tau) else float(tau[0])
            w = 1.0 / ((np.abs(legacy_for_W[can_bootstrap]) - tl) ** 2 + 1e-6) \
                / np.maximum(obs_k[can_bootstrap].astype(float), 1.0)
            stopping = np.zeros(M.shape[1])
            np.add.at(stopping, obs_i[can_bootstrap], w)
            np.add.at(stopping, obs_j[can_bootstrap], w)
        order = _gene_processing_order(gene_order, k=k, legacy_l1=legacy_l1,
                                       stopping_mass=stopping)
        pos = np.empty(order.size, dtype=np.int64); pos[order] = np.arange(order.size)
        own_indptr, own_partner, own_pairref = _owned_partners(
            obs_i, obs_j, can_bootstrap, pos)
        owned_counts = np.diff(own_indptr)[order]
        batches = _gene_batches(order, owned_counts,
                                gene_batch_peak_gb=gene_batch_peak_gb,
                                coarse_block=coarse_block)
        rows4, cols4, vals4 = _bootstrap_gene_rows(
            M, order, own_indptr, own_partner, own_pairref, legacy_for_W,
            batches=batches, checkpoint_path=checkpoint_path, G=M.shape[1],
            tau=tau, ci_level=ci_level, max_bootstraps=max_bootstraps,
            coarse_block=coarse_block, refine_block=refine_block,
            min_samples_for_ci=min_samples_for_ci, alpha=alpha, metric=metric,
            seed=seed, show_progress=show_progress)
        settled_kind = None; n_boot = None
```

(`_gene_batches` lands in Task 5; for Task 4 use a temporary single-batch stub: `batches = [(0, order.size)]`.)

- [ ] **Step 5: Run parity test**

Run: `python -m pytest tests/test_pmi_bootstrap.py::test_gene_row_kernel_parity_with_pair_gather -v`
Expected: PASS

- [ ] **Step 6: Run full module**

Run: `python -m pytest tests/test_pmi_bootstrap.py -v`
Expected: PASS (all)

- [ ] **Step 7: Commit**

```bash
git add src/tracer/metrics.py tests/test_pmi_bootstrap.py
git commit -m "feat(metrics): gene-row bootstrap kernel via resample-multiplicity matvec"
```

---

## Task 5: Memory-budgeted gene batching

**Files:**
- Modify: `src/tracer/metrics.py` (add `_gene_batches`; replace the single-batch stub)
- Test: `tests/test_pmi_bootstrap.py`

- [ ] **Step 1: Write failing test**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_pmi_bootstrap.py::test_gene_batches_respect_pair_cap_and_cover_all -v`
Expected: FAIL — `_gene_batches` not defined.

- [ ] **Step 3: Implement `_gene_batches`**

```python
def _gene_batches(order, owned_counts, *, gene_batch_peak_gb, coarse_block):
    """Greedy contiguous batching over `order`. Memory in the gene-row kernel
    is accumulator-dominated: peak ~ (owned pairs in batch) * coarse_block * 32B.
    Cap batch owned-pair count at gene_batch_peak_gb of that budget."""
    cap = max(1, int(gene_batch_peak_gb * 1e9 / (coarse_block * 32)))
    batches = []
    n = order.size
    s = 0
    while s < n:
        tot = 0; e = s
        while e < n and (e == s or tot + owned_counts[e] <= cap):
            tot += owned_counts[e]; e += 1
        batches.append((s, e))
        s = e
    return batches
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_pmi_bootstrap.py::test_gene_batches_respect_pair_cap_and_cover_all -v`
Expected: PASS

- [ ] **Step 5: Replace the single-batch stub in Task 4 Step 4**

Already calls `_gene_batches(...)`; remove the temporary `batches = [(0, order.size)]` line if present.

- [ ] **Step 6: Multi-batch coverage test**

```python
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
```

- [ ] **Step 7: Run + commit**

Run: `python -m pytest tests/test_pmi_bootstrap.py -k "gene_row or batches" -v`
Expected: PASS

```bash
git add src/tracer/metrics.py tests/test_pmi_bootstrap.py
git commit -m "feat(metrics): memory-budgeted gene batching for the gene-row kernel"
```

---

## Task 6: Checkpoint / resume

**Files:**
- Modify: `src/tracer/metrics.py` (add `_write_checkpoint`, `_read_checkpoint`)
- Test: `tests/test_pmi_bootstrap.py`

- [ ] **Step 1: Write failing test**

```python
def test_checkpoint_roundtrip(tmp_path):
    import numpy as np
    from tracer.metrics import _write_checkpoint, _read_checkpoint
    p = tmp_path / "ck.npz"
    _write_checkpoint(str(p), [0, 1], [2, 3], [0.5, -0.5], G=4, cursor=2)
    rows, cols, vals, cursor = _read_checkpoint(str(p))
    assert rows == [0, 1] and cols == [2, 3]
    assert vals == [0.5, -0.5] and cursor == 2
    assert _read_checkpoint(str(tmp_path / "missing.npz")) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_pmi_bootstrap.py::test_checkpoint_roundtrip -v`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement checkpoint helpers (atomic write)**

```python
def _write_checkpoint(path, rows, cols, vals, G, cursor):
    import os
    tmp = f"{path}.tmp"
    np.savez(tmp,
             rows=np.asarray(rows, dtype=np.int64),
             cols=np.asarray(cols, dtype=np.int64),
             vals=np.asarray(vals, dtype=np.float64),
             G=np.int64(G), cursor=np.int64(cursor))
    os.replace(tmp + ".npz" if not tmp.endswith(".npz") else tmp, path) \
        if os.path.exists(tmp) else os.replace(tmp + ".npz", path)


def _read_checkpoint(path):
    import os
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return (d["rows"].tolist(), d["cols"].tolist(),
            d["vals"].tolist(), int(d["cursor"]))
```

> NOTE: `np.savez` appends `.npz`; simplify by requiring `path` to end in `.npz` and writing to `path + ".tmp.npz"` then `os.replace`. Implementer: use the simplest variant that makes the test green; the test asserts the round-trip contract.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_pmi_bootstrap.py::test_checkpoint_roundtrip -v`
Expected: PASS

- [ ] **Step 5: Resume-equivalence test**

```python
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
```

- [ ] **Step 6: Run + commit**

Run: `python -m pytest tests/test_pmi_bootstrap.py -k checkpoint -v`
Expected: PASS

```bash
git add src/tracer/metrics.py tests/test_pmi_bootstrap.py
git commit -m "feat(metrics): per-batch checkpoint/resume for gene-row kernel"
```

---

## Task 7: Kidney driver — bootstrap mode (matrix input)

**Files:**
- Modify: `tutorials/kidney/compute_pmi_bootstrap.py`

- [ ] **Step 1: Add `--mode` and matrix-input bootstrap path**

Add an argparse (or env) `--mode {legacy,bootstrap}` (default `legacy`). In `main`, branch:

```python
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["legacy", "bootstrap"], default="legacy")
ap.add_argument("--no-checkpoint", action="store_true")
args = ap.parse_args()
```

For bootstrap mode, read counts directly and pass `counts=` (no `build_long_df`):

```python
import anndata as ad
a = ad.read_h5ad(H5AD)
X = a.layers["counts"] if "counts" in a.layers else a.X
ckpt = None if args.no_checkpoint else os.path.join(OUT_DIR, "bootstrap_pmi.ckpt.npz")
result = metrics.compute_pmi_bootstrap(
    None,
    counts=(X, a.var_names.astype(str).to_numpy(), a.obs_names.astype(str).to_numpy()),
    metric="pmi",
    min_occurrences_per_context=MIN_OCCURRENCES_PER_CONTEXT,
    min_expected_cooccur_for_evidence=MIN_EXPECTED_COOCCUR_FOR_EVIDENCE,
    bootstrap_kernel="gene_row", gene_order="prob_ascending",
    gene_batch_peak_gb=16.0, checkpoint_path=ckpt,
    seed=SEED, show_progress=True,
)
OUT_PREFIX = "bootstrap_pmi"
```

Keep the existing W/long/genes/diagnostics writers (they already use `OUT_PREFIX`).

- [ ] **Step 2: Smoke-run on a tiny subset (manual)**

Run (subset to keep it fast):

```bash
cd tutorials/kidney
python - <<'PY'
# quick smoke: 300 genes x all cells, bootstrap mode, gene_row
import os, sys, types, importlib, numpy as np, scipy.sparse as sp, anndata as ad
SRC=os.path.abspath("../../src")
pkg=types.ModuleType("tracer"); pkg.__path__=[os.path.join(SRC,"tracer")]; sys.modules["tracer"]=pkg
sys.modules.setdefault("geopandas", types.ModuleType("geopandas"))
m=importlib.import_module("tracer.metrics")
a=ad.read_h5ad("kidney_reference_harmonized.h5ad")
X=a.layers["counts"][:, :300]
res=m.compute_pmi_bootstrap(None, counts=(X, a.var_names[:300].astype(str).to_numpy()),
        metric="pmi", min_expected_cooccur_for_evidence=10.0,
        bootstrap_kernel="gene_row", gene_batch_peak_gb=16.0, seed=0, show_progress=True)
print("W nnz:", res.W_sparse.nnz, "genes:", res.genes.size)
PY
```

Expected: completes in seconds, prints a nonzero `W nnz`.

- [ ] **Step 3: Commit**

```bash
git add tutorials/kidney/compute_pmi_bootstrap.py
git commit -m "feat(kidney): bootstrap mode using gene-row kernel + matrix input"
```

- [ ] **Step 4: Full run (background, separate session)**

```bash
cd tutorials/kidney && python -u compute_pmi_bootstrap.py --mode bootstrap > output/run_bootstrap.log 2>&1 &
```

Expected: runs to completion (strongest gene-batches first, checkpointed); writes `output/bootstrap_pmi_{W.npz,long.csv,genes.txt,diagnostics.json}`.

---

## Self-Review

**Spec coverage:** §2 gene-row kernel → Task 4. §3 dedup shrinking window → Task 3 (`_owned_partners`). §4 ordering (prob/l1/stopping_mass/index) → Task 3 + Task 4 Step 4. §5 API (`bootstrap_kernel`, `gene_order`, `gene_batch_peak_gb`, `checkpoint_path`) → Tasks 1,2,4,5,6. §5.1 input handling (df vs matrix) → Tasks 1,2. §6 algorithm → Tasks 2–5. §7 reproducibility (pair_gather parity bitwise; gene_row statistical) → Task 2 (parity), Task 4 (parity). §8 tests → all tasks. §9 driver → Task 7. Checkpoint §5/§10 → Task 6.

**Placeholder scan:** code shown in every code step; the two `> NOTE` callouts (Task 1 Step 7 sequencing, Task 6 Step 3 `np.savez` suffix) are implementation guidance, not deferred work — both have green-test criteria.

**Type consistency:** `_bootstrap_from_presence`, `_bootstrap_gene_rows`, `_owned_partners` (returns `own_indptr, own_partner, own_pairref`), `_gene_batches` (returns list of `(start,end)`), `_classify_ci` (returns `(kind, lo, hi, median)`), checkpoint `(rows, cols, vals, cursor)` — names/signatures match across Tasks 2–6. Settled-pair value stored = `legacy_for_W` (consistent with `pair_gather`).

**Open risk flagged:** the gene-row inner Python `for li in zip(loc, val)` per-gene appends are O(owned pairs · iterations); for the full kidney run this is the wall-clock cost. Acceptable for v1 (checkpointed, strongest-first); a vectorized accumulator is a follow-up optimization noted in spec §10.
