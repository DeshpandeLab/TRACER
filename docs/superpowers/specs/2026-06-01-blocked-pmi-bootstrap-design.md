# Whole-transcriptome bootstrap PMI via a gene-row (streamed-matmul) kernel

**Date:** 2026-06-01
**Branch:** `optimization/core-refactor`
**Status:** design, pending implementation
**Files:** `src/tracer/metrics.py`, `tests/test_npmi_bootstrap.py`, `tutorials/kidney/`
**Supersedes:** the earlier pair-blocking-by-|PMI| draft in this same doc (rejected — see §11).

## 1. Problem

`compute_pmi_bootstrap` bootstraps every `can_bootstrap` candidate pair with a
per-pair **column-gather** kernel ([`_bootstrap_npmi_for_pairs`](../../../src/tracer/metrics.py)):

```python
Mi = M_sample[:, pairs_i]; Mj = M_sample[:, pairs_j]; co = Mi.multiply(Mj).sum(0)
```

`pairs_i` carries gene indices *with repetition*, so a gene in `d` pairs has its
presence column **re-materialized `d` times**. At whole-transcriptome scRNA scale
(kidney: 15,203 cells × 16,095 genes → 24.9M candidate pairs) that is
**~290 GB/iteration → SIGSEGV** (observed). scRNA cells are dense contexts
(~288 genes/cell at presence ≥ 2), so co-occurrence explodes.

The legacy-only sparse path (`min_expected_cooccur_for_bootstrap=inf`) already
ships point estimates without CIs (`tutorials/kidney/output/legacy_pmi_*`). This
spec adds **bootstrap CIs at full-transcriptome scale** by replacing the kernel,
not by working around it.

## 2. Core idea — a gene-row / streamed-matmul kernel

Compute one gene's entire co-occurrence row in a single **row-sum**
(`M_sampleᵀ·M_sample`, one row at a time) instead of gathering a column per pair:

```
co(g, partners) = M_sample[rows where g is present].sum(0)[partners]
```

You touch gene `g`'s `k_g` cells **once** and read off all partners; no column is
duplicated. Measured on the kidney candidate set:

| per bootstrap iteration | gene-row kernel | column-gather (current) |
|---|---:|---:|
| compute | **3.6 G ops** | 36.3 G ops (~10×) |
| peak memory | **~1 GB** (bounded by `MᵀM` support) | ~290 GB (∝ candidate count) |

The pathological genes flip: a ubiquitous gene (k = 15,203) is the *worst* for
column-gather (column copied per partner) and the *cheapest* here (its row ≈ the
marginal vector).

## 3. Dedup — upper-triangle via a shrinking window

Process genes in a fixed order; each gene pairs **only with the not-yet-processed
remainder**. So pair `(g,h)` is owned (computed/bootstrapped) exactly once, by
whichever endpoint comes first. The scan window is strictly triangular:

```
gene at position p:  remaining (G − p) partners   → window shrinks by exactly 1 each step
```

This is ~½ the work of the naive square and prevents any pair being bootstrapped
twice. (A single full `MᵀM` matmul does **not** capture this — it computes both
triangles and discards one; the row-by-row streaming form does.) The current code
already restricts co-occurrence to `obs_i < obs_j`; we carry that forward as the
moving lower bound.

Caveat to state honestly in code/docs: "one fewer per row" is guaranteed for the
**scan window** (index range), not for the **realized nonzero count** of each row,
which depends on each gene's connectivity and is not monotonic.

## 4. Gene order — a second-order knob (order-invariance proof)

**Base co-occurrence compute is order-invariant.** Per cell `c` with `m_c` present
genes, one pass costs `Σ_{g∈S_c}(# S_c-genes after g) = m_c(m_c−1)/2` — a sum over
a permutation of ranks `1..m_c`, identical for any order. Total `= Σ_c C(m_c,2)`.
A high-`k` gene processed last only looks cheap because its cost was prepaid by its
earlier-processed partners. **So ordering does not reduce the base scan.**

**Order only matters through early-stopping:** a gene processed later sees a
smaller remainder, so each of *its* bootstrap iterations is cheaper. The objective
is therefore "process the genes that need the most iterations last," i.e. sort by
`(iterations_g × k_g)` ascending.

- Uniform iteration counts ⇒ reduces to **ascending detection probability**
  (high-`k` genes last).
- Real data is **not** uniform: low-`k` genes have sparse co-occurrence → wider
  CIs → tend to need *more* iterations, which pulls the other way.

**Proxies for "iterations" (all from legacy outputs, computed pre-bootstrap):**
Stopping time per pair is **peaked at |PMI| ≈ τ** and worsened by low support
`k_ij` — both strong (|PMI| ≫ τ) and near-chance (|PMI| ≈ 0) pairs settle fast.
So candidate keys differ in shape:

- `prob_ascending` — sort by detection `k_g`. Monotone, simplest; matches the
  `(iters×k)` rule under the uniform-iterations assumption.
- `l1_pmi` — sort by `Σ_h |PMI(g,h)|`. A **magnitude** proxy; note it does *not*
  match stopping time (it is monotone in magnitude, but slowness is non-monotone
  — it upweights the strong/fast tail and leaves the slow near-τ genes mid-
  schedule). Offered for comparison, not recommended for stopping time.
- `stopping_mass` — sort by `Σ_h 1/((|PMI_gh|−τ)² + ε) · 1/k_ij`. A **near-τ /
  low-support** mass that actually concentrates slow pairs late. The principled
  stopping-time key, still cheap.

Decision: `gene_order` is a **parameter**, default `"prob_ascending"`, documented
as a heuristic whose true target is "slowest-settling last." `l1_pmi` and
`stopping_mass` are selectable alternatives to measure against. Not claimed
optimal. The structural wins (§2, §3) are order-invariant regardless, so this is a
second-order knob.

## 5. API

```python
# replaces the per-pair Stage-4 kernel; same public return type
compute_pmi_bootstrap(
    ...,
    bootstrap_kernel: str = "gene_row",      # "gene_row" (new) | "pair_gather" (legacy, small panels)
    gene_order: str = "prob_ascending",      # "prob_ascending"|"prob_descending"|"l1_pmi"|"stopping_mass"|"index"
    gene_batch_peak_gb: float = 16.0,        # memory budget per processed gene-batch
    checkpoint_path: str | os.PathLike | None = None,
)
```

- `bootstrap_kernel="pair_gather"` preserves the exact current code path for small
  panels / regression parity. `"gene_row"` is the new default.
- `gene_batch_peak_gb` (default 16) caps the working set of a gene-batch; batches
  are sized so accumulators + co-occurrence rows stay under budget (§6).
- `checkpoint_path` = None → no disk I/O (clean library default). The kidney driver
  passes a path → checkpoint after each batch (realizes "default-on" at the driver
  layer; `--no-checkpoint` to disable).

## 5.1 Input handling — long-form df vs. matrix / h5ad

The bootstrap consumes a **contexts × genes presence matrix `M`**, not a DataFrame.
The long-form `df` is only one way to specify `M` (transcript/spatial data, where
you must aggregate transcripts into cells). A single-cell h5ad is *already* a
cells×genes matrix, so converting it to long-form just to `groupby` it back is a
wasteful round-trip (the current kidney driver explodes 16.5M nonzeros for nothing).

**One normalization layer, two front-ends → `(M, genes, contexts)`:**

| input | build `M` | notes |
|---|---|---|
| `df_subset` (1 row/transcript) | `groupby(group_key, feature_col).sum(count_col) ≥ min_occ` — existing `_build_presence_matrix` | spatial path, unchanged |
| `counts=(X, var_names[, obs_names])` | `M = (X ≥ min_occurrences_per_context)` (binarize CSR); `genes=var_names`, `contexts=obs_names` | matrix/h5ad path, no groupby |

- **Refactor**: extract `_bootstrap_from_presence(M, genes, contexts, ...)` as the
  core; `compute_pmi_bootstrap` accepts **exactly one** of `df_subset` or `counts`
  and validates that. One bootstrap, two builders — no duplicated logic.
- **Threshold semantics are identical**: "present" = count ≥
  `min_occurrences_per_context` (≥2 UMIs ≡ ≥2 transcripts).
- **`anndata` stays out of `tracer`**: the library takes a scipy sparse matrix +
  `var_names`; the kidney driver reads the h5ad and passes
  `(adata.layers["counts"], adata.var_names, adata.obs_names)`.
- **Transcript-only kwargs do not apply to the matrix path**: `nuclear_only`/
  `nucleus_col`, `percentile_filter`, `per_gene_percentile_filter`, and the
  `exclude_contexts` sentinels are transcript-table properties. With `counts=`
  they are ignored (warn) or rejected; context exclusion is optional via an
  `obs_names` mask. Document explicitly.
- **Driver change**: `tutorials/kidney/compute_pmi_bootstrap.py` passes the counts
  matrix directly (drop `build_long_df`), avoiding the 16.5M-row DataFrame.

## 6. Algorithm

Stages 0–3 unchanged (presence matrix `M`, legacy point estimates, evidence
classification, `neg_one`/`low_evidence`/`indeterminate`). Then:

1. Order genes per `gene_order`. Each gene owns pairs with later (remaining) genes.
2. Walk genes; accumulate into gene-**batches** bounded by `gene_batch_peak_gb`
   (a batch's owned-candidate-pair accumulators ≤ ~30% of budget; co-occurrence
   working set ≤ ~70%). Early genes own many partners → small batches; late genes
   own few → large batches.
3. Per batch, run the active sampler with the **gene-row kernel**: each iteration
   resample cells → for each gene in the batch, row-sum over its resampled cells
   restricted to its *unsettled* remaining partners; update accumulators; settle
   via the existing tau / CI / early-stop rules; free settled.
4. Append settled `(i, j, value)` to the global sparse output; aggregate
   diagnostics; if `checkpoint_path`, flush `W` + a cursor JSON.
5. Assemble one upper-triangle CSR `W_sparse` (G×G float32) + diagnostics.
   `PmiBootstrapResult` schema unchanged.

Per-batch seed = `seed + batch_index` (deterministic).

## 7. Reproducibility / equivalence

- `bootstrap_kernel="pair_gather"` ⇒ bitwise-identical to today (regression test).
- `"gene_row"` computes the **same per-pair estimator/CI**, but with a different
  RNG stream (per-batch resamples), so it is **statistically** equivalent, not
  bitwise. Validated by: (a) settled-pair *set* matches `pair_gather` on a shared
  fixture; (b) per-pair medians agree within bootstrap tolerance; (c) both agree
  with the legacy point estimates for well-supported pairs.

## 8. Testing (TDD, `tests/test_npmi_bootstrap.py`)

1. **Kernel parity**: on the existing synthetic fixture, `gene_row` and
   `pair_gather` settle the same pair set; medians within tolerance.
2. **Dedup**: every candidate pair owned exactly once; scan window shrinks by one
   per gene; no pair bootstrapped twice.
3. **Order-invariance of base scan**: total co-occurrence op count equal across
   `gene_order` values (instrument a counter).
4. **Memory probe**: high-detection synthetic genes; batch peak RSS within
   ~`gene_batch_peak_gb` (loose factor) — guards the budget constants.
5. **Checkpoint/resume**: kill after batch k, resume → final `W` == uninterrupted.
6. **Determinism**: same `(seed, gene_order, gene_batch_peak_gb)` → identical `W`.

## 9. Kidney driver

`tutorials/kidney/compute_pmi_bootstrap.py` gains `--mode {legacy,bootstrap}`.
bootstrap mode: `metric="pmi"`, `min_expected_cooccur_for_evidence=10`,
`bootstrap_kernel="gene_row"`, `gene_order="prob_ascending"`,
`gene_batch_peak_gb=16`, `checkpoint_path=output/bootstrap_pmi.ckpt.npz`,
`show_progress=True` → `output/bootstrap_pmi_{W.npz,long.csv,genes.txt,
diagnostics.json}`. Legacy mode unchanged. Run in background.

## 10. Risks / open questions

- **Kernel rewrite risk**: `gene_row` is a new core, not a wrapper — mitigated by
  the parity test against `pair_gather` and the retained legacy point estimates.
- **Late-tail efficiency**: when only a few stubborn pairs remain unsettled,
  per-pair gather on that tiny set is cheaper than re-scanning rows. A future
  hybrid (gene-row coarse → pair-gather tail) is possible; out of scope for v1.
- **Budget-constant calibration** (`MULTIPLY_OVERHEAD`, 70/30 split): set by the
  memory-probe test; lower the slice fraction if peak overshoots.
- **`gene_order` default**: `prob_ascending` is a heuristic (§4), not proven
  optimal; it is a tunable parameter.

## 11. Rejected alternative — pair-blocking by |PMI|

Chunk the candidate pair list into memory-budgeted blocks ordered by |legacy PMI|,
reusing the existing column-gather kernel. Lower implementation risk, but it
**inherits the ~10× column-duplication waste** and only works *around* the bad
kernel (needs 40–60 dual-capped blocks for 24.9M pairs). The gene-row kernel fixes
the root cause (~1 GB, ~10× less compute, dedup triangle) and aligns with the
sparse-native-kernel direction in the project notes, so it is preferred despite the
larger change.
