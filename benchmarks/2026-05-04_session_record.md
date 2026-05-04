# Session record — 2026-05-03 / 2026-05-04

End-of-day record of pipeline architecture + performance work on
`optimization/core-refactor`. Branch is now **30 commits ahead of origin**;
**45/45 pytest passing**.

## Commits landed this session

| SHA       | Time (local)        | Subject                                                                    |
|-----------|---------------------|----------------------------------------------------------------------------|
| `24043d5` | 2026-05-04 00:05    | perf(prune): batch nuclear-seed Prune in Cython (14–23× faster)            |
| `e049535` | 2026-05-03 22:38    | feat(pipeline): swap to variant C as default — Group → Rescue (yield-optimal) |
| `e7a9dc3` | 2026-05-03 22:35    | test(refs): regenerate regression-test snapshots for unassigned-label refactor |
| `aafffba` | 2026-05-03 22:31    | refactor(pipeline): unassigned-label cleanup + iterative IR + cell_id↔UNASSIGNED invariant |

## Architectural changes

1. **Variant C is the new default.** `run_segmented_pipeline` order is now:
   `Prune → Group → Rescue (3-pass) → Stitch → Demote → Final Rescue → Finalize`.
   Previously was variant A: `Prune → Initial Rescue → Group → Stitch → Demote → Final Rescue`.
   Validated on lung 500 µm ROI as the yield-optimal config.

2. **Production parameters tightened:**
   - `PMI_THR = 1e-5` (was `log(1.5)` ≈ 0.405) — validated on cell-37962 EMT case
   - `RESCUE_NEG_THR = -0.05` (was `log(1/3)` ≈ -1.10)
   - Group `min_comp_size = 4` (was 1)
   - `RESCUE_MAX_PASSES = 3` (Initial Rescue 1-pass → 3-pass with early-stop)
   - Hard floor: `min_comp_size ≥ 2` enforced via `ValueError` in `annotate_unassigned_components_fast`

3. **DROP retired as a mid-pipeline sentinel.** Replaced with stage-rejected
   diagnostic labels (`prune_rejected`, `group_rejected`, `demote_rejected`).
   New centralized `UNASSIGNED_LABELS` set + helpers (`is_unassigned_label`,
   `unassigned_mask`, `finalize_unassigned`) in `spatial.py`.

4. **`cell_id ↔ UNASSIGNED` invariant** enforced at pipeline end via
   `finalize_unassigned()` — collapses all unassigned-class labels to
   `"UNASSIGNED"` AND resets `cell_id` to `"-1"` for those rows. Makes
   both-assigned and either-assigned ARI scopes equivalent.

5. **`infer_entity_type` recognizes `*_rejected` suffix** as `"unknown"` so
   Stitch's type whitelist excludes them uniformly without label-specific
   gating. Plot helpers (`is_whole_cell_id`, `is_partial_or_pseudocell_id`)
   updated to exclude bare `"UNASSIGNED"`.

6. **`prune_transcripts_nuclear_seed` Cython batch.** Replaces Python
   per-cell loop (O(n_cells × n_tx) DataFrame masking) with a single
   `_cy_prune.prune_cells_nuclear_seed` call. Phase 1a/1b/1c run in C.
   Output byte-equivalent to the Python reference impl. Adds optional
   `skip_phase_1c` parameter (default `False` — keeping 1c is a runtime
   *win* at scale, since it carves work out of Group/Rescue).

## Headline benchmarks

### Lung 500 µm ROI (51,569 tx, 2,977 cells)

| Variant                | Runtime  | ARI (both-assigned) | Final unassigned |
|------------------------|---------:|---------------------:|-----------------:|
| Origin (38ecc5e, A/mcs=1) | 10.1 s | 0.3254              |              783 |
| **C/mcs=4 default (NEW)** | **5.7 s (with Cython)** | **0.7930** | 5,557 (10.8 %) |
| C/mcs=4 (no 1c)        | 5.6 s    | 0.7880               |            5,587 |

### Lung 1000 µm ROI

| Variant         | Runtime  | ARI    | n_partials |
|-----------------|---------:|-------:|-----------:|
| C/mcs=4 (default, with 1c) | **28.2 s** | **0.7975** | 2,007 |
| C/mcs=4 (no 1c)            | 32.4 s     | 0.7930     |     0 |

→ Phase 1c is a net runtime *win* at this scale (decentralized work
  saves Group+Rescue load).

### Lung full dataset (1,436,900 tx, 58,406 cells)

| Variant                | Runtime  | ARI     | Final unas (%) |
|------------------------|---------:|--------:|---------------:|
| Origin (A, whole-cell prune, mcs=1) | 826 s (13:46) | 0.2842 | 14,057 (1.0 %) |
| C/mcs=4 default        | 1,919 s (32:00) — pre-Cython-Prune | 0.7937 | 107,256 (7.5 %) |
| C/mcs=4 default (post-Cython-Prune, projected) | ~1,840 s | 0.7937 | 107,256 |

Group is **70 % of origin's runtime, larger share of C's** — the dominant
single-threaded bottleneck on full data. Cython Prune speedup saves
~70-150 s but doesn't move the needle at full scale until Group is
parallelized.

## Cython batch — Prune

Per-stage Prune timing (with Phase 1c, ROI scale):

| Scale  | Python loop (old) | Cython batch (new) | Speedup |
|--------|------------------:|-------------------:|--------:|
| 500 µm | 4.66 s            | 0.33 s             | **14×**  |
| 1000 µm | est. ~20 s        | 0.75 s             | ~25×     |

Output is **byte-equivalent** to the Python reference impl on both ROI
scales (ARI, n_partials, unassigned counts all match exactly).

## Open follow-ups

1. **Cython batch for Rescue** — `pre_stage2_rescue` / `reassign_unassigned_grid_pool`
   has the same Python-per-tx-loop pattern as Prune did. Per-tx PMI eval
   is the bottleneck. Estimated 5–15× speedup on Rescue stage.
2. **Cython batch for Group** — per-comp NPMI greedy prune is a Python
   loop calling Cython `prune_single` per comp. Same pattern. ~5–10×
   estimated.
3. **Group is single-threaded.** Adding `prange` / `nogil` to its
   per-comp loop would give a parallelism speedup *on top of* the batch.
   Highest-leverage perf work for full-data scale.
4. **`tiling.py` infrastructure** exists for tile-parallel pipelines but
   isn't engaged by `run_segmented_pipeline`. Tile-level parallelism
   could give near-linear 16-core scaling on tx-grid-buildable stages.

## Files of record

- `tests/_pipeline_runner.py` — runner with C/mcs=4 default + iterative IR
- `src/tracer/spatial.py` — `UNASSIGNED_LABELS`, `finalize_unassigned`, `min_comp_size ≥ 2` guard
- `src/tracer/stitching.py` — `infer_entity_type` recognizes `*_rejected` + bare UNASSIGNED
- `src/tracer/pruning.py` — `prune_transcripts_nuclear_seed` calls Cython batch
- `src/tracer/_cy_prune.pyx` — `prune_cells_nuclear_seed` batch (~150 lines)
- `src/tracer/plot.py` — bare-UNASSIGNED guard in cell-id classifiers
- `tests/references/{segmented,noseg,segmented_section,seg_vs_noseg}.json` — regenerated for new pipeline behaviour
