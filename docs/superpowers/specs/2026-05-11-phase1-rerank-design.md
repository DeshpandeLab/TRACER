# Phase1-Rerank — Design

**Date:** 2026-05-11
**Status:** Spec — not yet implemented
**Branching home (when implementation lands):** new feature worktree off `optimization/core-refactor` per current branching model.

---

## 1. Goal

Add a new pipeline stage `Phase1-Rerank` (opt-in, default off) that, within each input parent cell, promotes the depth-1 entity with the largest nuclear-tx count to the canonical "main" label `{cell_id}`, renaming the remaining depth-1 entities in size order. Defuses Phase 1's greedy 1a→1b→1c privilege when a partial ends up with more nuclear tx than the main, and recovers cells that would otherwise vanish at Phase 1-QC when only a partial sibling is healthy.

Both originally-distinct cases collapse into one rule:

- **Case A** — main and partial both above `PHASE1_QC_MIN_TX`, partial larger. Today the main retains the cell-id title. After Rerank, labels swap; downstream Phase 1-QC leaves both surviving.
- **Case B** — main below `PHASE1_QC_MIN_TX`, a partial above it. Today the entire parent demotes (main → unassigned at Phase 1-QC; partial survives only as a non-canonical partial). After Rerank, the partial promotes to main; Phase 1-QC demotes the deposed-main fragment under its new partial label.

## 2. Rationale

### Why tx count is a legitimate signal in this regime

Phase 1 operates **nuclear-tx only** (`nuclear_only_admit=1`). A tx tagged `cell_id=N` AND `overlaps_nucleus=True` is asserted to be inside cell N's 2D nuclear mask — the high-confidence channel of the input segmentation. Neighbor-cell nucleus tx receive the neighbor's `cell_id`; they do not appear under N. So a large nuclear-tx partial under parent N is one of:

- A genuine second program co-expressed in the same nucleus, or
- A 2D nuclear segmentation error fusing two nuclei into one mask.

Both are real signal, not cytoplasmic leakage.

### Why the main isn't structurally privileged

Phase 1a picks the most-internally-coherent K-gene seed first; 1b admits to it; 1c forms a second-best seed from rejects. The "main" wins by going first in a greedy procedure. Under joint optimization, a larger coherent program would have absorbed more tx via 1b-style admissions. The main-vs-partial labeling is contingent on greedy ordering, not on intrinsic identity.

### Why no extra coherence gate

Phase 1c's sub-seed must already pass `SEED_COHERENCE_FLOOR=0.10`. Anything that survives to be a Rerank candidate has cleared this bar. The only theoretical pathology — "small but very coherent identity (1a) vs. large moderately coherent housekeeping/stress (1c)" — is bounded by the seed-coherence floor: pure housekeeping piles have near-zero pairwise PMI and fail the floor; stress programs that pass are real cellular states worth surfacing.

## 3. Behavior

For each parent `cell_id`:

1. **Identify depth-1 roots:** the main `{cell_id}` (if present) plus every label of the form `{cell_id}-k`. Sub-partials `{cell_id}-k-j` (from Split-Phase1's z-split) follow their depth-1 parent and are not candidates for the main slot on their own.
2. **Compute subtree nuclear-tx counts:** for each depth-1 root, count all tx whose label is that root *or* descends from it (matches the pattern `{cell_id}-k(-j)?`).
3. **Sort depth-1 roots by subtree size desc.** **Strict `>` only** — ties leave the original ranking untouched, so the original `{cell_id}` retains the main slot on equality.
4. **If sort order changed**, apply renaming map `{old_depth1 → new_depth1}`. To prevent collisions when the new main has sub-partials that would otherwise rename to the same slots as deposed depth-1 entities, reserve the first `n_rank0_subs` suffix slots for the new main's sub-partials. So: `new_depth1[0] = "{cell_id}"`; `new_depth1[k] = "{cell_id}-{k + n_rank0_subs}"` for `k ≥ 1`. Sub-partial suffixes are renumbered per old depth-1 starting at 1 (uniform rule applied to every depth-1, not just rank-0; in practice this is a no-op for non-rank-0 entities because Split-Phase1 emits contiguous suffixes — keeping the uniform rule simplifies the code without observable behavior change on production data).

Pure relabeling — no tx demoted, no entities deleted, no coordinates moved. Downstream Phase 1-QC's size demotion runs after, unchanged.

Idempotent — re-running Rerank on its own output is a no-op.

## 4. Placement

```
Prune
  → [Phase1-Reassign-1c (opt-in, default off)]
  → Split-Phase1
  → [Phase1-Rerank (NEW, opt-in, default off)]
  → Phase1-QC (size demotion)
  → Rescue
  → ...
```

- After Split-Phase1 so subtree sizes are stable post-z-split.
- After Reassign-1c (if on) so tx counts reflect corrected admissions.
- Before Phase 1-QC so the new main is what's subject to / protected from size demotion.

## 5. Configuration surface

### New dataclass

`src/tracer/config.py`:

```python
@dataclass(frozen=True)
class Phase1RerankConfig:
    enabled: bool = False     # opt-in, default off
    margin_tx: int = 1        # minimum (n_largest - n_runner_up) required
                              # to swap. margin_tx=1 ⇒ strict >.
```

### TOML mirror

`defaults.toml`:

```toml
[phase1_rerank]
enabled = false
margin_tx = 1
```

### Registration

- Add to `_SECTION_TO_CLS` in `src/tracer/config.py`.
- Add to `PipelineConfig` and `__all__`.
- Runner constant `PHASE1_RERANK_ENABLED: bool = False` in `tests/_pipeline_runner.py` (Phase A — runner reads from module-level constants, not from `load_config()` yet, per the project-wide Phase B follow-up).

### Sunset path

Per the project's config-trim direction: when this stage promotes to 🟢 FROZEN, retire `margin_tx` to a code-level constant if it remains at 1 across all benched platforms. `enabled` likely retires to `True` and disappears from the user-facing config at the same promotion.

## 6. Edge cases

| Case | Behavior |
|---|---|
| Only one depth-1 entity under parent | No-op. |
| Largest is already the original main | No-op; emit no relabel. |
| Ties between top entities by count | Original main retains slot (strict `>`). |
| Largest is `< PHASE1_QC_MIN_TX` | Rerank proceeds anyway; Phase 1-QC demotes it on the next stage. |
| Sub-partial `{cell_id}-k-j` is the largest single entity under the parent | Cannot win the main slot. Its depth-1 ancestor `{cell_id}-k` is what's ranked. |
| Reassign-1c is on and moved tx | Rerank reads the post-Reassign counts. Natural composition. |
| `UNASSIGNED_*` labels (from cascade residual handler) | Untouched. Rerank's depth-1 enumeration filters them out by label regex. |

## 7. Tests

Parametric invariants in `tests/test_config.py`, mirroring the `GroupConfig` pattern:

1. Default-off — runner output byte-identical to current production.
2. Default-on, no swap candidate — output identical to off.
3. Default-on, single swap candidate — labels permute as expected; tx counts conserved per entity; sub-partial suffixes preserved.
4. Tie at the top — original main retains slot (strict-`>` enforcement).
5. Three-way reorder (main + 2 partials, partial-2 largest, partial-1 middle, main smallest) — full permutation correct.
6. Sub-partial present — sub-partial follows parent's new label.
7. Interaction with Reassign-1c on — Rerank reads post-Reassign counts, not pre-Reassign.
8. Interaction with Phase 1-QC:
   - Entity that promotes via Rerank but was below `min_tx` does *not* survive QC.
   - Entity that demoted via Rerank but is ≥ `min_tx` *does* survive QC under a non-main suffix.
9. `UNASSIGNED_*` labels are untouched.
10. Idempotence — running Rerank twice in a row is a no-op the second time.

Lockstep test auto-extends as it does for every typed stage.

## 8. Bench plan

Once landed (default off), repeat the Reassign-1c bench shape on NW + C + SE ROIs (500 µm boxes per `bench_per_stage_param.py`), full 2×2 sweep:

| | Reassign-1c off | Reassign-1c on |
|---|---|---|
| **Rerank off** | production today (already benched) | already benched 2026-05-10 |
| **Rerank on**  | NEW | NEW |

**Per cell:**
- N input cells, N output cells (depth-1 mains), N output partials.
- Coverage (% tx assigned).
- End-of-pipeline per-entity coherence: Q1, median, Q3, IQR, mean, std.
- ARI vs off/off baseline.
- For cells common to both runs: paired Δcoherence distribution.

**Promotion-to-default-on gated on:**
- Demonstrable cell-count recovery from Case B (positive Δ in `n_output_cells`).
- No coherence regression for shared cells.
- ARI stability above some threshold (e.g. ≥ 0.97) vs the off/off baseline so downstream Stitch/Group are not destabilized.

Artifacts to `benchmarks/phase1_rerank_sweep.{json,log}` plus a partitions parquet keyed by `(roi, reassign, rerank)` for offline reuse.

## 9. Out of scope

- Coherence-aware criterion (option (c) / (d) from the brainstorming). Deferred unless the bench shows tx-count alone produces an unacceptable mis-promote rate.
- Recursive rerank at sub-partial depth. Only depth-1 is reranked in V1.
- Modifying `Phase1QcConfig` or any other frozen stage. Rerank is additive and standalone.
- Mid-QC / Demote / Stitch promote-partial logic — those stages have their own demotion paths; if a similar pattern proves useful there, it's a separate spec.

## 10. Open questions for next session

- Worktree / branch naming for the implementation work. Suggested: new worktree off `optimization/core-refactor`, branch `feature/phase1-rerank`.
- Whether to commit this spec to that feature branch or leave it untracked alongside the existing handoff docs in the stoic-feynman worktree.
