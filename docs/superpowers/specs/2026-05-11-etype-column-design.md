# Entity-type categorical column — Design

**Date:** 2026-05-11
**Status:** Spec — not yet implemented
**Branch:** `feature/etype-column` off `optimization/core-refactor`

---

## 1. Goal

Decouple entity-kind classification from `tracer_id` label-string parsing by introducing a parallel `_etype` pandas `Categorical` column. Stages populate it at emit time; readers consume it via sibling functions; the legacy string-parsing helpers stay as a fallback during migration and are deleted once nothing references them.

## 2. Motivation

The PDAC_io bench on 2026-05-11 surfaced a latent correctness bug: the existing label-string encoding (`42` = main, `42-1` = partial, `42-1-1` = sub-partial) and its classifier `tracer.stitching.infer_entity_type` assume cell_id contains no dash. Xenium FFPE / IO outputs use dash-containing alphanumeric cell_ids (e.g., `adohnpem-1`). On such data:

- `infer_entity_type("adohnpem-1")` returns `"partial"` (wrong — it's a main).
- Phase1-Rerank's regex `^(\d+)(?:-(\d+)(?:-(\d+))?)?$` fails to match → silently no-op.
- Every consumer of `infer_entity_type` in `spatial.py`, `stitching.py`, `plot.py`, and bench scripts produces wrong classifications.

The PDAC bench reported `n_cells=0 / n_partials=452,740 / Δ=0` as a measurement artifact of this bug, not a real "rerank had no effect" finding.

The fix is to stop encoding kind in the label string for classification purposes. The label remains a useful unique identifier with hierarchical encoding for human readability, but its structure is no longer load-bearing for type decisions.

## 3. Design

### Categories

Five values, matching the existing `infer_entity_type` return space:

```
"cell"        — main entity for an input cell_id (or cascade_main, treated symmetrically)
"partial"     — sub-seed emitted by Phase 1c, or a cascade partial
"component"   — UNASSIGNED_<n> (legacy spatial-CC Group fallback) or similar pseudo-cells
"drop"        — explicitly demoted entity (reserved; not currently produced)
"unknown"     — unassigned tx or unrecognized; sentinel values like "-1", "DROP", "UNASSIGNED", "nan", "*_rejected"
```

Stored as `pd.Categorical` with these five categories. Internally a uint8 code (5 categories × 20.7M tx PDAC = ~20 MB; negligible).

### Column name: `_etype`

Underscore prefix marks it as a pipeline-managed sidecar (matches `_is_nuc` and `_etype` already used transiently in `stitching.py`). Lives alongside `tracer_id` throughout the pipeline.

### Sibling-function convention

For every reader that currently parses the label, introduce a new sibling that operates on `_etype`:

| Current (string-parse) | Sibling (`_etype`-aware) |
|---|---|
| `infer_entity_type(s)` | `infer_entity_type_etype(df)` |
| `_classify(label)` in `_pipeline_runner.py` | `_classify_etype(df)` |
| Regex bucketing in `_phase1_rerank_within_parent` | `_phase1_rerank_within_parent_etype` |
| Regex bucketing in `_reassign_nuclear_post_1c` | `_reassign_nuclear_post_1c_etype` |
| `is_whole_cell_id(s)`, `is_partial_or_pseudocell_id(s)` | column-based equivalents |
| Bench `_entity_counts(labels)` | `_entity_counts_etype(df)` |

Old functions are never edited; new functions live independently. Migration is opt-in per call site.

### Migration flag

A module-level constant in `tests/_pipeline_runner.py`:

```python
USE_ETYPE_COLUMN: bool = False    # default off during migration
```

Each call site of a legacy reader gains a branch:

```python
if USE_ETYPE_COLUMN and "_etype" in df.columns:
    kinds = infer_entity_type_etype(df)
else:
    kinds = df[entity_col].map(infer_entity_type)
```

With flag off, production behavior is byte-identical to today.

### Stage emitters

Each stage that creates or transforms entities also writes `_etype`:

- **Phase 1 (`pruning.py:prune_transcripts_fast`)**: maps the Cython kernel's per-tx codes (`0=main`, `1=partial`, `2=unassigned`) to `_etype` values. The kernel already returns the codes; we just stop discarding them.
- **Phase 1c → Reassign-1c**: moving a nuclear tx from main to partial swaps its `_etype` from `"cell"` to `"partial"`.
- **Phase1-Rerank**: when the relabeling swaps a main and a partial, swap their `_etype` values accordingly.
- **Split-Phase1**: a sub-partial emitted from z-split inherits its parent's `_etype` (still `"partial"`).
- **Phase1-QC**: demoted entities have `_etype` set to `"unknown"` (matching the `"-1"` sentinel).
- **Group / cascade**: cascade emits `"partial"` for `cascade_<n>-1` labels; legacy `UNASSIGNED_<n>` is `"component"`.
- **Stitch**: when merging entities, the merged entity takes the surviving label's `_etype`.
- **Demote / Final Rescue / Finalize**: pass-through; demotions update to `"unknown"`.

### Categorical preservation under pandas operations

`df.copy()`, `df.loc[mask, "_etype"] = "value"`, `df.assign(_etype=...)` all preserve the Categorical dtype as long as new values are in the registered category set. Per-stage emit code must ensure assignments use only the five canonical strings. Sibling helpers can validate via `assert df["_etype"].dtype.name == "category"` in dev.

## 4. Phased rollout

Six steps, each independently testable. Steps 1–3 are non-breaking additions.

### Step 1 — Sibling functions

Write the `_etype`-aware versions in the relevant files. No edits to existing functions.

### Step 2 — Stage emitters

Each stage that creates or transforms entities writes `_etype` alongside `tracer_id`. Always-on (the column is just additive). Per-stage emission can land independently; downstream stages tolerate the column being missing.

### Step 3 — Branched call sites with flag

Add `USE_ETYPE_COLUMN: bool = False` constant. Branch every call site of a legacy reader. With flag off, production behavior is unchanged.

### Step 4 — Parity testing

Run the full test suite and the PDAC bench with `USE_ETYPE_COLUMN=True`. Verify:
- Lung cancer outputs match between the two flags (no regression on integer cell_ids).
- PDAC outputs are now sensible (n_cells > 0, correct partial counts).
- Phase1-Rerank actually fires moves on PDAC.

### Step 5 — Hardwire to True

Flip `USE_ETYPE_COLUMN` default to `True`. Collapse the branching at each call site to just the new path. One-line per site.

### Step 6 — Remove legacy

Delete `infer_entity_type` and its callers' regex parsing helpers. Tests reference partitions stay valid (the label format is unchanged).

## 5. Tests

- **Step 1**: each sibling function has unit tests on synthetic frames. Compares vs the legacy version on integer cell_ids.
- **Step 2**: each stage's emit logic has a test asserting `_etype` is populated correctly and matches a re-classification via the legacy `infer_entity_type` on integer cell_ids.
- **Step 3**: full suite passes with flag off (byte-identical).
- **Step 4**: full suite passes with flag on for integer cell_ids; PDAC bench produces sensible numbers.
- **Step 5+6**: trivial — code shrinks, behavior unchanged.

## 6. Out of scope

- Changing the label-string encoding scheme. Labels keep their `{cell_id}-{k}` form for human readability and natural split idioms.
- Refactoring `is_whole_cell_id` / `is_partial_or_pseudocell_id` in `plot.py` (visualization only — can migrate independently in a follow-up).
- Sanitizing input cell_ids. The bug is fixed by the categorical, not by mutating input data.
- NOSEG pipeline analog. NOSEG's cascade emits all-partial output by design (per the SEG/NOSEG-asymmetry memory); no SEG-style main vs partial dichotomy applies.

## 7. Open questions

- **Cascade entity classification.** Should `cascade_<n>` (no dash) labels be `"cell"` and `cascade_<n>-1` be `"partial"`? Or should cascade entities have their own type (`"cascade_cell"`, `"cascade_partial"`)? Per the existing `infer_entity_type`, they're treated as `cell` / `partial` symmetrically with Phase 1 entities — preserve that.
- **`Phase1ReassignConfig` and `Phase1RerankConfig` interaction.** These dataclasses don't reference `_etype` directly. They stay as-is; only the runner-level helpers change.

## 8. Success criteria

1. PDAC bench (re-run with `USE_ETYPE_COLUMN=True`) produces non-zero `n_cells`, sensible coherence numbers, and Phase1-Rerank fires real moves.
2. Lung cancer bench produces output byte-identical to the pre-refactor state (with flag on OR off).
3. Test suite green at every step.
4. `infer_entity_type` is removed at the end without dangling references.
