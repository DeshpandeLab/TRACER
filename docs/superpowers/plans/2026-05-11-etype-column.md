# Entity-type column refactor — Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax. The work is split across six steps; each step is independently testable and commit-able.

**Goal:** Add `_etype` Categorical column as the canonical source of entity-kind classification, with flag-gated migration from the legacy label-string parsing.

**Spec:** [`docs/superpowers/specs/2026-05-11-etype-column-design.md`](../specs/2026-05-11-etype-column-design.md)

**Branch:** `feature/etype-column` off `optimization/core-refactor`.

---

## File touch list

**Create:**
- `src/tracer/_etype.py` — central helpers: `ETYPE_CATEGORIES`, the categorical-builder constructor, and the column-based reader functions.
- `tests/test_etype.py` — unit tests for the new helpers.

**Modify (additive only through step 4):**
- `src/tracer/stitching.py` — sibling readers next to `infer_entity_type`.
- `src/tracer/pruning.py` — Phase 1 emitter writes `_etype` after Cython kernel.
- `tests/_pipeline_runner.py` — emit `_etype` after `_reassign_nuclear_post_1c`, `_phase1_rerank_within_parent`, `_spatial_split_phase1_entities`, `_qc_demote_small_phase1_entities`; add `USE_ETYPE_COLUMN` flag and branched call sites.
- `src/tracer/spatial.py` — emit `_etype` after Group / cascade; add sibling readers.

**Modify (step 5 — collapsing branches):**
- All branched call sites collapse to the `_etype` path.

**Modify (step 6 — deletion):**
- Delete `infer_entity_type` and string-parsing helpers; remove from `__all__`.

---

## Step 1 — Sibling functions (additive, no behavior change)

- [ ] **1.1 Create `src/tracer/_etype.py`** with the central definitions:

```python
"""Entity-type categorical column — canonical kind classification.

Replaces label-string parsing (see infer_entity_type in stitching.py).
The Categorical column `_etype` is populated by every stage that emits
or transforms entities; readers consume it directly without parsing
the label.

Categories: cell, partial, component, drop, unknown.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ETYPE_CATEGORIES: list[str] = ["cell", "partial", "component", "drop", "unknown"]
ETYPE_DTYPE: pd.CategoricalDtype = pd.CategoricalDtype(
    categories=ETYPE_CATEGORIES, ordered=False
)


def empty_etype(n: int) -> pd.Categorical:
    """Build an all-`unknown` etype column of length n."""
    return pd.Categorical(["unknown"] * n, dtype=ETYPE_DTYPE)


def etype_from_codes(codes: np.ndarray) -> pd.Categorical:
    """Map Cython per-tx codes from prune_cells_nuclear_seed to etypes.

    Codes from the kernel:
      0 = main      → cell
      1 = partial   → partial
      2 = unassigned → unknown
      3 = fallback-needed → unknown  (caller handles fallback)
    """
    cat_codes = np.full(codes.shape, ETYPE_CATEGORIES.index("unknown"), dtype=np.int8)
    cat_codes[codes == 0] = ETYPE_CATEGORIES.index("cell")
    cat_codes[codes == 1] = ETYPE_CATEGORIES.index("partial")
    return pd.Categorical.from_codes(cat_codes, dtype=ETYPE_DTYPE)


def infer_etype_from_label(labels: pd.Series | np.ndarray) -> pd.Categorical:
    """Parity helper: classify a label series via the same rules as
    `stitching.infer_entity_type`. Used during migration to verify
    stage emitters produce a column consistent with legacy parsing
    on integer cell_ids.

    Categories follow the legacy convention:
      sentinels (`-1`, `DROP`, `UNASSIGNED`, `nan`, `*_rejected`) → unknown
      `UNASSIGNED_<n>` → component
      contains `-` → partial
      else → cell
    """
    s = pd.Series(labels).astype(str)
    out = np.full(len(s), "unknown", dtype=object)
    is_sentinel = s.isin({"-1", "DROP", "UNASSIGNED", "nan"}) | s.str.endswith("_rejected")
    out[is_sentinel] = "unknown"
    is_component = ~is_sentinel & s.str.startswith("UNASSIGNED_")
    out[is_component] = "component"
    is_partial = ~is_sentinel & ~is_component & s.str.contains("-", regex=False)
    out[is_partial] = "partial"
    is_cell = ~is_sentinel & ~is_component & ~is_partial
    out[is_cell] = "cell"
    return pd.Categorical(out, dtype=ETYPE_DTYPE)
```

- [ ] **1.2 Create `tests/test_etype.py`** with unit tests:

```python
import numpy as np
import pandas as pd
import pytest

from tracer._etype import (
    ETYPE_CATEGORIES, ETYPE_DTYPE,
    empty_etype, etype_from_codes, infer_etype_from_label,
)


def test_categories_canonical():
    assert ETYPE_CATEGORIES == ["cell", "partial", "component", "drop", "unknown"]


def test_empty_etype():
    e = empty_etype(5)
    assert isinstance(e, pd.Categorical)
    assert e.dtype == ETYPE_DTYPE
    assert (e == "unknown").all()


def test_etype_from_codes():
    codes = np.array([0, 1, 2, 0, 1], dtype=np.int8)
    e = etype_from_codes(codes)
    assert list(e) == ["cell", "partial", "unknown", "cell", "partial"]


def test_infer_etype_from_label_integer_ids():
    labels = pd.Series(["42", "42-1", "42-1-1", "UNASSIGNED_3", "-1", "DROP"])
    e = infer_etype_from_label(labels)
    assert list(e) == ["cell", "partial", "partial", "component", "unknown", "unknown"]


def test_infer_etype_from_label_ffpe_dash_in_cell_id():
    # PDAC-style alphanumeric cell_id with native -1 suffix.
    # The legacy parsing rule will misclassify the main as a partial —
    # this test DOCUMENTS the legacy bug for posterity. The parity
    # helper matches legacy intentionally; the bug is fixed via stage
    # emitters, not by changing this helper.
    labels = pd.Series(["adohnpem-1", "adohnpem-1-1"])
    e = infer_etype_from_label(labels)
    # Legacy says both are "partial" (the bug).
    assert list(e) == ["partial", "partial"]


def test_dtype_after_concat():
    # Categorical preserves dtype when values are in the category set.
    a = pd.Series(empty_etype(3))
    b = pd.Series(pd.Categorical(["cell", "partial", "unknown"], dtype=ETYPE_DTYPE))
    c = pd.concat([a, b]).reset_index(drop=True)
    assert c.dtype == ETYPE_DTYPE


def test_can_assign_categorical_value():
    df = pd.DataFrame({"x": [1, 2, 3]})
    df["_etype"] = empty_etype(3)
    df.loc[df["x"] == 2, "_etype"] = "cell"
    assert df["_etype"].dtype == ETYPE_DTYPE
    assert list(df["_etype"]) == ["unknown", "cell", "unknown"]
```

- [ ] **1.3 Run tests:**

```bash
cd /Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/etype-column
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate genesis_env
pytest tests/test_etype.py -v
```

Expected: 7 PASSED.

- [ ] **1.4 Commit:**

```bash
git add src/tracer/_etype.py tests/test_etype.py
git commit -m "feat(etype): central categorical helpers (_etype column foundation)"
```

---

## Step 2 — Phase 1 emitter writes `_etype`

The Cython kernel returns per-tx codes 0/1/2 (main/partial/unassigned). Currently `pruning.py:600` consumes the codes to write the label-string column but discards them. We add a parallel `_etype` write.

- [ ] **2.1 Locate the prune call site:** `src/tracer/pruning.py` line ~600. Read around it to understand how `codes` is used.

- [ ] **2.2 Add `_etype` population after labels are written:**

```python
from ._etype import etype_from_codes  # add to imports

# ... after the existing codes-to-label conversion ...
df["_etype"] = etype_from_codes(codes)
```

(Verify the placement: `_etype` must be written into `df` at the same level that `out_col`/`tracer_id` is written.)

- [ ] **2.3 Add a parity test in `tests/test_etype.py`:**

```python
def test_phase1_emitter_writes_etype_consistent_with_labels():
    """On integer cell_ids, the new `_etype` column must match the
    legacy infer_etype_from_label() applied to the label string.

    This is the parity gate for the Phase 1 emitter. If it diverges
    on integer cell_ids, the emitter is buggy."""
    # Construct a minimal synthetic input — reuse the smoke fixture if
    # convenient, or build a tiny 3-cell, 10-gene case from scratch.
    # Run prune_transcripts_fast and assert:
    #   df["_etype"] consistent with infer_etype_from_label(df[entity_col])
    # on integer cell_ids.
    ...
```

(Concrete test body to be filled in by the implementer — use the synthetic fixture from `tests/conftest.py` if available.)

- [ ] **2.4 Run tests:**

```bash
pytest tests/test_etype.py tests/test_pipeline_smoke.py tests/test_pipeline_regression.py -v
```

Expected: all PASS. The `_etype` column is present in pruning output but no consumer reads it yet.

- [ ] **2.5 Commit:**

```bash
git add src/tracer/pruning.py tests/test_etype.py
git commit -m "feat(etype): Phase 1 emitter populates _etype from Cython codes"
```

---

## Step 3 — Other stage emitters

Each emitter is an independent commit. Order doesn't matter beyond Phase 1 being first (it produces the initial `_etype`; downstream stages update it).

For each of these, find the emit logic and add `_etype` writes:

### 3a — `_reassign_nuclear_post_1c`

When a nuclear tx is moved from main to partial, its `_etype` flips from `"cell"` to `"partial"`.

```python
# in the tx-move loop (around line 330 in tests/_pipeline_runner.py)
labels[tx_idx] = best_p
etype_arr[tx_idx] = "partial"  # was "cell"; now belongs to a partial entity
```

Both the legacy and vectorized `_reassign_nuclear_post_1c` need this update.

- [ ] Implementation, parity test, commit.

### 3b — `_phase1_rerank_within_parent`

When a main and partial swap, swap their `_etype` values too. Actually: under the current dash-encoded label scheme, the labels swap means the new `42` was old `42-1`, so the original `_etype` (partial) becomes the new main's `_etype` (cell). Apply the same renaming map to `_etype` that's applied to labels.

- [ ] Implementation, parity test, commit.

### 3c — `_spatial_split_phase1_entities`

When a sub-partial is minted via z-split, inherit the parent's `_etype` (still `"partial"` for a partial parent, `"cell"` for a main parent — sub-partials of mains keep the main's kind unless we add a `"subpartial"` category).

Per the spec, sub-partials of partials stay `"partial"`. Sub-partials of mains... unclear (do any production paths produce these?). For now, inherit `"partial"` for all z-split outputs and revisit if a use case emerges.

- [ ] Implementation, parity test, commit.

### 3d — `_qc_demote_small_phase1_entities`

Demoted entities go to `"unknown"` (matching the `"-1"` sentinel).

- [ ] Implementation, parity test, commit.

### 3e — Group / cascade

`cascade_<n>` (no dash) → `"cell"`; `cascade_<n>-1` → `"partial"`; `UNASSIGNED_<n>` → `"component"`. The cascade and legacy spatial-CC emit logic in `src/tracer/spatial.py` and `src/tracer/density_cascade.py`.

- [ ] Implementation, parity test, commit.

### 3f — Stitch, Demote, Final Rescue, Finalize

Pass-through stages. Where they preserve labels, they preserve `_etype`. Where they demote (Demote), they update `_etype` to `"unknown"`.

- [ ] Implementation, parity test, commit.

After step 3, the post-`run_segmented_pipeline` DataFrame carries `_etype` end-to-end. With flag still off, behavior is unchanged.

---

## Step 4 — Flag + branched call sites

- [ ] **4.1 Add the flag constant** in `tests/_pipeline_runner.py` near the other PHASE1_* knobs:

```python
USE_ETYPE_COLUMN: bool = False    # opt-in: classify entities via _etype
                                    # column instead of label-string parsing
```

- [ ] **4.2 Add sibling readers** in `src/tracer/stitching.py`, next to `infer_entity_type`:

```python
def infer_entity_type_etype(df: pd.DataFrame, type_col: str = "_etype") -> pd.Series:
    """Read entity kind from the `_etype` column instead of parsing
    the label string. Returns the same string categories as
    `infer_entity_type` for parity."""
    return df[type_col].astype(str)
```

Similar siblings for `_classify`, `is_whole_cell_id`, etc.

- [ ] **4.3 Add branched call sites.** For each consumer of `infer_entity_type`:

```python
# Was:
types = entity_ids.map(infer_entity_type)
# Becomes:
if USE_ETYPE_COLUMN and "_etype" in df.columns:
    types = infer_entity_type_etype(df).reindex(entity_ids.index)
else:
    types = entity_ids.map(infer_entity_type)
```

Land one call site per commit so each is reviewable.

- [ ] **4.4 Full suite with flag off:** must be byte-identical to pre-refactor. `pytest tests/ -v`.

- [ ] **4.5 Full suite with flag on:** run as `USE_ETYPE_COLUMN=1 pytest tests/ -v` (after wiring env-var override at the top of `_pipeline_runner.py`), OR temporarily flip the flag default and re-run. Must also pass.

- [ ] **4.6 PDAC bench with flag on.** Re-run `benchmarks/bench_pdac_io_phase1_rerank.py` after monkeypatching `USE_ETYPE_COLUMN=True`. Expected outcomes:
  - `n_cells > 0` (no longer 0).
  - Phase1-Rerank produces real moves on PDAC.
  - Coherence quartiles look like the Xenium-lung pattern.

If PDAC produces sensible results AND lung cancer matches its pre-refactor output, step 4 passes.

- [ ] **4.7 Commit:**

```bash
git commit -m "feat(etype): flag-gated migration; readers branch on _etype column"
```

---

## Step 5 — Hardwire flag to True

- [ ] **5.1 Flip default:**

```python
USE_ETYPE_COLUMN: bool = True
```

- [ ] **5.2 Collapse branched call sites** to just the `_etype` path. Each is a one-line diff per call.

- [ ] **5.3 Run full suite** — should still PASS.

- [ ] **5.4 Re-bench PDAC** — should still produce the sensible results from step 4.6.

- [ ] **5.5 Commit:**

```bash
git commit -m "refactor(etype): hardwire _etype path; collapse branches"
```

---

## Step 6 — Remove legacy

- [ ] **6.1 Delete `infer_entity_type` and dash-parsing regex helpers** from `src/tracer/stitching.py`, `src/tracer/plot.py`, and bench scripts.

- [ ] **6.2 Verify no dangling references** via `grep -rn "infer_entity_type\b\| in s.startswith.UNASSIGNED" src tests benchmarks`.

- [ ] **6.3 Run full suite.**

- [ ] **6.4 Commit:**

```bash
git commit -m "refactor(etype): remove legacy infer_entity_type / dash parsing"
```

---

## Step 7 — Land it

After all steps pass:

- [ ] Merge to `optimization/core-refactor` via `--no-ff` (matching the project branching model).
- [ ] Update `.context/freeze_status.md`: note that Phase1-Rerank, Reassign-1c, and all entity-classification helpers now use `_etype`.
- [ ] Re-run the PDAC second-tissue bench on the merged result; record proper second-tissue numbers for Phase1-Rerank's freeze gate.
