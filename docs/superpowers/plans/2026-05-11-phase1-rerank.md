# Phase1-Rerank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new opt-in pipeline stage `Phase1-Rerank` between Split-Phase1 and Phase1-QC that, within each input parent cell, promotes the depth-1 entity with the most nuclear tx to the canonical "main" `{cell_id}` label.

**Architecture:** Pure relabeling pass. New typed config `Phase1RerankConfig` mirrors the `GroupConfig` Phase-A pattern (dataclass + TOML section + lockstep + invariants). New helper `_phase1_rerank_within_parent` in the runner, hooked behind a `PHASE1_RERANK_ENABLED` constant defaulting to `False`. SEG pipeline only (NOSEG has no Phase 1).

**Tech Stack:** Python 3.11, pandas, numpy, pytest, conda `genesis_env`.

**Spec:** [`docs/superpowers/specs/2026-05-11-phase1-rerank-design.md`](../specs/2026-05-11-phase1-rerank-design.md)

**Worktree:** This plan should be executed in a feature worktree branched off `optimization/core-refactor`. Suggested branch name: `feature/phase1-rerank`. The stoic-feynman-587f37 worktree where this plan was written is a separate active worktree and should not host the implementation. Per the project branching rule, branch from `optimization/core-refactor` and merge back into it (not main).

---

## File Structure

**Modify:**
- `src/tracer/config.py` — add `Phase1RerankConfig` dataclass, register in `_SECTION_TO_CLS` and `PipelineConfig`.
- `src/tracer/configs/defaults.toml` — add `[phase1_rerank]` section.
- `tests/_pipeline_runner.py` — add `_phase1_rerank_within_parent` helper (after `_qc_demote_small_phase1_entities` at current line 89), add `PHASE1_RERANK_ENABLED` constant near current line 635, hook stage in `run_segmented_pipeline` between Split-Phase1 and Phase1-QC (between current lines 829 and 833).
- `tests/test_config.py` — add `test_phase1_rerank_invariants` parametric.

**Create:**
- `tests/test_phase1_rerank.py` — unit tests for `_phase1_rerank_within_parent`.

**Out of scope for this plan** (separate bench plan, see Task 7):
- `benchmarks/bench_phase1_rerank.py`

---

## Task 1: Add Phase1RerankConfig dataclass + invariants test

**Files:**
- Modify: `src/tracer/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1.1: Write the failing dataclass-invariant test**

Append to `tests/test_config.py` after `test_group_invariants` (current line 188). At the top of the file, add `Phase1RerankConfig` to the imports list (current import block around line 24 imports `Phase1QcConfig`, `GroupConfig`, etc. — add `Phase1RerankConfig` alongside them).

```python
@pytest.mark.parametrize("kwargs, match", [
    ({"margin_tx": 0}, "margin_tx"),
    ({"margin_tx": -1}, "margin_tx"),
])
def test_phase1_rerank_invariants(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Phase1RerankConfig(**kwargs)


def test_phase1_rerank_defaults():
    cfg = Phase1RerankConfig()
    assert cfg.enabled is False
    assert cfg.margin_tx == 1
```

- [ ] **Step 1.2: Run the test to verify it fails**

Run:
```bash
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate genesis_env
pytest tests/test_config.py::test_phase1_rerank_invariants tests/test_config.py::test_phase1_rerank_defaults -v
```

Expected: ImportError / NameError on `Phase1RerankConfig`.

- [ ] **Step 1.3: Add the dataclass**

In `src/tracer/config.py`, insert after the closing of `Phase1QcConfig` (current line 92) and before `class RescueConfig` (current line 95):

```python
@dataclass(frozen=True)
class Phase1RerankConfig:
    """Re-rank depth-1 entities under each parent cell by nuclear-tx
    count; promote the largest to the main `{cell_id}` slot.

    Opt-in (default off). Defuses Phase 1's greedy 1a→1b→1c privilege
    when a partial ends up with more nuclear tx than the main. See
    `docs/superpowers/specs/2026-05-11-phase1-rerank-design.md`.
    """
    enabled: bool = False
    margin_tx: int = 1   # minimum (n_largest - n_runner_up) required
                         # to swap. margin_tx=1 ⇒ strict >.

    def __post_init__(self) -> None:
        if self.margin_tx < 1:
            raise ValueError(
                f"phase1_rerank.margin_tx must be >= 1; got {self.margin_tx}"
            )
```

- [ ] **Step 1.4: Add to `PipelineConfig` and `_SECTION_TO_CLS`**

In the same file:

1. In the `PipelineConfig` definition (current line 331-344), add after the `phase1_qc` field:

```python
    phase1_rerank: Phase1RerankConfig = field(default_factory=Phase1RerankConfig)
```

The full block becomes (preserving order: phase1 → split_phase1 → phase1_qc → phase1_rerank → rescue → group → final_rescue → bootstrap):

```python
@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline config. ..."""
    phase1: Phase1Config = field(default_factory=Phase1Config)
    split_phase1: SplitPhase1Config = field(default_factory=SplitPhase1Config)
    phase1_qc: Phase1QcConfig = field(default_factory=Phase1QcConfig)
    phase1_rerank: Phase1RerankConfig = field(default_factory=Phase1RerankConfig)
    rescue: RescueConfig = field(default_factory=RescueConfig)
    group: GroupConfig = field(default_factory=GroupConfig)
    final_rescue: RescueConfig = field(
        default_factory=lambda: RescueConfig(small_entity_guard_n=0)
    )
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
```

2. In `_SECTION_TO_CLS` (current line 355-363), add the entry between `"phase1_qc"` and `"rescue"`:

```python
_SECTION_TO_CLS: dict[str, type] = {
    "phase1": Phase1Config,
    "split_phase1": SplitPhase1Config,
    "phase1_qc": Phase1QcConfig,
    "phase1_rerank": Phase1RerankConfig,
    "rescue": RescueConfig,
    "group": GroupConfig,
    "final_rescue": RescueConfig,
    "bootstrap": BootstrapConfig,
}
```

- [ ] **Step 1.5: Run the test to verify it passes**

```bash
pytest tests/test_config.py::test_phase1_rerank_invariants tests/test_config.py::test_phase1_rerank_defaults -v
```

Expected: PASS (3 cases for invariants + 1 default = 4 PASSED).

- [ ] **Step 1.6: Run the full test_config.py suite to confirm no other tests broke**

```bash
pytest tests/test_config.py -v
```

Expected: all existing tests PASS, plus the new ones. The lockstep test `test_defaults_toml_matches_dataclass_defaults` will FAIL — that's expected and fixed in Task 2.

- [ ] **Step 1.7: Commit**

```bash
git add src/tracer/config.py tests/test_config.py
git commit -m "feat(config): scaffold Phase1RerankConfig dataclass + invariants"
```

---

## Task 2: Mirror config in defaults.toml

**Files:**
- Modify: `src/tracer/configs/defaults.toml`

- [ ] **Step 2.1: Confirm lockstep test currently fails**

```bash
pytest tests/test_config.py::test_defaults_toml_matches_dataclass_defaults -v
```

Expected: FAIL with message mentioning `phase1_rerank` is missing from the loaded TOML.

- [ ] **Step 2.2: Add the TOML section**

In `src/tracer/configs/defaults.toml`, after the `[phase1_qc]` section (currently ends at line 53 with `min_tx = 3`) and before `# Rescue —` (currently line 56), insert:

```toml

# ---------------------------------------------------------------------
# Phase1-Rerank — opt-in; within each parent cell, re-rank depth-1
# entities by nuclear-tx count and promote the largest to the main
# `{cell_id}` label. Defuses Phase 1's greedy 1a→1b→1c privilege.
# Default off; flip after a bench shows demonstrable cell-count
# recovery without coherence regression.
# ---------------------------------------------------------------------
[phase1_rerank]
enabled = false
margin_tx = 1
```

- [ ] **Step 2.3: Run the lockstep test to verify it passes**

```bash
pytest tests/test_config.py::test_defaults_toml_matches_dataclass_defaults -v
```

Expected: PASS.

- [ ] **Step 2.4: Run the full test_config.py suite**

```bash
pytest tests/test_config.py -v
```

Expected: all PASS.

- [ ] **Step 2.5: Commit**

```bash
git add src/tracer/configs/defaults.toml
git commit -m "feat(config): mirror Phase1RerankConfig in defaults.toml"
```

---

## Task 3: Implement `_phase1_rerank_within_parent` (TDD)

**Files:**
- Create: `tests/test_phase1_rerank.py`
- Modify: `tests/_pipeline_runner.py`

### Task 3a: Skeleton + no-op test

- [ ] **Step 3a.1: Write the no-op test**

Create `tests/test_phase1_rerank.py` with:

```python
"""Unit tests for `_phase1_rerank_within_parent`.

The function re-ranks depth-1 entities under each parent cell_id by
nuclear-tx count and promotes the largest to the main `{cell_id}` label.
Pure relabeling; no tx demotion, no coordinate changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests._pipeline_runner import _phase1_rerank_within_parent


def _df(rows: list[tuple]) -> pd.DataFrame:
    """Build a minimal test frame: rows of (entity, cell_id, nuclear)."""
    return pd.DataFrame(
        rows, columns=["tracer_id", "cell_id", "overlaps_nucleus"]
    )


def test_no_partials_is_noop():
    """One depth-1 entity under parent → no relabel."""
    df = _df([
        ("42", "42", True),
        ("42", "42", True),
        ("42", "42", True),
    ])
    out, stats = _phase1_rerank_within_parent(
        df, entity_col="tracer_id", cell_id_col="cell_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    assert (out["tracer_id"] == df["tracer_id"]).all()
    assert stats["n_parents_reranked"] == 0
    assert stats["n_tx_relabeled"] == 0
```

- [ ] **Step 3a.2: Run to verify it fails**

```bash
pytest tests/test_phase1_rerank.py::test_no_partials_is_noop -v
```

Expected: FAIL with `ImportError: cannot import name '_phase1_rerank_within_parent'`.

- [ ] **Step 3a.3: Add the skeleton implementation**

In `tests/_pipeline_runner.py`, insert immediately after `_qc_demote_small_phase1_entities` (current line 89, ending function block) and before `_spatial_split_phase1_entities` (current line 92):

```python
def _phase1_rerank_within_parent(df_in: pd.DataFrame, *,
                                   entity_col: str,
                                   cell_id_col: str = "cell_id",
                                   nuclear_col: str = "overlaps_nucleus",
                                   margin_tx: int = 1,
                                   ) -> tuple[pd.DataFrame, dict]:
    """Within each parent cell, re-rank depth-1 entities by nuclear-tx
    count and promote the largest to the main `{cell_id}` label.

    Spec: docs/superpowers/specs/2026-05-11-phase1-rerank-design.md
    """
    import re as _re

    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)

    stats = {"n_parents_reranked": 0, "n_tx_relabeled": 0}
    return df_out, stats
```

- [ ] **Step 3a.4: Run the test to verify it passes**

```bash
pytest tests/test_phase1_rerank.py::test_no_partials_is_noop -v
```

Expected: PASS.

- [ ] **Step 3a.5: Commit**

```bash
git add tests/_pipeline_runner.py tests/test_phase1_rerank.py
git commit -m "feat(rerank): skeleton _phase1_rerank_within_parent + first test"
```

### Task 3b: Single-swap case (real implementation)

- [ ] **Step 3b.1: Write the single-swap test**

Append to `tests/test_phase1_rerank.py`:

```python
def test_single_swap_promotes_larger_partial():
    """Partial `42-1` has 5 nuclear tx, main `42` has 3 → swap."""
    df = _df([
        ("42",   "42", True),
        ("42",   "42", True),
        ("42",   "42", True),
        ("42-1", "42", True),
        ("42-1", "42", True),
        ("42-1", "42", True),
        ("42-1", "42", True),
        ("42-1", "42", True),
    ])
    out, stats = _phase1_rerank_within_parent(
        df, entity_col="tracer_id", cell_id_col="cell_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    # Old `42-1` (5 tx) now wears the main label `42`
    # Old `42` (3 tx) is now `42-1`
    counts = out["tracer_id"].value_counts().to_dict()
    assert counts == {"42": 5, "42-1": 3}
    assert stats["n_parents_reranked"] == 1
    assert stats["n_tx_relabeled"] == 8  # all 8 tx under this parent
```

- [ ] **Step 3b.2: Run to verify it fails**

```bash
pytest tests/test_phase1_rerank.py::test_single_swap_promotes_larger_partial -v
```

Expected: FAIL — counts come back as `{"42": 3, "42-1": 5}` (the input).

- [ ] **Step 3b.3: Implement the rerank algorithm**

Replace the skeleton body in `tests/_pipeline_runner.py` (the `_phase1_rerank_within_parent` function body):

```python
def _phase1_rerank_within_parent(df_in: pd.DataFrame, *,
                                   entity_col: str,
                                   cell_id_col: str = "cell_id",
                                   nuclear_col: str = "overlaps_nucleus",
                                   margin_tx: int = 1,
                                   ) -> tuple[pd.DataFrame, dict]:
    """Within each parent cell, re-rank depth-1 entities by nuclear-tx
    count and promote the largest to the main `{cell_id}` label.

    Spec: docs/superpowers/specs/2026-05-11-phase1-rerank-design.md
    """
    import re as _re

    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    labels = df_out[entity_col].to_numpy(dtype=object).copy()
    is_nuclear = df_out[nuclear_col].to_numpy(dtype=bool)

    # Pattern: {cell_id}, {cell_id}-{k}, {cell_id}-{k}-{j}
    _re_label = _re.compile(r"^(\d+)(?:-(\d+)(?:-(\d+))?)?$")

    # Bucket tx-row-indices by (parent_cell_id, depth1_label).
    # depth1_label is the {cell_id}-{k} or just {cell_id} root.
    parent_to_depth1_rows: dict[str, dict[str, list[int]]] = {}
    for i, lab in enumerate(labels):
        m = _re_label.match(str(lab))
        if not m:
            continue  # UNASSIGNED_*, -1, etc.
        parent = m.group(1)
        d1 = m.group(2)
        depth1 = parent if d1 is None else f"{parent}-{d1}"
        parent_to_depth1_rows.setdefault(parent, {}).setdefault(
            depth1, []
        ).append(i)

    stats = {"n_parents_reranked": 0, "n_tx_relabeled": 0}

    for parent, depth1_map in parent_to_depth1_rows.items():
        if len(depth1_map) < 2:
            continue  # only one depth-1 entity, nothing to rerank

        # Count NUCLEAR tx per depth-1 (subtree counts: all rows under
        # depth1 root, since we bucketed by depth1 ancestor).
        sizes: list[tuple[str, int]] = []
        for d1, rows in depth1_map.items():
            n_nuc = int(sum(1 for r in rows if is_nuclear[r]))
            sizes.append((d1, n_nuc))

        # Current main is the depth-1 root that equals `parent` exactly
        # (or None if there isn't one — every depth-1 here is a partial).
        current_main = parent if parent in depth1_map else None

        # Sort by (nuclear count desc, current-main-priority desc, label asc).
        # Current-main-priority breaks ties in favor of the original main
        # → strict-> semantics fall out without an explicit check.
        def _sort_key(d1_size: tuple[str, int]) -> tuple:
            d1, n = d1_size
            is_curr_main = 1 if d1 == current_main else 0
            return (-n, -is_curr_main, d1)
        sizes.sort(key=_sort_key)

        # Apply margin_tx: only swap if (n_largest - n_runner_up) >= margin_tx.
        n_largest = sizes[0][1]
        n_runner_up = sizes[1][1]
        if (n_largest - n_runner_up) < margin_tx:
            continue
        # Edge: if largest is the current main already, nothing changes.
        if sizes[0][0] == current_main:
            continue

        # Build renaming map: rank-0 → parent, rank-k → f"{parent}-{k}"
        # for k = 1, 2, ...
        new_depth1: dict[str, str] = {}
        for k, (d1, _) in enumerate(sizes):
            new_depth1[d1] = parent if k == 0 else f"{parent}-{k}"

        # Apply renaming: every label under each old depth-1 prefix gets
        # the new depth-1 prefix. Sub-partial suffixes preserved.
        # Rebuild from the old label string.
        all_rows = [r for rows in depth1_map.values() for r in rows]
        for r in all_rows:
            old_lab = str(labels[r])
            m = _re_label.match(old_lab)
            assert m is not None  # bucketed above means it matched
            old_parent = m.group(1)
            assert old_parent == parent
            d1k = m.group(2)
            d2j = m.group(3)
            old_d1 = old_parent if d1k is None else f"{old_parent}-{d1k}"
            new_d1 = new_depth1[old_d1]
            if d2j is None:
                labels[r] = new_d1
            else:
                labels[r] = f"{new_d1}-{d2j}"

        stats["n_parents_reranked"] += 1
        stats["n_tx_relabeled"] += len(all_rows)

    df_out[entity_col] = labels
    return df_out, stats
```

- [ ] **Step 3b.4: Run the test to verify it passes**

```bash
pytest tests/test_phase1_rerank.py -v
```

Expected: both tests PASS.

- [ ] **Step 3b.5: Commit**

```bash
git add tests/_pipeline_runner.py tests/test_phase1_rerank.py
git commit -m "feat(rerank): implement nuclear-tx rerank with strict-> tiebreak"
```

### Task 3c: Tie case

- [ ] **Step 3c.1: Write the tie test**

Append to `tests/test_phase1_rerank.py`:

```python
def test_tie_keeps_original_main():
    """Main `42` and partial `42-1` both have 4 nuclear tx → strict >
    means original main wins; no relabel."""
    df = _df([
        ("42",   "42", True),  ("42",   "42", True),
        ("42",   "42", True),  ("42",   "42", True),
        ("42-1", "42", True),  ("42-1", "42", True),
        ("42-1", "42", True),  ("42-1", "42", True),
    ])
    out, stats = _phase1_rerank_within_parent(
        df, entity_col="tracer_id", cell_id_col="cell_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    counts = out["tracer_id"].value_counts().to_dict()
    assert counts == {"42": 4, "42-1": 4}
    assert stats["n_parents_reranked"] == 0
```

- [ ] **Step 3c.2: Run — should already pass from Task 3b implementation**

```bash
pytest tests/test_phase1_rerank.py::test_tie_keeps_original_main -v
```

Expected: PASS (the `n_largest - n_runner_up >= margin_tx` check with both = 4 and `margin_tx=1` → 0 >= 1 is False → no swap).

- [ ] **Step 3c.3: Commit**

```bash
git add tests/test_phase1_rerank.py
git commit -m "test(rerank): tie keeps original main"
```

### Task 3d: Three-way reorder

- [ ] **Step 3d.1: Write the three-way test**

Append to `tests/test_phase1_rerank.py`:

```python
def test_three_way_reorder():
    """Main 42 has 2, partial 42-1 has 4, partial 42-2 has 7 →
    new order: 42-2 (7) → main; 42-1 (4) → -1; 42 (2) → -2."""
    df = _df([
        ("42",   "42", True), ("42",   "42", True),
        ("42-1", "42", True), ("42-1", "42", True),
        ("42-1", "42", True), ("42-1", "42", True),
        ("42-2", "42", True), ("42-2", "42", True),
        ("42-2", "42", True), ("42-2", "42", True),
        ("42-2", "42", True), ("42-2", "42", True),
        ("42-2", "42", True),
    ])
    out, stats = _phase1_rerank_within_parent(
        df, entity_col="tracer_id", cell_id_col="cell_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    counts = out["tracer_id"].value_counts().to_dict()
    # Old 42-2 (7 tx) → main "42"
    # Old 42-1 (4 tx) → "42-1"
    # Old 42   (2 tx) → "42-2"
    assert counts == {"42": 7, "42-1": 4, "42-2": 2}
    assert stats["n_parents_reranked"] == 1
    assert stats["n_tx_relabeled"] == 13
```

- [ ] **Step 3d.2: Run**

```bash
pytest tests/test_phase1_rerank.py::test_three_way_reorder -v
```

Expected: PASS.

- [ ] **Step 3d.3: Commit**

```bash
git add tests/test_phase1_rerank.py
git commit -m "test(rerank): three-way reorder"
```

### Task 3e: Sub-partial follow (with collision-safe renumber)

**Background:** Naming collision arises when promoting a partial that itself has sub-partials. Example: old `42-1` (subtree 6) promotes to new main `42`. Its sub-partial `42-1-1` should attach to the new main as `42-1`. But the deposed old-main `42` (which becomes rank-1 by size order) would also map to `42-1`. **Resolution: reserve suffix slots for the rank-0 entity's sub-partials first; bump subsequent depth-1 entities past the reserved range.**

So if rank-0 has `n_sub` sub-partials, they renumber to `{parent}-1` ... `{parent}-n_sub`; rank-1 depth-1 entity gets `{parent}-(n_sub+1)`; rank-2 gets `{parent}-(n_sub+2)`; etc.

- [ ] **Step 3e.1: Write the sub-partial test (with bump-on-collision expectation)**

Append to `tests/test_phase1_rerank.py`:

```python
def test_subpartial_follows_parent_with_bump_on_collision():
    """Promoted partial brings its sub-partials along; deposed main
    bumps past the reserved sub-suffix slots.

    Before:
      42       × 2   (main)
      42-1     × 4   (partial; direct tx)
      42-1-1   × 2   (sub-partial of 42-1)
      subtree-size: 42=2, 42-1=6 → 42-1 wins.

    After:
      42       × 4   (was 42-1 direct)
      42-1     × 2   (was 42-1-1; sub-partial of new main)
      42-2     × 2   (was 42; bumped past the reserved -1 slot)
    """
    df = _df([
        ("42",     "42", True), ("42",     "42", True),
        ("42-1",   "42", True), ("42-1",   "42", True),
        ("42-1",   "42", True), ("42-1",   "42", True),
        ("42-1-1", "42", True), ("42-1-1", "42", True),
    ])
    out, stats = _phase1_rerank_within_parent(
        df, entity_col="tracer_id", cell_id_col="cell_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    counts = out["tracer_id"].value_counts().to_dict()
    assert counts == {"42": 4, "42-1": 2, "42-2": 2}
    assert stats["n_parents_reranked"] == 1
    assert stats["n_tx_relabeled"] == 8
```

- [ ] **Step 3e.2: Run to verify it fails**

```bash
pytest tests/test_phase1_rerank.py::test_subpartial_follows_parent_with_bump_on_collision -v
```

Expected: FAIL. The Task 3b implementation does not yet handle sub-partials — the assertion at `assert m is not None` may pass but the relabeling will produce wrong labels (sub-partial suffix preserved verbatim leads to collision or wrong placement).

- [ ] **Step 3e.3: Update `_phase1_rerank_within_parent` to handle sub-partials**

Replace the entire body of `_phase1_rerank_within_parent` (in `tests/_pipeline_runner.py`) with this collision-safe version. This is a complete replacement of the Task 3b body:

```python
def _phase1_rerank_within_parent(df_in: pd.DataFrame, *,
                                   entity_col: str,
                                   cell_id_col: str = "cell_id",
                                   nuclear_col: str = "overlaps_nucleus",
                                   margin_tx: int = 1,
                                   ) -> tuple[pd.DataFrame, dict]:
    """Within each parent cell, re-rank depth-1 entities by nuclear-tx
    count and promote the largest to the main `{cell_id}` label.

    Sub-partials follow their depth-1 ancestor's renaming. Naming
    collisions (deposed main vs renumbered sub-partials of new main)
    are resolved by reserving sub-suffix slots for rank-0's sub-partials
    first and bumping deposed depth-1 entities past the reserved range.

    Spec: docs/superpowers/specs/2026-05-11-phase1-rerank-design.md
    """
    import re as _re

    df_out = df_in.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)
    labels = df_out[entity_col].to_numpy(dtype=object).copy()
    is_nuclear = df_out[nuclear_col].to_numpy(dtype=bool)

    _re_label = _re.compile(r"^(\d+)(?:-(\d+)(?:-(\d+))?)?$")

    # Bucket tx-row-indices by (parent_cell_id, depth1_label).
    parent_to_depth1_rows: dict[str, dict[str, list[int]]] = {}
    for i, lab in enumerate(labels):
        m = _re_label.match(str(lab))
        if not m:
            continue
        parent = m.group(1)
        d1 = m.group(2)
        depth1 = parent if d1 is None else f"{parent}-{d1}"
        parent_to_depth1_rows.setdefault(parent, {}).setdefault(
            depth1, []
        ).append(i)

    stats = {"n_parents_reranked": 0, "n_tx_relabeled": 0}

    for parent, depth1_map in parent_to_depth1_rows.items():
        if len(depth1_map) < 2:
            continue

        # Sort depth-1 roots by nuclear-tx subtree size (desc), with
        # original-main tiebreak (current main beats partials on ties).
        sizes: list[tuple[str, int]] = []
        current_main = parent if parent in depth1_map else None
        for d1, rows in depth1_map.items():
            n_nuc = int(sum(1 for r in rows if is_nuclear[r]))
            sizes.append((d1, n_nuc))

        def _sort_key(d1_size: tuple[str, int]) -> tuple:
            d1, n = d1_size
            is_curr_main = 1 if d1 == current_main else 0
            return (-n, -is_curr_main, d1)
        sizes.sort(key=_sort_key)

        # margin_tx gate
        n_largest = sizes[0][1]
        n_runner_up = sizes[1][1]
        if (n_largest - n_runner_up) < margin_tx:
            continue
        if sizes[0][0] == current_main:
            continue

        # Count rank-0's sub-partials (so we know how many suffix slots
        # to reserve under the new main name).
        rank0_old_d1 = sizes[0][0]
        rank0_subs: set[str] = set()
        for r in depth1_map[rank0_old_d1]:
            m = _re_label.match(str(labels[r]))
            assert m is not None
            if m.group(3) is not None:
                rank0_subs.add(m.group(3))
        n_rank0_subs = len(rank0_subs)

        # Build the depth-1 rename map.
        # rank-0 → parent
        # rank-k (k>=1) → parent-(k + n_rank0_subs)
        new_depth1: dict[str, str] = {}
        for k, (d1, _) in enumerate(sizes):
            if k == 0:
                new_depth1[d1] = parent
            else:
                new_depth1[d1] = f"{parent}-{k + n_rank0_subs}"

        # Build the sub-suffix renumber map per old depth-1.
        # Sub-partials of rank-0 use suffixes 1..n_rank0_subs.
        # Sub-partials of other depth-1 entities use 1..n_subs starting
        # fresh under their new depth-1 name (no collision because their
        # new depth-1 name is distinct from the new main name).
        sub_rename: dict[tuple[str, str], str] = {}
        for d1, _ in sizes:
            old_d2js: list[str] = []
            for r in depth1_map[d1]:
                m = _re_label.match(str(labels[r]))
                assert m is not None
                if m.group(3) is not None and m.group(3) not in old_d2js:
                    old_d2js.append(m.group(3))
            old_d2js.sort(key=int)
            for new_idx, old_d2j in enumerate(old_d2js, start=1):
                sub_rename[(d1, old_d2j)] = str(new_idx)

        # Apply renames to every row under this parent.
        all_rows = [r for rows in depth1_map.values() for r in rows]
        for r in all_rows:
            m = _re_label.match(str(labels[r]))
            assert m is not None
            d1k = m.group(2)
            d2j = m.group(3)
            old_d1 = parent if d1k is None else f"{parent}-{d1k}"
            new_d1 = new_depth1[old_d1]
            if d2j is None:
                labels[r] = new_d1
            else:
                labels[r] = f"{new_d1}-{sub_rename[(old_d1, d2j)]}"

        stats["n_parents_reranked"] += 1
        stats["n_tx_relabeled"] += len(all_rows)

    df_out[entity_col] = labels
    return df_out, stats
```

- [ ] **Step 3e.4: Run all rerank tests**

```bash
pytest tests/test_phase1_rerank.py -v
```

Expected: all five tests PASS. Critically, the three-way reorder test from Task 3d should still PASS — when there are no sub-partials, `n_rank0_subs=0` and the new code reduces to the same behavior as Task 3b.

- [ ] **Step 3e.5: Commit**

```bash
git add tests/_pipeline_runner.py tests/test_phase1_rerank.py
git commit -m "feat(rerank): sub-partial follow with bump-on-collision renumber"
```

### Task 3f: UNASSIGNED untouched + cyto tx don't count

- [ ] **Step 3f.1: Write the UNASSIGNED test**

Append:

```python
def test_unassigned_labels_untouched():
    """Labels matching UNASSIGNED_*, -1, etc. are not candidates for
    rerank under any parent."""
    df = _df([
        ("42",            "42", True), ("42",            "42", True),
        ("42-1",          "42", True), ("42-1",          "42", True),
        ("42-1",          "42", True), ("42-1",          "42", True),
        ("UNASSIGNED_7", "42", True),
        ("-1",            "42", False),
        ("UNASSIGNED",   "42", True),
    ])
    out, stats = _phase1_rerank_within_parent(
        df, entity_col="tracer_id", cell_id_col="cell_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    counts = out["tracer_id"].value_counts().to_dict()
    # Rerank happens between 42 (2) and 42-1 (4): swap.
    # UNASSIGNED_7, -1, UNASSIGNED labels unchanged.
    assert counts["42"] == 4
    assert counts["42-1"] == 2
    assert counts["UNASSIGNED_7"] == 1
    assert counts["-1"] == 1
    assert counts["UNASSIGNED"] == 1
```

- [ ] **Step 3f.2: Run — should pass (label regex already filters non-matching labels)**

```bash
pytest tests/test_phase1_rerank.py::test_unassigned_labels_untouched -v
```

Expected: PASS.

- [ ] **Step 3f.3: Write the nuclear-vs-cyto test**

```python
def test_cyto_tx_dont_count_toward_size():
    """Only nuclear tx count toward the size used for ranking.
    Partial has more total tx but fewer nuclear → main wins."""
    df = _df([
        # Main 42: 3 nuclear, 0 cyto
        ("42",   "42", True),  ("42",   "42", True),  ("42",   "42", True),
        # Partial 42-1: 2 nuclear, 5 cyto = 7 total but only 2 nuclear
        ("42-1", "42", True),  ("42-1", "42", True),
        ("42-1", "42", False), ("42-1", "42", False),
        ("42-1", "42", False), ("42-1", "42", False), ("42-1", "42", False),
    ])
    out, stats = _phase1_rerank_within_parent(
        df, entity_col="tracer_id", cell_id_col="cell_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    counts = out["tracer_id"].value_counts().to_dict()
    # No swap: main has more nuclear tx
    assert counts == {"42": 3, "42-1": 7}
    assert stats["n_parents_reranked"] == 0
```

- [ ] **Step 3f.4: Run**

```bash
pytest tests/test_phase1_rerank.py::test_cyto_tx_dont_count_toward_size -v
```

Expected: PASS (nuclear-only counting was implemented in Task 3b).

- [ ] **Step 3f.5: Commit**

```bash
git add tests/test_phase1_rerank.py
git commit -m "test(rerank): UNASSIGNED untouched + cyto tx excluded from sizing"
```

### Task 3g: Idempotence + margin_tx

- [ ] **Step 3g.1: Write the idempotence test**

Append:

```python
def test_idempotent():
    """Running rerank twice in a row is a no-op the second time."""
    df = _df([
        ("42",   "42", True),
        ("42",   "42", True),
        ("42-1", "42", True),  ("42-1", "42", True),
        ("42-1", "42", True),  ("42-1", "42", True),
    ])
    out1, stats1 = _phase1_rerank_within_parent(
        df, entity_col="tracer_id", cell_id_col="cell_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    assert stats1["n_parents_reranked"] == 1
    out2, stats2 = _phase1_rerank_within_parent(
        out1, entity_col="tracer_id", cell_id_col="cell_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    assert (out2["tracer_id"] == out1["tracer_id"]).all()
    assert stats2["n_parents_reranked"] == 0


def test_margin_tx_blocks_close_swaps():
    """margin_tx=3 blocks a +1 difference."""
    df = _df([
        ("42",   "42", True), ("42",   "42", True), ("42",   "42", True),
        ("42-1", "42", True), ("42-1", "42", True),
        ("42-1", "42", True), ("42-1", "42", True),
    ])
    out, stats = _phase1_rerank_within_parent(
        df, entity_col="tracer_id", cell_id_col="cell_id",
        nuclear_col="overlaps_nucleus", margin_tx=3,
    )
    counts = out["tracer_id"].value_counts().to_dict()
    # main=3, partial=4 → diff=1 < margin_tx=3 → no swap
    assert counts == {"42": 3, "42-1": 4}
    assert stats["n_parents_reranked"] == 0
```

- [ ] **Step 3g.2: Run**

```bash
pytest tests/test_phase1_rerank.py -v
```

Expected: all PASS.

- [ ] **Step 3g.3: Commit**

```bash
git add tests/test_phase1_rerank.py
git commit -m "test(rerank): idempotence + margin_tx threshold"
```

---

## Task 4: Hook the stage into `run_segmented_pipeline`

**Files:**
- Modify: `tests/_pipeline_runner.py`
- Modify: `tests/test_pipeline_smoke.py` (or add minimal smoke test if convention is elsewhere — confirm by listing tests in the file before adding)

- [ ] **Step 4.1: Write a failing smoke test**

First, locate the smoke test for the SEG pipeline. Run:

```bash
grep -n -E "def test_.*(seg|segmented)" tests/test_pipeline_smoke.py
```

Add to that file (or to `tests/test_phase1_rerank.py` if smoke conventions differ):

```python
def test_rerank_off_matches_baseline_seg_smoke():
    """With PHASE1_RERANK_ENABLED=False (default), SEG pipeline output
    is byte-identical to current production."""
    # Find the existing SEG smoke harness in this file and reuse it;
    # the assertion is: the recorded Phase1-Rerank stage MUST NOT appear
    # in the progression list (because the toggle is off).
    from tests._pipeline_runner import run_segmented_pipeline
    df, panel = _build_seg_smoke_inputs()   # use existing fixture / helper
    _df_out, progression = run_segmented_pipeline(df, panel)
    stage_names = [p["stage"] for p in progression]
    assert "Phase1-Rerank" not in stage_names


def test_rerank_on_records_stage_seg_smoke(monkeypatch):
    """Flipping PHASE1_RERANK_ENABLED=True records the new stage."""
    import tests._pipeline_runner as runner
    monkeypatch.setattr(runner, "PHASE1_RERANK_ENABLED", True)
    df, panel = _build_seg_smoke_inputs()
    _df_out, progression = runner.run_segmented_pipeline(df, panel)
    stage_names = [p["stage"] for p in progression]
    # Must appear after Split-Phase1 and before Phase1-QC
    idx_split = stage_names.index("Split-Phase1")
    idx_qc = stage_names.index("Phase1-QC")
    idx_rerank = stage_names.index("Phase1-Rerank")
    assert idx_split < idx_rerank < idx_qc
```

If `_build_seg_smoke_inputs` doesn't exist, use whatever fixture the file's existing SEG tests use (read the file, replicate the call).

- [ ] **Step 4.2: Run to verify the second test fails (stage not yet hooked)**

```bash
pytest tests/test_pipeline_smoke.py::test_rerank_on_records_stage_seg_smoke tests/test_pipeline_smoke.py::test_rerank_off_matches_baseline_seg_smoke -v
```

Expected: the off-test PASSES (no hook yet, so stage isn't recorded — that's correct for off behavior); the on-test FAILS with `ValueError: 'Phase1-Rerank' is not in list`.

- [ ] **Step 4.3: Add the constant**

In `tests/_pipeline_runner.py`, find the existing constant `PHASE1_REASSIGN_AFTER_1C: bool = False` (current line 635). Immediately after it, add:

```python
PHASE1_RERANK_ENABLED: bool = False    # opt-in: rerank depth-1 entities
                                         # under each parent by nuclear-tx
                                         # count. See
                                         # docs/superpowers/specs/2026-05-11-phase1-rerank-design.md
PHASE1_RERANK_MARGIN_TX: int = 1
```

- [ ] **Step 4.4: Wire the stage call**

In `run_segmented_pipeline`, locate the block between `_record_stage(progression, "Split-Phase1", ...)` (current line 829) and `df_pruned, _qc_stats = _qc_demote_small_phase1_entities(...)` (current line 833). Insert immediately after line 829, BEFORE line 833:

```python
    # Phase1-Rerank (opt-in): within each parent cell_id, promote the
    # depth-1 entity with the most nuclear tx to the main `{cell_id}`
    # label. Defuses Phase 1's greedy 1a→1b→1c privilege.
    if PHASE1_RERANK_ENABLED:
        df_pruned, _rerank_stats = _phase1_rerank_within_parent(
            df_pruned, entity_col="tracer_id",
            cell_id_col="cell_id", nuclear_col="overlaps_nucleus",
            margin_tx=PHASE1_RERANK_MARGIN_TX,
        )
        _record_stage(progression, "Phase1-Rerank", df_pruned, "tracer_id")
```

- [ ] **Step 4.5: Run the smoke tests**

```bash
pytest tests/test_pipeline_smoke.py::test_rerank_off_matches_baseline_seg_smoke tests/test_pipeline_smoke.py::test_rerank_on_records_stage_seg_smoke -v
```

Expected: both PASS.

- [ ] **Step 4.6: Run the full smoke suite**

```bash
pytest tests/test_pipeline_smoke.py -v
```

Expected: all PASS. Default-off must not affect the production pipeline.

- [ ] **Step 4.7: Run the full regression suite**

```bash
pytest tests/ -v
```

Expected: all PASS (84+ existing + new rerank tests). Default-off → no behavioral change.

- [ ] **Step 4.8: Commit**

```bash
git add tests/_pipeline_runner.py tests/test_pipeline_smoke.py
git commit -m "feat(pipeline): hook Phase1-Rerank stage (opt-in, default off)"
```

---

## Task 5: Integration test — Rerank composes with Reassign-1c

**Files:**
- Modify: `tests/test_pipeline_smoke.py`

- [ ] **Step 5.1: Write the composition test**

Append:

```python
def test_rerank_composes_with_reassign_1c(monkeypatch):
    """Both opt-in stages on simultaneously: Rerank reads post-Reassign
    tx counts, not pre-Reassign."""
    import tests._pipeline_runner as runner
    monkeypatch.setattr(runner, "PHASE1_REASSIGN_AFTER_1C", True)
    monkeypatch.setattr(runner, "PHASE1_RERANK_ENABLED", True)
    df, panel = _build_seg_smoke_inputs()
    _df_out, progression = runner.run_segmented_pipeline(df, panel)
    stage_names = [p["stage"] for p in progression]
    idx_prune = stage_names.index("Prune")
    idx_reassign = stage_names.index("Phase1-Reassign-1c")
    idx_split = stage_names.index("Split-Phase1")
    idx_rerank = stage_names.index("Phase1-Rerank")
    idx_qc = stage_names.index("Phase1-QC")
    assert idx_prune < idx_reassign < idx_split < idx_rerank < idx_qc
```

- [ ] **Step 5.2: Run**

```bash
pytest tests/test_pipeline_smoke.py::test_rerank_composes_with_reassign_1c -v
```

Expected: PASS.

- [ ] **Step 5.3: Commit**

```bash
git add tests/test_pipeline_smoke.py
git commit -m "test(rerank): integration with Reassign-1c"
```

---

## Task 6: Phase 1-QC interaction test

**Files:**
- Modify: `tests/test_phase1_rerank.py`

- [ ] **Step 6.1: Write the QC-interaction test**

Append to `tests/test_phase1_rerank.py`:

```python
def test_qc_after_rerank_demotes_old_main_below_min_tx():
    """If old main had 2 tx (below min_tx=3) and partial had 5, after
    Rerank the partial wears `42` (survives QC); the old main wears
    `42-1` with 2 tx and gets demoted by the next Phase 1-QC pass."""
    from tests._pipeline_runner import (
        _phase1_rerank_within_parent,
        _qc_demote_small_phase1_entities,
        PHASE1_QC_MIN_TX,
    )
    df = _df([
        ("42",   "42", True), ("42",   "42", True),
        ("42-1", "42", True), ("42-1", "42", True),
        ("42-1", "42", True), ("42-1", "42", True), ("42-1", "42", True),
    ])
    rerk, _ = _phase1_rerank_within_parent(
        df, entity_col="tracer_id", cell_id_col="cell_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    qcd, qc_stats = _qc_demote_small_phase1_entities(
        rerk, entity_col="tracer_id", min_size=PHASE1_QC_MIN_TX,
        unassigned_id="-1",
    )
    counts = qcd["tracer_id"].value_counts().to_dict()
    # Rerank: old 42-1 (5) → "42"; old 42 (2) → "42-1"
    # QC: "42-1" has 2 tx < 3 → demote to "-1"
    assert counts["42"] == 5
    assert counts.get("-1", 0) == 2
    assert "42-1" not in counts
```

- [ ] **Step 6.2: Run**

```bash
pytest tests/test_phase1_rerank.py::test_qc_after_rerank_demotes_old_main_below_min_tx -v
```

Expected: PASS.

- [ ] **Step 6.3: Commit**

```bash
git add tests/test_phase1_rerank.py
git commit -m "test(rerank): QC composes correctly after rerank"
```

---

## Task 7: Hand off to bench

**Files:** none modified in this task. This is a hand-off note.

- [ ] **Step 7.1: Write the bench-plan stub**

Create `benchmarks/bench_phase1_rerank.md` (a placeholder note; the actual bench script is a separate plan):

```markdown
# Phase1-Rerank bench plan

Run after this implementation merges. See spec §8 for the design:
[docs/superpowers/specs/2026-05-11-phase1-rerank-design.md](../docs/superpowers/specs/2026-05-11-phase1-rerank-design.md).

The bench should reuse `benchmarks/bench_reassign_1c_coherence.py` as a
template — same ROI/loader/coherence definitions. Sweep both toggles
(`PHASE1_RERANK_ENABLED`, `PHASE1_REASSIGN_AFTER_1C`) × 3 ROIs (NW, C,
SE). Output: `benchmarks/phase1_rerank_sweep.{json,log,partitions.parquet}`.

Promotion-to-default-on gate: positive cell-count Δ from Case B + no
coherence regression on shared cells + ARI vs off/off baseline ≥ ~0.97.
```

- [ ] **Step 7.2: Commit**

```bash
git add benchmarks/bench_phase1_rerank.md
git commit -m "docs(rerank): bench-plan stub for follow-up"
```

---

## Task 8: Final regression + summary

- [ ] **Step 8.1: Run the full test suite one more time**

```bash
pytest tests/ -v
```

Expected: all PASS, including the original 84 + the 9-ish new rerank tests + 3 smoke tests.

- [ ] **Step 8.2: Confirm default-off byte-identicality with origin**

```bash
git diff origin/optimization/core-refactor -- tests/_pipeline_runner.py | head -40
```

Spot-check that with `PHASE1_RERANK_ENABLED=False` (the default), the only behavioral effect is the absence of the `Phase1-Rerank` stage block — no other code path was modified.

- [ ] **Step 8.3: Update `.context/freeze_status.md` and `TASKS.md`**

In the **GENESIS main root** (not the worktree — these files are part of the project's `.context/` convention and live in the main working tree):

In `/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.context/freeze_status.md`, in the SEG pipeline table, add a new row between row 4 (`Phase1-QC`) and row 5 (`Split` post-Phase1-QC):

```markdown
| 5 | `Phase1-Rerank` | 🔵 OPT-IN | **off** (`PHASE1_RERANK_ENABLED=False`) | `Phase1RerankConfig` | Within each parent cell, promote the depth-1 entity with the most nuclear tx to main. Bench-pending — see spec 2026-05-11-phase1-rerank-design.md. |
```

And renumber subsequent stages from 6 onward.

In `/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/TASKS.md`, mark the Q1 promote-partial item as superseded by this implementation and add the bench follow-up.

- [ ] **Step 8.4: Open PR back into `optimization/core-refactor`**

```bash
git push -u origin feature/phase1-rerank
gh pr create --base optimization/core-refactor --title "feat: Phase1-Rerank opt-in stage" --body "$(cat <<'EOF'
## Summary
- New opt-in stage `Phase1-Rerank` (default off) between Split-Phase1 and Phase1-QC.
- Within each parent cell, promotes the depth-1 entity with the most nuclear tx to the main `{cell_id}` slot.
- Defuses Phase 1's greedy 1a→1b→1c privilege; addresses the Q1 promote-partial backlog item.
- `Phase1RerankConfig` typed dataclass + `[phase1_rerank]` TOML section + 9 unit tests + 3 smoke/integration tests.
- Default-off: production behavior unchanged.

## Spec
[docs/superpowers/specs/2026-05-11-phase1-rerank-design.md](docs/superpowers/specs/2026-05-11-phase1-rerank-design.md)

## Test plan
- [ ] `pytest tests/test_config.py -v` — invariants + lockstep + defaults
- [ ] `pytest tests/test_phase1_rerank.py -v` — 9 algorithm unit tests
- [ ] `pytest tests/test_pipeline_smoke.py -v` — full smoke incl. rerank-off/on and Reassign+Rerank composition
- [ ] `pytest tests/ -v` — full regression

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Notes on conventions followed

- TDD: every code change is preceded by a failing test.
- Frequent commits: 12 commits across 8 tasks.
- No placeholders: every code block is a complete, runnable snippet.
- Default-off discipline: opt-in stages don't change production behavior until a focused bench justifies the flip — same pattern as Reassign-1c (commit `8454454`) and the NOSEG cascade flip (`1b20a06`).
- Branch hygiene: feature branch off `optimization/core-refactor`; PR back into `optimization/core-refactor`, not `main`.
- Sunset path acknowledged: per the config-trim direction, `enabled` and `margin_tx` should retire on freeze.
