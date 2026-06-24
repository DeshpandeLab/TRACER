# Transcript-flow Sankey Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Sankey/alluvial plot of per-transcript assignment flow through the SEG and NOSEG pipelines, with a lightweight `snapshot_phase` hook that pipeline drivers can call after each phase boundary.

**Architecture:**
- `src/tracer/sankey_log.py` — vectorized class classifier, snapshot helper, phase-key tier tables, display-label map. Pure-Python; numpy-only at import.
- `src/tracer/flow_plot.py` — `plot_transcript_flow(df, ...) -> Figure`. Reads `etype_at_<phase>` columns, builds tidy transition DataFrame, dispatches to plotly (default) or matplotlib backend. Plotly imported lazily.
- Snapshot calls live in the per-tutorial driver (no central runner). `tutorials/lung_cancer/run_lung_cancer.py` is the SEG demo driver; the same pattern documented for NOSEG notebook.

**Tech Stack:** numpy, pandas (existing); plotly (new optional dep, lazy import); matplotlib (existing). Tests use pytest + the existing standalone-load fallback in `tests/conftest.py`.

**Spec:** `docs/superpowers/specs/2026-06-09-transcript-flow-sankey-design.md`

**Working branch:** `feature/transcript-flow-sankey` (off `optimization/core-refactor`).

**Working directory for all paths below:** `/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/origin-core-refactor`.

---

## Task 1: Classifier + display-label module foundation

**Files:**
- Create: `src/tracer/sankey_log.py`
- Test: `tests/test_sankey_log.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sankey_log.py`:

```python
"""Tests for tracer.sankey_log — classifier, phase keys, display labels."""
import numpy as np
import pandas as pd
import pytest

try:
    from tracer import sankey_log as sl
except ImportError:
    import sankey_log as sl  # conftest standalone fallback


class TestClassifyEtypeVec:
    def test_main_cells(self):
        ids = np.array(["cell_42", "cell_7"], dtype=object)
        etypes = np.array(["cell", "cell"], dtype=object)
        out = sl._classify_etype_vec(ids, etypes)
        assert out.dtype == np.int8
        assert (out == sl.CLASS_MAIN).all()

    def test_partials(self):
        ids = np.array(["cell_42-1", "cell_7-3"], dtype=object)
        etypes = np.array(["partial", "partial"], dtype=object)
        assert (sl._classify_etype_vec(ids, etypes) == sl.CLASS_PARTIAL).all()

    def test_components(self):
        ids = np.array(["UNASSIGNED_base1", "UNASSIGNED_base2"], dtype=object)
        etypes = np.array(["component", "component"], dtype=object)
        assert (sl._classify_etype_vec(ids, etypes) == sl.CLASS_COMPONENT).all()

    def test_unassigned_sentinel(self):
        ids = np.array(["-1", "-1"], dtype=object)
        etypes = np.array(["unknown", "unknown"], dtype=object)
        assert (sl._classify_etype_vec(ids, etypes) == sl.CLASS_UNASSIGNED).all()

    def test_dropped_sentinels(self):
        ids = np.array(["DROP", "demote_rejected"], dtype=object)
        etypes = np.array(["unknown", "unknown"], dtype=object)
        assert (sl._classify_etype_vec(ids, etypes) == sl.CLASS_DROPPED).all()

    def test_etype_missing_falls_back(self):
        ids = np.array(["cell_42", "cell_42-1", "-1", "DROP"], dtype=object)
        # No etype passed; sentinels still classify, the rest become partial
        out = sl._classify_etype_vec(ids, None)
        assert out[0] == sl.CLASS_PARTIAL  # lossy default
        assert out[1] == sl.CLASS_PARTIAL
        assert out[2] == sl.CLASS_UNASSIGNED
        assert out[3] == sl.CLASS_DROPPED

    def test_sentinel_overrides_etype(self):
        # If id is "-1" but etype is stale "cell", classification is unassigned
        ids = np.array(["-1"], dtype=object)
        etypes = np.array(["cell"], dtype=object)
        assert sl._classify_etype_vec(ids, etypes)[0] == sl.CLASS_UNASSIGNED


class TestConstants:
    def test_class_codes_unique(self):
        codes = [sl.CLASS_MAIN, sl.CLASS_PARTIAL, sl.CLASS_COMPONENT,
                 sl.CLASS_UNASSIGNED, sl.CLASS_DROPPED]
        assert len(set(codes)) == 5
        assert all(0 <= c <= 4 for c in codes)

    def test_seg_default_phases(self):
        assert sl.PHASE_KEYS_SEG_DEFAULT[0] == "input"
        assert sl.PHASE_KEYS_SEG_DEFAULT[-1] == "finalize"
        assert "phase1" in sl.PHASE_KEYS_SEG_DEFAULT
        assert len(sl.PHASE_KEYS_SEG_DEFAULT) == 9

    def test_noseg_default_phases(self):
        assert sl.PHASE_KEYS_NOSEG_DEFAULT[0] == "input"
        assert sl.PHASE_KEYS_NOSEG_DEFAULT[-1] == "finalize"
        assert "cascade" in sl.PHASE_KEYS_NOSEG_DEFAULT
        assert len(sl.PHASE_KEYS_NOSEG_DEFAULT) == 8

    def test_seg_verbose_superset_of_default(self):
        # phase1 collapses to (prune + sub-steps) in verbose
        verbose_set = set(sl.PHASE_KEYS_SEG_VERBOSE)
        assert "prune" in verbose_set
        assert "reassign_1c" in verbose_set
        assert "phase1" not in verbose_set  # phase1 is default-tier shorthand
        assert len(sl.PHASE_KEYS_SEG_VERBOSE) == 14

    def test_collapse_map(self):
        assert sl.COLLAPSE_SEG["phase1+rescue"] == "rescue"
        assert sl.COLLAPSE_SEG["group+rescue"] == "post_group_rescue"
        assert sl.COLLAPSE_SEG["stitch+demote+rescue"] == "final_rescue"

    def test_display_labels_cover_all_keys(self):
        # Every key from every tier must have a display label
        all_keys = (set(sl.PHASE_KEYS_SEG_VERBOSE)
                    | set(sl.PHASE_KEYS_SEG_DEFAULT)
                    | set(sl.PHASE_KEYS_SEG_COLLAPSED)
                    | set(sl.PHASE_KEYS_NOSEG_VERBOSE)
                    | set(sl.PHASE_KEYS_NOSEG_DEFAULT)
                    | set(sl.PHASE_KEYS_NOSEG_COLLAPSED))
        missing = all_keys - set(sl.PHASE_DISPLAY_LABELS.keys())
        assert not missing, f"Missing display labels: {missing}"

    def test_phase1_displays_as_prune(self):
        assert sl.PHASE_DISPLAY_LABELS["phase1"] == "Prune"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/origin-core-refactor
pytest tests/test_sankey_log.py -v
```

Expected: ImportError or `ModuleNotFoundError: tracer.sankey_log`.

- [ ] **Step 3: Implement the module**

Create `src/tracer/sankey_log.py`:

```python
"""Per-phase transcript-state snapshot logging for the Sankey/flow plot.

Centralized classifier, phase-key tier tables, and display-label map.
Pure numpy/pandas; no plotting dependencies. Public API:
- snapshot_phase(df, phase, *, id_col)
- _classify_etype_vec(id_arr, etype_arr)
- PHASE_KEYS_{SEG,NOSEG}_{DEFAULT,VERBOSE,COLLAPSED}
- COLLAPSE_{SEG,NOSEG}
- PHASE_DISPLAY_LABELS
- CLASS_* constants
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

# ─── class codes (int8) ────────────────────────────────────────────────
CLASS_MAIN: int = 0
CLASS_PARTIAL: int = 1
CLASS_COMPONENT: int = 2
CLASS_UNASSIGNED: int = 3
CLASS_DROPPED: int = 4

CLASS_NAMES = {
    CLASS_MAIN: "main",
    CLASS_PARTIAL: "partial",
    CLASS_COMPONENT: "component",
    CLASS_UNASSIGNED: "unassigned",
    CLASS_DROPPED: "dropped",
}

# Three-class collapsed view: component→partial, dropped→unassigned
CLASS_COLLAPSE_3 = {
    CLASS_MAIN: CLASS_MAIN,
    CLASS_PARTIAL: CLASS_PARTIAL,
    CLASS_COMPONENT: CLASS_PARTIAL,
    CLASS_UNASSIGNED: CLASS_UNASSIGNED,
    CLASS_DROPPED: CLASS_UNASSIGNED,
}

# Sentinel id strings
_UNASSIGNED_SENTINELS = frozenset({"-1"})
_DROPPED_SENTINELS = frozenset({"DROP", "demote_rejected"})


# ─── classifier ────────────────────────────────────────────────────────
def _classify_etype_vec(
    id_arr: np.ndarray,
    etype_arr: Optional[np.ndarray],
) -> np.ndarray:
    """Vectorized class lookup for an id column ± etype Categorical.

    Sentinels (`"-1"`, `"DROP"`, `"demote_rejected"`) always win — they
    override whatever the etype column says (handles stale rows).
    Otherwise:
      etype == "cell"      → CLASS_MAIN
      etype == "partial"   → CLASS_PARTIAL
      etype == "component" → CLASS_COMPONENT
      etype absent / "unknown" → CLASS_PARTIAL (lossy fallback)
    """
    n = len(id_arr)
    out = np.full(n, CLASS_PARTIAL, dtype=np.int8)

    if etype_arr is not None:
        etype_str = np.asarray(etype_arr, dtype=object)
        out[etype_str == "cell"] = CLASS_MAIN
        out[etype_str == "partial"] = CLASS_PARTIAL
        out[etype_str == "component"] = CLASS_COMPONENT

    ids = np.asarray(id_arr, dtype=object)
    is_unassigned = np.array([s in _UNASSIGNED_SENTINELS for s in ids], dtype=bool)
    is_dropped = np.array([s in _DROPPED_SENTINELS for s in ids], dtype=bool)
    out[is_unassigned] = CLASS_UNASSIGNED
    out[is_dropped] = CLASS_DROPPED

    return out


# ─── phase tiers ───────────────────────────────────────────────────────
PHASE_KEYS_SEG_DEFAULT = [
    "input", "phase1", "rescue", "group", "post_group_rescue",
    "stitch", "demote", "final_rescue", "finalize",
]
PHASE_KEYS_NOSEG_DEFAULT = [
    "input", "cascade", "mid_qc", "post_group_rescue",
    "stitch", "demote", "final_rescue", "finalize",
]

PHASE_KEYS_SEG_VERBOSE = [
    "input", "prune", "reassign_1c", "split_p1", "rerank", "qc_p1",
    "maha_remerge", "rescue", "group", "mid_qc", "post_group_rescue",
    "stitch", "demote", "final_rescue", "finalize",
]
PHASE_KEYS_NOSEG_VERBOSE = list(PHASE_KEYS_NOSEG_DEFAULT)

# Tier A — display-time collapse. Value is the source column at the end
# of the collapsed group (i.e. take the snapshot at that boundary).
COLLAPSE_SEG = {
    "phase1+rescue":         "rescue",
    "group+rescue":          "post_group_rescue",
    "stitch+demote+rescue":  "final_rescue",
}
PHASE_KEYS_SEG_COLLAPSED = [
    "input", "phase1+rescue", "group+rescue", "stitch+demote+rescue", "finalize",
]
COLLAPSE_NOSEG = {
    "cascade+rescue":        "post_group_rescue",
    "stitch+demote+rescue":  "final_rescue",
}
PHASE_KEYS_NOSEG_COLLAPSED = [
    "input", "cascade+rescue", "stitch+demote+rescue", "finalize",
]


# ─── display labels ────────────────────────────────────────────────────
PHASE_DISPLAY_LABELS = {
    # Tier B (default)
    "input":              "Input",
    "phase1":             "Prune",
    "rescue":             "Rescue",
    "group":              "Group",
    "post_group_rescue":  "Post-Group Rescue",
    "stitch":             "Stitch",
    "demote":             "Demote",
    "final_rescue":       "Final Rescue",
    "finalize":           "Finalize",
    # Tier C verbose (SEG)
    "prune":              "Prune",
    "reassign_1c":        "Reassign 1c",
    "split_p1":           "Split P1",
    "rerank":             "Rerank",
    "qc_p1":              "QC P1",
    "maha_remerge":       "Maha Remerge",
    "mid_qc":             "Mid QC",
    # NOSEG
    "cascade":            "Cascade",
    # Tier A collapsed (both pipelines)
    "phase1+rescue":            "Prune + Rescue",
    "group+rescue":             "Group + Rescue",
    "stitch+demote+rescue":     "Stitch + Demote + Rescue",
    "cascade+rescue":           "Cascade + Rescue",
}


# ─── snapshot helper ───────────────────────────────────────────────────
def snapshot_phase(df: pd.DataFrame, phase: str, *, id_col: str) -> None:
    """In-place: write `etype_at_<phase>` int8 column on `df`.

    Reads the current `id_col` (e.g. `tracer_id` pre-Stitch, `stitched`
    post-Stitch) and `_etype` if present.
    """
    if id_col not in df.columns:
        raise KeyError(f"snapshot_phase: id_col {id_col!r} not in df.columns")
    etype_arr = df["_etype"].values if "_etype" in df.columns else None
    df[f"etype_at_{phase}"] = _classify_etype_vec(df[id_col].values, etype_arr)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_sankey_log.py -v
```

Expected: all 14+ tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tracer/sankey_log.py tests/test_sankey_log.py
git commit -m "feat(sankey_log): classifier, phase tiers, display labels

Adds src/tracer/sankey_log.py with:
- _classify_etype_vec (int8 codes: main/partial/component/unassigned/dropped)
- snapshot_phase(df, phase, *, id_col) hook
- PHASE_KEYS_{SEG,NOSEG}_{DEFAULT,VERBOSE,COLLAPSED} tier tables
- COLLAPSE_{SEG,NOSEG} display-time group maps
- PHASE_DISPLAY_LABELS (phase1 → 'Prune')

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: snapshot_phase round-trip tests + conservation invariant

**Files:**
- Modify: `tests/test_sankey_log.py`
- Modify: `src/tracer/sankey_log.py` (add `_validate_conservation`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sankey_log.py`:

```python
class TestSnapshotPhase:
    def _toy_df(self):
        return pd.DataFrame({
            "tracer_id": ["cell_1", "cell_1-1", "UNASSIGNED_b", "-1", "DROP"],
            "_etype": ["cell", "partial", "component", "unknown", "unknown"],
        })

    def test_writes_int8_column(self):
        df = self._toy_df()
        sl.snapshot_phase(df, "phase1", id_col="tracer_id")
        col = df["etype_at_phase1"]
        assert col.dtype == np.int8
        assert list(col) == [sl.CLASS_MAIN, sl.CLASS_PARTIAL,
                             sl.CLASS_COMPONENT, sl.CLASS_UNASSIGNED,
                             sl.CLASS_DROPPED]

    def test_inplace(self):
        df = self._toy_df()
        ret = sl.snapshot_phase(df, "rescue", id_col="tracer_id")
        assert ret is None
        assert "etype_at_rescue" in df.columns

    def test_missing_id_col_raises(self):
        df = self._toy_df()
        with pytest.raises(KeyError, match="id_col"):
            sl.snapshot_phase(df, "stitch", id_col="stitched")

    def test_no_etype_column(self):
        df = pd.DataFrame({"tracer_id": ["cell_1", "-1", "DROP"]})
        sl.snapshot_phase(df, "input", id_col="tracer_id")
        # Sentinels still classify correctly; "cell_1" falls back to partial
        assert list(df["etype_at_input"]) == [sl.CLASS_PARTIAL,
                                              sl.CLASS_UNASSIGNED,
                                              sl.CLASS_DROPPED]


class TestConservation:
    def test_size_preserved_across_two_snapshots(self):
        df = pd.DataFrame({"tracer_id": ["cell_1", "-1", "DROP"] * 100,
                           "_etype": ["cell", "unknown", "unknown"] * 100})
        sl.snapshot_phase(df, "input", id_col="tracer_id")
        df["tracer_id"] = df["tracer_id"].replace({"-1": "cell_99"})
        # Caller would refresh _etype too; for the test, also flip etype
        df.loc[df["tracer_id"] == "cell_99", "_etype"] = "cell"
        sl.snapshot_phase(df, "rescue", id_col="tracer_id")
        # Total rows unchanged
        assert df["etype_at_input"].size == df["etype_at_rescue"].size
        # Conservation: sum of class counts equal between snapshots
        c_in = pd.Series(df["etype_at_input"]).value_counts().sort_index()
        c_out = pd.Series(df["etype_at_rescue"]).value_counts().sort_index()
        assert c_in.sum() == c_out.sum() == 300
```

- [ ] **Step 2: Run tests to verify Task-2-new ones fail or pass**

```bash
pytest tests/test_sankey_log.py::TestSnapshotPhase tests/test_sankey_log.py::TestConservation -v
```

Expected: PASS (snapshot_phase already implemented in Task 1). If any fails, the implementation has a bug.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sankey_log.py
git commit -m "test(sankey_log): snapshot round-trip + conservation invariant

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Data-prep — tidy transition DataFrame

**Files:**
- Create: `src/tracer/flow_plot.py`
- Create: `tests/test_flow_plot.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_flow_plot.py`:

```python
"""Tests for tracer.flow_plot — data-prep, view resolution, backends."""
import numpy as np
import pandas as pd
import pytest

try:
    from tracer import flow_plot as fp
    from tracer import sankey_log as sl
except ImportError:
    import flow_plot as fp
    import sankey_log as sl


def _toy_snapshot_df(n_per_class=10):
    """3-phase df with hand-crafted transitions.
    input: 10 cell + 10 partial + 10 unassigned = 30
    phase1: same (no change)
    rescue: 10 unassigned → 5 partial + 5 dropped, others unchanged
    """
    base = ([sl.CLASS_MAIN] * n_per_class
            + [sl.CLASS_PARTIAL] * n_per_class
            + [sl.CLASS_UNASSIGNED] * n_per_class)
    after_rescue = ([sl.CLASS_MAIN] * n_per_class
                    + [sl.CLASS_PARTIAL] * n_per_class
                    + [sl.CLASS_PARTIAL] * 5 + [sl.CLASS_DROPPED] * 5)
    return pd.DataFrame({
        "etype_at_input":   np.array(base, dtype=np.int8),
        "etype_at_phase1":  np.array(base, dtype=np.int8),
        "etype_at_rescue":  np.array(after_rescue, dtype=np.int8),
    })


class TestPrepareFlowData:
    def test_tidy_shape(self):
        df = _toy_snapshot_df()
        tidy = fp._prepare_flow_data(
            df, phases=["input", "phase1", "rescue"], class_grouping="five",
        )
        assert set(tidy.columns) == {
            "phase_from", "phase_to", "class_from", "class_to", "n"
        }

    def test_conservation_input_to_phase1(self):
        df = _toy_snapshot_df()
        tidy = fp._prepare_flow_data(
            df, phases=["input", "phase1", "rescue"], class_grouping="five",
        )
        # input → phase1: identity, all 30 transcripts in same class
        sub = tidy[(tidy["phase_from"] == "input") &
                   (tidy["phase_to"] == "phase1")]
        assert sub["n"].sum() == 30

    def test_transition_phase1_to_rescue(self):
        df = _toy_snapshot_df()
        tidy = fp._prepare_flow_data(
            df, phases=["input", "phase1", "rescue"], class_grouping="five",
        )
        sub = tidy[(tidy["phase_from"] == "phase1") &
                   (tidy["phase_to"] == "rescue")]
        # 10 main→main, 10 partial→partial, 5 unassigned→partial, 5 unassigned→dropped
        edges = {(r.class_from, r.class_to): r.n for r in sub.itertuples()}
        assert edges[(sl.CLASS_MAIN, sl.CLASS_MAIN)] == 10
        assert edges[(sl.CLASS_PARTIAL, sl.CLASS_PARTIAL)] == 10
        assert edges[(sl.CLASS_UNASSIGNED, sl.CLASS_PARTIAL)] == 5
        assert edges[(sl.CLASS_UNASSIGNED, sl.CLASS_DROPPED)] == 5

    def test_class_grouping_three(self):
        df = _toy_snapshot_df()
        tidy = fp._prepare_flow_data(
            df, phases=["input", "phase1", "rescue"], class_grouping="three",
        )
        # 5 dropped collapses into unassigned bucket
        sub = tidy[(tidy["phase_from"] == "phase1") &
                   (tidy["phase_to"] == "rescue")]
        edges = {(r.class_from, r.class_to): r.n for r in sub.itertuples()}
        # 5 unassigned→partial stays; 5 unassigned→dropped becomes unassigned→unassigned
        assert edges[(sl.CLASS_UNASSIGNED, sl.CLASS_UNASSIGNED)] == 5

    def test_conservation_assertion_raises_on_mismatch(self):
        df = _toy_snapshot_df()
        # Corrupt: drop a row from the rescue column
        df.loc[0, "etype_at_rescue"] = -1  # invalid code
        with pytest.raises((ValueError, AssertionError)):
            fp._prepare_flow_data(
                df, phases=["input", "phase1", "rescue"],
                class_grouping="five", strict_conservation=True,
            )

    def test_min_flow_frac_drops_small(self):
        df = _toy_snapshot_df()  # 30 total tx
        tidy = fp._prepare_flow_data(
            df, phases=["input", "phase1", "rescue"],
            class_grouping="five", min_flow_frac=0.2,  # drops anything < 6 tx
        )
        # The 5-tx unassigned→dropped edge should be filtered out
        sub = tidy[(tidy["phase_from"] == "phase1") &
                   (tidy["phase_to"] == "rescue")]
        edges = {(r.class_from, r.class_to): r.n for r in sub.itertuples()}
        assert (sl.CLASS_UNASSIGNED, sl.CLASS_DROPPED) not in edges
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_flow_plot.py::TestPrepareFlowData -v
```

Expected: ImportError on `flow_plot`.

- [ ] **Step 3: Create flow_plot module with _prepare_flow_data**

Create `src/tracer/flow_plot.py`:

```python
"""Sankey/alluvial plot of per-transcript assignment flow through pipeline phases.

Public API:
- plot_transcript_flow(transcripts, ...) — added in later tasks

Internals:
- _prepare_flow_data — build tidy (phase_from, phase_to, class_from, class_to, n) df
- _collapse_classes — 5-class → 3-class collapse
- _resolve_view — select phase list from view + columns present
- _render_plotly / _render_matplotlib — backend dispatchers
"""
from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import sankey_log as sl


def _collapse_classes(codes: np.ndarray, grouping: str) -> np.ndarray:
    """5-class int8 → 3-class int8 if grouping=='three'; else passthrough."""
    if grouping == "five":
        return codes
    if grouping == "three":
        out = codes.copy()
        for src, dst in sl.CLASS_COLLAPSE_3.items():
            if src != dst:
                out[codes == src] = dst
        return out
    raise ValueError(f"class_grouping must be 'three' or 'five', got {grouping!r}")


def _prepare_flow_data(
    df: pd.DataFrame,
    *,
    phases: Sequence[str],
    class_grouping: str = "three",
    min_flow_frac: float = 0.0,
    strict_conservation: bool = False,
) -> pd.DataFrame:
    """Build the tidy transition DataFrame from snapshot columns.

    Returns columns: phase_from, phase_to, class_from, class_to, n.
    """
    cols = [f"etype_at_{p}" for p in phases]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"snapshot columns missing: {missing}")

    total = len(df)
    rows = []
    prev_codes = _collapse_classes(df[cols[0]].values.astype(np.int8),
                                   class_grouping)
    for i in range(1, len(phases)):
        curr_codes = _collapse_classes(df[cols[i]].values.astype(np.int8),
                                       class_grouping)
        if strict_conservation:
            if curr_codes.size != prev_codes.size:
                raise ValueError(
                    f"size mismatch at {phases[i-1]}→{phases[i]}: "
                    f"{prev_codes.size} vs {curr_codes.size}"
                )
        # Count transitions (from, to) → n
        edges = pd.crosstab(prev_codes, curr_codes)
        for cls_from in edges.index:
            for cls_to in edges.columns:
                n = int(edges.loc[cls_from, cls_to])
                if n == 0:
                    continue
                if total > 0 and (n / total) < min_flow_frac:
                    continue
                rows.append({
                    "phase_from": phases[i-1],
                    "phase_to": phases[i],
                    "class_from": int(cls_from),
                    "class_to": int(cls_to),
                    "n": n,
                })
        prev_codes = curr_codes

    return pd.DataFrame(rows, columns=[
        "phase_from", "phase_to", "class_from", "class_to", "n"
    ])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_flow_plot.py::TestPrepareFlowData -v
```

Expected: all 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tracer/flow_plot.py tests/test_flow_plot.py
git commit -m "feat(flow_plot): tidy transition DataFrame data-prep

Adds src/tracer/flow_plot.py with _prepare_flow_data and _collapse_classes.
Tidy schema: (phase_from, phase_to, class_from, class_to, n).
Supports 5↔3 class grouping, min_flow_frac filtering, strict conservation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: View resolution + auto-pipeline detection

**Files:**
- Modify: `src/tracer/flow_plot.py` (add `_resolve_view`)
- Modify: `tests/test_flow_plot.py` (add `TestResolveView`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_flow_plot.py`:

```python
class TestResolveView:
    def test_seg_default(self):
        df_cols = {f"etype_at_{p}" for p in sl.PHASE_KEYS_SEG_DEFAULT}
        phases = fp._resolve_view(df_cols, pipeline="seg", view="default")
        assert phases == sl.PHASE_KEYS_SEG_DEFAULT

    def test_noseg_default(self):
        df_cols = {f"etype_at_{p}" for p in sl.PHASE_KEYS_NOSEG_DEFAULT}
        phases = fp._resolve_view(df_cols, pipeline="noseg", view="default")
        assert phases == sl.PHASE_KEYS_NOSEG_DEFAULT

    def test_auto_detects_seg(self):
        # Presence of phase1 column → SEG
        df_cols = {f"etype_at_{p}" for p in sl.PHASE_KEYS_SEG_DEFAULT}
        phases = fp._resolve_view(df_cols, pipeline="auto", view="default")
        assert phases == sl.PHASE_KEYS_SEG_DEFAULT

    def test_auto_detects_noseg(self):
        # Presence of cascade column → NOSEG
        df_cols = {f"etype_at_{p}" for p in sl.PHASE_KEYS_NOSEG_DEFAULT}
        phases = fp._resolve_view(df_cols, pipeline="auto", view="default")
        assert phases == sl.PHASE_KEYS_NOSEG_DEFAULT

    def test_seg_collapsed_returns_source_columns(self):
        df_cols = {f"etype_at_{p}" for p in sl.PHASE_KEYS_SEG_DEFAULT}
        phases = fp._resolve_view(df_cols, pipeline="seg", view="collapsed")
        # Collapsed view maps to TIER B SOURCE columns:
        # input, rescue, post_group_rescue, final_rescue, finalize
        assert phases == ["input", "rescue", "post_group_rescue",
                          "final_rescue", "finalize"]

    def test_verbose_missing_raises(self):
        df_cols = {f"etype_at_{p}" for p in sl.PHASE_KEYS_SEG_DEFAULT}
        with pytest.raises(KeyError, match="verbose"):
            fp._resolve_view(df_cols, pipeline="seg", view="verbose")

    def test_optional_phase_skipped(self):
        # mid_qc absent → drop it from the resolved list
        cols = {f"etype_at_{p}" for p in sl.PHASE_KEYS_SEG_DEFAULT
                if p != "mid_qc"}  # mid_qc not in default anyway but keep test
        phases = fp._resolve_view(cols, pipeline="seg", view="default")
        assert "mid_qc" not in phases
        assert "phase1" in phases  # other required ones still present
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_flow_plot.py::TestResolveView -v
```

Expected: `_resolve_view` does not exist.

- [ ] **Step 3: Add _resolve_view to flow_plot.py**

Insert into `src/tracer/flow_plot.py` after `_collapse_classes`:

```python
def _detect_pipeline(df_cols: set) -> str:
    """Detect 'seg' vs 'noseg' from which snapshot columns are present."""
    if "etype_at_phase1" in df_cols or "etype_at_prune" in df_cols:
        return "seg"
    if "etype_at_cascade" in df_cols:
        return "noseg"
    # Fallback: assume SEG (the more common pipeline)
    return "seg"


def _resolve_view(
    df_cols: set,
    *,
    pipeline: str = "auto",
    view: str = "default",
) -> list[str]:
    """Resolve a phase-key list given the snapshot columns present in df.

    Filters out phase keys whose snapshot column isn't in df (optional or
    skipped phases). Raises KeyError if a *required* column for the
    requested view is missing.
    """
    if pipeline == "auto":
        pipeline = _detect_pipeline(df_cols)

    if view == "verbose":
        keys = (sl.PHASE_KEYS_SEG_VERBOSE if pipeline == "seg"
                else sl.PHASE_KEYS_NOSEG_VERBOSE)
        missing_required = [k for k in ("input", "phase1" if pipeline == "seg"
                                        else "cascade", "finalize")
                            if f"etype_at_{k}" not in df_cols
                            and k != "phase1"]
        # For verbose, require at least one verbose-only column (e.g. prune)
        if pipeline == "seg" and "etype_at_prune" not in df_cols:
            raise KeyError(
                "view='verbose' requested but no verbose snapshot columns "
                "found (e.g. etype_at_prune). Re-run pipeline with "
                "snapshot_level='verbose'."
            )
    elif view == "collapsed":
        # Map collapsed groups to their END-OF-GROUP source columns
        collapse = (sl.COLLAPSE_SEG if pipeline == "seg"
                    else sl.COLLAPSE_NOSEG)
        collapsed_keys = (sl.PHASE_KEYS_SEG_COLLAPSED if pipeline == "seg"
                          else sl.PHASE_KEYS_NOSEG_COLLAPSED)
        keys = []
        for k in collapsed_keys:
            keys.append(collapse.get(k, k))
    elif view == "default":
        keys = (sl.PHASE_KEYS_SEG_DEFAULT if pipeline == "seg"
                else sl.PHASE_KEYS_NOSEG_DEFAULT)
    else:
        raise ValueError(f"view must be 'default'|'collapsed'|'verbose', got {view!r}")

    # Drop phases whose snapshot is absent (optional / skipped)
    return [k for k in keys if f"etype_at_{k}" in df_cols]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_flow_plot.py::TestResolveView -v
```

Expected: all 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tracer/flow_plot.py tests/test_flow_plot.py
git commit -m "feat(flow_plot): view resolution + pipeline auto-detect

_resolve_view picks a phase list given:
- pipeline (seg/noseg/auto — auto sniffs which etype_at_* cols are present)
- view (default/collapsed/verbose)
collapsed view maps the display-label group to its end-of-group source column.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: plot_transcript_flow public API + matplotlib backend

**Files:**
- Modify: `src/tracer/flow_plot.py` (add public function + matplotlib renderer)
- Modify: `tests/test_flow_plot.py` (add `TestPlotMatplotlib`)

We do matplotlib first (already in the tracer dep stack), then plotly (new lazy dep). This way `pytest` doesn't need plotly to validate the core wiring.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_flow_plot.py`:

```python
import matplotlib
matplotlib.use("Agg")  # headless

class TestPlotMatplotlib:
    def _full_df(self):
        """Toy SEG-shaped df with all 9 default-tier snapshot columns."""
        n = 50
        base = ([sl.CLASS_MAIN] * 10 + [sl.CLASS_PARTIAL] * 20
                + [sl.CLASS_UNASSIGNED] * 20)
        df = pd.DataFrame({
            f"etype_at_{p}": np.array(base, dtype=np.int8)
            for p in sl.PHASE_KEYS_SEG_DEFAULT
        })
        # Simulate final_rescue moving 5 unassigned → dropped
        final = base.copy()
        final[-5:] = [sl.CLASS_DROPPED] * 5
        df["etype_at_finalize"] = np.array(final, dtype=np.int8)
        return df

    def test_returns_figure(self):
        df = self._full_df()
        fig = fp.plot_transcript_flow(df, backend="matplotlib")
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_return_data_tuple(self):
        df = self._full_df()
        fig, tidy = fp.plot_transcript_flow(
            df, backend="matplotlib", return_data=True
        )
        assert isinstance(tidy, pd.DataFrame)
        assert set(tidy.columns) >= {"phase_from", "phase_to",
                                     "class_from", "class_to", "n"}

    def test_output_file_png(self, tmp_path):
        df = self._full_df()
        out = tmp_path / "flow.png"
        fp.plot_transcript_flow(df, backend="matplotlib", output=str(out))
        assert out.exists()
        assert out.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_flow_plot.py::TestPlotMatplotlib -v
```

Expected: `plot_transcript_flow` does not exist.

- [ ] **Step 3: Add public function + matplotlib renderer to flow_plot.py**

Append to `src/tracer/flow_plot.py`:

```python
# ─── public API ─────────────────────────────────────────────────────────
def plot_transcript_flow(
    transcripts: pd.DataFrame,
    *,
    pipeline: str = "auto",
    view: str = "default",
    phases: Optional[Sequence[str]] = None,
    drop_unchanged: bool = True,
    min_flow_frac: float = 0.001,
    class_grouping: str = "three",
    color_by: str = "source",
    palette: Optional[dict] = None,
    title: Optional[str] = None,
    backend: str = "plotly",
    output=None,
    return_data: bool = False,
):
    """Sankey/alluvial plot of transcript assignment flow through pipeline phases.

    See docs/superpowers/specs/2026-06-09-transcript-flow-sankey-design.md
    for the full design.
    """
    df_cols = set(transcripts.columns)

    if phases is None:
        phases = _resolve_view(df_cols, pipeline=pipeline, view=view)

    if len(phases) < 2:
        raise ValueError(
            f"Need at least 2 phase snapshots to plot a flow; got {phases}"
        )

    tidy = _prepare_flow_data(
        transcripts,
        phases=phases,
        class_grouping=class_grouping,
        min_flow_frac=min_flow_frac,
        strict_conservation=False,
    )

    if drop_unchanged:
        # Drop phase boundaries where every transition is identity
        # (no class crosses class boundaries)
        keep_boundaries = set()
        for ph_from, ph_to, group in tidy.groupby(["phase_from", "phase_to"]):
            if (group["class_from"] != group["class_to"]).any():
                keep_boundaries.add((ph_from, ph_to))
        if keep_boundaries:
            tidy = tidy[tidy.apply(
                lambda r: (r["phase_from"], r["phase_to"]) in keep_boundaries,
                axis=1
            )].reset_index(drop=True)

    if backend == "matplotlib":
        fig = _render_matplotlib(tidy, phases, title=title,
                                 class_grouping=class_grouping,
                                 palette=palette, color_by=color_by)
    elif backend == "plotly":
        fig = _render_plotly(tidy, phases, title=title,
                             class_grouping=class_grouping,
                             palette=palette, color_by=color_by)
    else:
        raise ValueError(
            f"backend must be 'plotly' or 'matplotlib', got {backend!r}"
        )

    if output is not None:
        _save_figure(fig, output, backend)

    if return_data:
        return fig, tidy
    return fig


# ─── default palette ───────────────────────────────────────────────────
_DEFAULT_PALETTE_5 = {
    sl.CLASS_MAIN: "#1f77b4",       # blue
    sl.CLASS_PARTIAL: "#2ca02c",    # green
    sl.CLASS_COMPONENT: "#9467bd",  # purple
    sl.CLASS_UNASSIGNED: "#7f7f7f", # grey
    sl.CLASS_DROPPED: "#d62728",    # red
}
_DEFAULT_PALETTE_3 = {
    sl.CLASS_MAIN: "#1f77b4",
    sl.CLASS_PARTIAL: "#2ca02c",
    sl.CLASS_UNASSIGNED: "#7f7f7f",
}


def _palette_for(class_grouping: str, override: Optional[dict]) -> dict:
    base = (_DEFAULT_PALETTE_5 if class_grouping == "five"
            else _DEFAULT_PALETTE_3)
    if override:
        base = {**base, **override}
    return base


# ─── matplotlib backend ─────────────────────────────────────────────────
def _render_matplotlib(
    tidy: pd.DataFrame,
    phases: Sequence[str],
    *,
    title: Optional[str],
    class_grouping: str,
    palette: Optional[dict],
    color_by: str,
):
    """Hand-rolled flow polygons with matplotlib. No hover; PNG-friendly."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    palette = _palette_for(class_grouping, palette)
    n_phases = len(phases)
    classes = sorted(palette.keys())

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * n_phases), 6))
    x_pos = np.linspace(0, 1, n_phases)
    band_height = 0.9 / len(classes)

    # Compute per-phase column totals to layout y-bands
    per_phase_totals = {}
    for i, p in enumerate(phases):
        if i == 0:
            grp = tidy[tidy["phase_from"] == p].groupby("class_from")["n"].sum()
        else:
            grp = tidy[tidy["phase_to"] == p].groupby("class_to")["n"].sum()
        per_phase_totals[p] = grp.to_dict()
    total_tx = sum(per_phase_totals[phases[0]].values()) or 1

    # Draw nodes (vertical bars) — one per (phase, class) where count > 0
    node_y_top = {}  # (phase, class) → top-y for this band
    for i, p in enumerate(phases):
        y_cursor = 1.0
        for c in classes:
            n = per_phase_totals[p].get(c, 0)
            if n == 0:
                continue
            h = (n / total_tx) * 0.9
            y_top = y_cursor
            y_bot = y_cursor - h
            ax.bar(x_pos[i], h, bottom=y_bot, width=0.04,
                   color=palette[c], alpha=0.9, zorder=3)
            node_y_top[(p, c)] = (y_top, y_bot)
            y_cursor = y_bot - 0.005

    # Draw ribbons (smooth quadrilaterals) between consecutive phases
    for i in range(n_phases - 1):
        p_from, p_to = phases[i], phases[i + 1]
        sub = tidy[(tidy["phase_from"] == p_from) &
                   (tidy["phase_to"] == p_to)]
        # Track running y-offsets at each node so stacked ribbons don't overlap
        from_offset = {c: 0.0 for c in classes}
        to_offset = {c: 0.0 for c in classes}
        for _, r in sub.iterrows():
            c_from, c_to, n = r.class_from, r.class_to, r.n
            if (p_from, c_from) not in node_y_top: continue
            if (p_to, c_to) not in node_y_top: continue
            top_from, _ = node_y_top[(p_from, c_from)]
            top_to, _ = node_y_top[(p_to, c_to)]
            h = (n / total_tx) * 0.9
            yf_top = top_from - from_offset[c_from]
            yf_bot = yf_top - h
            yt_top = top_to - to_offset[c_to]
            yt_bot = yt_top - h
            from_offset[c_from] += h
            to_offset[c_to] += h
            verts = [
                (x_pos[i] + 0.02, yf_top),
                ((x_pos[i] + x_pos[i+1]) / 2, yf_top),
                ((x_pos[i] + x_pos[i+1]) / 2, yt_top),
                (x_pos[i+1] - 0.02, yt_top),
                (x_pos[i+1] - 0.02, yt_bot),
                ((x_pos[i] + x_pos[i+1]) / 2, yt_bot),
                ((x_pos[i] + x_pos[i+1]) / 2, yf_bot),
                (x_pos[i] + 0.02, yf_bot),
                (x_pos[i] + 0.02, yf_top),
            ]
            codes = ([Path.MOVETO]
                     + [Path.CURVE4] * 3
                     + [Path.LINETO]
                     + [Path.CURVE4] * 3
                     + [Path.CLOSEPOLY])
            color_class = c_from if color_by == "source" else c_to
            ax.add_patch(PathPatch(
                Path(verts, codes),
                facecolor=palette[color_class], alpha=0.35,
                edgecolor="none", zorder=1
            ))

    # Phase labels at top
    for i, p in enumerate(phases):
        ax.text(x_pos[i], 1.04, sl.PHASE_DISPLAY_LABELS.get(p, p),
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, 1.1)
    ax.set_axis_off()
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def _save_figure(fig, output, backend: str) -> None:
    from pathlib import Path
    p = Path(output)
    if backend == "matplotlib":
        fig.savefig(p, dpi=150, bbox_inches="tight")
    elif backend == "plotly":
        if p.suffix.lower() == ".html":
            fig.write_html(str(p))
        else:
            fig.write_image(str(p))  # requires kaleido for non-html
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_flow_plot.py::TestPlotMatplotlib -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tracer/flow_plot.py tests/test_flow_plot.py
git commit -m "feat(flow_plot): plot_transcript_flow + matplotlib backend

Adds the public plot_transcript_flow function. Matplotlib backend draws
flow polygons via PathPatch — no interactivity but works headless and
in any matplotlib env. Default palette covers 5 + 3 class groupings.
drop_unchanged collapses identity-only phase boundaries.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Plotly backend (lazy import)

**Files:**
- Modify: `src/tracer/flow_plot.py` (add `_render_plotly`)
- Modify: `tests/test_flow_plot.py` (add `TestPlotPlotly`, skip if not installed)

- [ ] **Step 1: Check plotly availability**

```bash
python -c "import plotly; print(plotly.__version__)" 2>&1
```

If not installed, the plotly tests will be skipped via `pytest.importorskip`. The plotly backend code must still be written so importing `flow_plot` itself does not require plotly.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_flow_plot.py`:

```python
class TestPlotPlotly:
    def setup_method(self):
        self.go = pytest.importorskip("plotly.graph_objects")

    def _full_df(self):
        n = 50
        base = ([sl.CLASS_MAIN] * 10 + [sl.CLASS_PARTIAL] * 20
                + [sl.CLASS_UNASSIGNED] * 20)
        df = pd.DataFrame({
            f"etype_at_{p}": np.array(base, dtype=np.int8)
            for p in sl.PHASE_KEYS_SEG_DEFAULT
        })
        final = base.copy()
        final[-5:] = [sl.CLASS_DROPPED] * 5
        df["etype_at_finalize"] = np.array(final, dtype=np.int8)
        return df

    def test_returns_plotly_figure(self):
        df = self._full_df()
        fig = fp.plot_transcript_flow(df, backend="plotly")
        assert isinstance(fig, self.go.Figure)

    def test_has_sankey_trace(self):
        df = self._full_df()
        fig = fp.plot_transcript_flow(df, backend="plotly")
        assert any(isinstance(t, self.go.Sankey) for t in fig.data)

    def test_node_count_matches_phases_times_classes(self):
        df = self._full_df()
        fig = fp.plot_transcript_flow(df, backend="plotly",
                                       class_grouping="three")
        sankey = next(t for t in fig.data if isinstance(t, self.go.Sankey))
        # Each (phase, class) with nonzero count gets a node
        # Should be > 0 and finite
        assert len(sankey.node.label) > 0

    def test_html_output(self, tmp_path):
        df = self._full_df()
        out = tmp_path / "flow.html"
        fp.plot_transcript_flow(df, backend="plotly", output=str(out))
        assert out.exists()
        assert b"Sankey" in out.read_bytes()
```

- [ ] **Step 3: Append _render_plotly to flow_plot.py**

```python
# ─── plotly backend (lazy import) ──────────────────────────────────────
def _render_plotly(
    tidy: pd.DataFrame,
    phases: Sequence[str],
    *,
    title: Optional[str],
    class_grouping: str,
    palette: Optional[dict],
    color_by: str,
):
    """Plotly Sankey — interactive HTML, hover tooltips."""
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "plotly is required for backend='plotly'. "
            "Install via `pip install plotly` or use backend='matplotlib'."
        ) from e

    palette = _palette_for(class_grouping, palette)

    # Build node list: one per (phase, class). Node index = phase_idx * K + class_idx
    classes = sorted(palette.keys())
    class_idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    n_nodes = len(phases) * K

    node_labels = []
    node_colors = []
    for p in phases:
        ph_label = sl.PHASE_DISPLAY_LABELS.get(p, p)
        for c in classes:
            node_labels.append(f"{ph_label}: {sl.CLASS_NAMES.get(c, str(c))}")
            node_colors.append(palette[c])

    def _idx(phase_pos: int, c: int) -> int:
        return phase_pos * K + class_idx[c]

    phase_pos = {p: i for i, p in enumerate(phases)}

    src, tgt, val, link_colors, hover = [], [], [], [], []
    total_tx = int(tidy["n"].sum()) or 1
    # Per-boundary totals to compute fraction; use to_class incoming total
    for _, r in tidy.iterrows():
        i_from = phase_pos[r["phase_from"]]
        i_to = phase_pos[r["phase_to"]]
        src.append(_idx(i_from, r["class_from"]))
        tgt.append(_idx(i_to, r["class_to"]))
        val.append(int(r["n"]))
        color_class = (r["class_from"] if color_by == "source"
                       else r["class_to"])
        # Translucent hex w/ alpha — use rgba string
        hex_ = palette[color_class].lstrip("#")
        rr, gg, bb = (int(hex_[i:i+2], 16) for i in (0, 2, 4))
        link_colors.append(f"rgba({rr},{gg},{bb},0.45)")
        pct = 100 * r["n"] / total_tx
        hover.append(
            f"{r['phase_from']} → {r['phase_to']}<br>"
            f"{sl.CLASS_NAMES.get(int(r['class_from']),'?')} → "
            f"{sl.CLASS_NAMES.get(int(r['class_to']),'?')}<br>"
            f"{int(r['n']):,} transcripts ({pct:.2f}%)"
        )

    sankey = go.Sankey(
        node=dict(
            pad=15, thickness=18,
            line=dict(color="black", width=0.3),
            label=node_labels, color=node_colors,
        ),
        link=dict(
            source=src, target=tgt, value=val,
            color=link_colors, customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
    )
    fig = go.Figure(data=[sankey])
    fig.update_layout(
        title_text=title or "Transcript-assignment flow through pipeline phases",
        font_size=11, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_flow_plot.py::TestPlotPlotly -v
```

Expected: 4 PASS (or all skipped if plotly absent — that's acceptable).

If plotly is missing, install it:

```bash
pip install plotly
```

- [ ] **Step 5: Commit**

```bash
git add src/tracer/flow_plot.py tests/test_flow_plot.py
git commit -m "feat(flow_plot): plotly Sankey backend with lazy import

backend='plotly' (default) renders an interactive go.Sankey with per-link
hover (count + percent). plotly imported lazily inside _render_plotly;
flow_plot module import has no plotly requirement. HTML output via
fig.write_html.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Wire exports in tracer.__init__

**Files:**
- Modify: `src/tracer/__init__.py`
- Modify: `tests/test_flow_plot.py` (add `TestPublicAPI`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_flow_plot.py`:

```python
class TestPublicAPI:
    def test_plot_transcript_flow_importable(self):
        # Skip if standalone-load fallback was used
        try:
            import tracer
        except ImportError:
            pytest.skip("full tracer import unavailable in this env")
        assert hasattr(tracer, "plot_transcript_flow")
        assert hasattr(tracer, "snapshot_phase")
```

- [ ] **Step 2: Add exports to `src/tracer/__init__.py`**

Find the line where existing public names are exported and add:

```python
from .sankey_log import (
    snapshot_phase,
    PHASE_KEYS_SEG_DEFAULT,
    PHASE_KEYS_NOSEG_DEFAULT,
    PHASE_KEYS_SEG_VERBOSE,
    PHASE_DISPLAY_LABELS,
)
from .flow_plot import plot_transcript_flow
```

Append the new names to `__all__` if present (check by reading the file first).

- [ ] **Step 3: Run all tests**

```bash
pytest tests/test_sankey_log.py tests/test_flow_plot.py -v
```

Expected: All previous tests + the new one PASS (or PublicAPI test is skipped in standalone mode).

- [ ] **Step 4: Commit**

```bash
git add src/tracer/__init__.py tests/test_flow_plot.py
git commit -m "feat(tracer): export snapshot_phase + plot_transcript_flow

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Wire snapshots into the lung_cancer SEG driver

**Files:**
- Modify: `tutorials/lung_cancer/run_lung_cancer.py`

This is the SEG demo. After each phase boundary, call `snapshot_phase` against the canonical phase key. NOSEG snapshots are documented but not in scope (notebook).

- [ ] **Step 1: Read the current driver to find phase boundaries**

```bash
cd /Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/origin-core-refactor
grep -n "Stage \|prune_transcripts\|annotate_unassigned\|apply_stitching\|demote_small\|reassign_unassigned_grid\|finalize_unassigned\|pre_stage2_rescue" tutorials/lung_cancer/run_lung_cancer.py
```

Note the line numbers of each stage call.

- [ ] **Step 2: Insert snapshot_phase calls**

After each phase primitive's output DataFrame is assigned, call `snapshot_phase`. The id column flips from `tracer_id` (or whatever column the driver currently uses for the assignment label — likely something like `cell_id_npmi_cons_p2`) to `stitched` after Stitch.

The driver uses several different column names per phase (`cell_id_npmi_cons_p2`, `cell_id_spatial`, `cell_id_stitched`, etc.). At each phase boundary, identify the canonical assignment column and pass it as `id_col`. Per the spec §5.2:

```python
# At the top of main():
from tracer.sankey_log import snapshot_phase

# After df construction (input phase)
snapshot_phase(df, "input", id_col="cell_id")  # or whatever id is available pre-prune

# After prune (Stage 1)
df_pruned, aux = prune_transcripts_fast(...)
snapshot_phase(df_pruned, "phase1", id_col="cell_id_npmi_cons_p2")  # canonical phase-1 output col

# After rescue / annotate_unassigned (Stage 2)
df_final = annotate_unassigned_components_fast(...)
snapshot_phase(df_final, "rescue", id_col="cell_id_npmi_cons_p2")
snapshot_phase(df_final, "group", id_col="cell_id_spatial")  # post-grouping

# After stitch (Stage 3)
df_stitched, _ = apply_stitching_to_transcripts_memory_efficient(...)
snapshot_phase(df_stitched, "stitch", id_col="cell_id_stitched")

# Before finetune stitch (Stage 5), record demote + final_rescue if those run
# After finalize
snapshot_phase(df_finetuned, "finalize", id_col="cell_id_finetuned")
```

The driver may have fewer phases than the canonical 9 — that's fine. Missing snapshot columns are skipped at plot time. **Add only the snapshots whose source columns the driver actually produces**.

If a phase isn't represented in this driver, document the gap with a comment:

```python
# NOTE: this driver runs Stage 1+2+3+5 only; phase keys
# {post_group_rescue, demote, final_rescue} are not snapshot here.
```

- [ ] **Step 3: Smoke test — run the driver on a tiny ROI**

The driver typically expects a config / data path. Find an existing small test fixture if one exists:

```bash
ls tests/fixtures/ tests/data/ 2>/dev/null
grep -rn "fixture\|tiny\|sample.*parquet" tutorials/lung_cancer/ | head -5
```

If no fixture: skip running the driver in this task and instead verify the imports and call sites compile:

```bash
python -c "
import ast, sys
with open('tutorials/lung_cancer/run_lung_cancer.py') as f:
    src = f.read()
ast.parse(src)
assert 'snapshot_phase' in src
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add tutorials/lung_cancer/run_lung_cancer.py
git commit -m "feat(tutorial): wire snapshot_phase into lung_cancer SEG driver

Adds snapshot_phase calls after each Stage 1/2/3/5 boundary in the
canonical SEG demo driver, with id_col tracking the active assignment
column at each stage. Drives the transcript-flow Sankey artifact.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Integration smoke + plot artifact

**Files:**
- Create: `tests/test_sankey_integration.py`

End-to-end test: hand-craft a small df with all 9 default-tier columns, run the plot, save HTML, verify the artifact.

- [ ] **Step 1: Write the test**

Create `tests/test_sankey_integration.py`:

```python
"""End-to-end smoke for the transcript-flow Sankey plot."""
import numpy as np
import pandas as pd
import pytest

try:
    from tracer import flow_plot as fp
    from tracer import sankey_log as sl
except ImportError:
    import flow_plot as fp
    import sankey_log as sl

import matplotlib
matplotlib.use("Agg")


def _synthetic_seg_df(n=200, seed=0):
    """Synthetic SEG snapshot df spanning all 9 default-tier phases."""
    rng = np.random.default_rng(seed)
    cols = sl.PHASE_KEYS_SEG_DEFAULT  # 9 phases
    # Start: ~70% main, ~20% partial, ~10% unassigned
    init = rng.choice([sl.CLASS_MAIN, sl.CLASS_PARTIAL, sl.CLASS_UNASSIGNED],
                      size=n, p=[0.7, 0.2, 0.1])
    df = pd.DataFrame({f"etype_at_{cols[0]}": init.astype(np.int8)})
    state = init.copy()
    # Random walk: at each phase, ~5% of transcripts change class
    for phase in cols[1:]:
        new = state.copy()
        flip_mask = rng.random(n) < 0.05
        new[flip_mask] = rng.integers(0, 5, size=flip_mask.sum())
        df[f"etype_at_{phase}"] = new.astype(np.int8)
        state = new
    return df


class TestSankeyIntegration:
    def test_full_seg_matplotlib(self, tmp_path):
        df = _synthetic_seg_df()
        out = tmp_path / "seg_flow.png"
        fig = fp.plot_transcript_flow(
            df, backend="matplotlib", output=str(out), title="SEG demo"
        )
        assert out.exists()
        assert out.stat().st_size > 1000  # nontrivial png

    def test_full_seg_plotly(self, tmp_path):
        pytest.importorskip("plotly.graph_objects")
        df = _synthetic_seg_df()
        out = tmp_path / "seg_flow.html"
        fig = fp.plot_transcript_flow(
            df, backend="plotly", output=str(out), title="SEG demo"
        )
        assert out.exists()
        # Has a Sankey trace embedded
        assert b"Sankey" in out.read_bytes()

    def test_collapsed_view_works(self):
        df = _synthetic_seg_df()
        fig = fp.plot_transcript_flow(
            df, view="collapsed", backend="matplotlib"
        )
        # Just checks no error; figure shape verified in earlier unit tests

    def test_class_grouping_three_vs_five_have_different_node_counts(self):
        pytest.importorskip("plotly.graph_objects")
        df = _synthetic_seg_df()
        fig3 = fp.plot_transcript_flow(df, backend="plotly",
                                        class_grouping="three")
        fig5 = fp.plot_transcript_flow(df, backend="plotly",
                                        class_grouping="five")
        n3 = len(fig3.data[0].node.label)
        n5 = len(fig5.data[0].node.label)
        assert n5 >= n3  # 5-class has at least as many bands
```

- [ ] **Step 2: Run integration tests**

```bash
pytest tests/test_sankey_integration.py -v
```

Expected: all 4 PASS (the plotly test is skipped if plotly absent).

- [ ] **Step 3: Generate a real artifact for visual inspection**

```bash
mkdir -p /tmp/sankey_demo
python -c "
import sys; sys.path.insert(0, 'src')
from tests.test_sankey_integration import _synthetic_seg_df
from tracer import flow_plot as fp
df = _synthetic_seg_df(n=2000, seed=42)
fp.plot_transcript_flow(df, backend='plotly',
    output='/tmp/sankey_demo/seg_demo.html', title='SEG synthetic demo')
fp.plot_transcript_flow(df, backend='matplotlib',
    output='/tmp/sankey_demo/seg_demo.png', title='SEG synthetic demo')
print('Wrote /tmp/sankey_demo/')
"
ls -la /tmp/sankey_demo/
```

Open the HTML in a browser to manually verify it looks right.

- [ ] **Step 4: Commit**

```bash
git add tests/test_sankey_integration.py
git commit -m "test(sankey): end-to-end integration smoke

Synthetic 9-phase SEG snapshot df → both backends render without error,
artifacts save to disk, plotly file contains a Sankey trace, class
grouping three/five produce sensible node counts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Add plotly to optional deps + final smoke

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Check pyproject.toml structure**

```bash
grep -n "optional-dependencies\|\[project\]\|dependencies" pyproject.toml | head -20
```

- [ ] **Step 2: Add plotly to an optional `viz` extra**

Open `pyproject.toml` and locate `[project.optional-dependencies]` (create if absent):

```toml
[project.optional-dependencies]
viz = ["plotly>=5.0"]
```

This means `pip install -e ".[viz]"` pulls plotly; bare install doesn't. The matplotlib backend works without it.

- [ ] **Step 3: Run full test suite**

```bash
cd /Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/origin-core-refactor
pytest tests/test_sankey_log.py tests/test_flow_plot.py tests/test_sankey_integration.py -v
```

Expected: all tests PASS (plotly tests skipped if plotly absent).

- [ ] **Step 4: Final commit**

```bash
git add pyproject.toml
git commit -m "build: add plotly as optional [viz] extra

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

After completing all tasks, verify against the spec (`docs/superpowers/specs/2026-06-09-transcript-flow-sankey-design.md`):

**Spec coverage:**
- §4 Class vocabulary (5 classes, 3-class collapse): Tasks 1, 3 ✓
- §5.1 Phase tiers + snapshot helper: Task 1 ✓
- §5.2 Hook insertion sites (canonical SEG driver): Task 8 ✓
- §5.3 Conservation invariant: Task 2 (size), Task 3 (strict_conservation flag) ✓
- §6 Plot API (all 11 params): Tasks 5, 6 ✓
- §6.2 Visual layout (5/3 Y-bands, ribbon coloring): Tasks 5, 6 ✓
- §6.3 Both backends (plotly default + matplotlib): Tasks 5, 6 ✓
- §6.4 Return contract (Figure or (Figure, tidy_df)): Tasks 5, 6 ✓
- §7 Verification (conservation, smoke, perf): Tasks 2, 9 ✓
- §8 File layout: matches Tasks 1, 3, 7, 8 ✓

**Placeholder scan:** None of the tasks contain TBD/TODO/"implement later" — all code blocks are concrete.

**Type consistency:**
- `snapshot_phase` signature: `(df, phase, *, id_col)` — used identically in Tasks 1, 2, 8.
- `_classify_etype_vec`: `(id_arr, etype_arr) -> np.ndarray[int8]` — consistent across Tasks 1, 3.
- `_prepare_flow_data`: returns `(phase_from, phase_to, class_from, class_to, n)` — same columns referenced in Tasks 3, 5, 9.
- `plot_transcript_flow` signature matches spec §6.
- `_resolve_view`: returns `list[str]` of phase keys — consistent with `phases` param shape.

**Out-of-scope but documented:**
- NOSEG driver snapshot insertion: deferred to user (notebook diffs aren't TDD-friendly). Pattern documented in spec §5.2.
- Per-gene Sankey: not in scope (spec §3).
- Performance perf-test (<2% wall clock): qualitative only; Task 9 stops at functional smoke. Defer rigorous benchmark to a follow-up.

**Plan complete and saved to `docs/superpowers/plans/2026-06-09-transcript-flow-sankey.md`.**
