# Endpoints Alluvial View + Neighboring-Cell Class Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `plot_endpoints_flow` — a post-hoc alluvial of initial→final transcript proportions computed from any TRACER partition (no snapshots) — that surfaces a first-class "neighboring cell" class for transcripts whose final base cell_id differs from their original input cell_id.

**Architecture:** Pure post-hoc, on `feature/endpoints-alluvial-view` (off `feature/transcript-flow-sankey`). `sankey_log.py` gains the `CLASS_MAIN_NEIGHBOR = 5` class (labels/collapse/order), an `_is_original_match` provenance helper, and a `classify_endpoints` function that turns a partition's origin-cell_id + final-label columns into (initial, final) code arrays. `flow_plot.py` gains a reusable `phase_labels` override, a lone-boundary `drop_unchanged` guard, a code-5 palette color, and `plot_endpoints_flow`, which builds a 2-column frame and delegates to the existing renderer via `phases=["input","final"]`. The snapshot machinery and multi-phase views are untouched.

**Tech Stack:** Python, numpy, pandas, matplotlib, plotly, pytest.

**Test command (run from repo root):** `PYTHONPATH=src python -m pytest <path> -v`

---

### Task 1: Add neighboring-cell constants, labels, collapse & order to `sankey_log`

**Files:**
- Modify: `src/tracer/sankey_log.py` (constants block, lines ~19-57)
- Test: `tests/test_sankey_log.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sankey_log.py`:

```python
class TestNeighborClassConstants:
    def test_neighbor_code_and_name(self):
        assert sl.CLASS_MAIN_NEIGHBOR == 5
        assert sl.CLASS_NAMES[sl.CLASS_MAIN_NEIGHBOR] == "neighboring cell"

    def test_renamed_labels(self):
        assert sl.CLASS_NAMES[sl.CLASS_MAIN] == "original cell"
        assert sl.CLASS_NAMES[sl.CLASS_PARTIAL] == "partial cell"
        assert sl.CLASS_NAMES[sl.CLASS_COMPONENT] == "component"
        assert sl.CLASS_NAMES[sl.CLASS_UNASSIGNED] == "unassigned"
        assert sl.CLASS_NAMES[sl.CLASS_DROPPED] == "dropped"

    def test_collapse_3_folds_neighbor_into_main(self):
        assert sl.CLASS_COLLAPSE_3[sl.CLASS_MAIN_NEIGHBOR] == sl.CLASS_MAIN

    def test_semantic_order_neighbor_after_main(self):
        order = sl.CLASS_SEMANTIC_ORDER
        assert sl.CLASS_MAIN in order and sl.CLASS_MAIN_NEIGHBOR in order
        assert order.index(sl.CLASS_MAIN_NEIGHBOR) == order.index(sl.CLASS_MAIN) + 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_sankey_log.py::TestNeighborClassConstants -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'CLASS_MAIN_NEIGHBOR'`

- [ ] **Step 3: Edit the constants block**

In `src/tracer/sankey_log.py`, add the new code after `CLASS_DROPPED`:

```python
CLASS_MAIN: int = 0
CLASS_PARTIAL: int = 1
CLASS_COMPONENT: int = 2
CLASS_UNASSIGNED: int = 3
CLASS_DROPPED: int = 4
CLASS_MAIN_NEIGHBOR: int = 5  # cell assigned to a different cell_id than origin
```

Replace the `CLASS_NAMES` dict with:

```python
CLASS_NAMES = {
    CLASS_MAIN: "original cell",
    CLASS_PARTIAL: "partial cell",
    CLASS_COMPONENT: "component",
    CLASS_UNASSIGNED: "unassigned",
    CLASS_DROPPED: "dropped",
    CLASS_MAIN_NEIGHBOR: "neighboring cell",
}
```

Add one line to `CLASS_COLLAPSE_3` (3-class view folds neighbor → original):

```python
CLASS_COLLAPSE_3 = {
    CLASS_MAIN: CLASS_MAIN,
    CLASS_PARTIAL: CLASS_PARTIAL,
    CLASS_COMPONENT: CLASS_PARTIAL,
    CLASS_UNASSIGNED: CLASS_UNASSIGNED,
    CLASS_DROPPED: CLASS_UNASSIGNED,
    CLASS_MAIN_NEIGHBOR: CLASS_MAIN,
}
```

Replace the `CLASS_SEMANTIC_ORDER` list so neighbor sits right after main:

```python
CLASS_SEMANTIC_ORDER = [
    CLASS_MAIN,
    CLASS_MAIN_NEIGHBOR,
    CLASS_PARTIAL,
    CLASS_COMPONENT,
    CLASS_UNASSIGNED,
    CLASS_DROPPED,
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_sankey_log.py::TestNeighborClassConstants -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tracer/sankey_log.py tests/test_sankey_log.py
git commit -m "feat(sankey): add neighboring-cell class code 5 + renamed labels"
```

---

### Task 2: Add `_is_original_match` provenance helper

**Files:**
- Modify: `src/tracer/sankey_log.py` (add import + sentinel set + helper)
- Test: `tests/test_sankey_log.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sankey_log.py`:

```python
class TestIsOriginalMatch:
    def test_same_cell(self):
        assert sl._is_original_match("42", "42") is True

    def test_partial_of_same_cell(self):
        assert sl._is_original_match("42-tr-1", "42") is True

    def test_different_cell(self):
        assert sl._is_original_match("57", "42") is False

    def test_ffpe_dash_cell_id_same(self):
        assert sl._is_original_match("dafehkie-1-tr-2", "dafehkie-1") is True

    def test_ffpe_dash_cell_id_different(self):
        assert sl._is_original_match("dafehkie-12", "dafehkie-1") is False

    def test_unassigned_origin_is_no_match(self):
        assert sl._is_original_match("42", "-1") is False
        assert sl._is_original_match("42", "UNASSIGNED") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_sankey_log.py::TestIsOriginalMatch -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_is_original_match'`

- [ ] **Step 3: Add the import, sentinel set, and helper**

In `src/tracer/sankey_log.py`, after the existing `import pandas as pd` line, add:

```python
from ._etype import split_entity_label, ENTITY_DELIMITER
```

After the `_DROPPED_SENTINELS = frozenset({...})` definition, add:

```python
# Origin tokens that are NOT a real cell — gate neighbor detection.
_NEIGHBOR_NONCELL_TOKENS = frozenset(
    _UNASSIGNED_SENTINELS | _DROPPED_SENTINELS | {""}
)


def _is_original_match(current_id: str, orig_cell_id: str) -> bool:
    """True iff the tx's CURRENT entity id shares the same base cell_id as
    its ORIGINAL (input) segmentation cell_id.

    Base cell_id is the part before the `-tr-` entity delimiter, so a tx that
    moved from cell ``A`` to a partial of the same cell (``A-tr-1``) still
    matches, while a move to cell ``B`` (or ``B-tr-1``) does not. Returns
    False if either id is a non-cell sentinel. `split_entity_label` only
    splits on `-tr-`, so FFPE dash cell_ids (e.g. ``dafehkie-1``) compare
    correctly.
    """
    if current_id in _NEIGHBOR_NONCELL_TOKENS or orig_cell_id in _NEIGHBOR_NONCELL_TOKENS:
        return False
    base_curr, _ = split_entity_label(current_id)
    base_orig, _ = split_entity_label(orig_cell_id)
    return base_curr == base_orig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_sankey_log.py::TestIsOriginalMatch -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tracer/sankey_log.py tests/test_sankey_log.py
git commit -m "feat(sankey): add _is_original_match base-cell_id provenance helper"
```

---

### Task 3: Add `classify_endpoints` post-hoc classifier

**Files:**
- Modify: `src/tracer/sankey_log.py` (add function near `snapshot_phase`)
- Test: `tests/test_sankey_log.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sankey_log.py`:

```python
class TestClassifyEndpoints:
    def _df(self, with_etype):
        # cur/orig pairs covering every case:
        #  B   / A   cell, moved        -> neighbor
        #  A   / A   cell, stayed       -> original
        #  B-tr-1 / A partial of other  -> partial (etype wins)
        #  C   / -1  cell, no real orig -> original
        #  -1  / D   unassigned final   -> unassigned
        cur  = ["B", "A", "B-tr-1", "C", "-1"]
        orig = ["A", "A", "A",      "-1", "D"]
        data = {"label": cur, "cell_id": orig}
        if with_etype:
            data["_etype"] = pd.Categorical(
                ["cell", "cell", "partial", "cell", "unknown"])
        return pd.DataFrame(data)

    def test_initial_is_original_or_unassigned(self):
        df = self._df(with_etype=True)
        initial, _ = sl.classify_endpoints(
            df, orig_id_col="cell_id", label_col="label")
        # origins: A,A,A,-1,D -> real,real,real,sentinel,real
        assert list(initial) == [sl.CLASS_MAIN, sl.CLASS_MAIN, sl.CLASS_MAIN,
                                 sl.CLASS_UNASSIGNED, sl.CLASS_MAIN]

    def test_final_with_etype(self):
        df = self._df(with_etype=True)
        _, final = sl.classify_endpoints(
            df, orig_id_col="cell_id", label_col="label")
        assert list(final) == [
            sl.CLASS_MAIN_NEIGHBOR,  # B from A
            sl.CLASS_MAIN,           # A from A
            sl.CLASS_PARTIAL,        # B-tr-1 (etype wins)
            sl.CLASS_MAIN,           # C from unassigned origin
            sl.CLASS_UNASSIGNED,     # -1
        ]

    def test_final_without_etype_derives_from_label(self):
        df = self._df(with_etype=False)  # no _etype column
        _, final = sl.classify_endpoints(
            df, orig_id_col="cell_id", label_col="label")
        assert list(final) == [
            sl.CLASS_MAIN_NEIGHBOR,  # B (no -tr-) moved
            sl.CLASS_MAIN,           # A
            sl.CLASS_PARTIAL,        # B-tr-1 has -tr-
            sl.CLASS_MAIN,           # C
            sl.CLASS_UNASSIGNED,     # -1
        ]

    def test_parity_with_is_original_match(self):
        df = self._df(with_etype=True)
        _, final = sl.classify_endpoints(
            df, orig_id_col="cell_id", label_col="label")
        for i, (l, o) in enumerate(zip(df["label"], df["cell_id"])):
            if final[i] == sl.CLASS_MAIN_NEIGHBOR:
                assert not sl._is_original_match(str(l), str(o))

    def test_missing_columns_raise(self):
        df = self._df(with_etype=True)
        with pytest.raises(KeyError):
            sl.classify_endpoints(df, orig_id_col="nope", label_col="label")
        with pytest.raises(KeyError):
            sl.classify_endpoints(df, orig_id_col="cell_id", label_col="nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_sankey_log.py::TestClassifyEndpoints -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'classify_endpoints'`

- [ ] **Step 3: Add the function**

In `src/tracer/sankey_log.py`, add (after `_classify_etype_vec`):

```python
def classify_endpoints(
    df: pd.DataFrame,
    *,
    orig_id_col: str = "cell_id",
    label_col: str,
    etype_col: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Post-hoc (initial, final) class-code arrays for an endpoints flow.

    No snapshot columns required — works on any TRACER partition that carries
    the original input cell_id and the final assigned label.

    initial: real origin cell_id -> CLASS_MAIN, else CLASS_UNASSIGNED.
    final:   5-class of the assigned label (using `etype_col`/`_etype` if
             available, else derived from the `-tr-` label structure), then
             whole-cell rows whose base cell_id differs from a real origin are
             promoted to CLASS_MAIN_NEIGHBOR ("etype wins": `-tr-` partials stay
             partial regardless of which cell).
    """
    if orig_id_col not in df.columns:
        raise KeyError(f"classify_endpoints: orig_id_col {orig_id_col!r} not in df.columns")
    if label_col not in df.columns:
        raise KeyError(f"classify_endpoints: label_col {label_col!r} not in df.columns")

    orig = df[orig_id_col].astype(str).to_numpy()
    label = df[label_col].astype(str).to_numpy()
    noncell = list(_NEIGHBOR_NONCELL_TOKENS)
    real_orig = ~np.isin(orig, noncell)

    # initial
    initial = np.where(real_orig, CLASS_MAIN, CLASS_UNASSIGNED).astype(np.int8)

    # final: base etype classification
    if etype_col is None and "_etype" in df.columns:
        etype_col = "_etype"
    etype_arr = (df[etype_col].values
                 if etype_col is not None and etype_col in df.columns else None)
    final = _classify_etype_vec(label, etype_arr)

    if etype_arr is None:
        # `_classify_etype_vec` lumps non-sentinel rows into PARTIAL when no
        # etype is given. Re-derive cell vs partial vs component from labels.
        s = pd.Series(label)
        non_sentinel = ~np.isin(final, [CLASS_UNASSIGNED, CLASS_DROPPED])
        has_tr = s.str.contains(ENTITY_DELIMITER, regex=False).to_numpy()
        comp_pref = s.str.startswith("UNASSIGNED").to_numpy()
        final[non_sentinel & comp_pref] = CLASS_UNASSIGNED
        final[non_sentinel & ~comp_pref & has_tr] = CLASS_PARTIAL
        final[non_sentinel & ~comp_pref & ~has_tr] = CLASS_MAIN

    # neighbor promotion (vectorized equivalent of _is_original_match):
    # whole-cell + real origin + base(label) != base(origin).
    base_label = pd.Series(label).str.split(ENTITY_DELIMITER, n=1).str[0].to_numpy()
    base_orig = pd.Series(orig).str.split(ENTITY_DELIMITER, n=1).str[0].to_numpy()
    real_label = ~np.isin(label, noncell)
    promote = (final == CLASS_MAIN) & real_orig & real_label & (base_label != base_orig)
    final[promote] = CLASS_MAIN_NEIGHBOR

    return initial, final.astype(np.int8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_sankey_log.py -v`
Expected: PASS (new class + all existing sankey_log tests still green)

- [ ] **Step 5: Commit**

```bash
git add src/tracer/sankey_log.py tests/test_sankey_log.py
git commit -m "feat(sankey): classify_endpoints — post-hoc initial/final class codes"
```

---

### Task 4: Reusable `phase_labels` + lone-boundary guard + code-5 palette

**Files:**
- Modify: `src/tracer/flow_plot.py` (`plot_transcript_flow` label & drop blocks; `_DEFAULT_PALETTE_5`)
- Test: `tests/test_flow_plot.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_flow_plot.py`:

```python
class TestRendererEndpointsSupport:
    def _two_phase_df(self):
        base = [sl.CLASS_MAIN] * 8
        final = ([sl.CLASS_MAIN] * 3 + [sl.CLASS_MAIN_NEIGHBOR] * 2
                 + [sl.CLASS_PARTIAL] * 2 + [sl.CLASS_UNASSIGNED] * 1)
        return pd.DataFrame({
            "etype_at_input": np.array(base, dtype=np.int8),
            "etype_at_final": np.array(final, dtype=np.int8),
        })

    def test_default_palette_has_neighbor_color(self):
        assert sl.CLASS_MAIN_NEIGHBOR in fp._DEFAULT_PALETTE_5

    def test_lone_boundary_kept_with_drop_unchanged(self):
        df = self._two_phase_df()
        fig, tidy = fp.plot_transcript_flow(
            df, phases=["input", "final"], class_grouping="five",
            drop_unchanged=True, backend="matplotlib", return_data=True,
        )
        assert len(tidy) > 0
        assert set(tidy["phase_from"]) == {"input"}
        assert set(tidy["phase_to"]) == {"final"}

    def test_phase_labels_override(self):
        df = self._two_phase_df()
        # display labels feed the matplotlib x tick labels; smoke that the
        # override is accepted and the figure builds.
        fig = fp.plot_transcript_flow(
            df, phases=["input", "final"], class_grouping="five",
            backend="matplotlib",
            phase_labels={"input": "Initial", "final": "Final"},
        )
        labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        assert "Initial" in labels and "Final" in labels
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_flow_plot.py::TestRendererEndpointsSupport -v`
Expected: FAIL — `test_default_palette_has_neighbor_color` asserts False, and `phase_labels` is an unexpected kwarg.

- [ ] **Step 3a: Add code 5 to the 5-class palette**

In `src/tracer/flow_plot.py`, extend `_DEFAULT_PALETTE_5`:

```python
_DEFAULT_PALETTE_5 = {
    sl.CLASS_MAIN: "#1f77b4",       # blue
    sl.CLASS_PARTIAL: "#2ca02c",    # green
    sl.CLASS_COMPONENT: "#9467bd",  # purple
    sl.CLASS_UNASSIGNED: "#7f7f7f", # grey
    sl.CLASS_DROPPED: "#d62728",    # red
    sl.CLASS_MAIN_NEIGHBOR: "#ff7f0e",  # orange
}
```

- [ ] **Step 3b: Add the `phase_labels` parameter**

In `plot_transcript_flow`'s signature, add `phase_labels=None` (place it after
`label_target`):

```python
    label_target: str = "columns",
    phase_labels: Optional[dict] = None,
    output=None,
```

After the `display_labels = [...]` list comprehension (the one calling
`sl.display_label_for(...)`), add:

```python
    if phase_labels:
        display_labels = [phase_labels.get(p, dl)
                          for p, dl in zip(phases, display_labels)]
```

- [ ] **Step 3c: Guard the `drop_unchanged` block**

Change the condition so a single-boundary plot is never emptied:

```python
    if drop_unchanged and len(phases) > 2:
```

(Leave the block body unchanged. Comment: skipped for a lone boundary — nothing
to compress and first→final always moves.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_flow_plot.py -v`
Expected: PASS (new class + existing flow_plot tests green)

- [ ] **Step 5: Commit**

```bash
git add src/tracer/flow_plot.py tests/test_flow_plot.py
git commit -m "feat(flow): phase_labels override, lone-boundary guard, neighbor color"
```

---

### Task 5: Add `plot_endpoints_flow`

**Files:**
- Modify: `src/tracer/flow_plot.py` (add `_resolve_label_col` + `plot_endpoints_flow`)
- Test: `tests/test_flow_plot.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_flow_plot.py`:

```python
class TestPlotEndpointsFlow:
    def _partition(self):
        # 10 tx: origins + final labels (mix of stay/move/partial/unassigned)
        return pd.DataFrame({
            "transcript_id": range(10),
            "cell_id":  ["A", "A", "A", "B", "B", "C", "C", "D", "-1", "E"],
            "stitched": ["A", "B", "A-tr-1", "B", "X", "C", "-1", "D-tr-2", "F", "E"],
        })

    def test_label_col_autodetect_and_build(self):
        fig, tidy = fp.plot_endpoints_flow(
            self._partition(), orig_id_col="cell_id",
            backend="matplotlib", return_data=True,
        )
        assert set(tidy["phase_from"]) == {"input"}
        assert set(tidy["phase_to"]) == {"final"}

    def test_neighbor_visible_default_five_class(self):
        # A->B (move) and B->X (move) are neighbors; default grouping shows them
        _, tidy = fp.plot_endpoints_flow(
            self._partition(), orig_id_col="cell_id",
            backend="matplotlib", return_data=True,
        )
        assert sl.CLASS_MAIN_NEIGHBOR in set(tidy["class_to"])

    def test_three_class_collapses_neighbor(self):
        _, tidy = fp.plot_endpoints_flow(
            self._partition(), orig_id_col="cell_id", class_grouping="three",
            backend="matplotlib", return_data=True,
        )
        assert sl.CLASS_MAIN_NEIGHBOR not in set(tidy["class_to"])

    def test_explicit_label_col(self):
        df = self._partition().rename(columns={"stitched": "final_label"})
        fig = fp.plot_endpoints_flow(
            df, orig_id_col="cell_id", label_col="final_label",
            backend="matplotlib",
        )
        assert fig is not None

    def test_no_label_col_raises(self):
        df = self._partition().drop(columns=["stitched"])
        with pytest.raises(KeyError):
            fp.plot_endpoints_flow(df, orig_id_col="cell_id",
                                   backend="matplotlib")

    def test_plotly_backend(self):
        pytest.importorskip("plotly")
        fig = fp.plot_endpoints_flow(
            self._partition(), orig_id_col="cell_id", backend="plotly")
        assert fig is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_flow_plot.py::TestPlotEndpointsFlow -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'plot_endpoints_flow'`

- [ ] **Step 3: Add the helper and public function**

In `src/tracer/flow_plot.py`, add near the other public API:

```python
def _resolve_label_col(df: pd.DataFrame, label_col: Optional[str]) -> str:
    """Resolve the final-assignment label column, auto-detecting common names."""
    if label_col is not None:
        if label_col not in df.columns:
            raise KeyError(f"label_col {label_col!r} not in df.columns")
        return label_col
    for cand in ("stitched", "tracer_id", "label"):
        if cand in df.columns:
            return cand
    raise KeyError(
        "plot_endpoints_flow: no label column found "
        "(tried 'stitched', 'tracer_id', 'label'); pass label_col=..."
    )


def plot_endpoints_flow(
    df: pd.DataFrame,
    *,
    orig_id_col: str = "cell_id",
    label_col: Optional[str] = None,
    etype_col: Optional[str] = None,
    **plot_kwargs,
):
    """Post-hoc alluvial of INITIAL -> FINAL transcript proportions.

    Works on any TRACER partition (live df or reloaded) carrying the original
    input cell_id (`orig_id_col`) and the final assigned label (`label_col`,
    auto-detected from 'stitched'/'tracer_id'/'label' when None). Classifies
    each tx into original cell / neighboring cell / partial cell / component /
    unassigned at both ends, then renders via `plot_transcript_flow`.

    Defaults `class_grouping="five"` so the neighboring class is visible (the
    3-class grouping folds it back into original). Remaining kwargs (backend,
    palette, class_order, title, output, return_data, ...) pass through.
    """
    label_col = _resolve_label_col(df, label_col)
    initial, final = sl.classify_endpoints(
        df, orig_id_col=orig_id_col, label_col=label_col, etype_col=etype_col,
    )
    df2 = pd.DataFrame({
        "etype_at_input": initial,
        "etype_at_final": final,
    })
    plot_kwargs.setdefault("class_grouping", "five")
    plot_kwargs.setdefault("phase_labels", {"input": "Initial", "final": "Final"})
    return plot_transcript_flow(df2, phases=["input", "final"], **plot_kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_flow_plot.py::TestPlotEndpointsFlow -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tracer/flow_plot.py tests/test_flow_plot.py
git commit -m "feat(flow): plot_endpoints_flow — post-hoc initial/final alluvial"
```

---

### Task 6: Full-suite regression + module docstrings

**Files:**
- Modify: `src/tracer/flow_plot.py` (module docstring), `src/tracer/sankey_log.py` (module docstring)
- Test: (no new tests)

- [ ] **Step 1: Update the public-API docstrings**

In `src/tracer/flow_plot.py` module docstring `Public API:` block, add:
`- plot_endpoints_flow(df, ...) — post-hoc initial→final alluvial`.

In `src/tracer/sankey_log.py` module docstring `Public API:` block, add:
`- classify_endpoints(df, *, orig_id_col, label_col, etype_col=None)`.

- [ ] **Step 2: Run the full flow/sankey/etype suite**

Run: `PYTHONPATH=src python -m pytest tests/test_flow_plot.py tests/test_sankey_log.py tests/test_sankey_integration.py tests/test_etype.py -v`
Expected: PASS (no regressions)

- [ ] **Step 3: Commit**

```bash
git add src/tracer/flow_plot.py src/tracer/sankey_log.py
git commit -m "docs(flow): list endpoints API in module docstrings"
```

---

### Task 7: Add an endpoints render to the pdac runner

**Files:**
- Modify: `tutorials/pdac_io/run_pdac_full_sankey.py`

- [ ] **Step 1: Reconcile local neighbor constants with core**

In `tutorials/pdac_io/run_pdac_full_sankey.py`:

1. Delete the local `CLASS_MAIN_NEIGHBOR = 5` line and the
   `sl.CLASS_NAMES.update({...})` / `sl.CLASS_COLLAPSE_3.update({...})` blocks
   (now canonical in core). Replace every later use of the bare
   `CLASS_MAIN_NEIGHBOR` with `sl.CLASS_MAIN_NEIGHBOR`.
2. Leave the existing multi-phase snapshot monkey-patch and Tier A/B renders
   unchanged — that machinery is out of scope here.

- [ ] **Step 2: Add the post-hoc endpoints render**

After the four existing Tier A/B `fp.plot_transcript_flow(...)` calls, add:

```python
    out_e_html = OUT_DIR / "pdac_full_endpoints.html"
    out_e_png  = OUT_DIR / "pdac_full_endpoints.png"
    title_e = f"pdac_io full sample — Initial → Final (n={len(df_out):,} tx)"
    label_col = "stitched" if "stitched" in df_out.columns else "tracer_id"
    fp.plot_endpoints_flow(df_out, orig_id_col="cell_id", label_col=label_col,
                           backend="plotly", palette=EXT_PALETTE,
                           class_order=ORDER, output=str(out_e_html),
                           title=title_e)
    fp.plot_endpoints_flow(df_out, orig_id_col="cell_id", label_col=label_col,
                           backend="matplotlib", palette=EXT_PALETTE,
                           class_order=ORDER, output=str(out_e_png),
                           title=title_e)
    for p in (out_e_html, out_e_png):
        print(f"  wrote {p}")
```

- [ ] **Step 3: Byte-compile check (full env needed to actually run the pipeline)**

Run: `PYTHONPATH=src python -c "import ast; ast.parse(open('tutorials/pdac_io/run_pdac_full_sankey.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Full suite once more**

Run: `PYTHONPATH=src python -m pytest tests/test_flow_plot.py tests/test_sankey_log.py tests/test_sankey_integration.py tests/test_etype.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tutorials/pdac_io/run_pdac_full_sankey.py
git commit -m "feat(tutorial): post-hoc endpoints render via plot_endpoints_flow"
```

---

## Self-Review notes

- **Spec coverage:** class vocabulary → Task 1; `_is_original_match` → Task 2;
  `classify_endpoints` (initial/final, etype-present & derived, etype-wins,
  unassigned-origin, parity) → Task 3; `phase_labels`/lone-boundary/palette →
  Task 4; `plot_endpoints_flow` (+ label autodetect, five-default,
  three-collapse, both backends) → Task 5; docstrings/regression → Task 6;
  runner endpoints render → Task 7. All spec sections mapped.
- **Decisions covered by tests:** "etype wins" (`test_final_with_etype` row 3),
  "originally-unassigned → original" (row 4), 3-class neighbor collapse
  (`test_three_class_collapses_neighbor`).
- **No snapshot changes:** `snapshot_phase` is untouched; the feature is purely
  post-hoc, matching the spec's key insight.
- **Type consistency:** `CLASS_MAIN_NEIGHBOR`, `_is_original_match`,
  `_NEIGHBOR_NONCELL_TOKENS`, `classify_endpoints(orig_id_col,label_col,etype_col)`,
  `_DEFAULT_PALETTE_5`, `phase_labels`, `plot_endpoints_flow`,
  `_resolve_label_col` used consistently across tasks.
- **Backward compatibility:** `phase_labels=None` and 3-class default both keep
  existing renderer output; no existing signature broken (new params are
  keyword-only with defaults).
