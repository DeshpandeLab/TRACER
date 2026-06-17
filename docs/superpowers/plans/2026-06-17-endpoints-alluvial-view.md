# Endpoints Alluvial View + Neighboring-Cell Class Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `view="endpoints"` mode to `plot_transcript_flow` that plots only initial→final proportions, and promote the runner's neighboring-cell prototype into the core library so cell-etype transcripts assigned to a different cell than their origin render as a first-class "neighboring cell" class.

**Architecture:** Two coupled pieces on `feature/endpoints-alluvial-view` (off `feature/transcript-flow-sankey`). (1) `sankey_log.py` gains a `CLASS_MAIN_NEIGHBOR = 5` class, updated labels/palette/order/collapse maps, an `_is_original_match` provenance helper, and an `orig_id_col` argument to `snapshot_phase` that promotes moved mains to code 5 at snapshot time. (2) `flow_plot.py` gains a `"endpoints"` view that resolves to `[first, last]` phases, a guard so a lone boundary is never dropped, a 6-class default palette, and `Initial`/`Final` column labels. The runner is simplified to call core instead of monkey-patching.

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

Add one line to the `CLASS_COLLAPSE_3` dict (3-class view folds neighbor → original):

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
- Modify: `src/tracer/sankey_log.py` (add import + helper near the classifier)
- Test: `tests/test_sankey_log.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sankey_log.py`:

```python
class TestIsOriginalMatch:
    def test_same_cell(self):
        assert sl._is_original_match("42", "42") is True

    def test_partial_of_same_cell(self):
        # partials use the -tr- delimiter; same base cell_id
        assert sl._is_original_match("42-tr-1", "42") is True

    def test_different_cell(self):
        assert sl._is_original_match("57", "42") is False

    def test_ffpe_dash_cell_id_same(self):
        assert sl._is_original_match("dafehkie-1-tr-2", "dafehkie-1") is True

    def test_ffpe_dash_cell_id_different(self):
        # "dafehkie-12" must NOT match origin "dafehkie-1"
        assert sl._is_original_match("dafehkie-12", "dafehkie-1") is False

    def test_unassigned_origin_is_no_match(self):
        assert sl._is_original_match("42", "-1") is False
        assert sl._is_original_match("42", "UNASSIGNED") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_sankey_log.py::TestIsOriginalMatch -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_is_original_match'`

- [ ] **Step 3: Add the import and helper**

In `src/tracer/sankey_log.py`, after the existing `import pandas as pd` line, add:

```python
from ._etype import split_entity_label
```

After the `_DROPPED_SENTINELS = frozenset({...})` definition, add:

```python
# Origin tokens that are NOT a real cell — used to gate neighbor detection.
# Union of unassigned + dropped sentinels plus empty string, mirroring the
# runner prototype's _UNASSIGNED_TOKENS.
_NEIGHBOR_NONCELL_TOKENS = frozenset(
    _UNASSIGNED_SENTINELS | _DROPPED_SENTINELS | {""}
)


def _is_original_match(current_id: str, orig_cell_id: str) -> bool:
    """True iff the tx's CURRENT entity id shares the same base cell_id as
    its ORIGINAL (input) segmentation cell_id.

    Base cell_id is the part before the `-tr-` entity delimiter, so a tx that
    moved from cell ``A`` to a partial of the same cell (``A-tr-1``) still
    matches, while a move to cell ``B`` (or ``B-tr-1``) does not. Returns
    False if either id is a non-cell sentinel — handles FFPE dash-containing
    cell_ids (e.g. ``dafehkie-1``) correctly because `split_entity_label`
    only splits on `-tr-`, not bare dashes.
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

### Task 3: Extend `snapshot_phase` with `orig_id_col` neighbor promotion

**Files:**
- Modify: `src/tracer/sankey_log.py` (`snapshot_phase`, end of file)
- Test: `tests/test_sankey_log.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sankey_log.py`:

```python
class TestSnapshotNeighborPromotion:
    def _df(self):
        # 4 tx, all etype "cell", with origin vs current cell_id pairs:
        #   moved A->B            -> neighboring
        #   stayed A->A           -> original
        #   moved to partial A->A-tr-1 -> original (same base)
        #   origin unassigned -1->C   -> original (no real origin)
        return pd.DataFrame({
            "cur":   ["B", "A", "A-tr-1", "C"],
            "orig":  ["A", "A", "A",      "-1"],
            "_etype": pd.Categorical(["cell", "cell", "cell", "cell"]),
        })

    def test_promotes_only_moved_real_origin(self):
        df = self._df()
        sl.snapshot_phase(df, "final", id_col="cur", orig_id_col="orig")
        codes = df["etype_at_final"].to_numpy()
        assert codes[0] == sl.CLASS_MAIN_NEIGHBOR   # A->B moved
        assert codes[1] == sl.CLASS_MAIN            # A->A stayed
        assert codes[2] == sl.CLASS_MAIN            # A->A-tr-1 same base
        assert codes[3] == sl.CLASS_MAIN            # origin unassigned

    def test_orig_id_col_none_is_unchanged(self):
        df = self._df()
        sl.snapshot_phase(df, "final", id_col="cur")  # no orig_id_col
        codes = df["etype_at_final"].to_numpy()
        assert (codes == sl.CLASS_MAIN).all()        # never code 5

    def test_missing_orig_id_col_raises(self):
        df = self._df()
        with pytest.raises(KeyError):
            sl.snapshot_phase(df, "final", id_col="cur", orig_id_col="nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_sankey_log.py::TestSnapshotNeighborPromotion -v`
Expected: FAIL — `snapshot_phase() got an unexpected keyword argument 'orig_id_col'`

- [ ] **Step 3: Replace `snapshot_phase`**

In `src/tracer/sankey_log.py`, replace the existing `snapshot_phase` function with:

```python
def snapshot_phase(
    df: pd.DataFrame,
    phase: str,
    *,
    id_col: str,
    orig_id_col: Optional[str] = None,
) -> None:
    """In-place: write `etype_at_<phase>` int8 column on `df`.

    Reads the current `id_col` (e.g. `tracer_id` pre-Stitch, `stitched`
    post-Stitch) and `_etype` if present.

    When `orig_id_col` is given, any row whose base 5-class code is
    `CLASS_MAIN` but whose current base cell_id differs from its original
    (real) cell_id is promoted to `CLASS_MAIN_NEIGHBOR`. When `orig_id_col`
    is None the output is identical to the legacy 5-class behavior.
    """
    if id_col not in df.columns:
        raise KeyError(f"snapshot_phase: id_col {id_col!r} not in df.columns")
    etype_arr = df["_etype"].values if "_etype" in df.columns else None
    codes = _classify_etype_vec(df[id_col].values, etype_arr)

    if orig_id_col is not None:
        if orig_id_col not in df.columns:
            raise KeyError(
                f"snapshot_phase: orig_id_col {orig_id_col!r} not in df.columns"
            )
        curr = df[id_col].astype(str).to_numpy()
        orig = df[orig_id_col].astype(str).to_numpy()
        real_orig = ~np.isin(orig, list(_NEIGHBOR_NONCELL_TOKENS))
        not_match = np.fromiter(
            (not _is_original_match(c, o) for c, o in zip(curr, orig)),
            dtype=bool, count=len(curr),
        )
        promote = (codes == CLASS_MAIN) & real_orig & not_match
        codes[promote] = CLASS_MAIN_NEIGHBOR

    df[f"etype_at_{phase}"] = codes.astype(np.int8)
```

Update the module docstring API line (top of file) from
`- snapshot_phase(df, phase, *, id_col)` to
`- snapshot_phase(df, phase, *, id_col, orig_id_col=None)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_sankey_log.py -v`
Expected: PASS (new class + all existing sankey_log tests still green)

- [ ] **Step 5: Commit**

```bash
git add src/tracer/sankey_log.py tests/test_sankey_log.py
git commit -m "feat(sankey): snapshot_phase orig_id_col promotes moved mains to neighbor"
```

---

### Task 4: Add `view="endpoints"` to `_resolve_view`

**Files:**
- Modify: `src/tracer/flow_plot.py` (`_resolve_view`, lines ~46-87)
- Test: `tests/test_flow_plot.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_flow_plot.py`:

```python
class TestResolveViewEndpoints:
    def test_seg_endpoints_first_last(self):
        cols = {f"etype_at_{k}" for k in
                ["input", "phase1", "rescue", "stitch", "final_rescue"]}
        keys = fp._resolve_view(cols, pipeline="seg", view="endpoints")
        assert keys == ["input", "final_rescue"]

    def test_noseg_endpoints_first_last(self):
        cols = {f"etype_at_{k}" for k in
                ["input", "cascade", "stitch", "final_rescue"]}
        keys = fp._resolve_view(cols, pipeline="noseg", view="endpoints")
        assert keys == ["input", "final_rescue"]

    def test_endpoints_needs_two_columns(self):
        with pytest.raises(KeyError):
            fp._resolve_view({"etype_at_input"}, pipeline="seg",
                             view="endpoints")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_flow_plot.py::TestResolveViewEndpoints -v`
Expected: FAIL — `ValueError: view must be 'default'|'collapsed'|'verbose'`

- [ ] **Step 3: Add the endpoints branch**

In `src/tracer/flow_plot.py`, inside `_resolve_view`, add a branch before the final `else` that raises `ValueError`. Place it after the `elif view == "default":` block:

```python
    elif view == "endpoints":
        # First and last of the pipeline's default phase list, restricted
        # to snapshot columns actually present. Returns early (already
        # filtered) so it skips the trailing absent-column filter below.
        default_keys = (sl.PHASE_KEYS_SEG_DEFAULT if pipeline == "seg"
                        else sl.PHASE_KEYS_NOSEG_DEFAULT)
        present = [k for k in default_keys if f"etype_at_{k}" in df_cols]
        if len(present) < 2:
            raise KeyError(
                "view='endpoints' needs at least 2 snapshot columns; "
                f"found {present}"
            )
        return [present[0], present[-1]]
```

Also update the final `else` message to include the new view name:

```python
    else:
        raise ValueError(
            f"view must be 'default'|'collapsed'|'verbose'|'endpoints', "
            f"got {view!r}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_flow_plot.py::TestResolveViewEndpoints -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tracer/flow_plot.py tests/test_flow_plot.py
git commit -m "feat(flow): add view='endpoints' resolving to [first,last] phases"
```

---

### Task 5: Lone-boundary `drop_unchanged` guard + endpoints labels + code-5 palette

**Files:**
- Modify: `src/tracer/flow_plot.py` (`plot_transcript_flow` drop_unchanged block & label block; `_DEFAULT_PALETTE_5`)
- Test: `tests/test_flow_plot.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_flow_plot.py`:

```python
class TestEndpointsPlot:
    def _endpoint_df(self):
        # input: all original cell; final: mix incl. neighbor (code 5)
        base = [sl.CLASS_MAIN] * 8
        final = ([sl.CLASS_MAIN] * 3 + [sl.CLASS_MAIN_NEIGHBOR] * 2
                 + [sl.CLASS_PARTIAL] * 2 + [sl.CLASS_UNASSIGNED] * 1)
        return pd.DataFrame({
            "etype_at_input": np.array(base, dtype=np.int8),
            "etype_at_final_rescue": np.array(final, dtype=np.int8),
        })

    def test_lone_boundary_kept_with_drop_unchanged(self):
        df = self._endpoint_df()
        fig, tidy = fp.plot_transcript_flow(
            df, pipeline="seg", view="endpoints", class_grouping="five",
            drop_unchanged=True, backend="matplotlib", return_data=True,
        )
        # the single input->final_rescue boundary must survive
        assert len(tidy) > 0
        assert set(tidy["phase_from"]) == {"input"}
        assert set(tidy["phase_to"]) == {"final_rescue"}

    def test_neighbor_code_renders_in_five_class(self):
        df = self._endpoint_df()
        fig, tidy = fp.plot_transcript_flow(
            df, pipeline="seg", view="endpoints", class_grouping="five",
            backend="matplotlib", return_data=True,
        )
        assert sl.CLASS_MAIN_NEIGHBOR in set(tidy["class_to"])

    def test_default_palette_has_neighbor_color(self):
        assert sl.CLASS_MAIN_NEIGHBOR in fp._DEFAULT_PALETTE_5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_flow_plot.py::TestEndpointsPlot -v`
Expected: FAIL — `test_default_palette_has_neighbor_color` (KeyError/assert), and the render tests may raise on the missing palette entry for code 5.

- [ ] **Step 3a: Add code 5 to the 5-class palette**

In `src/tracer/flow_plot.py`, add the neighbor color to `_DEFAULT_PALETTE_5`:

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

- [ ] **Step 3b: Guard the `drop_unchanged` block**

In `plot_transcript_flow`, change the drop_unchanged condition so a 2-phase (single-boundary) plot is never emptied:

```python
    if drop_unchanged and len(phases) > 2:
        # Drop phase boundaries where every transition is identity
        # (no class crosses class boundaries). Skipped when there is only
        # one boundary (endpoints view) — there is nothing to compress and
        # first->final always has movement.
        keep_boundaries = set()
```

(Leave the body of the block unchanged.)

- [ ] **Step 3c: Override endpoints column labels**

In `plot_transcript_flow`, immediately after the `display_labels = [...]` list comprehension that calls `sl.display_label_for(...)`, add:

```python
    if view == "endpoints":
        display_labels = ["Initial", "Final"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_flow_plot.py -v`
Expected: PASS (new class + existing flow_plot tests green)

- [ ] **Step 5: Commit**

```bash
git add src/tracer/flow_plot.py tests/test_flow_plot.py
git commit -m "feat(flow): endpoints labels, lone-boundary guard, neighbor palette color"
```

---

### Task 6: Verify both backends render endpoints end-to-end (smoke)

**Files:**
- Test: `tests/test_flow_plot.py`

- [ ] **Step 1: Write the smoke test**

Add to `tests/test_flow_plot.py`:

```python
class TestEndpointsBackends:
    def _df(self):
        base = [sl.CLASS_MAIN] * 6 + [sl.CLASS_UNASSIGNED] * 4
        final = ([sl.CLASS_MAIN] * 3 + [sl.CLASS_MAIN_NEIGHBOR] * 3
                 + [sl.CLASS_MAIN] * 2 + [sl.CLASS_UNASSIGNED] * 2)
        return pd.DataFrame({
            "etype_at_input": np.array(base, dtype=np.int8),
            "etype_at_final_rescue": np.array(final, dtype=np.int8),
        })

    def test_matplotlib_endpoints(self):
        fig = fp.plot_transcript_flow(
            self._df(), pipeline="seg", view="endpoints",
            class_grouping="five", backend="matplotlib",
        )
        assert fig is not None

    def test_plotly_endpoints(self):
        pytest.importorskip("plotly")
        fig = fp.plot_transcript_flow(
            self._df(), pipeline="seg", view="endpoints",
            class_grouping="five", backend="plotly",
        )
        assert fig is not None

    def test_three_class_collapses_neighbor(self):
        # In the default 3-class grouping, neighbor folds back into original,
        # so code 5 must NOT appear in the tidy output.
        _, tidy = fp.plot_transcript_flow(
            self._df(), pipeline="seg", view="endpoints",
            class_grouping="three", backend="matplotlib", return_data=True,
        )
        assert sl.CLASS_MAIN_NEIGHBOR not in set(tidy["class_from"])
        assert sl.CLASS_MAIN_NEIGHBOR not in set(tidy["class_to"])
```

- [ ] **Step 2: Run test to verify it fails (or errors)**

Run: `PYTHONPATH=src python -m pytest tests/test_flow_plot.py::TestEndpointsBackends -v`
Expected: PASS already if Tasks 4-5 are correct; if `test_three_class_collapses_neighbor` fails, the `CLASS_COLLAPSE_3` entry from Task 1 is missing — fix Task 1 first.

- [ ] **Step 3: (No new implementation expected)**

These exercise existing code paths. If a test fails, fix the underlying task rather than adding code here.

- [ ] **Step 4: Run full flow + sankey suite**

Run: `PYTHONPATH=src python -m pytest tests/test_flow_plot.py tests/test_sankey_log.py tests/test_sankey_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_flow_plot.py
git commit -m "test(flow): endpoints backend smoke + 3-class neighbor-collapse coverage"
```

---

### Task 7: Simplify the runner to use core + add an endpoints render

**Files:**
- Modify: `tutorials/pdac_io/run_pdac_full_sankey.py`

- [ ] **Step 1: Remove the in-runner neighbor prototype, delegate to core**

In `tutorials/pdac_io/run_pdac_full_sankey.py`:

1. Delete the local `CLASS_MAIN_NEIGHBOR = 5` definition and the
   `sl.CLASS_NAMES.update({...})` / `sl.CLASS_COLLAPSE_3.update({...})` blocks
   (now defined in core). Replace references to the local `CLASS_MAIN_NEIGHBOR`
   with `sl.CLASS_MAIN_NEIGHBOR`.
2. Delete the local `_is_original_match` function and the `_UNASSIGNED_TOKENS`
   set (now in core as `sl._is_original_match` / `sl._NEIGHBOR_NONCELL_TOKENS`).
3. Keep `EXT_PALETTE` and `ORDER` as-is (the runner still passes them
   explicitly; they now match the core defaults but explicit is fine).
4. In `_patched_record_stage`, replace the snapshot + neighbor-flag block with
   a single core call that does the promotion at snapshot time:

```python
    def _patched_record_stage(progression, stage_name, df_p, col):
        phase_key = STAGE_TO_PHASE_KEY.get(stage_name)
        if phase_key is not None:
            orig = "cell_id" if (phase_key in POST_RESCUE_PHASES
                                 and "cell_id" in df_p.columns) else None
            try:
                sl.snapshot_phase(df_p, phase_key, id_col=col, orig_id_col=orig)
            except Exception as e:
                print(f"  ⚠️  snapshot_phase failed for {stage_name!r}: {e}",
                      flush=True)
        return _orig_record(progression, stage_name, df_p, col)
```

5. Delete the `_neighboring_cols` list and the entire post-run
   "apply original-vs-neighbor distinction (mains only)" loop (lines around
   `for phase_key in _neighboring_cols:`) — promotion now happens in
   `snapshot_phase`. Set `n_neighbor_total` from the snapshot columns instead:

```python
    n_neighbor_total = sum(
        int((df_out[f"etype_at_{k}"] == sl.CLASS_MAIN_NEIGHBOR).sum())
        for k in POST_RESCUE_PHASES if f"etype_at_{k}" in df_out.columns
    )
    print(f"  marked {n_neighbor_total:,} (tx × phase) cells as 'neighbor'",
          flush=True)
```

6. In the partition-persist `keep_extra` list, drop the
   `c.startswith("neighboring_at_")` clause (those columns no longer exist):

```python
    keep_extra = [c for c in df_out.columns if c.startswith("etype_at_")]
```

- [ ] **Step 2: Add an endpoints render after the Tier A/B renders**

After the four existing `fp.plot_transcript_flow(...)` calls (Tier A/B), add:

```python
    out_e_html = OUT_DIR / "pdac_full_endpoints.html"
    out_e_png  = OUT_DIR / "pdac_full_endpoints.png"
    title_e = f"pdac_io full sample — Initial → Final (n={len(df_out):,} tx)"
    fp.plot_transcript_flow(df_out, backend="plotly", view="endpoints",
                            output=str(out_e_html), title=title_e, **common)
    fp.plot_transcript_flow(df_out, backend="matplotlib", view="endpoints",
                            output=str(out_e_png), title=title_e, **common)
    for p in (out_e_html, out_e_png):
        print(f"  wrote {p}")
```

(`common` already sets `class_grouping="five"`, so neighboring cell is visible.)

- [ ] **Step 3: Byte-compile check (runner needs the full env to actually run)**

Run: `PYTHONPATH=src python -c "import ast; ast.parse(open('tutorials/pdac_io/run_pdac_full_sankey.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Run the full test suite once more**

Run: `PYTHONPATH=src python -m pytest tests/test_flow_plot.py tests/test_sankey_log.py tests/test_sankey_integration.py tests/test_etype.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tutorials/pdac_io/run_pdac_full_sankey.py
git commit -m "refactor(tutorial): use core neighbor classification + add endpoints render"
```

---

## Self-Review notes

- **Spec coverage:** Part 1 (sankey_log: constants/labels/collapse/order → Task 1; `_is_original_match` → Task 2; `snapshot_phase` orig_id_col → Task 3). Part 2 (endpoints view → Task 4; drop_unchanged guard + labels + palette → Task 5). Behavior/backends → Task 6. Runner simplification + endpoints render → Task 7. All spec sections mapped.
- **Edge case** (originally-unassigned not neighboring) is covered by `TestSnapshotNeighborPromotion.test_promotes_only_moved_real_origin` (4th row).
- **Type consistency:** `CLASS_MAIN_NEIGHBOR`, `_is_original_match`, `_NEIGHBOR_NONCELL_TOKENS`, `snapshot_phase(..., orig_id_col=)`, `_DEFAULT_PALETTE_5`, `view="endpoints"` used consistently across tasks.
- **Backward compatibility:** `snapshot_phase` with `orig_id_col=None` and the 3-class default both keep existing output (Tasks 3 & 6 assert this).
