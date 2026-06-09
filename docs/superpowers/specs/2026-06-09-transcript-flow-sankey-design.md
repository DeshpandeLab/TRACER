# Transcript-flow Sankey/alluvial visualization

**Status:** draft — design phase
**Branch:** `feature/transcript-flow-sankey` (off `optimization/core-refactor`)
**Date:** 2026-06-09

## 1. Problem

The SEG and NOSEG pipelines move every transcript through ~14 (SEG) or ~8 (NOSEG)
phases that promote, demote, rescue, merge, and drop assignments. Today the only
post-hoc audit is a flat counts CSV (`tutorials/entity_type_stage_counts.py`) at
five hand-picked stages plus a handful of sparse snapshot columns
(`cell_id_npmi_cons_p1`, `cell_id_npmi_cons_p2`, `cell_id_spatial`,
`cell_id_stitched`, `cell_id_final`). Neither tells you *where* a transcript
that ended up `DROP` originally entered — which phase promoted it, which
demoted it, and which rescue passes did or didn't catch it. A Sankey/alluvial
plot answers that directly.

## 2. Goals

- One interactive HTML figure per run that shows every per-phase transition of
  every transcript, grouped into a small fixed set of classes.
- A matplotlib backend for headless / publication PNG output.
- Drill-down: ribbons hover with absolute count + percentage; the underlying
  tidy DataFrame is returnable.
- Per-phase snapshot columns persisted to the same parquet the pipeline
  already writes, so plots can be regenerated offline without re-running.

## 3. Non-goals

- Per-gene Sankey (gene-stratified flow — 16k-fanout problem).
- Per-region / tile-level Sankey.
- Animation through phases.
- Refactoring `tutorials/entity_type_stage_counts.py`. Its classifier is
  shared (see §5) but its CSV-emitter is left alone.

## 4. Class vocabulary

Ribbons connect *classes*, not entity-ids — ids change every phase. Five base
classes, fixed Y-position across all columns:

| Code | Class            | Match rule (against the active id column + `_etype`)                                          |
|-----:|------------------|-----------------------------------------------------------------------------------------------|
| 0    | `main`           | `_etype == "cell"` (Phase-1 main or rerank-promoted main)                                     |
| 1    | `partial`        | `_etype == "partial"` (depth-1, Stitch-merged sub-partial, cascade synthetic)                 |
| 2    | `component`      | `_etype == "component"` (i.e. `UNASSIGNED_*` Group component)                                 |
| 3    | `unassigned`     | id ∈ {`"-1"`} (in-flight, not yet committed)                                                  |
| 4    | `dropped`        | id ∈ {`"DROP"`, `"demote_rejected"`} (final or interim eviction)                               |

Stored as `int8` so the per-phase snapshot column is 1 byte / transcript.

**Default plot grouping (`class_grouping="three"`):** collapse to three classes
for the headline view —
- `main` ← `main`
- `partial` ← `partial` ∪ `component`
- `unassigned` ← `unassigned` ∪ `dropped`

Snapshot columns *always* store full 5-class resolution; `class_grouping="five"`
re-renders without re-running. The classifier is a single vectorized helper
`_classify_etype_vec(id_series, etype_series) -> np.ndarray[int8]`.

## 5. Logging hook

### 5.1 Snapshot helper + phase tiers

Three nested phase-key tiers. The default Sankey shows Tier B; Tier A is a
display-time collapse of Tier B; Tier C unlocks the Phase-1 internals for
debugging.

```python
# src/tracer/sankey_log.py (new module)

# Tier B — DEFAULT: 9 SEG phases, 8 NOSEG phases. Hooks run at these
# boundaries by default. Phase-1 sub-mutations are folded into one "phase1"
# snapshot.
PHASE_KEYS_SEG_DEFAULT = [
    "input", "phase1", "rescue", "group", "post_group_rescue",
    "stitch", "demote", "final_rescue", "finalize",
]
PHASE_KEYS_NOSEG_DEFAULT = [
    "input", "cascade", "mid_qc", "post_group_rescue",
    "stitch", "demote", "final_rescue", "finalize",
]

# Tier C — VERBOSE: 14 SEG phases. Hooks run at these only when
# snapshot_level="verbose". NOSEG verbose == default (no Phase-1 to expand).
PHASE_KEYS_SEG_VERBOSE = [
    "input", "prune", "reassign_1c", "split_p1", "rerank", "qc_p1",
    "maha_remerge", "rescue", "group", "mid_qc", "post_group_rescue",
    "stitch", "demote", "final_rescue", "finalize",
]
PHASE_KEYS_NOSEG_VERBOSE = PHASE_KEYS_NOSEG_DEFAULT

# Tier A — COLLAPSED: 5 SEG nodes / 4 NOSEG nodes. Display-only. No new
# snapshot columns; nodes are sourced from existing Tier B columns at the
# end-of-group boundary.
COLLAPSE_SEG = {
    "phase1+rescue":         "rescue",            # source column at group end
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

def snapshot_phase(df: pd.DataFrame, phase: str, *, id_col: str) -> None:
    """In-place; writes etype_at_<phase> int8 column."""
    df[f"etype_at_{phase}"] = _classify_etype_vec(
        df[id_col].values,
        df["_etype"].values if "_etype" in df.columns else None,
    )

# Display labels — internal keys stay terse for code/columns; user-facing
# Sankey nodes render the values from this map. Tier B's "phase1" displays
# as "Prune" since pruning is the headline Phase-1 operation. Tier C's
# verbose `prune` sub-step also renders as "Prune" — no collision because
# the two never appear in the same view.
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
    # Tier C verbose extras (SEG only)
    "prune":              "Prune",
    "reassign_1c":        "Reassign 1c",
    "split_p1":           "Split P1",
    "rerank":             "Rerank",
    "qc_p1":              "QC P1",
    "maha_remerge":       "Maha Remerge",
    "mid_qc":             "Mid QC",
    # NOSEG
    "cascade":            "Cascade",
    # Tier A collapsed groups (display-only)
    "phase1+rescue":            "Prune + Rescue",
    "group+rescue":             "Group + Rescue",
    "stitch+demote+rescue":     "Stitch + Demote + Rescue",
    "cascade+rescue":           "Cascade + Rescue",
}
```

Snapshot-time tiers are controlled by the runner; display tiers and labels
are controlled by `plot_transcript_flow`.

`id_col` is `tracer_id` for phases 1-10 (SEG) / 1-4 (NOSEG) and `stitched`
from Stitch onward. A central phase-config table maps each key to
`{long_label, optional, id_col, expected_in_pipeline: {"seg", "noseg"}}`.

### 5.2 Hook insertion sites

This codebase has **no central pipeline runner** — each tutorial driver
(`tutorials/lung_cancer/run_lung_cancer.py`,
`tutorials/mouse_ileum/run_mouse_ileum.py`, etc.) composes phases by hand
from primitives in `spatial.py` / `pruning.py` / `stitching.py`. The
snapshot is therefore inserted **at the call-site in each driver**, not
inside the library functions. `snapshot_phase` is a public helper users
plug into their own drivers.

Canonical demo driver for SEG: `tutorials/lung_cancer/run_lung_cancer.py`.
Canonical demo driver for NOSEG: `tutorials/lung_cancer/noseg_workflow.ipynb`
(notebook — pattern documented but not TDD-tested by this plan).

Tier column: `D` = snapshot at default level; `V` = snapshot only at
`snapshot_level="verbose"`; `–` = no hook in that pipeline.

| Phase key            | Tier | After which primitive (SEG)                                       | After which primitive (NOSEG)               |
|----------------------|------|-------------------------------------------------------------------|---------------------------------------------|
| `input`              | D    | df construction                                                   | df construction                             |
| `prune`              | V    | `prune_transcripts_fast` / nuclear-seed prune                     | —                                           |
| `reassign_1c`        | V    | `_reassign_nuclear_post_1c_etype` (optional)                      | —                                           |
| `split_p1`           | V    | `_spatial_split_phase1_entities` (optional)                       | —                                           |
| `rerank`             | V    | `_phase1_rerank_within_parent_etype` (optional)                   | —                                           |
| `qc_p1`              | V    | `_qc_demote_small_phase1_entities` (optional)                     | —                                           |
| `maha_remerge`       | V    | `phase1_maha_remerge` (optional)                                  | —                                           |
| `phase1`             | D    | last Phase-1 primitive (typically `prune_transcripts_fast`)       | —                                           |
| `rescue`             | D    | `pre_stage2_rescue` loop                                          | —                                           |
| `cascade`            | D    | —                                                                 | `cascade_as_residual_handler`               |
| `group`              | D    | `annotate_unassigned_components_fast`                             | (cascade performs grouping; skip)           |
| `mid_qc`             | D    | mid-QC step if present                                            | mid-QC step if present                      |
| `post_group_rescue`  | D    | second `pre_stage2_rescue` loop                                   | second `pre_stage2_rescue` loop             |
| `stitch`             | D    | `apply_stitching_to_transcripts_memory_efficient` (id_col=`stitched`) | same                                    |
| `demote`             | D    | `demote_small_entities`                                           | same                                        |
| `final_rescue`       | D    | `reassign_unassigned_grid_pool` loop                              | same                                        |
| `finalize`           | D    | `finalize_unassigned`                                             | same                                        |

The `phase1` snapshot is the same data as the verbose-tier final Phase-1
snapshot — implementation emits both column names in verbose mode so
default-view plotting keeps working unchanged.

Optional phases that skip leave the column **absent** (sentinel for
"phase did not execute"). The plot's data-prep treats missing-column ==
identity transition.

### 5.3 Conservation invariant

After every snapshot, `assert df[f"etype_at_{phase}"].size == n_total_tx`. The
plot's data-prep also asserts that `sum_classes(out_k) == sum_classes(in_k)`
for every transition; mismatch raises with a labelled warning identifying
which phase leaked.

### 5.4 Cost

`int8 × n_phases × n_tx`:
- **Default** (Tier B, 9 SEG / 8 NOSEG): ~90 MB SEG / ~80 MB NOSEG at 10 M tx
- **Verbose** (Tier C, 14 SEG): ~140 MB SEG at 10 M tx

Wall-clock < 2 %: each snapshot is one vectorized categorical map.
Snapshots ride the existing parquet write at the end of the pipeline; no
new file format.

## 6. Plot API

```python
# src/tracer/plot.py — new public function (does not touch existing plot_cc)
def plot_transcript_flow(
    transcripts: pd.DataFrame,
    *,
    pipeline: Literal["seg", "noseg", "auto"] = "auto",
    view: Literal["default", "collapsed", "verbose"] = "default",
    phases: list[str] | None = None,
    drop_unchanged: bool = True,
    min_flow_frac: float = 0.001,
    class_grouping: Literal["three", "five"] = "three",
    color_by: Literal["source", "target", "transition_kind"] = "source",
    palette: dict[str, str] | None = None,
    title: str | None = None,
    backend: Literal["plotly", "matplotlib"] = "plotly",
    output: str | Path | None = None,
    return_data: bool = False,
) -> "plotly.graph_objects.Figure | matplotlib.figure.Figure | tuple[Figure, pd.DataFrame]":
    ...
```

### 6.1 Parameters

- `pipeline`: `"auto"` detects which `etype_at_*` columns are present.
- `view`: which phase tier to render.
  - `"default"` (Tier B, 9 SEG / 8 NOSEG)
  - `"collapsed"` (Tier A, 5 SEG / 4 NOSEG): groups `phase1+rescue`,
    `group+rescue`, `stitch+demote+rescue` into single columns. Sourced from
    the same Tier-B snapshots — needs no `snapshot_level="verbose"` run.
  - `"verbose"` (Tier C, 14 SEG): requires the pipeline to have been run with
    `snapshot_level="verbose"`; raises a clear error if those columns are
    absent.
- `phases`: explicit override (subset or reorder); when given,
  takes precedence over `view`.
- `drop_unchanged`: collapse a phase that moved zero transcripts (e.g.
  optional `maha_remerge` that was disabled in a verbose run).
- `min_flow_frac`: ribbons below this fraction of total tx are hidden
  (default 0.001 = 0.1 %).
- `class_grouping`: `"three"` (default) or `"five"` (see §4).
- `color_by`: ribbon colored by source class (default), target class, or
  transition kind (`promotion / demotion / merge / drop / stay`).
- `palette`: override per-class color dict.
- `backend`: `"plotly"` returns `go.Figure`; `"matplotlib"` returns
  `plt.Figure` via a hand-rolled flow-polygon renderer.
- `output`: file path; extension picks format (`.html`, `.png`, `.pdf`,
  `.svg`).
- `return_data`: also return tidy DataFrame
  `(phase_from, phase_to, class_from, class_to, n)`.

### 6.2 Visual layout

- Columns evenly spaced left-to-right by phase order; phase name + total-count
  badge above each.
- 5 (or 3) fixed Y-bands per column, top → bottom: `main`, `partial`,
  `component`, `unassigned`, `dropped`. Fixed Y so ribbons read directionally.
- Ribbon color = source class (default); ribbon opacity scaled to
  `log(count)` so the tail stays visible.
- Hover: `Phase X → Phase Y: 12,403 transcripts (3.2 % of total)`.

### 6.3 Backends

- **plotly (default):** `plotly.graph_objects.Sankey`. Lazy import behind
  `try: import plotly` — only required when this function is called.
- **matplotlib:** hand-rolled flow polygons using
  `matplotlib.patches.PathPatch`. No new dependency. Same node/Y-band layout;
  no hover.

### 6.4 Return contract

Default: `Figure` only. With `return_data=True`: `(Figure, tidy_df)` where
tidy_df is the same shape the existing `_record_stage` diagnostics use, so
downstream tooling can ingest both.

## 7. Verification

- Conservation: `sum_over_classes(in) == sum_over_classes(out)` at every
  transition; mismatch raises.
- Round-trip: known SEG run on PDAC 50 µm ROI; final `etype_at_finalize`
  counts via the snapshot path must equal `entity_type_stage_counts.py`'s
  classification of `cell_id_finetuned`.
- Performance: snapshot wall-clock < 2 % of full SEG run on 50 µm ROI.
- Plot smoke: render a NOSEG kidney bootstrap parquet to HTML; manual
  inspection of every ribbon's count vs `_record_stage` diagnostics.

## 8. File layout

```
src/tracer/sankey_log.py                   # NEW — phase-key tables + snapshot_phase + classifier + display labels
src/tracer/flow_plot.py                    # NEW — plot_transcript_flow + data-prep + backends
src/tracer/__init__.py                     # EDITED — export plot_transcript_flow, snapshot_phase
src/tracer/plot.py                         # EDITED — single-line re-export of plot_transcript_flow
tutorials/lung_cancer/run_lung_cancer.py   # EDITED — insert snapshot_phase calls (SEG demo)
tests/test_sankey_log.py                   # NEW — classifier + snapshot + conservation tests
tests/test_flow_plot.py                    # NEW — data-prep + view resolution + backend smoke
```

`plot_transcript_flow` and `snapshot_phase` are both public. `sankey_log`'s
phase-key tables and `PHASE_DISPLAY_LABELS` are also public so users
plumbing snapshots into their own drivers can import them.

NOSEG snapshots: the pattern is documented in the spec but inserting the
calls into `tutorials/lung_cancer/noseg_workflow.ipynb` is deferred
(notebook diffs are hard to TDD-test). Users can follow the SEG demo
verbatim.

## 9. Scope boundaries (recap)

In: three nested phase tiers (collapsed/default/verbose), 5-class vocabulary
collapsible to 3, plotly + matplotlib, persist to existing parquet.

Out: per-gene Sankey, per-tile Sankey, animation, refactoring
`entity_type_stage_counts.py`.

## 10. Open questions

- Should `snapshot_phase` be a no-op when `_etype` is missing (older
  pipelines), or raise? Tentative answer: warn + fall back to id-only
  classification (still correct for the 3 sentinel classes
  `unassigned/dropped`, lossy for `main/partial/component`).
- Default `min_flow_frac` — 0.1 % keeps the plot readable on 10 M tx but
  hides the very tail. Make it user-tunable; document in plot docstring.
- Runner kwarg name for snapshot tier: `snapshot_level` vs `verbose` vs a
  generic `transcript_flow_log={"off","default","verbose"}` — defer to plan
  phase. Default is `"default"`.
