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

### 5.1 Snapshot helper

```python
# src/tracer/sankey_log.py (new module)
PHASE_KEYS_SEG = [
    "input", "prune", "reassign_1c", "split_p1", "rerank", "qc_p1",
    "maha_remerge", "rescue", "group", "mid_qc", "post_group_rescue",
    "stitch", "demote", "final_rescue", "finalize",
]
PHASE_KEYS_NOSEG = [
    "input", "cascade", "mid_qc", "post_group_rescue",
    "stitch", "demote", "final_rescue", "finalize",
]

def snapshot_phase(df: pd.DataFrame, phase: str, *, id_col: str) -> None:
    """In-place; writes etype_at_<phase> int8 column."""
    df[f"etype_at_{phase}"] = _classify_etype_vec(
        df[id_col].values,
        df["_etype"].values if "_etype" in df.columns else None,
    )
```

`id_col` is `tracer_id` for phases 1-10 (SEG) / 1-4 (NOSEG) and `stitched`
from Stitch onward. A central phase-config table maps each key to
`{long_label, optional, id_col, expected_in_pipeline: {"seg", "noseg"}}`.

### 5.2 Hook insertion sites

| Phase key            | SEG insert site (file:fn)                                           | NOSEG insert site                          |
|----------------------|---------------------------------------------------------------------|--------------------------------------------|
| `input`              | start of `run_pipeline` (after df construction)                     | start of `run_noseg_pipeline`              |
| `prune`              | after `prune_transcripts_nuclear_seed`                              | —                                          |
| `reassign_1c`        | after `_reassign_nuclear_post_1c_etype` (optional)                  | —                                          |
| `split_p1`           | after `_spatial_split_phase1_entities`                              | —                                          |
| `rerank`             | after `_phase1_rerank_within_parent_etype` (optional)               | —                                          |
| `qc_p1`              | after `_qc_demote_small_phase1_entities`                            | —                                          |
| `maha_remerge`       | after `phase1_maha_remerge` (optional)                              | —                                          |
| `rescue`             | after Rescue loop (all passes)                                      | —                                          |
| `cascade`            | —                                                                   | after `cascade_as_residual_handler`        |
| `group`              | after `annotate_unassigned_components_fast`                         | (cascade also performs grouping; skip)     |
| `mid_qc`             | after Mid-QC if it ran                                              | after Mid-QC if it ran                     |
| `post_group_rescue`  | after Post-Group-Rescue loop                                        | after Post-Group-Rescue loop               |
| `stitch`             | after `apply_stitching_to_transcripts_memory_efficient` (id_col=`stitched`) | same                              |
| `demote`             | after `demote_small_entities`                                       | same                                       |
| `final_rescue`       | after Final-Rescue loop                                             | same                                       |
| `finalize`           | after `finalize_unassigned`                                         | same                                       |

Optional phases that skip leave the column **absent** (sentinel for
"phase did not execute"). The plot's data-prep treats missing-column ==
identity transition.

### 5.3 Conservation invariant

After every snapshot, `assert df[f"etype_at_{phase}"].size == n_total_tx`. The
plot's data-prep also asserts that `sum_classes(out_k) == sum_classes(in_k)`
for every transition; mismatch raises with a labelled warning identifying
which phase leaked.

### 5.4 Cost

`int8 × n_phases × n_tx` = 14 × 10 MB = **140 MB** at 10 M tx (SEG) or 80 MB
(NOSEG). Wall-clock < 2 %: each snapshot is one vectorized categorical map.
Snapshots ride the existing parquet write at the end of the pipeline; no
new file format.

## 6. Plot API

```python
# src/tracer/plot.py — new public function (does not touch existing plot_cc)
def plot_transcript_flow(
    transcripts: pd.DataFrame,
    *,
    pipeline: Literal["seg", "noseg", "auto"] = "auto",
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
- `phases`: subset / reorder; defaults to `PHASE_KEYS_<mode>` filtered to
  what's present.
- `drop_unchanged`: collapse a phase that moved zero transcripts (e.g.
  optional `maha_remerge` that was disabled).
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
src/tracer/sankey_log.py        # NEW — phase-key tables + snapshot_phase + classifier
src/tracer/plot.py              # EXTENDED — add plot_transcript_flow
src/tracer/_pipeline_runner.py  # EDITED — insert snapshot_phase calls
src/tracer/spatial.py           # EDITED — snapshot_phase calls in run_noseg_pipeline
tests/test_sankey_log.py        # NEW — snapshot + classifier + conservation tests
tests/test_plot_flow.py         # NEW — backend smoke + tidy-df shape
```

`plot_transcript_flow` is exported from `tracer.__init__`. `sankey_log` is
internal but `snapshot_phase` is part of the public hook contract — third-
party pipelines (e.g. branches under development) can call it directly.

## 9. Scope boundaries (recap)

In: 14/8 canonical phases, 5-class vocabulary collapsible to 3, plotly +
matplotlib, persist to existing parquet.

Out: per-gene Sankey, per-tile Sankey, animation, refactoring
`entity_type_stage_counts.py`.

## 10. Open questions

- Should `snapshot_phase` be a no-op when `_etype` is missing (older
  pipelines), or raise? Tentative answer: warn + fall back to id-only
  classification (still correct for the 3 sentinel classes
  `unassigned/dropped`, lossy for `main/partial/component`).
- Default `min_flow_frac` — 0.1 % keeps the plot readable on 10 M tx but
  hides the very tail. Make it user-tunable; document in plot docstring.
