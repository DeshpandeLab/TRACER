# Endpoints alluvial view + first-class neighboring-cell class

**Date:** 2026-06-17
**Branch:** `feature/endpoints-alluvial-view` (off `feature/transcript-flow-sankey`)
**Status:** Design — approved (post-hoc revision), pending spec review

## Goal

Add an alluvial that plots **only the initial and final proportions** — two
columns, no intermediate pipeline stages — with smooth tracked ribbons, matching
the pre/post phenotypic-trajectory style the user provided as a reference.

Surface a **"neighboring cell"** class distinguishing transcripts that ended up
assigned to a *different* cell than their original (input) segmentation
`cell_id`.

## Key insight: endpoints is fully post-hoc — no snapshots needed

Unlike the multi-phase views (`default`/`collapsed`/`verbose`), the endpoints
view needs **no pipeline instrumentation**. Every endpoint class is recoverable
from any TRACER output partition using just two per-transcript columns that are
always present:

- the **original** input `cell_id` (assignments are written to
  `tracer_id`/`stitched`; `cell_id` is never overwritten), and
- the **final** assigned label (the `-tr-` entity id).

Therefore the endpoints feature is a pure post-hoc function over a saved/in-memory
partition. It does **not** modify `snapshot_phase`, does **not** require the
runner's `_record_stage` monkey-patch, and does **not** add a snapshot-based
`view` branch. Intermediate-stage states are genuinely unrecoverable post-hoc and
remain the domain of the existing snapshot machinery — which is left untouched.

## Context / current state

- The alluvial implementation lives **only** on `feature/transcript-flow-sankey`
  (`src/tracer/flow_plot.py`, `src/tracer/sankey_log.py`); not merged into
  `upstream/main`, `main`, or `optimization/core-refactor`.
- The engine already builds **tracked transitions** via
  `crosstab(class_from, class_to)` for any 2+ phase set, and the matplotlib
  backend already draws the smooth crossing ribbons.
- The canonical entity-type vocabulary (`src/tracer/_etype.py`) is exactly
  `["cell", "partial", "component", "drop", "unknown"]`; there is **no**
  `neighbor` etype. "Neighboring cell" is a derived class, not a relabel.
- `_etype.split_entity_label(label)` returns `(base_cell_id, [depth_indices])`,
  splitting only on the `-tr-` delimiter (`ENTITY_DELIMITER`), so it is robust to
  dash-containing FFPE cell_ids like `dafehkie-1`.

## Definitions

### Initial column

For each transcript: real input `cell_id` → **original cell**; non-cell sentinel
(`-1`/`UNASSIGNED`/`DROP`/`nan`/empty/`*_rejected`) → **unassigned**. (At input
there are no partials/components/neighbors — every in-cell tx is its own original
cell.)

### Final column

Classify the final assigned label into the 5-class vocabulary (using the `_etype`
column when present, otherwise deriving cell-vs-partial from the `-tr-` structure
of the label). Then, **only for whole-cell rows** (`CLASS_MAIN`), promote to
**neighboring cell** when the label's base cell_id differs from a *real* origin
cell_id.

**etype wins (decided):** any `-tr-` label is `partial cell` regardless of which
cell it belongs to. The original/neighboring split applies only to whole-cell
(no `-tr-`) labels. So origin `A` → final `B-tr-1` is **partial cell**, not
neighboring.

**Originally-unassigned (decided):** a tx whose origin is a sentinel and which
later lands in a cell is **not** neighboring (no real origin to move from) — it is
`original cell` of its new home.

## Class vocabulary (`sankey_log.py`)

- Add `CLASS_MAIN_NEIGHBOR = 5`.
- `CLASS_NAMES`: `original cell` / `partial cell` / `component` / `unassigned`
  / `dropped` / `neighboring cell`.
- `CLASS_COLLAPSE_3[CLASS_MAIN_NEIGHBOR] = CLASS_MAIN` (3-class folds neighbor →
  original).
- `CLASS_SEMANTIC_ORDER`: `CLASS_MAIN_NEIGHBOR` immediately after `CLASS_MAIN`.

## New API

### `sankey_log.classify_endpoints`

```python
classify_endpoints(df, *, orig_id_col="cell_id", label_col, etype_col=None)
    -> tuple[np.ndarray, np.ndarray]   # (initial_codes int8, final_codes int8)
```

Computes the two endpoint code arrays per the definitions above. Uses
`df[etype_col]` (or `df["_etype"]` if present and `etype_col` is None) for the
final etype; otherwise derives cell/partial/unassigned from the label structure.
The neighbor promotion is the vectorized equivalent of `_is_original_match`
(base-cell_id comparison with a real-origin gate), with a unit test asserting
parity against the scalar helper.

### `sankey_log._is_original_match`

```python
_is_original_match(current_id, orig_cell_id) -> bool
```

True iff both ids are real cells and share the same base cell_id (via
`split_entity_label`). Scalar predicate; documented rule + unit-tested.

### `flow_plot.plot_endpoints_flow`

```python
plot_endpoints_flow(df, *, orig_id_col="cell_id", label_col=None,
                    etype_col=None, **plot_kwargs)
```

Auto-detects `label_col` (`stitched` → `tracer_id` → `label`) when None. Calls
`classify_endpoints`, builds a 2-column frame
(`etype_at_input`, `etype_at_final`), and delegates to `plot_transcript_flow`
with `phases=["input", "final"]` and `phase_labels={"input": "Initial",
"final": "Final"}`. Defaults `class_grouping="five"` so the neighboring class is
visible (the 3-class grouping intentionally folds it into original). All other
plot kwargs (backend, palette, class_order, title, output, return_data, …) pass
through.

## Renderer changes (`flow_plot.py`)

- Add `phase_labels: Optional[dict] = None` to `plot_transcript_flow`. When
  given, override per-phase display labels by phase key. Reusable, not
  endpoints-specific.
- Guard the `drop_unchanged` block so a lone boundary (`len(phases) == 2`) is
  never dropped (prevents the degenerate empty plot; first→final always moves).
- Add `CLASS_MAIN_NEIGHBOR` → `#ff7f0e` (orange) to `_DEFAULT_PALETTE_5` so the
  class renders without a caller-supplied palette.

## Behavior summary

- **Initial:** original cell / unassigned only.
- **Final:** original cell / neighboring cell / partial cell / component /
  unassigned / dropped (legend auto-filters to classes present).
- Ribbons are tracked crosstab transitions, drawn as the existing smooth S-curves.

## Other touches

- Add an endpoints render to `tutorials/pdac_io/run_pdac_full_sankey.py` via
  `plot_endpoints_flow(df_out, orig_id_col="cell_id")` (demonstrates post-hoc on a
  live df). Reconcile the runner's local `CLASS_MAIN_NEIGHBOR`/label overrides
  with the now-canonical core definitions.

## Testing

- `_is_original_match`: same cell, partial-of-same, different cell, FFPE dash ids
  (same/different), unassigned origin.
- `classify_endpoints`:
  - initial = original/unassigned only
  - final whole-cell moved → neighbor; stayed → original; partial-of-other-cell →
    partial (etype wins); originally-unassigned → original; sentinel → unassigned
  - `_etype` absent → derives cell/partial from `-tr-`
  - parity: neighbor decision matches scalar `_is_original_match` on a sample
- `plot_endpoints_flow`: builds 2-column flow; ribbons present; neighbor visible
  under `class_grouping="five"`; folds into original under `class_grouping="three"`;
  both backends smoke via `return_data`; `Initial`/`Final` labels applied.
- renderer: `phase_labels` override; lone-boundary `drop_unchanged` guard;
  `CLASS_MAIN_NEIGHBOR` in `_DEFAULT_PALETTE_5`.

## Out of scope

- Generalizing to arbitrary categorical columns (non-entity-class).
- Showing neighboring cell in the multi-phase snapshot views (would require
  per-phase origin comparison; separate concern — the multi-phase machinery is
  untouched here).
- Merging the alluvial work into `upstream/main` / `optimization/core-refactor`.
- Emitting a `neighbor` value into the canonical `_etype` column.
