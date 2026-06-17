# Endpoints alluvial view + first-class neighboring-cell class

**Date:** 2026-06-17
**Branch:** `feature/endpoints-alluvial-view` (off `feature/transcript-flow-sankey`)
**Status:** Design — approved, pending spec review

## Goal

Extend the transcript-flow alluvial (`plot_transcript_flow`) so it can plot
**only the initial and final proportions** — two columns, no intermediate
pipeline stages — with smooth tracked ribbons, matching the pre/post
phenotypic-trajectory style the user provided as a reference.

As part of this, surface a **"neighboring cell"** class as a first-class
entity type, distinguishing transcripts that ended up assigned to a *different*
cell than their original (input) segmentation `cell_id` from those that stayed
in their original cell.

## Context / current state

- The alluvial implementation lives **only** on `feature/transcript-flow-sankey`
  (`src/tracer/flow_plot.py`, `src/tracer/sankey_log.py`); it is not merged into
  `upstream/main`, `main`, or `optimization/core-refactor`. We build on the
  feature branch and rebase/merge upstream later.
- The engine already builds **tracked transitions** via
  `crosstab(class_from, class_to)` for any `phases` list, and the matplotlib
  backend already draws the smooth crossing ribbons. So an endpoints view is a
  thin, correct convenience layer — not new rendering.
- The canonical entity-type vocabulary (`src/tracer/_etype.py`) is exactly
  `["cell", "partial", "component", "drop", "unknown"]`. There is **no**
  `neighbor` etype, and no stage emits one. "Neighboring cell" is therefore a
  new derived class, not a relabel.
- A working **prototype** of the neighboring-cell logic already exists as a
  per-run monkey-patch in `tutorials/pdac_io/run_pdac_full_sankey.py`
  (`CLASS_MAIN_NEIGHBOR = 5`, label/palette/order, `_is_original_match`, and a
  post-hoc promotion of `cell → neighboring cell`). This work promotes that
  prototype into the core library.

## Definition: neighboring cell

A transcript is a **neighboring cell** transcript at a given phase iff:

1. its current etype is `cell` (it is assigned to a cell entity), AND
2. its **original** (input) `cell_id` is a real cell (not a sentinel/unassigned
   token), AND
3. its current assigned **base** `cell_id` differs from its original base
   `cell_id`.

"Base cell_id" is computed via `_etype.split_entity_label`, so a transcript that
moved from cell `A` to a *partial of the same cell* `A-tr-1` is **not**
neighboring (same base `A`), while one that moved to cell `B` (or `B-tr-1`) is.
This is robust to dash-containing FFPE cell_ids (e.g. `dafehkie-1`), unlike the
prototype's `startswith(orig + "-")` proxy.

**Edge case (decided):** a transcript that was *originally unassigned* and later
lands in a cell is **not** neighboring — it had no origin cell to move from, so
it counts as `original cell` of its new home. (Matches the prototype's
`is_real_orig` gate.)

Neighbor detection applies to `cell`-etype rows only (mains); partials and
components are unaffected.

## Part 1 — Promote neighboring-cell into core (`sankey_log.py`)

- Add constant `CLASS_MAIN_NEIGHBOR = 5`.
- Update `CLASS_NAMES`:
  - `CLASS_MAIN` → `"original cell"`
  - `CLASS_PARTIAL` → `"partial cell"`
  - `CLASS_COMPONENT` → `"component"`
  - `CLASS_UNASSIGNED` → `"unassigned"`
  - `CLASS_DROPPED` → `"dropped"`
  - `CLASS_MAIN_NEIGHBOR` → `"neighboring cell"`
- Add `CLASS_COLLAPSE_3[CLASS_MAIN_NEIGHBOR] = CLASS_MAIN` so the 3-class
  (`class_grouping="three"`) view folds neighboring back into original.
- Insert `CLASS_MAIN_NEIGHBOR` into `CLASS_SEMANTIC_ORDER` immediately after
  `CLASS_MAIN` (so it stacks adjacent to original cell by default).
- Add helper `_is_original_match(current_id, orig_cell_id)` using
  `split_entity_label` for base-id comparison; returns `False` when either id is
  an unassigned sentinel.
- Extend `snapshot_phase`:

  ```python
  def snapshot_phase(df, phase, *, id_col, orig_id_col=None): ...
  ```

  When `orig_id_col` is provided, after computing the base 5-class code, promote
  rows that are `CLASS_MAIN` but fail `_is_original_match(df[id_col], df[orig_id_col])`
  (and whose origin is a real cell) to `CLASS_MAIN_NEIGHBOR`. When `orig_id_col`
  is `None`, behavior is unchanged — **backward-compatible** for existing call
  sites. This removes the need for the runner's separate `neighboring_at_{phase}`
  columns and post-hoc promotion loop.

## Part 2 — `view="endpoints"` (`flow_plot.py`)

- `_resolve_view`: add an `"endpoints"` case. Resolve the pipeline's `default`
  phase list (filtered to snapshot columns present in the df), then return
  `[keys[0], keys[-1]]`. SEG → `["input", "final_rescue"]`; NOSEG → first and
  last present snapshot. Rides the existing `view=` selector
  (`default` / `collapsed` / `verbose` / `endpoints`); no new parameter.
- `plot_transcript_flow`: guard the `drop_unchanged` block so that when there is
  exactly one boundary (`len(phases) == 2`) the sole boundary is always kept
  (prevents the degenerate empty plot; harmless because first→final always has
  movement).
- Palette: add a 6-class default palette (or extend the 5-class default) with
  `CLASS_MAIN_NEIGHBOR` → `#ff7f0e` (orange) so the new class renders without a
  caller-supplied palette override. `_resolve_class_order` already appends
  palette codes not in `CLASS_SEMANTIC_ORDER`, but since we add `5` to the
  semantic order it slots next to `MAIN`.
- Endpoints default column labels: `"Initial"` and `"Final"`. Still overridable
  through the existing `title` / label machinery.

## Behavior summary

- **Initial** column: every in-cell transcript is `original cell`
  (current == origin); off-cell transcripts are `unassigned`.
- **Final** column: in-cell transcripts that moved to a different cell_id are
  `neighboring cell`; the rest split across
  `original cell` / `partial cell` / `unassigned` / `dropped`.
- Ribbons are tracked crosstab transitions (who moved where), drawn as the
  existing smooth S-curve polygons.

## Other touches

- Simplify `tutorials/pdac_io/run_pdac_full_sankey.py`: drop the monkey-patched
  neighbor flag / post-hoc promotion now living in core; call
  `snapshot_phase(..., orig_id_col="cell_id")`; add an endpoints render
  producing `*_endpoints.html` and `*_endpoints.png`.

## Testing

Unit tests (pytest):

- `snapshot_phase` with `orig_id_col`:
  - moved to a different cell → `CLASS_MAIN_NEIGHBOR`
  - stayed in same cell → `CLASS_MAIN`
  - moved to a *partial of the same* cell (`A` → `A-tr-1`) → `CLASS_MAIN`
  - originally unassigned, now in a cell → `CLASS_MAIN` (not neighbor)
  - `orig_id_col=None` → identical to current output (no code `5`)
- `_is_original_match` with dash-containing FFPE ids
  (`dafehkie-1` vs `dafehkie-1-tr-2` vs `dafehkie-12`).
- `_resolve_view(view="endpoints")` returns exactly `[first, last]` for both SEG
  and NOSEG column sets.
- `drop_unchanged` lone-boundary guard: a 2-phase plot keeps its only boundary.
- 3-class collapse folds `CLASS_MAIN_NEIGHBOR` into `CLASS_MAIN`.

Both backends smoke-tested via `return_data=True`.

## Out of scope

- Generalizing the alluvial to arbitrary categorical columns (non-entity-class).
- Merging the alluvial work into `upstream/main` / `optimization/core-refactor`
  (separate integration task).
- Emitting a `neighbor` value into the canonical `_etype` column itself — the
  class is derived at snapshot time from cell_id provenance, not stored as an
  etype.
