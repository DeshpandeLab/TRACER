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


def _check_valid_codes(codes: np.ndarray, phase_name: str) -> None:
    """Raise ValueError if any code is outside the valid 5-class range."""
    valid = {sl.CLASS_MAIN, sl.CLASS_PARTIAL, sl.CLASS_COMPONENT,
             sl.CLASS_UNASSIGNED, sl.CLASS_DROPPED}
    bad = set(int(c) for c in np.unique(codes)) - valid
    if bad:
        raise ValueError(
            f"phase {phase_name!r} contains invalid class codes {sorted(bad)}; "
            f"expected subset of {sorted(valid)}"
        )


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
    if strict_conservation:
        _check_valid_codes(prev_codes, phases[0])
    for i in range(1, len(phases)):
        curr_codes = _collapse_classes(df[cols[i]].values.astype(np.int8),
                                       class_grouping)
        if strict_conservation:
            if curr_codes.size != prev_codes.size:
                raise ValueError(
                    f"size mismatch at {phases[i-1]}→{phases[i]}: "
                    f"{prev_codes.size} vs {curr_codes.size}"
                )
            _check_valid_codes(curr_codes, phases[i])
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
    class_order: Optional[Sequence[int]] = None,
    color_by: str = "source",
    palette: Optional[dict] = None,
    title: Optional[str] = None,
    backend: str = "plotly",
    label_target: str = "columns",
    phase_labels: Optional[dict] = None,
    output=None,
    return_data: bool = False,
):
    """Sankey/alluvial plot of transcript assignment flow through pipeline phases.

    See docs/superpowers/specs/2026-06-09-transcript-flow-sankey-design.md
    for the full design.

    Parameters
    ----------
    class_order : list of int, optional
        Top-to-bottom visual order of class codes within each column. By
        default classes are sorted ascending by integer code (which matches
        the semantic order of the 5-class vocabulary: main → partial →
        component → unassigned → dropped). Override with a custom list to
        place extended codes (e.g. ``main_neighbor``) next to their semantic
        sibling. Every code in the list must be in ``palette``; codes in
        ``palette`` not listed are appended at the bottom.
    label_target : {"columns", "ribbons", "both"}, default "columns"
        Where to place phase labels.
        - ``"columns"``: state labels under each column (current default).
        - ``"ribbons"``: stage labels at midpoints between columns, above
          the ribbons. Conceptually correct since ribbons ARE the stages
          (a column is the state after its preceding stage ran).
        - ``"both"``: state labels at columns AND stage labels above ribbons.
        Stage labels bypass any ``column_label_prefix`` the caller may
        have set via patching ``sl.display_label_for`` — they use the raw
        ``PHASE_DISPLAY_LABELS`` lookup. matplotlib backend only for now;
        plotly currently ignores ``label_target``.
    phase_labels : dict, optional
        Map of phase key -> display label, overriding the computed column
        labels for those phases (others keep their computed label). Used e.g.
        by ``plot_endpoints_flow`` to show "Initial"/"Final".
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

    if drop_unchanged and len(phases) > 2:
        # Drop phase boundaries where every transition is identity
        # (no class crosses class boundaries)
        keep_boundaries = set()
        for (ph_from, ph_to), group in tidy.groupby(["phase_from", "phase_to"]):
            if (group["class_from"] != group["class_to"]).any():
                keep_boundaries.add((ph_from, ph_to))
        if keep_boundaries:
            tidy = tidy[tidy.apply(
                lambda r: (r["phase_from"], r["phase_to"]) in keep_boundaries,
                axis=1
            )].reset_index(drop=True)

    # Resolve display labels per phase (Tier A inverts COLLAPSE maps so a
    # source column like `rescue` shows as `Prune`, not `Rescue`).
    resolved_pipeline = (_detect_pipeline(df_cols) if pipeline == "auto"
                        else pipeline)
    display_labels = [
        sl.display_label_for(p, pipeline=resolved_pipeline, view=view)
        for p in phases
    ]
    if phase_labels:
        display_labels = [phase_labels.get(p, dl)
                          for p, dl in zip(phases, display_labels)]
    # Stage labels for the ribbons (action names) — bypass any caller
    # column-label prefix by looking up PHASE_DISPLAY_LABELS directly,
    # with the same Tier-A inverse-collapse logic as display_label_for.
    def _raw_stage_label(phase_key: str) -> str:
        if view == "collapsed":
            inverse = (sl._INVERSE_COLLAPSE_SEG if resolved_pipeline == "seg"
                       else sl._INVERSE_COLLAPSE_NOSEG)
            key = inverse.get(phase_key, phase_key)
            return sl.PHASE_DISPLAY_LABELS.get(key, key)
        return sl.PHASE_DISPLAY_LABELS.get(phase_key, phase_key)
    stage_labels = [_raw_stage_label(p) for p in phases[1:]]

    if backend == "matplotlib":
        fig = _render_matplotlib(tidy, phases, display_labels=display_labels,
                                 stage_labels=stage_labels,
                                 label_target=label_target,
                                 title=title, class_grouping=class_grouping,
                                 class_order=class_order,
                                 palette=palette, color_by=color_by)
    elif backend == "plotly":
        fig = _render_plotly(tidy, phases, display_labels=display_labels,
                             title=title, class_grouping=class_grouping,
                             class_order=class_order,
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
    # Neighbor (code 5) is intentionally absent from CLASS_SEMANTIC_ORDER, so by
    # default it would land at the bottom. Place it right after original cell.
    # Only safe to inject when using the default palette AND the 5-class
    # grouping; if the caller supplied their own palette, they own the ordering
    # too, and under "three" grouping codes 5/2/4 are absent from the palette
    # (avoids a class_order-not-in-palette ValueError).
    if "palette" not in plot_kwargs and plot_kwargs["class_grouping"] == "five":
        plot_kwargs.setdefault("class_order", [
            sl.CLASS_MAIN, sl.CLASS_MAIN_NEIGHBOR, sl.CLASS_PARTIAL,
            sl.CLASS_COMPONENT, sl.CLASS_UNASSIGNED, sl.CLASS_DROPPED,
        ])
    return plot_transcript_flow(df2, phases=["input", "final"], **plot_kwargs)


# ─── default palette ───────────────────────────────────────────────────
_DEFAULT_PALETTE_5 = {
    sl.CLASS_MAIN: "#1f77b4",       # blue
    sl.CLASS_PARTIAL: "#2ca02c",    # green
    sl.CLASS_COMPONENT: "#9467bd",  # purple
    sl.CLASS_UNASSIGNED: "#7f7f7f", # grey
    sl.CLASS_DROPPED: "#d62728",    # red
    sl.CLASS_MAIN_NEIGHBOR: "#E69F00",  # orange (Okabe-Ito)
}
_DEFAULT_PALETTE_3 = {
    sl.CLASS_MAIN: "#1f77b4",
    sl.CLASS_PARTIAL: "#2ca02c",
    sl.CLASS_UNASSIGNED: "#7f7f7f",
}


def _resolve_class_order(palette: dict,
                         class_order: Optional[Sequence[int]]) -> list[int]:
    """Resolve top-to-bottom visual class ordering.

    Default (None): the semantic order defined in `sl.CLASS_SEMANTIC_ORDER`
    (main → partial → component → unassigned → dropped), filtered to codes
    present in `palette`. Palette codes outside that list are appended at
    the bottom, sorted ascending — extended palettes (e.g. main_neighbor=5)
    should pass an explicit `class_order` to slot new codes near their
    semantic sibling rather than landing at the bottom by default.

    With a user-supplied list: validate all codes are in the palette, then
    append any palette codes the user omitted at the bottom (so they're
    still drawn — silent loss would be worse).
    """
    palette_keys = set(palette.keys())
    if class_order is None:
        in_canonical = [c for c in sl.CLASS_SEMANTIC_ORDER if c in palette_keys]
        leftover = sorted(palette_keys - set(in_canonical))
        return in_canonical + leftover
    bad = [c for c in class_order if c not in palette_keys]
    if bad:
        raise ValueError(
            f"class_order contains codes not in palette: {bad}; "
            f"palette has {sorted(palette_keys)}"
        )
    seen = set(class_order)
    leftover = [c for c in sorted(palette_keys) if c not in seen]
    return list(class_order) + leftover


def _palette_for(class_grouping: str, override: Optional[dict]) -> dict:
    base = (_DEFAULT_PALETTE_5 if class_grouping == "five"
            else _DEFAULT_PALETTE_3)
    if override:
        base = {**base, **override}
    return base


# ─── plotly backend (lazy import) ──────────────────────────────────────
def _render_plotly(
    tidy: pd.DataFrame,
    phases: Sequence[str],
    *,
    display_labels: Optional[Sequence[str]] = None,
    title: Optional[str],
    class_grouping: str,
    class_order: Optional[Sequence[int]] = None,
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
    classes = _resolve_class_order(palette, class_order)
    class_idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    if display_labels is None:
        display_labels = [sl.PHASE_DISPLAY_LABELS.get(p, p) for p in phases]

    # Node labels show only the class (column header carries the phase).
    node_labels = []
    node_colors = []
    for _ in phases:
        for c in classes:
            node_labels.append(sl.CLASS_NAMES.get(c, str(c)))
            node_colors.append(palette[c])

    def _idx(phase_pos: int, c: int) -> int:
        return phase_pos * K + class_idx[c]

    phase_pos = {p: i for i, p in enumerate(phases)}

    src, tgt, val, link_colors, hover = [], [], [], [], []
    # Total tx = mass in any one column (conserved). `tidy["n"].sum()`
    # would multiply this by the number of boundaries, giving percentages
    # ~Nboundaries× too small.
    total_tx = int(tidy[tidy["phase_from"] == phases[0]]["n"].sum()) or 1
    for _, r in tidy.iterrows():
        i_from = phase_pos[r["phase_from"]]
        i_to = phase_pos[r["phase_to"]]
        src.append(_idx(i_from, r["class_from"]))
        tgt.append(_idx(i_to, r["class_to"]))
        val.append(int(r["n"]))
        color_class = (r["class_from"] if color_by == "source"
                       else r["class_to"])
        hex_ = palette[color_class].lstrip("#")
        rr, gg, bb = (int(hex_[i:i+2], 16) for i in (0, 2, 4))
        link_colors.append(f"rgba({rr},{gg},{bb},0.45)")
        pct = 100 * r["n"] / total_tx
        # Use display labels in hover too.
        hover.append(
            f"{display_labels[i_from]} → {display_labels[i_to]}<br>"
            f"{sl.CLASS_NAMES.get(int(r['class_from']),'?')} → "
            f"{sl.CLASS_NAMES.get(int(r['class_to']),'?')}<br>"
            f"{int(r['n']):,} transcripts ({pct:.2f}%)"
        )

    sankey = go.Sankey(
        arrangement="snap",  # respect node.x positions
        node=dict(
            pad=15, thickness=18,
            line=dict(color="black", width=0.3),
            label=node_labels, color=node_colors,
            # Pin x so columns stay aligned with the header annotations.
            # Don't pin y — mass-weighted pinning broke flow conservation
            # on real data; let plotly auto-balance y per column. (Tradeoff:
            # plotly may not respect class_order in HTML; matplotlib does.)
            x=[max(0.001, min(0.999, phase_pos[p] / max(1, len(phases) - 1)))
               for p in phases for _ in classes],
        ),
        link=dict(
            source=src, target=tgt, value=val,
            color=link_colors, customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
    )

    # Column-header annotations along the X axis (top of the figure).
    n_phases = len(phases)
    annotations = []
    for i, lbl in enumerate(display_labels):
        xref_val = (i / max(1, n_phases - 1)) if n_phases > 1 else 0.5
        annotations.append(dict(
            x=xref_val, y=1.06, xref="paper", yref="paper",
            text=f"<b>{lbl}</b>", showarrow=False,
            font=dict(size=12), xanchor="center",
        ))

    # Legend via invisible scatter traces — auto-filtered to classes that
    # actually appear in tidy (otherwise dead swatches appear for codes the
    # data never produces, e.g. component/dropped on modern SEG runs).
    present = set(tidy["class_from"]) | set(tidy["class_to"])
    legend_classes = [c for c in classes if c in present]
    legend_traces = [
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=palette[c]),
            name=sl.CLASS_NAMES.get(c, str(c)),
            showlegend=True, hoverinfo="skip",
        )
        for c in legend_classes
    ]

    fig = go.Figure(data=[sankey, *legend_traces])
    fig.update_layout(
        title_text=title or "Transcript-assignment flow through pipeline phases",
        font_size=11,
        margin=dict(l=20, r=20, t=80, b=40),
        annotations=annotations,
        legend=dict(
            title="Entity class",
            orientation="h", x=0.5, y=-0.08,
            xanchor="center", yanchor="top",
        ),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
    )
    return fig


# ─── matplotlib backend ─────────────────────────────────────────────────
def _layout_nodes_mpl(
    ax,
    tidy: pd.DataFrame,
    phases: Sequence[str],
    x_pos: np.ndarray,
    palette: dict,
    classes: Sequence[int],
) -> tuple[dict, int]:
    """Draw node bars and return:
    - node_y_top: dict[(phase, class)] -> (y_top, y_bot)
    - total_tx: int (total transcripts in the first phase)
    """
    # 1) Compute per-phase per-class totals from tidy.
    #    Use BOTH incoming and outgoing edges so a phase that lost an
    #    entire incoming boundary to `drop_unchanged` (because it was
    #    pure identity) still gets a complete node layout sourced from
    #    its outgoing-edge mass.
    per_phase_totals = {}
    n_phases = len(phases)
    for i, p in enumerate(phases):
        grp_in = (tidy[tidy["phase_to"] == p]
                  .groupby("class_to")["n"].sum() if i > 0
                  else pd.Series(dtype="int64"))
        grp_out = (tidy[tidy["phase_from"] == p]
                   .groupby("class_from")["n"].sum() if i < n_phases - 1
                   else pd.Series(dtype="int64"))
        # Either direction is a valid total (they agree when nothing was
        # dropped). When drop_unchanged ate one side, use the other.
        totals: dict[int, int] = {}
        for c in set(grp_in.index) | set(grp_out.index):
            totals[int(c)] = int(max(grp_in.get(c, 0), grp_out.get(c, 0)))
        per_phase_totals[p] = totals
    total_tx = sum(per_phase_totals[phases[0]].values()) or 1

    # 2) Draw vertical node bars and record y-extents
    node_y_top: dict = {}
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
    return node_y_top, total_tx


def _draw_ribbons_mpl(
    ax,
    tidy: pd.DataFrame,
    phases: Sequence[str],
    x_pos: np.ndarray,
    node_y_top: dict,
    total_tx: int,
    palette: dict,
    classes: Sequence[int],
    color_by: str,
) -> None:
    """Draw smooth ribbon polygons between consecutive phases."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    # Map class code → its top-to-bottom rank in `classes`. Lower rank ==
    # higher in the column.
    class_pos = {c: i for i, c in enumerate(classes)}

    for i in range(len(phases) - 1):
        p_from, p_to = phases[i], phases[i + 1]
        sub = tidy[(tidy["phase_from"] == p_from) &
                   (tidy["phase_to"] == p_to)]
        # Vertical-order-preserving stacking: order ribbons by (src_pos,
        # tgt_pos) so within each source node, outgoing ribbons going to
        # top-of-column targets emerge from the top; symmetrically,
        # incoming ribbons at each target stack in their source's
        # vertical-position order. Without this, ribbons to a top target
        # (e.g. neighboring_cell) emerge from the bottom of their source
        # and bend awkwardly across the column.
        sub = sub.assign(
            _src_pos=sub["class_from"].map(class_pos),
            _tgt_pos=sub["class_to"].map(class_pos),
        ).sort_values(["_src_pos", "_tgt_pos"])
        # Track running y-offsets at each node so stacked ribbons don't overlap
        from_offset = {c: 0.0 for c in classes}
        to_offset = {c: 0.0 for c in classes}
        for _, r in sub.iterrows():
            c_from, c_to, n = r.class_from, r.class_to, r.n
            # If a tidy row points at a node we never drew, that's a
            # data-pipeline bug, not a normal flow — raise rather than
            # silently dropping the ribbon.
            if (p_from, c_from) not in node_y_top:
                raise RuntimeError(
                    f"ribbon source node not in layout: phase={p_from!r}, "
                    f"class={c_from}; this indicates a tidy-df bug"
                )
            if (p_to, c_to) not in node_y_top:
                raise RuntimeError(
                    f"ribbon target node not in layout: phase={p_to!r}, "
                    f"class={c_to}; this indicates a tidy-df bug"
                )
            top_from, _ = node_y_top[(p_from, c_from)]
            top_to, _ = node_y_top[(p_to, c_to)]
            h = (n / total_tx) * 0.9
            yf_top = top_from - from_offset[c_from]
            yf_bot = yf_top - h
            yt_top = top_to - to_offset[c_to]
            yt_bot = yt_top - h
            from_offset[c_from] += h
            to_offset[c_to] += h
            # Ribbon polygon: 9 vertices forming a smooth S-curve quad.
            # Vertices 0..3 = top S-curve (left→right via control points)
            # Vertex  3..4 = right-edge drop (top → bottom on right side)
            # Vertices 4..7 = bottom S-curve (right→left)
            # Vertex  8     = close back to start
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


def _render_matplotlib(
    tidy: pd.DataFrame,
    phases: Sequence[str],
    *,
    display_labels: Optional[Sequence[str]] = None,
    stage_labels: Optional[Sequence[str]] = None,
    label_target: str = "columns",
    title: Optional[str],
    class_grouping: str,
    class_order: Optional[Sequence[int]] = None,
    palette: Optional[dict],
    color_by: str,
):
    """Hand-rolled flow polygons with matplotlib. No hover; PNG-friendly."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    palette = _palette_for(class_grouping, palette)
    n_phases = len(phases)
    classes = _resolve_class_order(palette, class_order)
    if display_labels is None:
        display_labels = [sl.PHASE_DISPLAY_LABELS.get(p, p) for p in phases]
    if stage_labels is None:
        stage_labels = [sl.PHASE_DISPLAY_LABELS.get(p, p) for p in phases[1:]]
    if label_target not in {"columns", "ribbons", "both"}:
        raise ValueError(f"label_target must be 'columns'|'ribbons'|'both', got {label_target!r}")

    fig, ax = plt.subplots(figsize=(max(9, 1.8 * n_phases), 6.5))
    x_pos = np.linspace(0, 1, n_phases)

    node_y_top, total_tx = _layout_nodes_mpl(
        ax, tidy, phases, x_pos, palette, classes,
    )
    _draw_ribbons_mpl(
        ax, tidy, phases, x_pos, node_y_top, total_tx, palette, classes, color_by,
    )

    # Column (state) labels on the X axis
    if label_target in {"columns", "both"}:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_labels, fontsize=10, fontweight="bold",
                           rotation=20, ha="right")
    else:
        ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.05, 1.05)
    # Reserve extra headroom when ribbon labels are placed above ribbons
    top_ylim = 1.12 if label_target in {"ribbons", "both"} else 1.02
    ax.set_ylim(0.0, top_ylim)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=4)

    # Stage (action) labels above the ribbons, at midpoints between columns
    if label_target in {"ribbons", "both"}:
        for i, lbl in enumerate(stage_labels):
            x_mid = (x_pos[i] + x_pos[i + 1]) / 2
            ax.text(x_mid, 1.04, lbl, ha="center", va="bottom",
                    fontsize=10, fontweight="bold", style="italic")

    # Legend for entity classes — auto-filter to classes that actually
    # appear in tidy (otherwise the legend shows dead swatches for codes
    # the data never produces, e.g. component/dropped on modern SEG runs).
    present = set(tidy["class_from"]) | set(tidy["class_to"])
    legend_classes = [c for c in classes if c in present]
    legend_handles = [
        Patch(facecolor=palette[c], edgecolor="none",
              label=sl.CLASS_NAMES.get(c, str(c)))
        for c in legend_classes
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=max(1, len(legend_classes)), frameon=False, fontsize=10,
        title="Entity class", title_fontsize=10,
    )

    if title:
        fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return fig


def _save_figure(fig, output, backend: str) -> None:
    # Local import: `pathlib.Path` would clash with `matplotlib.path.Path`
    # used in the ribbon helpers if hoisted to module scope.
    from pathlib import Path
    p = Path(output)
    if backend == "matplotlib":
        fig.savefig(p, dpi=150, bbox_inches="tight")
    elif backend == "plotly":
        if p.suffix.lower() == ".html":
            fig.write_html(str(p))
        else:
            fig.write_image(str(p))  # requires kaleido for non-html
