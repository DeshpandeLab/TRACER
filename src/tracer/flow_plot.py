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
        for (ph_from, ph_to), group in tidy.groupby(["phase_from", "phase_to"]):
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


# ─── plotly backend (stub — implemented in Task 6) ──────────────────────
def _render_plotly(*args, **kwargs):
    """Stub — implemented in Task 6."""
    raise NotImplementedError("_render_plotly is added in Task 6")


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
    # 1) Compute per-phase per-class totals from tidy
    per_phase_totals = {}
    for i, p in enumerate(phases):
        if i == 0:
            grp = tidy[tidy["phase_from"] == p].groupby("class_from")["n"].sum()
        else:
            grp = tidy[tidy["phase_to"] == p].groupby("class_to")["n"].sum()
        per_phase_totals[p] = grp.to_dict()
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

    for i in range(len(phases) - 1):
        p_from, p_to = phases[i], phases[i + 1]
        sub = tidy[(tidy["phase_from"] == p_from) &
                   (tidy["phase_to"] == p_to)]
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
    title: Optional[str],
    class_grouping: str,
    palette: Optional[dict],
    color_by: str,
):
    """Hand-rolled flow polygons with matplotlib. No hover; PNG-friendly."""
    import matplotlib.pyplot as plt

    palette = _palette_for(class_grouping, palette)
    n_phases = len(phases)
    classes = sorted(palette.keys())

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * n_phases), 6))
    x_pos = np.linspace(0, 1, n_phases)

    node_y_top, total_tx = _layout_nodes_mpl(
        ax, tidy, phases, x_pos, palette, classes,
    )
    _draw_ribbons_mpl(
        ax, tidy, phases, x_pos, node_y_top, total_tx, palette, classes, color_by,
    )

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
