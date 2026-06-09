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
