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
        # etype == "partial" is the default fill — no assignment needed
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
