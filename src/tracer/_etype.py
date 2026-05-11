"""Entity-type categorical column — canonical kind classification.

Replaces label-string parsing (see `stitching.infer_entity_type`) as the
canonical mechanism for asking "what kind of entity is this row".

The `_etype` column is populated by every stage that emits or transforms
entities; readers consume it directly via `infer_entity_type_etype` and
related sibling helpers, without parsing the label.

Categories (string-valued for readability; backed by uint8 codes):
  - ``cell``       — main entity for an input cell_id (or a cascade main).
  - ``partial``    — sub-seed emitted by Phase 1c, or a cascade partial.
  - ``component``  — UNASSIGNED_<n> (legacy spatial-CC Group fallback) or
                     similar pseudo-cells.
  - ``drop``       — explicitly demoted entity. Reserved; not produced
                     by any stage today but kept for symmetry.
  - ``unknown``    — unassigned tx or unrecognized; sentinel values
                     like "-1", "DROP", "UNASSIGNED", "nan", "*_rejected".

Memory: 5 categories → uint8 codes; 20.7M tx × 1 byte ≈ 20 MB. Negligible.

See `docs/superpowers/specs/2026-05-11-etype-column-design.md` for the
full migration plan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ETYPE_CATEGORIES: list[str] = ["cell", "partial", "component", "drop", "unknown"]

ETYPE_DTYPE: pd.CategoricalDtype = pd.CategoricalDtype(
    categories=ETYPE_CATEGORIES, ordered=False
)


def empty_etype(n: int) -> pd.Categorical:
    """Build an all-`unknown` etype column of length ``n``."""
    return pd.Categorical(["unknown"] * n, dtype=ETYPE_DTYPE)


def etype_from_codes(codes: np.ndarray) -> pd.Categorical:
    """Map Cython per-tx codes from ``prune_cells_nuclear_seed`` to etypes.

    Codes returned by the kernel:
      0 = main             → ``cell``
      1 = partial          → ``partial``
      2 = unassigned       → ``unknown``
      3 = fallback-needed  → ``unknown`` (caller handles fallback path)
    """
    cat_codes = np.full(
        codes.shape, ETYPE_CATEGORIES.index("unknown"), dtype=np.int8
    )
    cat_codes[codes == 0] = ETYPE_CATEGORIES.index("cell")
    cat_codes[codes == 1] = ETYPE_CATEGORIES.index("partial")
    return pd.Categorical.from_codes(cat_codes, dtype=ETYPE_DTYPE)


def infer_etype_from_label(labels) -> pd.Categorical:
    """Parity helper: classify a label series via the same rules as
    `stitching.infer_entity_type`. Used during migration to verify
    stage emitters produce a column consistent with legacy parsing
    *on integer cell_ids*.

    On dash-containing cell_ids (Xenium FFPE / IO), the legacy rule
    misclassifies mains as partials — this helper preserves that
    behavior intentionally so it can be used as a regression baseline.
    The bug is fixed in production by stage emitters that write the
    correct `_etype` directly from kernel codes / stage semantics,
    not by changing the parsing rule here.

    Categories returned:
      - sentinels (``-1``, ``DROP``, ``UNASSIGNED``, ``nan``,
        ``*_rejected``) → ``unknown``
      - starts with ``UNASSIGNED_``                       → ``component``
      - contains ``-``                                    → ``partial``
      - else                                              → ``cell``
    """
    s = pd.Series(labels).astype(str).reset_index(drop=True)
    out = np.full(len(s), "unknown", dtype=object)

    is_sentinel = s.isin({"-1", "DROP", "UNASSIGNED", "nan"}) | s.str.endswith(
        "_rejected"
    )
    is_component = ~is_sentinel & s.str.startswith("UNASSIGNED_")
    is_partial = (
        ~is_sentinel & ~is_component & s.str.contains("-", regex=False)
    )
    is_cell = ~is_sentinel & ~is_component & ~is_partial

    out[is_sentinel.to_numpy()] = "unknown"
    out[is_component.to_numpy()] = "component"
    out[is_partial.to_numpy()] = "partial"
    out[is_cell.to_numpy()] = "cell"

    return pd.Categorical(out, dtype=ETYPE_DTYPE)


def infer_entity_type_etype(
    df: pd.DataFrame, type_col: str = "_etype"
) -> pd.Series:
    """Sibling reader: return entity kind from the ``_etype`` column.

    Drop-in for the label-parsing ``stitching.infer_entity_type`` at
    call sites that have access to the DataFrame. Returns a string
    Series with the same vocabulary as the legacy helper.
    """
    return df[type_col].astype(str)
