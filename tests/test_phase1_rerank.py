"""Unit tests for `_phase1_rerank_within_parent`.

The function re-ranks depth-1 entities under each parent cell_id by
nuclear-tx count and promotes the largest to the main `{cell_id}` label.
Pure relabeling; no tx demotion, no coordinate changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests._pipeline_runner import _phase1_rerank_within_parent


def _df(rows: list[tuple]) -> pd.DataFrame:
    """Build a minimal test frame: rows of (entity, cell_id, nuclear)."""
    return pd.DataFrame(
        rows, columns=["tracer_id", "cell_id", "overlaps_nucleus"]
    )


def test_no_partials_is_noop():
    """One depth-1 entity under parent → no relabel."""
    df = _df([
        ("42", "42", True),
        ("42", "42", True),
        ("42", "42", True),
    ])
    out, stats = _phase1_rerank_within_parent(
        df, entity_col="tracer_id", cell_id_col="cell_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    assert (out["tracer_id"] == df["tracer_id"]).all()
    assert stats["n_parents_reranked"] == 0
    assert stats["n_tx_relabeled"] == 0
