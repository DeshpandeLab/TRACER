"""Unit tests for `tracer._etype` foundation helpers.

These tests verify the foundation alone — no stage emitters yet.
Parity vs the legacy `infer_entity_type` parser is checked on integer
cell_ids; on dash-containing FFPE-style cell_ids the helper
*intentionally* reproduces the legacy bug (so it serves as a
regression baseline for the column-based emitters that follow).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tracer._etype import (
    ETYPE_CATEGORIES,
    ETYPE_DTYPE,
    empty_etype,
    etype_from_codes,
    infer_etype_from_label,
    infer_entity_type_etype,
)


def test_categories_canonical():
    assert ETYPE_CATEGORIES == ["cell", "partial", "component", "drop", "unknown"]


def test_dtype_uses_canonical_categories():
    assert list(ETYPE_DTYPE.categories) == ETYPE_CATEGORIES
    assert not ETYPE_DTYPE.ordered


def test_empty_etype():
    e = empty_etype(5)
    assert isinstance(e, pd.Categorical)
    assert e.dtype == ETYPE_DTYPE
    assert (np.asarray(e) == "unknown").all()


def test_etype_from_codes_basic():
    codes = np.array([0, 1, 2, 0, 1], dtype=np.int8)
    e = etype_from_codes(codes)
    assert list(np.asarray(e)) == ["cell", "partial", "unknown", "cell", "partial"]


def test_etype_from_codes_fallback_maps_to_unknown():
    codes = np.array([0, 1, 2, 3], dtype=np.int8)
    e = etype_from_codes(codes)
    # Codes 2 and 3 both → unknown
    assert list(np.asarray(e)) == ["cell", "partial", "unknown", "unknown"]


def test_infer_etype_from_label_integer_ids():
    labels = pd.Series(["42", "42-1", "42-1-1", "UNASSIGNED_3", "-1", "DROP", "nan"])
    e = infer_etype_from_label(labels)
    assert list(np.asarray(e)) == [
        "cell", "partial", "partial", "component", "unknown", "unknown", "unknown"
    ]


def test_infer_etype_from_label_rejected_sentinels():
    labels = pd.Series(["prune_rejected", "group_rejected", "demote_rejected"])
    e = infer_etype_from_label(labels)
    assert list(np.asarray(e)) == ["unknown", "unknown", "unknown"]


def test_infer_etype_from_label_ffpe_dash_in_cell_id_documents_legacy_bug():
    """PDAC-style alphanumeric cell_id with native `-1` suffix.

    The legacy parsing rule misclassifies the main as a partial because
    the cell_id contains a dash. This test DOCUMENTS the legacy bug so
    that stage emitters using kernel codes (which avoid the bug
    entirely) can verify they produce CORRECT classifications even on
    these labels.
    """
    labels = pd.Series(["adohnpem-1", "adohnpem-1-1"])
    e = infer_etype_from_label(labels)
    # Legacy says both are "partial" (the bug — `adohnpem-1` is really
    # a main, but the parser sees a dash and calls it a partial).
    assert list(np.asarray(e)) == ["partial", "partial"]


def test_concat_preserves_dtype():
    a = pd.Series(empty_etype(3))
    b = pd.Series(pd.Categorical(["cell", "partial", "unknown"], dtype=ETYPE_DTYPE))
    c = pd.concat([a, b]).reset_index(drop=True)
    assert c.dtype == ETYPE_DTYPE


def test_can_assign_categorical_value_via_loc():
    df = pd.DataFrame({"x": [1, 2, 3]})
    df["_etype"] = empty_etype(3)
    df.loc[df["x"] == 2, "_etype"] = "cell"
    assert df["_etype"].dtype == ETYPE_DTYPE
    assert list(df["_etype"].astype(str)) == ["unknown", "cell", "unknown"]


def test_invalid_category_assignment_raises_or_becomes_nan():
    """Assigning a string not in ETYPE_CATEGORIES should NOT silently
    produce a wrong category. pandas raises TypeError or sets NaN."""
    df = pd.DataFrame({"_etype": empty_etype(2)})
    # pandas behavior on out-of-category assignment differs by version;
    # whichever happens, the value must NOT silently become a valid
    # different category.
    try:
        df.loc[0, "_etype"] = "not_a_real_category"
    except (TypeError, ValueError):
        return  # acceptable: pandas refused
    # If pandas allowed the assignment, it must have NaN'd it.
    assert pd.isna(df["_etype"].iloc[0]) or str(df["_etype"].iloc[0]) != "cell"


def test_infer_entity_type_etype_reads_column():
    df = pd.DataFrame({
        "tracer_id": ["42", "42-1", "UNASSIGNED_7"],
        "_etype": pd.Categorical(
            ["cell", "partial", "component"], dtype=ETYPE_DTYPE
        ),
    })
    kinds = infer_entity_type_etype(df)
    assert list(kinds) == ["cell", "partial", "component"]


# ---------------------------------------------------------------------------
# Step 2 — Phase 1 emitter parity
# ---------------------------------------------------------------------------


def test_phase1_emitter_writes_etype_consistent_with_labels():
    """On integer cell_ids, the Phase 1 emitter's `_etype` column must
    agree with `infer_etype_from_label` applied to the produced label
    column. This is the parity gate: if it diverges on integer
    cell_ids, the emitter is buggy. (On FFPE/IO dash-containing
    cell_ids, the emitter would be CORRECT and the parity helper would
    be WRONG — the inversion that motivates the refactor — but we
    don't exercise that case here; that's the job of the PDAC
    re-bench in Step 4.)
    """
    from tests.synthetic import (
        make_synthetic_transcripts,
        make_synthetic_npmi_panel_for_transcripts,
    )
    from tracer.pruning import prune_transcripts_fast

    df, gt = make_synthetic_transcripts(
        n_cells=10, n_types=2, seed=42,
    )
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    df_out, _aux = prune_transcripts_fast(
        df.copy(), panel,
        cell_id_col="cell_id", gene_col="feature_name",
        threshold=0.05, unassigned_id="-1",
        nan_fill=0.0, n_jobs=-1, show_progress=False,
    )
    # `_etype` column is present and a Categorical
    assert "_etype" in df_out.columns
    assert df_out["_etype"].dtype == ETYPE_DTYPE

    # Parity against legacy label parsing on integer cell_ids.
    # The label column produced by prune_transcripts_fast is `tracer_id`
    # by default (or whatever `out_col` is set to). Check the columns
    # to find it.
    label_cols = [c for c in df_out.columns if c in ("tracer_id", "out", "_out")]
    # Default out_col in the package is `tracer_id`; if that's missing,
    # fall back to whatever the function actually wrote. Worst case
    # we'll find no match and the test surfaces a real bug.
    assert label_cols, f"expected a label column; got {list(df_out.columns)}"
    label_col = label_cols[0]

    legacy_kinds = infer_etype_from_label(df_out[label_col])
    new_kinds = df_out["_etype"]
    # Compare as strings (Categorical equality requires same dtype)
    assert (
        np.asarray(legacy_kinds).astype(str) == np.asarray(new_kinds).astype(str)
    ).all(), (
        "Phase 1 _etype emitter diverges from legacy label classification "
        "on integer cell_ids — this is the parity gate failure."
    )


def test_phase1_nuclear_seed_path_writes_etype_from_kernel_codes():
    """The Cython-batched nuclear-seed prune path (production for
    Xenium FFPE / IO) writes `_etype` directly from kernel codes,
    bypassing label-string parsing. On integer cell_ids the result
    must agree with `infer_etype_from_label` applied to the label
    column (parity gate); the bug-free behavior on FFPE cell_ids is
    exercised separately in the PDAC re-bench (Step 4)."""
    from tests.synthetic import (
        make_synthetic_transcripts,
        make_synthetic_npmi_panel_for_transcripts,
    )
    from tracer.pruning import prune_transcripts_nuclear_seed

    df, gt = make_synthetic_transcripts(
        n_cells=10, n_types=2, seed=42,
    )
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    # The nuclear-seed path requires the column the runner expects.
    df = df.rename(columns={"is_nuclear": "overlaps_nucleus"})

    df_out, _aux = prune_transcripts_nuclear_seed(
        df.copy(), panel,
        cell_id_col="cell_id", gene_col="feature_name",
        nuclear_col="overlaps_nucleus",
        threshold=0.05, unassigned_id="-1",
        metric_col="NPMI",  # synthetic panel uses NPMI column name
        nan_fill=0.0, n_jobs=-1, show_progress=False,
    )
    assert "_etype" in df_out.columns
    assert df_out["_etype"].dtype == ETYPE_DTYPE

    label_cols = [c for c in df_out.columns if c == "tracer_id"]
    assert label_cols, f"expected tracer_id; got {list(df_out.columns)}"
    label_col = label_cols[0]

    legacy_kinds = infer_etype_from_label(df_out[label_col])
    new_kinds = df_out["_etype"]
    assert (
        np.asarray(legacy_kinds).astype(str) == np.asarray(new_kinds).astype(str)
    ).all(), (
        "Nuclear-seed _etype emitter diverges from legacy classification "
        "on integer cell_ids — kernel-code mapping is buggy."
    )
