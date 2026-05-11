"""Lock-down + speedup tests for the vectorized rewrite of `_reassign_nuclear_post_1c`.

The strategy:
  - `_reassign_nuclear_post_1c_legacy` is the current (Python-loop) implementation,
    kept as a fallback reference.
  - `_reassign_nuclear_post_1c` is the new vectorized version.

Tests:
  1. On a small synthetic case, both versions return byte-identical labels
     and stats.
  2. On the NW 500 µm ROI (real data), both versions return byte-identical
     labels and stats. The vectorized version must be ≥ 2× faster than
     legacy on this input.

The "byte-identical" requirement matters because reassign's existing
behavior is encoded in downstream benchmarks; if we change semantics we
invalidate those.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _build_synthetic_case() -> tuple[pd.DataFrame, dict]:
    """A two-parent toy case with known reassign behavior.

    Parent 42:
      - main `42` has 3 nuclear tx of genes GA, GB, GC (the "alveolar" program).
      - partial `42-1` has 3 nuclear tx of genes GX, GY, GZ (the "immune" program).
      - Plus one extra nuclear tx of GX in main (this is the candidate for reassign).

    Parent 99 has only a main, no partials → reassign is a no-op for it.

    W matrix is constructed so mean_pmi(GX, S_partial \\ {GX}) > mean_pmi(GX, S_main) + margin.
    """
    rows = [
        # parent 42 main: GA, GB, GC + GX (the candidate)
        ("42",   "42", "GA", True),
        ("42",   "42", "GB", True),
        ("42",   "42", "GC", True),
        ("42",   "42", "GX", True),  # ← candidate for reassign to 42-1
        # parent 42 partial: GX, GY, GZ
        ("42-1", "42", "GX", True),
        ("42-1", "42", "GY", True),
        ("42-1", "42", "GZ", True),
        # parent 99 main only
        ("99",   "99", "GA", True),
        ("99",   "99", "GB", True),
    ]
    df = pd.DataFrame(rows, columns=["tracer_id", "cell_id", "feature_name", "overlaps_nucleus"])

    # gene indexing
    genes = ["GA", "GB", "GC", "GX", "GY", "GZ"]
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)

    # PMI matrix: alveolar program (GA, GB, GC) has high within-PMI;
    # immune program (GX, GY, GZ) has high within-PMI;
    # GX vs alveolar is mid-low; GX vs immune is high → triggers reassign.
    W = np.full((n, n), 0.0, dtype=np.float32)
    # alveolar
    for a in ["GA", "GB", "GC"]:
        for b in ["GA", "GB", "GC"]:
            if a != b:
                W[gene_to_idx[a], gene_to_idx[b]] = 0.5
    # immune
    for a in ["GX", "GY", "GZ"]:
        for b in ["GX", "GY", "GZ"]:
            if a != b:
                W[gene_to_idx[a], gene_to_idx[b]] = 0.6
    # cross-program: low
    for a in ["GA", "GB", "GC"]:
        for b in ["GX", "GY", "GZ"]:
            W[gene_to_idx[a], gene_to_idx[b]] = 0.05
            W[gene_to_idx[b], gene_to_idx[a]] = 0.05

    aux = {"W": W, "gene_to_idx": gene_to_idx}
    return df, aux


# ---------------------------------------------------------------------------
# Synthetic equivalence
# ---------------------------------------------------------------------------


def test_synthetic_byte_identical_output():
    """Legacy and vectorized produce the same labels on a small case."""
    from tests._pipeline_runner import (
        _reassign_nuclear_post_1c,
        _reassign_nuclear_post_1c_legacy,
    )
    df, aux = _build_synthetic_case()

    out_legacy, stats_legacy = _reassign_nuclear_post_1c_legacy(
        df, entity_col="tracer_id", aux=aux,
    )
    out_vec, stats_vec = _reassign_nuclear_post_1c(
        df, entity_col="tracer_id", aux=aux,
    )

    # Labels match exactly
    assert (out_legacy["tracer_id"].to_numpy() == out_vec["tracer_id"].to_numpy()).all(), (
        "Vectorized output diverges from legacy on synthetic case"
    )
    # Stats match
    assert stats_legacy["n_tx_moved"] == stats_vec["n_tx_moved"]
    assert stats_legacy["n_parents_with_partials"] == stats_vec["n_parents_with_partials"]

    # Sanity: the GX-in-main row reassigned to 42-1
    assert stats_legacy["n_tx_moved"] == 1, (
        f"Expected 1 reassign; got {stats_legacy['n_tx_moved']}"
    )


def test_synthetic_no_partials_is_noop():
    """Parent with no partials → no reassignment."""
    from tests._pipeline_runner import (
        _reassign_nuclear_post_1c,
        _reassign_nuclear_post_1c_legacy,
    )
    # Reuse synthetic; isolate parent 99 (no partials)
    df, aux = _build_synthetic_case()
    df_99 = df[df["cell_id"] == "99"].reset_index(drop=True)

    out_legacy, stats_legacy = _reassign_nuclear_post_1c_legacy(
        df_99, entity_col="tracer_id", aux=aux,
    )
    out_vec, stats_vec = _reassign_nuclear_post_1c(
        df_99, entity_col="tracer_id", aux=aux,
    )

    assert (out_legacy["tracer_id"].to_numpy() == out_vec["tracer_id"].to_numpy()).all()
    assert stats_legacy["n_tx_moved"] == 0
    assert stats_vec["n_tx_moved"] == 0


def test_unmappable_gene_index_handled():
    """Tx with gene index = -1 (gene not in panel) is skipped, no crash."""
    from tests._pipeline_runner import (
        _reassign_nuclear_post_1c,
        _reassign_nuclear_post_1c_legacy,
    )
    df, aux = _build_synthetic_case()
    # Inject a tx with an unmappable gene name
    extra = pd.DataFrame([("42", "42", "GZZZ_UNKNOWN", True)],
                         columns=["tracer_id", "cell_id", "feature_name", "overlaps_nucleus"])
    df2 = pd.concat([df, extra], ignore_index=True)

    out_legacy, stats_legacy = _reassign_nuclear_post_1c_legacy(
        df2, entity_col="tracer_id", aux=aux,
    )
    out_vec, stats_vec = _reassign_nuclear_post_1c(
        df2, entity_col="tracer_id", aux=aux,
    )
    assert (out_legacy["tracer_id"].to_numpy() == out_vec["tracer_id"].to_numpy()).all()
    assert stats_legacy["n_tx_moved"] == stats_vec["n_tx_moved"]


# ---------------------------------------------------------------------------
# ROI equivalence + speed
# ---------------------------------------------------------------------------


_ROI_BBOX = (850.0, 1350.0, 2950.0, 3450.0)  # NW 500 µm
_PANEL = Path("/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/lung_cancer/data/pmi_bs_lung_cancer_C_5_95.csv")
_PARQUET = Path("/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/lung_cancer/data/lung_cancer_df.parquet")


@pytest.mark.skipif(not _PANEL.exists() or not _PARQUET.exists(),
                     reason="Local Xenium dataset not available in this checkout.")
def test_roi_byte_identical_labels_and_speed():
    """On the NW ROI:
    1. Legacy and vectorized produce byte-identical labels.
    2. Vectorized is at least 2× faster.
    """
    from tests._pipeline_runner import (
        _reassign_nuclear_post_1c,
        _reassign_nuclear_post_1c_legacy,
    )
    from tracer.pruning import prune_transcripts_fast

    # Load ROI
    df = pd.read_parquet(_PARQUET)
    x_lo, x_hi, y_lo, y_hi = _ROI_BBOX
    mask = ((df["x"] >= x_lo) & (df["x"] < x_hi) &
            (df["y"] >= y_lo) & (df["y"] < y_hi))
    df = df.loc[mask].reset_index(drop=True)
    panel = pd.read_csv(_PANEL)

    # Phase 1
    df_pruned, aux = prune_transcripts_fast(
        df, panel, cell_id_col="cell_id", gene_col="feature_name",
        threshold=0.05, unassigned_id="-1",
        nan_fill=0.0, n_jobs=-1, show_progress=False,
    )

    # Run both
    t0 = time.perf_counter()
    out_legacy, stats_legacy = _reassign_nuclear_post_1c_legacy(
        df_pruned, entity_col="tracer_id", aux=aux,
    )
    t_legacy = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_vec, stats_vec = _reassign_nuclear_post_1c(
        df_pruned, entity_col="tracer_id", aux=aux,
    )
    t_vec = time.perf_counter() - t0

    speedup = t_legacy / max(t_vec, 1e-9)
    print(f"\n[roi-eq] legacy={t_legacy:.3f}s  vec={t_vec:.3f}s  speedup={speedup:.1f}×")
    print(f"[roi-eq] legacy moves={stats_legacy['n_tx_moved']}  vec moves={stats_vec['n_tx_moved']}")

    # Byte-identical labels
    assert (out_legacy["tracer_id"].to_numpy() == out_vec["tracer_id"].to_numpy()).all(), (
        "ROI: vectorized output differs from legacy"
    )
    assert stats_legacy["n_tx_moved"] == stats_vec["n_tx_moved"]
    assert stats_legacy["n_parents_with_partials"] == stats_vec["n_parents_with_partials"]

    # Speed gate (≥ 2× on ROI; will likely be much more on full tissue)
    assert speedup >= 2.0, f"Speedup {speedup:.1f}× below 2× target on ROI"
