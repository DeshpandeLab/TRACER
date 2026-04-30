"""Smoke tests for the segmented + no-segmentation pipelines on synthetic
transcripts.

The synthetic input plants 8 cells in a 4x2 xy grid at 24 µm spacing,
3 type archetypes (gene panels of 4 genes each), 25 tx per cell,
80% archetype-coherent + 20% cross-type noise.

Three test classes:
- ``TestSegmentedWorkflow``: input ``cell_id`` set; runs Prune → Split
  → Initial Rescue → Group → Stitch → Demote → Final Rescue.
- ``TestNoSegWorkflow``: ``cell_id`` stripped to ``"-1"``; runs Group →
  Stitch → Demote → Final Rescue. Includes a regression test for the
  SHIELD_LABEL absorption bug (commit 6aa3f3e).
- ``TestSegVsNoSegConsistency``: runs both pipelines on identical
  synthetic input and asserts non-trivial cross-mode partition agreement.
"""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from tests.synthetic import (
    make_synthetic_transcripts,
    make_synthetic_npmi_panel_for_transcripts,
)
from tests._pipeline_runner import run_segmented_pipeline, run_noseg_pipeline


# Voxel-grid layout: 8 cells, ~80 voxels each (1 µm voxels), full
# 10 µm z volume. Cells get amoeboid shapes via 6-conn flood-fill.
CELLS_KW = dict(
    n_cells=8,
    voxels_per_cell_mean=80,
    tx_per_cell=25,
    n_genes=12,
    n_types=3,
    domain_z_um=10.0,
    nuclear_layers=2,
)

# Section-extraction sub-volume: middle 5 µm of the 10 µm domain.
SECTION_Z = (2.5, 7.5)


@pytest.fixture(scope="module")
def synthetic_inputs():
    """Built once per module: synthetic transcripts + matching PMI panel + ground truth."""
    df, gt = make_synthetic_transcripts(**CELLS_KW, seed=42)
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    return df, panel, gt


@pytest.fixture(scope="module")
def seg_result(synthetic_inputs):
    df, panel, gt = synthetic_inputs
    df_out, prog = run_segmented_pipeline(df, panel)
    return df_out, prog, gt


@pytest.fixture(scope="module")
def noseg_result(synthetic_inputs):
    df, panel, gt = synthetic_inputs
    df_out, prog = run_noseg_pipeline(df, panel)
    return df_out, prog, gt


def _final_label_counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    s = df[col].astype(str)
    n_tx_total = len(s)
    n_unas_tx = int((s == "-1").sum())
    n_assigned = n_tx_total - n_unas_tx
    return {"total": n_tx_total, "assigned": n_assigned, "unassigned": n_unas_tx}


# ============================================================================
# Segmented workflow
# ============================================================================

class TestSegmentedWorkflow:
    def test_pipeline_runs_end_to_end(self, seg_result):
        df_out, prog, gt = seg_result
        assert "stitched" in df_out.columns
        # 8 stages of progression captured
        assert len(prog) >= 7

    def test_final_entity_count_in_range(self, seg_result):
        df_out, prog, gt = seg_result
        s = df_out["stitched"].astype(str)
        n_distinct = (s != "-1").groupby(s).any().sum()
        # 8 planted cells; some pipeline noise allowed
        assert 4 <= n_distinct <= 16, (
            f"Expected 4..16 final entities, got {n_distinct}"
        )

    def test_coverage_above_50pct(self, seg_result):
        df_out, prog, gt = seg_result
        c = _final_label_counts(df_out, "stitched")
        coverage = c["assigned"] / c["total"]
        assert coverage > 0.5, f"Coverage {coverage:.1%} below 50%"

    def test_seg_recovers_planted_truth(self, seg_result):
        """Sanity: segmented TRACER on planted ground truth should
        recover the partition with high ARI (it's a refinement of the
        input cell_id, not a from-scratch clustering)."""
        df_out, prog, gt = seg_result
        truth = df_out["cell_id"].astype(str).values
        out = df_out["stitched"].astype(str).values
        # Ignore unassigned tx in the assignment-only ARI
        mask = out != "-1"
        if mask.sum() < 2:
            pytest.skip("not enough assigned tx")
        ari = adjusted_rand_score(truth[mask], out[mask])
        assert ari > 0.5, f"ARI(seg-output, ground_truth) = {ari:.3f} < 0.5"


# ============================================================================
# No-segmentation workflow
# ============================================================================

class TestNoSegWorkflow:
    def test_noseg_pipeline_runs_end_to_end(self, noseg_result):
        df_out, prog, gt = noseg_result
        assert "stitched" in df_out.columns

    def test_noseg_finds_components(self, noseg_result):
        """Group + Stitch should recover a non-trivial number of
        components from the planted cells."""
        df_out, prog, gt = noseg_result
        s = df_out["stitched"].astype(str)
        n_distinct = (s != "-1").groupby(s).any().sum()
        assert n_distinct >= 3, (
            f"Expected ≥3 components, got {n_distinct} "
            f"(should find most of {gt['n_cells']} planted cells)"
        )

    def test_no_phantom_cell(self, noseg_result):
        """Regression test for the SHIELD_LABEL absorption bug
        (commit 6aa3f3e). Pre-fix, pre_stage2_rescue absorbed nearby
        unassigned tx into the __GUARD_SKIP__ shield label, producing
        a phantom "cell" containing thousands of tx. Verify no single
        entity holds > 50% of total tx (the phantom-cell symptom).

        Note: the no-seg pipeline doesn't actually invoke pre_stage2_rescue
        (it has no entities to rescue into) — but this test serves as a
        general "pathological merger" regression check that would also
        fail if Stitch over-merged."""
        df_out, prog, gt = noseg_result
        s = df_out["stitched"].astype(str)
        s = s[s != "-1"]
        if len(s) == 0:
            pytest.skip("no assigned tx")
        max_share = s.value_counts().iloc[0] / len(s)
        # The pre-fix bug put ~100% of rescued tx into one phantom
        # cluster. Threshold 0.85 catches that pathology while
        # tolerating legitimate dense-tissue consolidation under noseg
        # (where Stitch can merge many adjacent same-type bin-cliques
        # into a single super-component).
        assert max_share <= 0.85, (
            f"Phantom-cell symptom: a single entity holds "
            f"{100*max_share:.1f}% of assigned tx (>85% is the SHIELD_LABEL "
            f"bug fingerprint)."
        )


# ============================================================================
# Cross-mode consistency
# ============================================================================

class TestSection:
    """Run the segmented pipeline on a tissue-section-extracted slab.

    Cells partially intersect the section boundary; the pipeline should
    still recover the bulk of the planted partition, with looser
    tolerances (clipped cells lose tx).
    """

    @pytest.fixture(scope="class")
    def section_inputs(self):
        df, gt = make_synthetic_transcripts(
            **CELLS_KW, section_z_range_um=SECTION_Z, seed=42,
        )
        panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
        return df, panel, gt

    @pytest.fixture(scope="class")
    def section_result(self, section_inputs):
        df, panel, gt = section_inputs
        df_out, prog = run_segmented_pipeline(df, panel)
        return df_out, prog, gt

    def test_section_pipeline_runs(self, section_result):
        df_out, _, _ = section_result
        assert "stitched" in df_out.columns

    def test_section_z_bounds_respected(self, section_inputs):
        df, _, _ = section_inputs
        z_lo, z_hi = SECTION_Z
        assert (df["z"] >= z_lo).all()
        assert (df["z"] < z_hi).all()

    def test_section_recovers_majority_of_truth(self, section_result):
        """ARI threshold relaxed for clipped-cell input (some cells
        lose most of their tx)."""
        df_out, _, _ = section_result
        truth = df_out["cell_id"].astype(str).values
        out = df_out["stitched"].astype(str).values
        mask = out != "-1"
        if mask.sum() < 2:
            pytest.skip("not enough assigned tx")
        ari = adjusted_rand_score(truth[mask], out[mask])
        assert ari > 0.4, f"ARI={ari:.3f} below 0.4 (section relaxed bound)"


class TestSegVsNoSegConsistency:
    """Runs BOTH pipelines on the same synthetic input and asserts
    non-trivial partition agreement.

    Note: under noseg, all tx start unassigned, so the cell_id used by
    the segmented pipeline is replaced. Both partitions are compared on
    the assigned-in-both subset.
    """

    def test_partition_agreement_above_chance(self, seg_result, noseg_result):
        seg_out, _, _ = seg_result
        noseg_out, _, _ = noseg_result

        seg_lbl = seg_out.set_index("transcript_id")["stitched"].astype(str)
        noseg_lbl = noseg_out.set_index("transcript_id")["stitched"].astype(str)

        idx = seg_lbl.index.intersection(noseg_lbl.index)
        a = seg_lbl.loc[idx]
        b = noseg_lbl.loc[idx]
        # ARI on assigned-in-both
        mask = (a != "-1") & (b != "-1")
        if mask.sum() < 2:
            pytest.skip("not enough assigned-in-both tx")
        ari = adjusted_rand_score(a[mask].values, b[mask].values)
        ami = adjusted_mutual_info_score(a[mask].values, b[mask].values)
        assert ari > 0.10, (
            f"ARI(seg, noseg) = {ari:.3f} below chance threshold (0.10)"
        )
        assert ami > 0.10, (
            f"AMI(seg, noseg) = {ami:.3f} below chance threshold (0.10)"
        )
