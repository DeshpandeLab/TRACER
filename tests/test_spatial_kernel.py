"""Unit tests for tracer.spatial_kernel.

Builds tiny synthetic tile-binned tx populations and verifies:
  - parse_xy_offsets / parse_xy_half_offsets correctness for all
    supported spec strings;
  - build_grid_index basic properties (n_tx, entity_n_tx);
  - enumerate_pair_witnesses output for a hand-computable layout;
  - neighbor_entities returns the right entities + bins.

Bit-equivalence with stitching.py's existing inline enumeration is
verified separately in the regression test (test_pipeline_smoke /
test_pipeline_regression) once the kernel is wired in. This file
covers the kernel-internal correctness only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tracer.graph import bin_xy
from tracer.spatial_kernel import (
    GridIndex,
    build_grid_index,
    enumerate_pair_witnesses,
    neighbor_entities,
    parse_xy_half_offsets,
    parse_xy_offsets,
)


# ---------------------------------------------------------------------
# parse_xy_offsets
# ---------------------------------------------------------------------
class TestParseXyOffsets:
    def test_zero(self):
        assert parse_xy_offsets("0") == ()

    def test_four(self):
        full = parse_xy_offsets("4")
        assert set(full) == {(1, 0), (-1, 0), (0, 1), (0, -1)}

    def test_eight(self):
        full = parse_xy_offsets("8")
        assert len(full) == 8
        assert (0, 0) not in full
        # All cells of the 3x3 ring except center
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx, dy) != (0, 0):
                    assert (dx, dy) in full

    def test_R2(self):
        full = parse_xy_offsets("R2")
        # 5x5 ring minus center = 24 cells
        assert len(full) == 24
        assert (0, 0) not in full
        # Includes Moore-1 plus Moore-2-only
        assert (2, 2) in full
        assert (-2, -2) in full

    def test_R3(self):
        full = parse_xy_offsets("R3")
        # 7x7 - 1 = 48 cells
        assert len(full) == 48

    def test_R1_equivalent_to_8(self):
        assert set(parse_xy_offsets("R1")) == set(parse_xy_offsets("8"))

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_xy_offsets("bad")
        with pytest.raises(ValueError):
            parse_xy_offsets("R0")
        with pytest.raises(ValueError):
            parse_xy_offsets("Rabc")


class TestParseXyHalfOffsets:
    def test_eight_has_4_half(self):
        half = parse_xy_half_offsets("8")
        assert len(half) == 4
        assert set(half) == {(0, 1), (1, -1), (1, 0), (1, 1)}

    def test_R2_has_12_half(self):
        # 24 full offsets → 12 half-plane offsets
        half = parse_xy_half_offsets("R2")
        assert len(half) == 12
        # Every (dx, dy) must satisfy dx > 0 OR (dx == 0 AND dy > 0)
        for (dx, dy) in half:
            assert dx > 0 or (dx == 0 and dy > 0)


# ---------------------------------------------------------------------
# build_grid_index
# ---------------------------------------------------------------------
class TestBuildGridIndex:
    def test_basic_counts(self):
        # 6 tx in 4 bins, 3 entities
        coords = np.array([
            [0.5, 0.5, 1.0],   # bin (0, 0), z=1   entity 0
            [0.6, 0.6, 1.0],   # bin (0, 0), z=1   entity 0
            [1.5, 0.5, 1.0],   # bin (1, 0), z=1   entity 0
            [0.5, 1.5, 1.0],   # bin (0, 1), z=1   entity 1
            [1.5, 1.5, 1.0],   # bin (1, 1), z=1   entity 1
            [0.5, 0.5, 2.0],   # bin (0, 0), z=2   entity 2
        ], dtype=np.float64)
        codes = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
        idx = build_grid_index(coords, codes, G_xy=1.0, G_z=1.0)

        assert idx.n_tx == 6
        assert idx.n_entities == 3
        assert idx.entity_n_tx.tolist() == [3, 2, 1]
        # bc_grouped should have 5 rows (4 unique xy-bins, but bin (0,0)
        # split into z=1 and z=2, and entity 0 has 2 tx in (0,0,z=1))
        # Total unique (bin_xy, bin_z, comp): (0,0,z=1, e=0)=2, (1,0,z=1, e=0)=1,
        # (0,1,z=1, e=1)=1, (1,1,z=1, e=1)=1, (0,0,z=2, e=2)=1
        assert len(idx.bc_grouped) == 5
        assert idx.bc_grouped["n_tx"].sum() == 6

    def test_skips_negative_entity_codes(self):
        coords = np.array([[0.5, 0.5], [1.5, 1.5]], dtype=np.float64)
        codes = np.array([0, -1], dtype=np.int64)
        idx = build_grid_index(coords, codes, G_xy=1.0)
        assert idx.n_tx == 1
        assert idx.n_entities == 1
        assert idx.entity_n_tx.tolist() == [1]

    def test_empty_input(self):
        coords = np.zeros((0, 3))
        codes = np.zeros(0, dtype=np.int64)
        idx = build_grid_index(coords, codes, G_xy=1.0, G_z=1.0)
        assert idx.n_tx == 0
        assert idx.n_entities == 0
        assert idx.bc_grouped.empty

    def test_2d_index(self):
        coords = np.array([[0.5, 0.5], [1.5, 1.5]], dtype=np.float64)
        codes = np.array([0, 1], dtype=np.int64)
        idx = build_grid_index(coords, codes, G_xy=1.0, G_z=None)
        assert idx.G_z is None
        assert (idx.transcript_bin_z == 0).all()

    def test_input_validation(self):
        coords = np.array([[0.5]])  # only 1 column
        codes = np.array([0])
        with pytest.raises(ValueError, match="coords must be"):
            build_grid_index(coords, codes, G_xy=1.0)

        with pytest.raises(ValueError, match="G_xy must be"):
            build_grid_index(np.array([[0.5, 0.5]]), np.array([0]), G_xy=0)

        with pytest.raises(ValueError, match="G_z is set"):
            build_grid_index(
                np.array([[0.5, 0.5]]), np.array([0]),
                G_xy=1.0, G_z=1.0,
            )


# ---------------------------------------------------------------------
# enumerate_pair_witnesses
# ---------------------------------------------------------------------
class TestEnumeratePairWitnesses:
    def test_same_bin_pair(self):
        # Two entities sharing one bin → one (0, 1) pair
        coords = np.array([[0.5, 0.5], [0.6, 0.6]], dtype=np.float64)
        codes = np.array([0, 1], dtype=np.int64)
        idx = build_grid_index(coords, codes, G_xy=1.0)
        pairs = enumerate_pair_witnesses(idx, neighborhood="8")
        assert len(pairs) == 1
        r = pairs.iloc[0]
        assert (r["lo"], r["hi"]) == (0, 1)
        assert r["n_lo"] == 1
        assert r["n_hi"] == 1
        assert r["n_records"] == 1  # 1 * 1

    def test_adjacent_bin_pair_8_moore(self):
        # Entity 0 in bin (0,0); entity 1 in bin (1,0). Diagonally
        # adjacent → in 8-Moore.
        coords = np.array([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64)
        codes = np.array([0, 1], dtype=np.int64)
        idx = build_grid_index(coords, codes, G_xy=1.0)
        pairs = enumerate_pair_witnesses(idx, neighborhood="8")
        assert len(pairs) == 1
        r = pairs.iloc[0]
        assert r["n_lo"] == 1
        assert r["n_hi"] == 1

    def test_no_pair_when_out_of_reach(self):
        # Entities 2 bins apart → 8-Moore should NOT pair them
        coords = np.array([[0.5, 0.5], [2.5, 0.5]], dtype=np.float64)
        codes = np.array([0, 1], dtype=np.int64)
        idx = build_grid_index(coords, codes, G_xy=1.0)
        pairs_8 = enumerate_pair_witnesses(idx, neighborhood="8")
        assert len(pairs_8) == 0
        # But R2 reach (Moore-2) should
        pairs_r2 = enumerate_pair_witnesses(idx, neighborhood="R2")
        assert len(pairs_r2) == 1

    def test_witness_counts_aggregate(self):
        # Entity 0 has 3 tx in bin (0,0); entity 1 has 2 tx in bin (1,0).
        # 8-Moore: pair (0,1) with n_lo=3, n_hi=2, n_records=3*2=6
        coords = np.array([
            [0.5, 0.5], [0.6, 0.6], [0.7, 0.7],   # entity 0
            [1.5, 0.5], [1.6, 0.6],                 # entity 1
        ], dtype=np.float64)
        codes = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        idx = build_grid_index(coords, codes, G_xy=1.0)
        pairs = enumerate_pair_witnesses(idx, neighborhood="8")
        assert len(pairs) == 1
        r = pairs.iloc[0]
        assert r["n_lo"] == 3
        assert r["n_hi"] == 2
        assert r["n_records"] == 6

    def test_witness_filter_capped(self):
        # 2-tx entity should pass even with witness_min=3 IF cap_at_n_tx
        coords = np.array([
            [0.5, 0.5], [0.6, 0.6],         # entity 0 (n=2)
            [1.5, 0.5], [1.6, 0.6], [1.7, 0.7], [1.8, 0.8],  # entity 1 (n=4)
        ], dtype=np.float64)
        codes = np.array([0, 0, 1, 1, 1, 1], dtype=np.int64)
        idx = build_grid_index(coords, codes, G_xy=1.0)
        # Without cap: filter requires both sides >= 3 → entity 0 has
        # only 2 tx, blocked.
        pairs_strict = enumerate_pair_witnesses(
            idx, neighborhood="8", witness_min=3, cap_at_n_tx=False,
        )
        assert len(pairs_strict) == 0
        # With cap: eff_min(0) = min(3, 2) = 2 → passes.
        pairs_capped = enumerate_pair_witnesses(
            idx, neighborhood="8", witness_min=3, cap_at_n_tx=True,
        )
        assert len(pairs_capped) == 1

    def test_z_depth(self):
        # Two entities in same xy-bin but z=0 vs z=3. z_depth=1 → no pair;
        # z_depth=3 → pair.
        coords = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 3.5],
        ], dtype=np.float64)
        codes = np.array([0, 1], dtype=np.int64)
        idx = build_grid_index(coords, codes, G_xy=1.0, G_z=1.0)
        assert enumerate_pair_witnesses(idx, neighborhood="8", z_depth=1).empty
        assert len(enumerate_pair_witnesses(idx, neighborhood="8", z_depth=3)) == 1


# ---------------------------------------------------------------------
# neighbor_entities
# ---------------------------------------------------------------------
class TestNeighborEntities:
    def test_basic(self):
        # Layout: e0 in bin (0,0); e1 in bin (1,0); e2 in bin (5,5)
        coords = np.array([
            [0.5, 0.5], [1.5, 0.5], [5.5, 5.5],
        ], dtype=np.float64)
        codes = np.array([0, 1, 2], dtype=np.int64)
        idx = build_grid_index(coords, codes, G_xy=1.0)

        # Query (0,0): in 8-Moore, e0 (self) + e1 (adjacent) reachable.
        # e2 is far away.
        q_bin = bin_xy(np.array([[0.5, 0.5]]), 1.0)[0]
        out = neighbor_entities(idx, int(q_bin), neighborhood="8")
        assert set(out.keys()) == {0, 1}

    def test_z_depth_filter(self):
        coords = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 3.5],
        ], dtype=np.float64)
        codes = np.array([0, 1], dtype=np.int64)
        idx = build_grid_index(coords, codes, G_xy=1.0, G_z=1.0)
        q_bin = bin_xy(np.array([[0.5, 0.5]]), 1.0)[0]

        out_narrow = neighbor_entities(idx, int(q_bin), bin_z=0,
                                          neighborhood="8", z_depth=1)
        assert set(out_narrow.keys()) == {0}

        out_wide = neighbor_entities(idx, int(q_bin), bin_z=0,
                                       neighborhood="8", z_depth=3)
        assert set(out_wide.keys()) == {0, 1}

    def test_empty_lookup(self):
        coords = np.zeros((0, 2))
        codes = np.zeros(0, dtype=np.int64)
        idx = build_grid_index(coords, codes, G_xy=1.0)
        out = neighbor_entities(idx, 0, neighborhood="8")
        assert out == {}
