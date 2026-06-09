"""Tests for tracer.sankey_log — classifier, phase keys, display labels."""
import numpy as np
import pandas as pd
import pytest

try:
    from tracer import sankey_log as sl
except ImportError:
    import sankey_log as sl  # conftest standalone fallback


class TestClassifyEtypeVec:
    def test_main_cells(self):
        ids = np.array(["cell_42", "cell_7"], dtype=object)
        etypes = np.array(["cell", "cell"], dtype=object)
        out = sl._classify_etype_vec(ids, etypes)
        assert out.dtype == np.int8
        assert (out == sl.CLASS_MAIN).all()

    def test_partials(self):
        ids = np.array(["cell_42-1", "cell_7-3"], dtype=object)
        etypes = np.array(["partial", "partial"], dtype=object)
        assert (sl._classify_etype_vec(ids, etypes) == sl.CLASS_PARTIAL).all()

    def test_components(self):
        ids = np.array(["UNASSIGNED_base1", "UNASSIGNED_base2"], dtype=object)
        etypes = np.array(["component", "component"], dtype=object)
        assert (sl._classify_etype_vec(ids, etypes) == sl.CLASS_COMPONENT).all()

    def test_unassigned_sentinel(self):
        ids = np.array(["-1", "-1"], dtype=object)
        etypes = np.array(["unknown", "unknown"], dtype=object)
        assert (sl._classify_etype_vec(ids, etypes) == sl.CLASS_UNASSIGNED).all()

    def test_dropped_sentinels(self):
        ids = np.array(["DROP", "demote_rejected"], dtype=object)
        etypes = np.array(["unknown", "unknown"], dtype=object)
        assert (sl._classify_etype_vec(ids, etypes) == sl.CLASS_DROPPED).all()

    def test_etype_missing_falls_back(self):
        ids = np.array(["cell_42", "cell_42-1", "-1", "DROP"], dtype=object)
        # No etype passed; sentinels still classify, the rest become partial
        out = sl._classify_etype_vec(ids, None)
        assert out[0] == sl.CLASS_PARTIAL  # lossy default
        assert out[1] == sl.CLASS_PARTIAL
        assert out[2] == sl.CLASS_UNASSIGNED
        assert out[3] == sl.CLASS_DROPPED

    def test_sentinel_overrides_etype(self):
        # If id is "-1" but etype is stale "cell", classification is unassigned
        ids = np.array(["-1"], dtype=object)
        etypes = np.array(["cell"], dtype=object)
        assert sl._classify_etype_vec(ids, etypes)[0] == sl.CLASS_UNASSIGNED


class TestConstants:
    def test_class_codes_unique(self):
        codes = [sl.CLASS_MAIN, sl.CLASS_PARTIAL, sl.CLASS_COMPONENT,
                 sl.CLASS_UNASSIGNED, sl.CLASS_DROPPED]
        assert len(set(codes)) == 5
        assert all(0 <= c <= 4 for c in codes)

    def test_seg_default_phases(self):
        assert sl.PHASE_KEYS_SEG_DEFAULT[0] == "input"
        assert sl.PHASE_KEYS_SEG_DEFAULT[-1] == "finalize"
        assert "phase1" in sl.PHASE_KEYS_SEG_DEFAULT
        assert len(sl.PHASE_KEYS_SEG_DEFAULT) == 9

    def test_noseg_default_phases(self):
        assert sl.PHASE_KEYS_NOSEG_DEFAULT[0] == "input"
        assert sl.PHASE_KEYS_NOSEG_DEFAULT[-1] == "finalize"
        assert "cascade" in sl.PHASE_KEYS_NOSEG_DEFAULT
        assert len(sl.PHASE_KEYS_NOSEG_DEFAULT) == 8

    def test_seg_verbose_superset_of_default(self):
        # phase1 collapses to (prune + sub-steps) in verbose
        verbose_set = set(sl.PHASE_KEYS_SEG_VERBOSE)
        assert "prune" in verbose_set
        assert "reassign_1c" in verbose_set
        assert "phase1" not in verbose_set  # phase1 is default-tier shorthand
        assert len(sl.PHASE_KEYS_SEG_VERBOSE) == 15

    def test_collapse_map(self):
        assert sl.COLLAPSE_SEG["phase1+rescue"] == "rescue"
        assert sl.COLLAPSE_SEG["group+rescue"] == "post_group_rescue"
        assert sl.COLLAPSE_SEG["stitch+demote+rescue"] == "final_rescue"

    def test_display_labels_cover_all_keys(self):
        # Every key from every tier must have a display label
        all_keys = (set(sl.PHASE_KEYS_SEG_VERBOSE)
                    | set(sl.PHASE_KEYS_SEG_DEFAULT)
                    | set(sl.PHASE_KEYS_SEG_COLLAPSED)
                    | set(sl.PHASE_KEYS_NOSEG_VERBOSE)
                    | set(sl.PHASE_KEYS_NOSEG_DEFAULT)
                    | set(sl.PHASE_KEYS_NOSEG_COLLAPSED))
        missing = all_keys - set(sl.PHASE_DISPLAY_LABELS.keys())
        assert not missing, f"Missing display labels: {missing}"

    def test_phase1_displays_as_prune(self):
        assert sl.PHASE_DISPLAY_LABELS["phase1"] == "Prune"
