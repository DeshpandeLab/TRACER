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

    def test_post_finalize_unassigned_token(self):
        # After finalize_unassigned, leftover -1 tx get id "UNASSIGNED"
        # (the canonical pipeline-output sentinel). Must still classify as
        # CLASS_UNASSIGNED — earlier the classifier missed this and the
        # tx fell through to CLASS_PARTIAL.
        ids = np.array(["UNASSIGNED", "nan",
                        "prune_rejected", "group_rejected"], dtype=object)
        etypes = np.array(["unknown"] * 4, dtype=object)
        out = sl._classify_etype_vec(ids, etypes)
        assert (out == sl.CLASS_UNASSIGNED).all()


class TestConstants:
    def test_class_codes_unique(self):
        codes = [sl.CLASS_MAIN, sl.CLASS_PARTIAL, sl.CLASS_COMPONENT,
                 sl.CLASS_UNASSIGNED, sl.CLASS_DROPPED]
        assert len(set(codes)) == 5
        assert all(0 <= c <= 4 for c in codes)

    def test_seg_default_phases(self):
        assert sl.PHASE_KEYS_SEG_DEFAULT[0] == "input"
        # `finalize` is dropped from the default tier — the runner's
        # Finalize step is a no-op on classification, so `final_rescue`
        # is the effective output column.
        assert sl.PHASE_KEYS_SEG_DEFAULT[-1] == "final_rescue"
        assert "finalize" not in sl.PHASE_KEYS_SEG_DEFAULT
        assert "phase1" in sl.PHASE_KEYS_SEG_DEFAULT
        assert len(sl.PHASE_KEYS_SEG_DEFAULT) == 8

    def test_noseg_default_phases(self):
        assert sl.PHASE_KEYS_NOSEG_DEFAULT[0] == "input"
        assert sl.PHASE_KEYS_NOSEG_DEFAULT[-1] == "final_rescue"
        assert "finalize" not in sl.PHASE_KEYS_NOSEG_DEFAULT
        assert "cascade" in sl.PHASE_KEYS_NOSEG_DEFAULT
        assert len(sl.PHASE_KEYS_NOSEG_DEFAULT) == 7

    def test_finalize_still_in_verbose_tier(self):
        # Verbose users can still see the runner's defensive Finalize step
        assert "finalize" in sl.PHASE_KEYS_SEG_VERBOSE
        assert "finalize" in sl.PHASE_KEYS_NOSEG_VERBOSE

    def test_final_rescue_column_reads_finalize(self):
        # Column override: state-after-final_rescue is the canonical
        # output, labeled "Finalize". Stage (ribbon) keeps "Final Rescue".
        assert sl.display_label_for("final_rescue", pipeline="seg") == "Finalize"
        assert sl.display_label_for("final_rescue", pipeline="noseg") == "Finalize"
        # Stage label (raw) still reads the action name
        assert sl.PHASE_DISPLAY_LABELS["final_rescue"] == "Final Rescue"

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


class TestSnapshotPhase:
    def _toy_df(self):
        return pd.DataFrame({
            "tracer_id": ["cell_1", "cell_1-1", "UNASSIGNED_b", "-1", "DROP"],
            "_etype": ["cell", "partial", "component", "unknown", "unknown"],
        })

    def test_writes_int8_column(self):
        df = self._toy_df()
        sl.snapshot_phase(df, "phase1", id_col="tracer_id")
        col = df["etype_at_phase1"]
        assert col.dtype == np.int8
        assert list(col) == [sl.CLASS_MAIN, sl.CLASS_PARTIAL,
                             sl.CLASS_COMPONENT, sl.CLASS_UNASSIGNED,
                             sl.CLASS_DROPPED]

    def test_inplace(self):
        df = self._toy_df()
        ret = sl.snapshot_phase(df, "rescue", id_col="tracer_id")
        assert ret is None
        assert "etype_at_rescue" in df.columns

    def test_missing_id_col_raises(self):
        df = self._toy_df()
        with pytest.raises(KeyError, match="id_col"):
            sl.snapshot_phase(df, "stitch", id_col="stitched")

    def test_no_etype_column(self):
        df = pd.DataFrame({"tracer_id": ["cell_1", "-1", "DROP"]})
        sl.snapshot_phase(df, "input", id_col="tracer_id")
        # Sentinels still classify correctly; "cell_1" falls back to partial
        assert list(df["etype_at_input"]) == [sl.CLASS_PARTIAL,
                                              sl.CLASS_UNASSIGNED,
                                              sl.CLASS_DROPPED]


class TestConservation:
    def test_size_preserved_across_two_snapshots(self):
        df = pd.DataFrame({"tracer_id": ["cell_1", "-1", "DROP"] * 100,
                           "_etype": ["cell", "unknown", "unknown"] * 100})
        sl.snapshot_phase(df, "input", id_col="tracer_id")
        df["tracer_id"] = df["tracer_id"].replace({"-1": "cell_99"})
        # Caller would refresh _etype too; for the test, also flip etype
        df.loc[df["tracer_id"] == "cell_99", "_etype"] = "cell"
        sl.snapshot_phase(df, "rescue", id_col="tracer_id")
        # Total rows unchanged
        assert df["etype_at_input"].size == df["etype_at_rescue"].size
        # Conservation: sum of class counts equal between snapshots
        c_in = pd.Series(df["etype_at_input"]).value_counts().sort_index()
        c_out = pd.Series(df["etype_at_rescue"]).value_counts().sort_index()
        assert c_in.sum() == c_out.sum() == 300


class TestNeighborClassConstants:
    def test_neighbor_code_and_name(self):
        assert sl.CLASS_MAIN_NEIGHBOR == 5
        assert sl.CLASS_NAMES[sl.CLASS_MAIN_NEIGHBOR] == "neighboring cell"

    def test_renamed_labels(self):
        assert sl.CLASS_NAMES[sl.CLASS_MAIN] == "original cell"
        assert sl.CLASS_NAMES[sl.CLASS_PARTIAL] == "partial cell"
        assert sl.CLASS_NAMES[sl.CLASS_COMPONENT] == "component"
        assert sl.CLASS_NAMES[sl.CLASS_UNASSIGNED] == "unassigned"
        assert sl.CLASS_NAMES[sl.CLASS_DROPPED] == "dropped"

    def test_collapse_3_folds_neighbor_into_main(self):
        assert sl.CLASS_COLLAPSE_3[sl.CLASS_MAIN_NEIGHBOR] == sl.CLASS_MAIN

    def test_semantic_order_neighbor_after_main(self):
        order = sl.CLASS_SEMANTIC_ORDER
        assert sl.CLASS_MAIN in order and sl.CLASS_MAIN_NEIGHBOR in order
        assert order.index(sl.CLASS_MAIN_NEIGHBOR) == order.index(sl.CLASS_MAIN) + 1


class TestIsOriginalMatch:
    def test_same_cell(self):
        assert sl._is_original_match("42", "42") is True

    def test_partial_of_same_cell(self):
        assert sl._is_original_match("42-tr-1", "42") is True

    def test_different_cell(self):
        assert sl._is_original_match("57", "42") is False

    def test_ffpe_dash_cell_id_same(self):
        assert sl._is_original_match("dafehkie-1-tr-2", "dafehkie-1") is True

    def test_ffpe_dash_cell_id_different(self):
        assert sl._is_original_match("dafehkie-12", "dafehkie-1") is False

    def test_unassigned_origin_is_no_match(self):
        assert sl._is_original_match("42", "-1") is False
        assert sl._is_original_match("42", "UNASSIGNED") is False


class TestClassifyEndpoints:
    def _df(self, with_etype):
        # cur/orig pairs covering every case:
        #  B   / A   cell, moved        -> neighbor
        #  A   / A   cell, stayed       -> original
        #  B-tr-1 / A partial of other  -> partial (etype wins)
        #  C   / -1  cell, no real orig -> original
        #  -1  / D   unassigned final   -> unassigned
        cur  = ["B", "A", "B-tr-1", "C", "-1"]
        orig = ["A", "A", "A",      "-1", "D"]
        data = {"label": cur, "cell_id": orig}
        if with_etype:
            data["_etype"] = pd.Categorical(
                ["cell", "cell", "partial", "cell", "unknown"])
        return pd.DataFrame(data)

    def test_initial_is_original_or_unassigned(self):
        df = self._df(with_etype=True)
        initial, _ = sl.classify_endpoints(
            df, orig_id_col="cell_id", label_col="label")
        assert list(initial) == [sl.CLASS_MAIN, sl.CLASS_MAIN, sl.CLASS_MAIN,
                                 sl.CLASS_UNASSIGNED, sl.CLASS_MAIN]

    def test_final_with_etype(self):
        df = self._df(with_etype=True)
        _, final = sl.classify_endpoints(
            df, orig_id_col="cell_id", label_col="label")
        assert list(final) == [
            sl.CLASS_MAIN_NEIGHBOR,  # B from A
            sl.CLASS_MAIN,           # A from A
            sl.CLASS_PARTIAL,        # B-tr-1 (etype wins)
            sl.CLASS_MAIN,           # C from unassigned origin
            sl.CLASS_UNASSIGNED,     # -1
        ]

    def test_final_without_etype_derives_from_label(self):
        df = self._df(with_etype=False)  # no _etype column
        _, final = sl.classify_endpoints(
            df, orig_id_col="cell_id", label_col="label")
        assert list(final) == [
            sl.CLASS_MAIN_NEIGHBOR,  # B (no -tr-) moved
            sl.CLASS_MAIN,           # A
            sl.CLASS_PARTIAL,        # B-tr-1 has -tr-
            sl.CLASS_MAIN,           # C
            sl.CLASS_UNASSIGNED,     # -1
        ]

    def test_parity_with_is_original_match(self):
        df = self._df(with_etype=True)
        _, final = sl.classify_endpoints(
            df, orig_id_col="cell_id", label_col="label")
        for i, (l, o) in enumerate(zip(df["label"], df["cell_id"])):
            if final[i] == sl.CLASS_MAIN_NEIGHBOR:
                assert not sl._is_original_match(str(l), str(o))

    def test_missing_columns_raise(self):
        df = self._df(with_etype=True)
        with pytest.raises(KeyError):
            sl.classify_endpoints(df, orig_id_col="nope", label_col="label")
        with pytest.raises(KeyError):
            sl.classify_endpoints(df, orig_id_col="cell_id", label_col="nope")
