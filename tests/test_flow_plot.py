"""Tests for tracer.flow_plot — data-prep, view resolution, backends."""
import numpy as np
import pandas as pd
import pytest

try:
    from tracer import flow_plot as fp
    from tracer import sankey_log as sl
except ImportError:
    import flow_plot as fp
    import sankey_log as sl


def _toy_snapshot_df(n_per_class=10):
    """3-phase df with hand-crafted transitions.
    input: 10 cell + 10 partial + 10 unassigned = 30
    phase1: same (no change)
    rescue: 10 unassigned → 5 partial + 5 dropped, others unchanged
    """
    base = ([sl.CLASS_MAIN] * n_per_class
            + [sl.CLASS_PARTIAL] * n_per_class
            + [sl.CLASS_UNASSIGNED] * n_per_class)
    after_rescue = ([sl.CLASS_MAIN] * n_per_class
                    + [sl.CLASS_PARTIAL] * n_per_class
                    + [sl.CLASS_PARTIAL] * 5 + [sl.CLASS_DROPPED] * 5)
    return pd.DataFrame({
        "etype_at_input":   np.array(base, dtype=np.int8),
        "etype_at_phase1":  np.array(base, dtype=np.int8),
        "etype_at_rescue":  np.array(after_rescue, dtype=np.int8),
    })


class TestPrepareFlowData:
    def test_tidy_shape(self):
        df = _toy_snapshot_df()
        tidy = fp._prepare_flow_data(
            df, phases=["input", "phase1", "rescue"], class_grouping="five",
        )
        assert set(tidy.columns) == {
            "phase_from", "phase_to", "class_from", "class_to", "n"
        }

    def test_conservation_input_to_phase1(self):
        df = _toy_snapshot_df()
        tidy = fp._prepare_flow_data(
            df, phases=["input", "phase1", "rescue"], class_grouping="five",
        )
        # input → phase1: identity, all 30 transcripts in same class
        sub = tidy[(tidy["phase_from"] == "input") &
                   (tidy["phase_to"] == "phase1")]
        assert sub["n"].sum() == 30

    def test_transition_phase1_to_rescue(self):
        df = _toy_snapshot_df()
        tidy = fp._prepare_flow_data(
            df, phases=["input", "phase1", "rescue"], class_grouping="five",
        )
        sub = tidy[(tidy["phase_from"] == "phase1") &
                   (tidy["phase_to"] == "rescue")]
        # 10 main→main, 10 partial→partial, 5 unassigned→partial, 5 unassigned→dropped
        edges = {(r.class_from, r.class_to): r.n for r in sub.itertuples()}
        assert edges[(sl.CLASS_MAIN, sl.CLASS_MAIN)] == 10
        assert edges[(sl.CLASS_PARTIAL, sl.CLASS_PARTIAL)] == 10
        assert edges[(sl.CLASS_UNASSIGNED, sl.CLASS_PARTIAL)] == 5
        assert edges[(sl.CLASS_UNASSIGNED, sl.CLASS_DROPPED)] == 5

    def test_class_grouping_three(self):
        df = _toy_snapshot_df()
        tidy = fp._prepare_flow_data(
            df, phases=["input", "phase1", "rescue"], class_grouping="three",
        )
        # 5 dropped collapses into unassigned bucket
        sub = tidy[(tidy["phase_from"] == "phase1") &
                   (tidy["phase_to"] == "rescue")]
        edges = {(r.class_from, r.class_to): r.n for r in sub.itertuples()}
        # 5 unassigned→partial stays; 5 unassigned→dropped becomes unassigned→unassigned
        assert edges[(sl.CLASS_UNASSIGNED, sl.CLASS_UNASSIGNED)] == 5

    def test_strict_conservation_raises_on_invalid_code(self):
        df = _toy_snapshot_df()
        # Inject an out-of-range class code; strict_conservation must catch it
        df.loc[0, "etype_at_rescue"] = -1  # invalid code (valid range: 0-4)
        with pytest.raises((ValueError, AssertionError)):
            fp._prepare_flow_data(
                df, phases=["input", "phase1", "rescue"],
                class_grouping="five", strict_conservation=True,
            )

    def test_min_flow_frac_drops_small(self):
        df = _toy_snapshot_df()  # 30 total tx
        tidy = fp._prepare_flow_data(
            df, phases=["input", "phase1", "rescue"],
            class_grouping="five", min_flow_frac=0.2,  # drops anything < 6 tx
        )
        # The 5-tx unassigned→dropped edge should be filtered out
        sub = tidy[(tidy["phase_from"] == "phase1") &
                   (tidy["phase_to"] == "rescue")]
        edges = {(r.class_from, r.class_to): r.n for r in sub.itertuples()}
        assert (sl.CLASS_UNASSIGNED, sl.CLASS_DROPPED) not in edges
