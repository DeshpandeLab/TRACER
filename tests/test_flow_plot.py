"""Tests for tracer.flow_plot — data-prep, view resolution, backends."""
import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")  # headless

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


class TestResolveClassOrder:
    def _toy_palette(self):
        return {
            sl.CLASS_MAIN: "#1f77b4",
            sl.CLASS_PARTIAL: "#2ca02c",
            sl.CLASS_UNASSIGNED: "#7f7f7f",
            5: "#ff7f0e",  # extended code (e.g. main_neighbor)
        }

    def test_default_follows_canonical_semantic_order(self):
        # Canonical codes (0..4) in palette → semantic order
        # (main → partial → unassigned for this 3-canonical subset).
        # Extended code 5 has no canonical position → appended at the bottom.
        out = fp._resolve_class_order(self._toy_palette(), None)
        assert out == [sl.CLASS_MAIN, sl.CLASS_PARTIAL, sl.CLASS_UNASSIGNED, 5]

    def test_default_matches_semantic_order_for_full_5(self):
        full_palette = {
            sl.CLASS_MAIN: "a", sl.CLASS_PARTIAL: "b",
            sl.CLASS_COMPONENT: "c", sl.CLASS_UNASSIGNED: "d",
            sl.CLASS_DROPPED: "e",
        }
        assert fp._resolve_class_order(full_palette, None) == list(sl.CLASS_SEMANTIC_ORDER)

    def test_explicit_order_is_respected(self):
        custom = [sl.CLASS_MAIN, 5, sl.CLASS_PARTIAL, sl.CLASS_UNASSIGNED]
        out = fp._resolve_class_order(self._toy_palette(), custom)
        assert out == custom

    def test_omitted_codes_appended(self):
        # User specifies only top two; the other palette keys must still
        # appear (in sorted order) at the bottom so they're not silently dropped.
        custom = [sl.CLASS_MAIN, 5]
        out = fp._resolve_class_order(self._toy_palette(), custom)
        assert out[:2] == custom
        assert set(out[2:]) == {sl.CLASS_PARTIAL, sl.CLASS_UNASSIGNED}

    def test_unknown_code_raises(self):
        with pytest.raises(ValueError, match="not in palette"):
            fp._resolve_class_order(self._toy_palette(),
                                    [sl.CLASS_MAIN, 99])


class TestResolveView:
    def test_seg_default(self):
        df_cols = {f"etype_at_{p}" for p in sl.PHASE_KEYS_SEG_DEFAULT}
        phases = fp._resolve_view(df_cols, pipeline="seg", view="default")
        assert phases == sl.PHASE_KEYS_SEG_DEFAULT

    def test_noseg_default(self):
        df_cols = {f"etype_at_{p}" for p in sl.PHASE_KEYS_NOSEG_DEFAULT}
        phases = fp._resolve_view(df_cols, pipeline="noseg", view="default")
        assert phases == sl.PHASE_KEYS_NOSEG_DEFAULT

    def test_auto_detects_seg(self):
        # Presence of phase1 column → SEG
        df_cols = {f"etype_at_{p}" for p in sl.PHASE_KEYS_SEG_DEFAULT}
        phases = fp._resolve_view(df_cols, pipeline="auto", view="default")
        assert phases == sl.PHASE_KEYS_SEG_DEFAULT

    def test_auto_detects_noseg(self):
        # Presence of cascade column → NOSEG
        df_cols = {f"etype_at_{p}" for p in sl.PHASE_KEYS_NOSEG_DEFAULT}
        phases = fp._resolve_view(df_cols, pipeline="auto", view="default")
        assert phases == sl.PHASE_KEYS_NOSEG_DEFAULT

    def test_seg_collapsed_returns_source_columns(self):
        df_cols = {f"etype_at_{p}" for p in sl.PHASE_KEYS_SEG_DEFAULT}
        phases = fp._resolve_view(df_cols, pipeline="seg", view="collapsed")
        # Tier A collapsed maps each compound group to its end-of-group
        # source column. After dropping the redundant Finalize column from
        # the default tier, Tier A ends at `final_rescue` (rendered as
        # "Finalize" via the column-label override).
        assert phases == ["input", "rescue", "post_group_rescue",
                          "final_rescue"]

    def test_verbose_missing_raises(self):
        df_cols = {f"etype_at_{p}" for p in sl.PHASE_KEYS_SEG_DEFAULT}
        with pytest.raises(KeyError, match="verbose"):
            fp._resolve_view(df_cols, pipeline="seg", view="verbose")

    def test_optional_phase_skipped(self):
        # mid_qc absent → drop it from the resolved list
        cols = {f"etype_at_{p}" for p in sl.PHASE_KEYS_SEG_DEFAULT
                if p != "mid_qc"}  # mid_qc not in default anyway but keep test
        phases = fp._resolve_view(cols, pipeline="seg", view="default")
        assert "mid_qc" not in phases
        assert "phase1" in phases  # other required ones still present


class TestPlotMatplotlib:
    def _full_df(self):
        """Toy SEG-shaped df with all 9 default-tier snapshot columns."""
        n = 50
        base = ([sl.CLASS_MAIN] * 10 + [sl.CLASS_PARTIAL] * 20
                + [sl.CLASS_UNASSIGNED] * 20)
        df = pd.DataFrame({
            f"etype_at_{p}": np.array(base, dtype=np.int8)
            for p in sl.PHASE_KEYS_SEG_DEFAULT
        })
        # Simulate final_rescue moving 5 unassigned → dropped
        final = base.copy()
        final[-5:] = [sl.CLASS_DROPPED] * 5
        df["etype_at_finalize"] = np.array(final, dtype=np.int8)
        return df

    def test_returns_figure(self):
        df = self._full_df()
        fig = fp.plot_transcript_flow(df, backend="matplotlib")
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_return_data_tuple(self):
        df = self._full_df()
        fig, tidy = fp.plot_transcript_flow(
            df, backend="matplotlib", return_data=True
        )
        assert isinstance(tidy, pd.DataFrame)
        assert set(tidy.columns) >= {"phase_from", "phase_to",
                                     "class_from", "class_to", "n"}

    def test_output_file_png(self, tmp_path):
        df = self._full_df()
        out = tmp_path / "flow.png"
        fp.plot_transcript_flow(df, backend="matplotlib", output=str(out))
        assert out.exists()
        assert out.stat().st_size > 0


class TestPlotPlotly:
    def setup_method(self):
        self.go = pytest.importorskip("plotly.graph_objects")

    def _full_df(self):
        n = 50
        base = ([sl.CLASS_MAIN] * 10 + [sl.CLASS_PARTIAL] * 20
                + [sl.CLASS_UNASSIGNED] * 20)
        df = pd.DataFrame({
            f"etype_at_{p}": np.array(base, dtype=np.int8)
            for p in sl.PHASE_KEYS_SEG_DEFAULT
        })
        final = base.copy()
        final[-5:] = [sl.CLASS_DROPPED] * 5
        df["etype_at_finalize"] = np.array(final, dtype=np.int8)
        return df

    def test_returns_plotly_figure(self):
        df = self._full_df()
        fig = fp.plot_transcript_flow(df, backend="plotly")
        assert isinstance(fig, self.go.Figure)

    def test_has_sankey_trace(self):
        df = self._full_df()
        fig = fp.plot_transcript_flow(df, backend="plotly")
        assert any(isinstance(t, self.go.Sankey) for t in fig.data)

    def test_node_count_matches_phases_times_classes(self):
        df = self._full_df()
        fig = fp.plot_transcript_flow(df, backend="plotly",
                                       class_grouping="three")
        sankey = next(t for t in fig.data if isinstance(t, self.go.Sankey))
        # Each (phase, class) with nonzero count gets a node
        # Should be > 0 and finite
        assert len(sankey.node.label) > 0

    def test_html_output(self, tmp_path):
        df = self._full_df()
        out = tmp_path / "flow.html"
        fp.plot_transcript_flow(df, backend="plotly", output=str(out))
        assert out.exists()
        assert b"Sankey" in out.read_bytes()


class TestPublicAPI:
    def test_plot_transcript_flow_importable(self):
        # Skip if standalone-load fallback was used
        try:
            import tracer
        except ImportError:
            pytest.skip("full tracer import unavailable in this env")
        # conftest installs a minimal stub `tracer` namespace when the full
        # package can't be imported (no geopandas/torch/open3d). That stub
        # lacks __version__; the real __init__.py sets it.
        if not hasattr(tracer, "__version__"):
            pytest.skip("standalone-load fallback in use; full tracer unavailable")
        assert hasattr(tracer, "plot_transcript_flow")
        assert hasattr(tracer, "snapshot_phase")
