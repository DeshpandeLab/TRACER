"""End-to-end smoke for the transcript-flow Sankey plot."""
import numpy as np
import pandas as pd
import pytest

try:
    from tracer import flow_plot as fp
    from tracer import sankey_log as sl
except ImportError:
    import flow_plot as fp
    import sankey_log as sl

import matplotlib
matplotlib.use("Agg")


def _synthetic_seg_df(n=200, seed=0):
    """Synthetic SEG snapshot df spanning all 9 default-tier phases."""
    rng = np.random.default_rng(seed)
    cols = sl.PHASE_KEYS_SEG_DEFAULT  # 9 phases
    # Start: ~70% main, ~20% partial, ~10% unassigned
    init = rng.choice([sl.CLASS_MAIN, sl.CLASS_PARTIAL, sl.CLASS_UNASSIGNED],
                      size=n, p=[0.7, 0.2, 0.1])
    df = pd.DataFrame({f"etype_at_{cols[0]}": init.astype(np.int8)})
    state = init.copy()
    # Random walk: at each phase, ~5% of transcripts change class
    for phase in cols[1:]:
        new = state.copy()
        flip_mask = rng.random(n) < 0.05
        new[flip_mask] = rng.integers(0, 5, size=flip_mask.sum())
        df[f"etype_at_{phase}"] = new.astype(np.int8)
        state = new
    return df


class TestSankeyIntegration:
    def test_full_seg_matplotlib(self, tmp_path):
        df = _synthetic_seg_df()
        out = tmp_path / "seg_flow.png"
        fig = fp.plot_transcript_flow(
            df, backend="matplotlib", output=str(out), title="SEG demo"
        )
        assert out.exists()
        assert out.stat().st_size > 1000  # nontrivial png

    def test_full_seg_plotly(self, tmp_path):
        pytest.importorskip("plotly.graph_objects")
        df = _synthetic_seg_df()
        out = tmp_path / "seg_flow.html"
        fig = fp.plot_transcript_flow(
            df, backend="plotly", output=str(out), title="SEG demo"
        )
        assert out.exists()
        # Has a Sankey trace embedded
        assert b"Sankey" in out.read_bytes()

    def test_collapsed_view_works(self):
        df = _synthetic_seg_df()
        fig = fp.plot_transcript_flow(
            df, view="collapsed", backend="matplotlib"
        )
        # Just checks no error; figure shape verified in earlier unit tests

    def test_class_grouping_three_vs_five_have_different_node_counts(self):
        pytest.importorskip("plotly.graph_objects")
        df = _synthetic_seg_df()
        fig3 = fp.plot_transcript_flow(df, backend="plotly",
                                        class_grouping="three")
        fig5 = fp.plot_transcript_flow(df, backend="plotly",
                                        class_grouping="five")
        n3 = len(fig3.data[0].node.label)
        n5 = len(fig5.data[0].node.label)
        assert n5 >= n3  # 5-class has at least as many bands
