from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GBM = ROOT / "tutorials" / "gbm"
if str(GBM) not in sys.path:
    sys.path.insert(0, str(GBM))

import plot_chromosome_heatmap as heatmap


def test_feature_tsv_candidates_handle_container_repo_root(monkeypatch):
    monkeypatch.setattr(heatmap, "_repo_root", lambda: Path("/app"))

    candidates = heatmap._feature_tsv_candidates()

    assert (
        Path(
            "/mnt/storage/dept/medonc/beroukhim/youyun/BTC_GBM/"
            "data/xenium/output-XETG00323__0023274__Patient4__20241004__181038/"
            "cell_feature_matrix/features.tsv.gz"
        )
        in candidates
    )
    assert Path("/app/tutorials/gbm/data/features.tsv.gz") in candidates


def test_feature_tsv_candidates_prefer_inferred_btc_gbm_root(monkeypatch):
    monkeypatch.setattr(
        heatmap, "_repo_root", lambda: Path("/project/BTC_GBM/code/TRACER")
    )

    candidates = heatmap._feature_tsv_candidates()

    assert candidates[0] == Path(
        "/project/BTC_GBM/"
        "data/xenium/output-XETG00323__0023274__Patient4__20241004__181038/"
        "cell_feature_matrix/features.tsv.gz"
    )


def test_parse_args_uses_coarse_leiden_without_chromosome_gap_option(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["plot_chromosome_heatmap.py"])

    args = heatmap._parse_args()

    assert args.leiden_resolution == 0.1
    assert args.bootstrap_iterations == heatmap.DEFAULT_BOOTSTRAP_ITERATIONS
    assert args.bootstrap_seed == 42
    assert not hasattr(args, "chromosome_gap_columns")


def test_chromosome_groups_use_real_gene_column_slices():
    positions = pd.DataFrame({"chromosome": ["1", "1", "2", "2", "2", "X"]})

    groups = heatmap._chromosome_groups(positions)

    assert groups == [("1", 0, 2), ("2", 2, 5), ("X", 5, 6)]
    assert [end - start for _, start, end in groups] == [2, 3, 1]


def test_chromosome_groups_handle_empty_positions():
    positions = pd.DataFrame({"chromosome": []})

    assert heatmap._chromosome_groups(positions) == []


def test_oligo_centered_chromosome_effects_have_expected_direction():
    obs = pd.DataFrame(
        {
            "cell_type": ["oligo", "oligo", "cancer_AC", "cancer_AC"],
            "heatmap_leiden": ["4", "4", "0", "0"],
        },
        index=["o1", "o2", "c1", "c2"],
    )
    log_matrix = np.array(
        [
            [1.0, 1.0, 4.0],
            [3.0, 3.0, 4.0],
            [4.0, 4.0, 2.0],
            [6.0, 6.0, 2.0],
        ],
        dtype=np.float32,
    )
    positions = pd.DataFrame(
        {
            "gene": ["g1", "g2", "g3"],
            "chromosome": ["2", "2", "7"],
        }
    )

    gene_scores, oligo_mask = heatmap._oligo_center_gene_scores(log_matrix, obs)
    chromosome_scores = heatmap._chromosome_score_frame(gene_scores, positions)
    effects = heatmap._chromosome_cluster_effects(
        chromosome_scores,
        obs,
        positions,
        bootstrap_iterations=20,
        bootstrap_seed=7,
    )

    assert oligo_mask.tolist() == [True, True, False, False]
    cluster0 = effects[effects["cluster"] == "0"].set_index("chromosome")
    assert cluster0.loc["2", "effect_vs_oligo"] > 0
    assert cluster0.loc["7", "effect_vs_oligo"] < 0
    assert cluster0.loc["2", "is_focus_chromosome"]


def test_chromosome_effect_bootstrap_is_deterministic():
    obs = pd.DataFrame(
        {
            "cell_type": ["oligo", "oligo", "cancer_AC", "cancer_AC"],
            "heatmap_leiden": ["4", "4", "0", "0"],
        }
    )
    chromosome_scores = pd.DataFrame(
        {
            "2": [0.0, 0.5, 1.0, 1.5],
            "7": [0.0, -0.5, -1.0, -1.5],
        }
    )
    positions = pd.DataFrame(
        {
            "gene": ["g1", "g2"],
            "chromosome": ["2", "7"],
        }
    )

    first = heatmap._chromosome_cluster_effects(
        chromosome_scores,
        obs,
        positions,
        bootstrap_iterations=25,
        bootstrap_seed=11,
    )
    second = heatmap._chromosome_cluster_effects(
        chromosome_scores,
        obs,
        positions,
        bootstrap_iterations=25,
        bootstrap_seed=11,
    )

    pd.testing.assert_frame_equal(first, second)
