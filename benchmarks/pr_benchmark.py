"""PR benchmark: TRACER recovery quality on synthetic data under two
scenarios.

1. **Easy mode** — full 10 µm volume + ground-truth cell_id. Measures
   "does the pipeline pass good input through cleanly." Should be near
   ARI = 1.0.
2. **Realistic mode** — 5 µm tissue section + simulated DAPI/Voronoi
   segmentation as input. Measures the actual signal we care about:
   how well TRACER **recovers** ground truth from a noisy Xenium-style
   segmenter. Errors include cells that lost their nucleus to clipping
   (no DAPI) and tx misassigned by the z-blind Voronoi tessellation.

Run before opening a PR; paste output into BENCHMARKS.md:

    python benchmarks/pr_benchmark.py >> BENCHMARKS.md

The output is a markdown block: a summary table plus a collapsed
``<details>`` block with per-stage progression.
"""
from __future__ import annotations

import contextlib
import datetime
import io
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Make project root importable when run as a standalone script
# (pytest handles this automatically via pyproject.toml, but a plain
# `python benchmarks/pr_benchmark.py` invocation does not).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SRC = _PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from tests.synthetic import (
    make_synthetic_npmi_panel_for_transcripts,
    make_synthetic_transcripts,
)
from tests.segmentation_sim import simulate_dapi_voronoi_segmentation
from tests._pipeline_runner import run_segmented_pipeline


CELLS_KW = dict(
    n_cells=8,
    voxels_per_cell_mean=80,
    tx_per_cell=25,
    n_genes=12,
    n_types=3,
    domain_z_um=10.0,
    nuclear_layers=2,
    # Decoding errors disabled. Real Xenium/MERFISH platforms report
    # per-tx misread rates ≤ 5%, but the dominant noise sources in
    # spatial transcriptomics are segmentation errors and z-projection
    # in sectioned tissue — both already modeled here independently
    # (DAPI/Voronoi sim + section_z_range_um). Layering an additional
    # 20% gene-level noise on top would double-count and make the
    # easy-mode ceiling artificially low.
    cross_type_noise_pct=0.0,
)
SECTION_Z = (2.5, 7.5)


def _git_describe() -> tuple[str, str]:
    """Return (branch, short-sha). Fallback to 'unknown' on any error
    (e.g. running outside a git checkout)."""
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return branch, sha
    except Exception:
        return "unknown", "unknown"


def _ari_ami_vs_truth(labels: np.ndarray, truth: np.ndarray
                      ) -> tuple[float, float]:
    """ARI/AMI on the subset where both labels and truth are assigned."""
    mask = (labels != "-1") & (truth != "-1")
    if mask.sum() < 2:
        return float("nan"), float("nan")
    return (
        float(adjusted_rand_score(truth[mask], labels[mask])),
        float(adjusted_mutual_info_score(truth[mask], labels[mask])),
    )


def _measure(scenario: str, df: pd.DataFrame,
             panel: pd.DataFrame) -> dict[str, Any]:
    """Run the segmented pipeline and compute recovery metrics vs the
    ground-truth partition.

    The caller is responsible for ensuring ``df`` has a
    ``cell_id_truth`` column carrying the ground-truth label. For the
    realistic-mode scenario this is set by
    :func:`simulate_dapi_voronoi_segmentation`. For the easy-mode
    scenario the caller copies ``cell_id`` (which already equals
    truth) into ``cell_id_truth`` before calling.

    We report ARI/AMI for **both** the input partition (the
    upstream segmentation that TRACER receives) and the output
    partition (TRACER's stitched labels), each vs ground truth. The
    delta is TRACER's value-add over the upstream segmenter.
    """
    if "cell_id_truth" not in df.columns:
        raise ValueError(
            "df must have a 'cell_id_truth' column carrying the "
            "ground-truth partition; ARI/AMI are always computed vs "
            "ground truth."
        )
    truth = df["cell_id_truth"].astype(str).to_numpy()
    inp = df["cell_id"].astype(str).to_numpy()

    # Input quality — how good is the segmentation we feed TRACER?
    input_ari, input_ami = _ari_ami_vs_truth(inp, truth)

    t0 = time.time()
    # Pipeline emits diagnostic prints; swallow them so the benchmark's
    # markdown output is the only thing on stdout.
    with contextlib.redirect_stdout(io.StringIO()):
        df_out, prog = run_segmented_pipeline(df, panel)
    dt = time.time() - t0

    out = df_out["stitched"].astype(str).to_numpy()
    output_ari, output_ami = _ari_ami_vs_truth(out, truth)
    coverage = float((out != "-1").mean())
    s_assigned = pd.Series(out)[pd.Series(out) != "-1"]
    n_ent = int(s_assigned.nunique())

    return {
        "scenario": scenario,
        "input_ari": input_ari,
        "input_ami": input_ami,
        "ari": output_ari,
        "ami": output_ami,
        "coverage": coverage,
        "n_ent": n_ent,
        "runtime": dt,
        "progression": prog,
    }


def _format_block(results: list[dict[str, Any]]) -> str:
    branch, sha = _git_describe()
    today = datetime.date.today().isoformat()

    def _fmt(v: float) -> str:
        return "n/a" if v != v else f"{v:.3f}"

    lines = [f"## {today} — {branch} @ {sha}", ""]
    lines.append("| scenario | input ARI | output ARI | output AMI | "
                 "coverage | n_ent | runtime |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['scenario']} | {_fmt(r['input_ari'])} | "
            f"{_fmt(r['ari'])} | {_fmt(r['ami'])} | "
            f"{100 * r['coverage']:.1f}% | {r['n_ent']} | "
            f"{r['runtime']:.2f}s |"
        )
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Per-stage progression</summary>")
    lines.append("")
    for r in results:
        lines.append(f"**{r['scenario']}**")
        lines.append("")
        lines.append("| stage | n_cells | n_partials | n_components | n_unassigned_tx |")
        lines.append("|---|---|---|---|---|")
        for s in r["progression"]:
            lines.append(
                f"| {s['stage']} | {s['n_cells']} | {s['n_partials']} | "
                f"{s['n_components']} | {s['n_unassigned_tx']} |"
            )
        lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    # Scenario 1: easy mode — full volume + ground-truth segmentation.
    # Copy cell_id → cell_id_truth so the comparison reference column
    # name is consistent across scenarios. The pipeline still receives
    # the ground-truth cell_id as its input segmentation.
    df_full, gt_full = make_synthetic_transcripts(**CELLS_KW, seed=42)
    df_full["cell_id_truth"] = df_full["cell_id"].astype(str)
    panel_full = make_synthetic_npmi_panel_for_transcripts(df_full, gt_full)
    r1 = _measure("full-volume + ground-truth", df_full, panel_full)

    # Scenario 2: section with ground-truth segmentation. Isolates the
    # effect of sectioning alone (clipped cells, lost cells) without
    # segmentation noise on top. Useful as a "section ceiling": the
    # best TRACER can do on this slab given a perfect upstream
    # segmenter.
    df_sec, gt_sec = make_synthetic_transcripts(
        **CELLS_KW, section_z_range_um=SECTION_Z, seed=42,
    )
    panel_sec = make_synthetic_npmi_panel_for_transcripts(df_sec, gt_sec)
    df_sec_gt = df_sec.copy()
    df_sec_gt["cell_id_truth"] = df_sec_gt["cell_id"].astype(str)
    r2 = _measure("section + ground-truth", df_sec_gt, panel_sec)

    # Scenario 3: realistic mode — section + simulated DAPI/Voronoi.
    # The sim overwrites cell_id with noisy labels and preserves the
    # original ground truth in cell_id_truth automatically.
    df_sec_seg = simulate_dapi_voronoi_segmentation(df_sec)
    r3 = _measure("section + DAPI/Voronoi", df_sec_seg, panel_sec)

    print(_format_block([r1, r2, r3]))


if __name__ == "__main__":
    main()
