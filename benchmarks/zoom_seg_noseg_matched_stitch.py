#!/usr/bin/env python3
"""Apples-to-apples SEG vs NOSEG bench at matched Stitch configs.

Three Stitch settings tested:
    G=2 µm, R=1 (production scale + bypass + capped witness)
    G=1 µm, R=2
    G=1 µm, R=3

For each:
    - Re-stitch SEG starting from its final partition (seg_lab as pre-Stitch input)
    - Re-stitch NOSEG starting from its final partition (noseg_lab as pre-Stitch input)
    - Compute:
        ARI(seg_lab original, NOSEG_re-stitched)   # what we were reporting
        ARI(SEG_re-stitched, NOSEG_re-stitched)    # apples-to-apples
        ARI(seg_lab original, SEG_re-stitched)     # how much SEG drifts under each Stitch

Notes:
    Re-stitching from the post-Stitch partition (not from the pre-Stitch state)
    underestimates how much SEG's Stitch would change, because the original
    Stitch's mergers can't be undone here. But it's the best available proxy
    without rerunning the full upstream pipeline on this ROI. Mergers it
    DOES make show drift; any genuine over-merge under aggressive settings
    will surface here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tracer.stitching import apply_stitching_to_transcripts_memory_efficient
from sklearn.metrics import adjusted_rand_score

PMI_THR = 0.2
SENT = {"-1", "DROP", "UNASSIGNED", "nan"}
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
PDAC = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
ZOOM = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg" / "zoom_worst_tx.parquet"


def _restitch(df_in, aux, input_col, G_xy, neighborhood):
    df_in = df_in.copy()
    df_in["tracer_id"] = df_in[input_col].astype(str)
    df_in.loc[df_in["tracer_id"].isin(SENT), "tracer_id"] = "-1"
    df_s, _ = apply_stitching_to_transcripts_memory_efficient(
        df_final=df_in, aux=aux,
        entity_col="tracer_id", gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        mode="count", threshold=PMI_THR, metric="pmi",
        penalize_simplicity=True, deltaC_min=0.03,
        c_union_bypass=0.9,
        dist_threshold=5.0, out_col="stitched", show_progress=False,
        candidate_source="grid", G=G_xy, stitch_neighborhood=neighborhood,
        G_z=1.0, z_neighbor_depth=1, min_local_tx_per_entity=3,
    )
    return df_s["stitched"].astype(str)


def main() -> int:
    zoom = pd.read_parquet(ZOOM)
    feats = pd.read_parquet(PDAC, columns=["transcript_id", "feature_name"])
    zcol = pd.read_parquet(PDAC, columns=["transcript_id", "z_location"]).rename(
        columns={"z_location": "z"})
    df = zoom.merge(feats, on="transcript_id", how="left").merge(
        zcol, on="transcript_id", how="left")
    df["feature_name"] = df["feature_name"].astype(str)
    df["seg_lab"] = df["seg_lab"].astype(str)
    df["noseg_lab"] = df["noseg_lab"].astype(str)
    df = df.reset_index(drop=True)

    panel_raw = pd.read_parquet(PANEL)
    all_genes = sorted(set(panel_raw["gene_i"].astype(str))
                       | set(panel_raw["gene_j"].astype(str))
                       | set(df["feature_name"].unique()))
    g2i = {g: i for i, g in enumerate(all_genes)}
    G = len(all_genes)
    W = np.full((G, G), np.nan, dtype=np.float32)
    gi = panel_raw["gene_i"].astype(str).map(g2i)
    gj = panel_raw["gene_j"].astype(str).map(g2i)
    have = gi.notna() & gj.notna()
    gi = gi[have].to_numpy(np.int64); gj = gj[have].to_numpy(np.int64)
    v = panel_raw.loc[have, "value"].to_numpy(np.float32)
    W[gi, gj] = v; W[gj, gi] = v
    np.fill_diagonal(W, np.nan)
    aux = {"W": W, "gene_to_idx": g2i}

    seg_orig = df["seg_lab"].copy()
    noseg_orig = df["noseg_lab"].copy()

    configs = [
        ("G=2, R=1",   2.0, "8"),
        ("G=1, R=2",   1.0, "R2"),
        ("G=1, R=3",   1.0, "R3"),
    ]

    print(f"{'config':<10s}  {'SEG ents':>8s}  {'NOSEG ents':>10s}  "
          f"{'ARI(seg_orig vs nosegΣ)':>22s}  {'ARI(segΣ vs nosegΣ)':>20s}  "
          f"{'ARI(seg_orig vs segΣ)':>21s}")
    for label, G_xy, neigh in configs:
        seg_re = _restitch(df, aux, "seg_lab", G_xy, neigh)
        noseg_re = _restitch(df, aux, "noseg_lab", G_xy, neigh)

        n_seg = seg_re[~seg_re.isin(SENT)].nunique()
        n_noseg = noseg_re[~noseg_re.isin(SENT)].nunique()

        both_old = (~seg_orig.isin(SENT)) & (~noseg_re.isin(SENT))
        ari_old = adjusted_rand_score(
            seg_orig[both_old].astype(str), noseg_re[both_old].astype(str)
        )
        both_match = (~seg_re.isin(SENT)) & (~noseg_re.isin(SENT))
        ari_match = adjusted_rand_score(
            seg_re[both_match].astype(str), noseg_re[both_match].astype(str)
        )
        both_seg = (~seg_orig.isin(SENT)) & (~seg_re.isin(SENT))
        ari_seg_drift = adjusted_rand_score(
            seg_orig[both_seg].astype(str), seg_re[both_seg].astype(str)
        )

        print(f"{label:<10s}  {n_seg:>8d}  {n_noseg:>10d}  "
              f"{ari_old:>22.4f}  {ari_match:>20.4f}  {ari_seg_drift:>21.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
