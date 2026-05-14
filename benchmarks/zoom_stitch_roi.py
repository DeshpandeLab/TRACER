#!/usr/bin/env python3
"""Run production Stitch on the worst-case ROI (cell nloapcgp-1) with the
current stitch-dist-threshold-fix worktree settings (capped witness +
c_union_bypass=0.9, deltaC_min=0.03), and compare the final stitched
components against SEG's (epi-main + CAF-partial) partition.

This invokes the REAL `apply_stitching_to_transcripts_memory_efficient`
function — not a simulation — so iterative boundary expansion, heap
ordering, and DSU lazy revalidation behave exactly as in production.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tracer.stitching import apply_stitching_to_transcripts_memory_efficient

PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
ZOOM_DIR = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg"

PMI_THR = 0.2
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}


def _build_W(panel_path, all_genes):
    panel = pd.read_parquet(panel_path).rename(columns={"value": "NPMI"})
    panel["gene_i"] = panel["gene_i"].astype(str)
    panel["gene_j"] = panel["gene_j"].astype(str)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    G = len(all_genes)
    W = np.full((G, G), np.nan, dtype=np.float32)
    gi = panel["gene_i"].map(gene_to_idx)
    gj = panel["gene_j"].map(gene_to_idx)
    have = gi.notna() & gj.notna()
    gi = gi[have].to_numpy(dtype=np.int64)
    gj = gj[have].to_numpy(dtype=np.int64)
    v = panel.loc[have, "NPMI"].to_numpy(dtype=np.float32)
    W[gi, gj] = v
    W[gj, gi] = v
    np.fill_diagonal(W, np.nan)
    return W, gene_to_idx


def main() -> int:
    zoom = pd.read_parquet(ZOOM_DIR / "zoom_worst_tx.parquet")
    feats = pd.read_parquet(
        PDAC_PARQUET, columns=["transcript_id", "feature_name"]
    )
    z_col = pd.read_parquet(
        PDAC_PARQUET, columns=["transcript_id", "z_location"]
    ).rename(columns={"z_location": "z"})
    df = zoom.merge(feats, on="transcript_id", how="left")
    df = df.merge(z_col, on="transcript_id", how="left")
    df["feature_name"] = df["feature_name"].astype(str)
    df["noseg_lab"] = df["noseg_lab"].astype(str)
    df["seg_lab"] = df["seg_lab"].astype(str)

    # Restrict to the 157 tx of cell nloapcgp-1
    cell = df[df["cell_id"].astype(str) == "nloapcgp-1"].copy().reset_index(
        drop=True
    )
    print(f"nloapcgp-1: {len(cell)} tx", flush=True)

    # Build W from union of genes seen
    panel_raw = pd.read_parquet(PANEL)
    all_genes = sorted(
        set(panel_raw["gene_i"].astype(str))
        | set(panel_raw["gene_j"].astype(str))
        | set(cell["feature_name"].unique())
    )
    W, gene_to_idx = _build_W(PANEL, all_genes)
    aux = {"W": W, "gene_to_idx": gene_to_idx}

    # Stitch's API needs an `entity_col` whose value is the pre-stitch
    # entity label. Use the cascade label (NOSEG output).
    cell["tracer_id"] = cell["noseg_lab"]

    # Sentinel tx must carry a recognisable label; mark them "-1".
    cell.loc[cell["tracer_id"].isin(SENTINELS), "tracer_id"] = "-1"

    n_pre = int((~cell["tracer_id"].isin(SENTINELS)).sum())
    pre_entities = sorted(
        cell.loc[~cell["tracer_id"].isin(SENTINELS), "tracer_id"].unique()
    )
    print(f"\nPRE-Stitch (NOSEG cascade entities): {len(pre_entities)} "
          f"entities covering {n_pre} tx", flush=True)
    for e in pre_entities:
        n = int((cell["tracer_id"] == e).sum())
        print(f"  {e:>22s}  n_tx={n}")

    # Run the real production stitch. Settings mirror tests/_pipeline_runner.py
    # for this worktree:
    #     deltaC_min=0.03, penalize_simplicity=True
    #     c_union_bypass=0.9          (new)
    #     min_local_tx_per_entity=3   (capped at n_tx internally — new)
    #     candidate_source="grid", G=2.0, G_z=1.0, neighborhood="8",
    #     z_neighbor_depth=1
    print(f"\nrunning Stitch...", flush=True)
    df_stitched, _ = apply_stitching_to_transcripts_memory_efficient(
        df_final=cell, aux=aux,
        entity_col="tracer_id", gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        mode="count", threshold=PMI_THR, metric="pmi",
        penalize_simplicity=True, deltaC_min=0.03,
        c_union_bypass=0.9,
        dist_threshold=5.0, out_col="stitched", show_progress=False,
        candidate_source="grid", G=2.0, stitch_neighborhood="8",
        G_z=1.0, z_neighbor_depth=1,
        min_local_tx_per_entity=3,
    )

    # Final stitched component breakdown
    df_stitched["stitched"] = df_stitched["stitched"].astype(str)
    post_assigned = df_stitched[~df_stitched["stitched"].isin(SENTINELS)]
    post_components = (
        post_assigned.groupby("stitched", observed=True).agg(
            n_tx=("transcript_id", "size"),
            pre_entities=("tracer_id", lambda s: sorted(set(s))),
            seg_labs=("seg_lab", lambda s: sorted(set(s))),
        ).reset_index().sort_values("n_tx", ascending=False)
    )
    print(f"\nPOST-Stitch components: {len(post_components)}", flush=True)
    print(f"  {'stitched':>26s}  {'n_tx':>5s}  pre_entities", flush=True)
    for _, r in post_components.iterrows():
        pre = ", ".join(p for p in r["pre_entities"] if p not in SENTINELS)
        print(f"  {r['stitched']:>26s}  {int(r['n_tx']):>5d}  {pre}")

    # SEG reference partition
    seg = cell[~cell["seg_lab"].isin(SENTINELS)]
    print(f"\nSEG reference partition: "
          f"{seg['seg_lab'].nunique()} entities, {len(seg)} tx", flush=True)
    for sl in sorted(seg["seg_lab"].unique()):
        n = int((cell["seg_lab"] == sl).sum())
        print(f"  {sl:>26s}  n_tx={n}")

    # Cross-tabulation: post-Stitch component vs SEG label, restricted
    # to the tx that BOTH partitioners assigned (both non-sentinel).
    both = df_stitched[
        (~df_stitched["stitched"].isin(SENTINELS))
        & (~df_stitched["seg_lab"].isin(SENTINELS))
    ]
    print(f"\nConfusion matrix (rows = post-Stitch comp, "
          f"cols = SEG label), n={len(both)} tx in both:", flush=True)
    ctab = (
        both.groupby(["stitched", "seg_lab"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print(ctab.to_string(), flush=True)

    # Quick agreement summary: for each post-component, what SEG label
    # carries its majority of tx? Compute "purity vs SEG".
    print(f"\nPer-Stitch-component SEG agreement:", flush=True)
    for sc, grp in both.groupby("stitched", observed=True):
        counts = grp["seg_lab"].value_counts()
        top = counts.idxmax()
        purity = counts.iloc[0] / counts.sum()
        print(f"  {sc:>26s}  n={len(grp):>3d}  "
              f"top_SEG={top}  purity={purity:.3f}  "
              f"(counts: {dict(counts)})")

    # And the reverse — for each SEG entity, how do its tx distribute
    # across Stitch components?
    print(f"\nPer-SEG-entity Stitch fragmentation:", flush=True)
    for sl, grp in both.groupby("seg_lab", observed=True):
        counts = grp["stitched"].value_counts()
        print(f"  {sl:>26s}  n={len(grp):>3d}  "
              f"n_stitch_comps={counts.size}  "
              f"(counts: {dict(counts)})")

    # Per-tx headline ARI vs SEG
    try:
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(both["seg_lab"].astype(str),
                                   both["stitched"].astype(str))
        print(f"\nARI (post-Stitch vs SEG, n={len(both)}): {ari:.4f}",
              flush=True)
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
