#!/usr/bin/env python3
"""Re-run production Stitch on the full 50µm ROI with G_xy=1.0 µm and
xy neighborhood R=3 (Moore-3, ±3 bins). All other knobs match the
current production config (G_z=1.0, z_depth=1, deltaC_min=0.03,
c_union_bypass=0.9, capped min_local_tx_per_entity=3,
penalize_simplicity=True).

Goal: see whether finer xy bins + wider xy reach merges the three CAF
clumps that the current G=2/R=1 setting leaves separated for cell
`nloapcgp-1`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tracer.stitching import apply_stitching_to_transcripts_memory_efficient

PMI_THR = 0.2
SENT = {"-1", "DROP", "UNASSIGNED", "nan"}
TARGETS = sorted({
    "cascade_103590-1", "cascade_103636-1", "cascade_103744-1-1",
    "cascade_135796-1", "cascade_135869-1-1", "cascade_136024-1",
    "cascade_25605-1", "cascade_25689-1", "cascade_38817-1",
    "cascade_5180-1", "cascade_55985-1", "cascade_77253-1-1",
    "cascade_77286-1",
})

PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
PDAC = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
ZOOM = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg" / "zoom_worst_tx.parquet"


def main() -> int:
    zoom = pd.read_parquet(ZOOM)
    feats = pd.read_parquet(PDAC, columns=["transcript_id", "feature_name"])
    zcol = pd.read_parquet(PDAC, columns=["transcript_id", "z_location"]).rename(
        columns={"z_location": "z"}
    )
    df = zoom.merge(feats, on="transcript_id", how="left").merge(
        zcol, on="transcript_id", how="left"
    )
    df["feature_name"] = df["feature_name"].astype(str)
    df["noseg_lab"] = df["noseg_lab"].astype(str)
    df = df.reset_index(drop=True)
    df["tracer_id"] = df["noseg_lab"]
    df.loc[df["tracer_id"].isin(SENT), "tracer_id"] = "-1"

    panel_raw = pd.read_parquet(PANEL)
    all_genes = sorted(
        set(panel_raw["gene_i"].astype(str))
        | set(panel_raw["gene_j"].astype(str))
        | set(df["feature_name"].unique())
    )
    g2i = {g: i for i, g in enumerate(all_genes)}
    G = len(all_genes)
    W = np.full((G, G), np.nan, dtype=np.float32)
    gi = panel_raw["gene_i"].astype(str).map(g2i)
    gj = panel_raw["gene_j"].astype(str).map(g2i)
    have = gi.notna() & gj.notna()
    gi = gi[have].to_numpy(np.int64)
    gj = gj[have].to_numpy(np.int64)
    v = panel_raw.loc[have, "value"].to_numpy(np.float32)
    W[gi, gj] = v; W[gj, gi] = v
    np.fill_diagonal(W, np.nan)

    print(f"Full ROI: {len(df)} tx, "
          f"{df['tracer_id'].nunique()} unique entities", flush=True)
    print(f"Running Stitch with G_xy=1.0, stitch_neighborhood='R3', "
          f"G_z=1.0, z_neighbor_depth=1 ...", flush=True)

    df_s, _ = apply_stitching_to_transcripts_memory_efficient(
        df_final=df, aux={"W": W, "gene_to_idx": g2i},
        entity_col="tracer_id", gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        mode="count", threshold=PMI_THR, metric="pmi",
        penalize_simplicity=True, deltaC_min=0.03,
        c_union_bypass=0.9,
        dist_threshold=5.0, out_col="stitched", show_progress=False,
        candidate_source="grid",
        G=1.0,                       # ← G_xy = 1 µm (was 2)
        stitch_neighborhood="R3",    # ← Moore-3 (was "8" = Moore-1)
        G_z=1.0, z_neighbor_depth=1,
        min_local_tx_per_entity=3,
    )
    df_s["stitched"] = df_s["stitched"].astype(str)

    mask = df_s["tracer_id"].isin(TARGETS)
    sub = df_s.loc[mask].copy()
    print(f"\nTx in 13 target fragments: {len(sub)}", flush=True)

    # Group by final stitched component, list which target fragments + extras
    comp_for_target = (
        sub.groupby("stitched", observed=True)
        .agg(n_tx_target=("stitched", "size"),
             targets_in=("tracer_id", lambda s: sorted(set(s))))
        .reset_index().sort_values("n_tx_target", ascending=False)
    )
    print(f"\nStitched components containing any of the 13 targets: "
          f"{len(comp_for_target)}", flush=True)
    print(f"  {'stitched':>28s}  {'n_tx':>5s}  pre_entities", flush=True)
    for _, r in comp_for_target.iterrows():
        full_set = sorted(
            p for p in df_s[df_s["stitched"] == r["stitched"]]["tracer_id"]
            .unique() if p not in SENT
        )
        outside = [p for p in full_set if p not in TARGETS]
        msg = f"target_members={r['targets_in']}"
        if outside:
            msg += f"  +absorbed-from-outside ({len(outside)}): {outside}"
        print(f"  {r['stitched']:>28s}  {int(r['n_tx_target']):>5d}  {msg}")

    # Did the three CAF clumps merge?
    caf_a = {"cascade_103636-1", "cascade_5180-1"}
    caf_b = {"cascade_135796-1", "cascade_135869-1-1", "cascade_77253-1-1"}
    caf_c = {"cascade_136024-1"}
    print(f"\nCAF clump merger check:", flush=True)
    for name, members in [("clump_A (103636+5180)", caf_a),
                           ("clump_B (135796+135869-1-1+77253-1-1)", caf_b),
                           ("clump_C (136024)", caf_c)]:
        st = sub.loc[sub["tracer_id"].isin(members), "stitched"].unique()
        print(f"  {name:<40s} → stitched components: {sorted(st)}")

    # Did all 3 CAFs end up in the same stitched comp?
    all_caf_members = caf_a | caf_b | caf_c
    all_caf_st = set(sub.loc[sub["tracer_id"].isin(all_caf_members),
                              "stitched"].unique())
    print(f"\nNumber of distinct stitched comps spanning the 3 CAF clumps: "
          f"{len(all_caf_st)}")
    if len(all_caf_st) == 1:
        print(f"  → ALL CAFs merged into one component ✓")
    else:
        print(f"  → CAFs remain split across {len(all_caf_st)} comps:")
        for st in sorted(all_caf_st):
            members_here = sorted(
                set(sub.loc[sub["stitched"] == st, "tracer_id"])
                & all_caf_members
            )
            print(f"      {st}: {members_here}")

    # Compare to SEG: per-Stitch-comp SEG purity
    both = df_s[(~df_s["stitched"].isin(SENT))
                & (~df_s["seg_lab"].astype(str).isin(SENT))]
    print(f"\nARI (post-Stitch vs SEG, full ROI, n={len(both)}):", flush=True)
    try:
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(
            both["seg_lab"].astype(str), both["stitched"].astype(str)
        )
        print(f"  {ari:.4f}", flush=True)
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
