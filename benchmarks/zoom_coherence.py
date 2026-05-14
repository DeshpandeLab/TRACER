#!/usr/bin/env python3
"""Compute coherence for nloapcgp-1 SEG entities and NOSEG fragments,
plus all pairwise unions, to test the 'deltaC_min blocks fragment absorption' hypothesis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
TAU = 0.2  # PMI scale, matches strict-PMI calibration
ZOOM_DIR = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg"


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
    W[gi, gj] = v; W[gj, gi] = v
    return W, gene_to_idx


def coherence(gene_set, W, gene_to_idx, tau=TAU):
    """C = purity - conflict over unique gene set."""
    gids = [gene_to_idx[g] for g in gene_set if g in gene_to_idx]
    if len(gids) < 2:
        return float("nan"), float("nan"), float("nan"), 0
    gids = np.asarray(sorted(set(gids)), dtype=np.int64)
    k = len(gids)
    sub = W[np.ix_(gids, gids)]
    iu = np.triu_indices(k, k=1)
    w = sub[iu]
    w = w[~np.isnan(w)]
    if w.size == 0:
        return float("nan"), float("nan"), float("nan"), k
    purity = float((w > tau).mean())
    conflict = float((w < -tau).mean())
    return purity - conflict, purity, conflict, k


def main() -> int:
    zoom = pd.read_parquet(ZOOM_DIR / "zoom_worst_tx.parquet")
    feats = pd.read_parquet(PDAC_PARQUET, columns=["transcript_id", "feature_name"])
    m = zoom.merge(feats, on="transcript_id", how="left")
    m["feature_name"] = m["feature_name"].astype(str)

    # Build W from union of all genes seen (panel + cell tx)
    panel_raw = pd.read_parquet(PANEL)
    all_genes = sorted(set(panel_raw["gene_i"].astype(str))
                        | set(panel_raw["gene_j"].astype(str))
                        | set(m["feature_name"].unique()))
    W, gtoi = _build_W(PANEL, all_genes)
    print(f"  W: {W.shape}, panel non-NaN pairs: {int((~np.isnan(W)).sum() // 2):,}",
          flush=True)
    print(f"  TAU = {TAU} (PMI scale)\n", flush=True)

    # 1. SEG entities
    sub_main = m[m["seg_lab"].astype(str) == "nloapcgp-1"]
    sub_part = m[m["seg_lab"].astype(str) == "nloapcgp-1-1"]
    genes_main = set(sub_main["feature_name"].unique())
    genes_part = set(sub_part["feature_name"].unique())
    C_main, P_main, X_main, k_main = coherence(genes_main, W, gtoi)
    C_part, P_part, X_part, k_part = coherence(genes_part, W, gtoi)
    print(f"SEG entities:")
    print(f"  nloapcgp-1   n_tx={len(sub_main)}  n_genes={k_main}  C={C_main:.4f}  purity={P_main:.4f}  conflict={X_main:.4f}")
    print(f"  nloapcgp-1-1 n_tx={len(sub_part)}  n_genes={k_part}  C={C_part:.4f}  purity={P_part:.4f}  conflict={X_part:.4f}")

    # 2. Hypothetical union (full cell_id nloapcgp-1)
    sub_all = m[m["cell_id"].astype(str) == "nloapcgp-1"]
    genes_all = set(sub_all["feature_name"].unique())
    C_all, P_all, X_all, k_all = coherence(genes_all, W, gtoi)
    print(f"\nIF MERGED (epi + CAF):")
    print(f"  union        n_tx={len(sub_all)}  n_genes={k_all}  C={C_all:.4f}  purity={P_all:.4f}  conflict={X_all:.4f}")
    # deltaC of the hypothetical merger (size-weighted average comparison)
    # Stitch's actual deltaC formula varies; the simplest comparison is
    # C(union) - max(C(main), C(part))
    deltaC_simple = C_all - max(C_main, C_part)
    print(f"  ΔC (union vs max-of-parts) = {deltaC_simple:+.4f}")

    # 3. NOSEG fragments — show C of each + best/worst pairwise union
    sub_noseg = m[m["cell_id"].astype(str) == "nloapcgp-1"]
    noseg_groups = sub_noseg.groupby("noseg_lab")
    print(f"\nNOSEG cascade fragments (nloapcgp-1's 157 tx, split into "
          f"{noseg_groups.ngroups} entities):", flush=True)
    SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}
    frag_records = []
    for nlab, grp in noseg_groups:
        if str(nlab) in SENTINELS:
            continue
        genes = set(grp["feature_name"].unique())
        C, P, X, k = coherence(genes, W, gtoi)
        frag_records.append({
            "label": nlab, "n_tx": len(grp), "n_genes": k,
            "C": C, "purity": P, "conflict": X,
            "genes": genes,
        })
    frag_df = pd.DataFrame(frag_records).sort_values("n_tx", ascending=False)
    print(f"  {'label':>22s}  {'n_tx':>4s}  {'n_genes':>7s}  {'C':>7s}  {'purity':>7s}  {'conflict':>8s}")
    for _, r in frag_df.iterrows():
        print(f"  {r['label']:>22s}  {r['n_tx']:>4d}  {r['n_genes']:>7d}  "
              f"{r['C']:>7.4f}  {r['purity']:>7.4f}  {r['conflict']:>8.4f}")

    # 4. Pairwise C of NOSEG fragment unions — what would Stitch see?
    # Production Stitch uses penalize_simplicity=True, which subtracts
    # 1/n to debias small entities:
    #   ΔC_pen = (C_union - 1/n_union) - max(C_u - 1/n_u, C_v - 1/n_v)
    print(f"\nNOSEG pairwise mergers — production Stitch uses penalize_simplicity=True:")
    print(f"  ΔC_raw = C(union) - max(C_A, C_B)")
    print(f"  ΔC_pen = (C(union) - 1/n_union) - max(C_A - 1/n_A, C_B - 1/n_B)")
    pair_records = []
    for i in range(len(frag_df)):
        for j in range(i+1, len(frag_df)):
            ra = frag_df.iloc[i]; rb = frag_df.iloc[j]
            genes_u = ra["genes"] | rb["genes"]
            Cu, Pu, Xu, ku = coherence(genes_u, W, gtoi)
            na = max(int(ra["n_genes"]), 1)
            nb = max(int(rb["n_genes"]), 1)
            n_union = max(ku, 1)
            dC_raw = Cu - max(ra["C"], rb["C"])
            C_sep_pen = max(ra["C"] - 1.0 / na, rb["C"] - 1.0 / nb)
            dC_pen = (Cu - 1.0 / n_union) - C_sep_pen
            pair_records.append((ra["label"], rb["label"], ra["C"], rb["C"],
                                 na, nb, Cu, n_union, dC_raw, dC_pen))
    pairs = pd.DataFrame(pair_records, columns=[
        "A","B","C_A","C_B","nA","nB","C_union","n_union","dC_raw","dC_pen"
    ])
    print(f"\nTop 10 by ΔC_pen (production-equivalent ranking):")
    print(f"  {'A':>22s}  {'B':>22s}  {'C_A':>6s}  {'C_B':>6s}  "
          f"{'nA':>3s} {'nB':>3s}  {'C_uni':>6s} {'n_uni':>5s}  "
          f"{'dC_raw':>7s} {'dC_pen':>7s}")
    for _, r in pairs.sort_values("dC_pen", ascending=False).head(10).iterrows():
        print(f"  {r['A']:>22s}  {r['B']:>22s}  {r['C_A']:>6.3f}  {r['C_B']:>6.3f}  "
              f"{int(r['nA']):>3d} {int(r['nB']):>3d}  "
              f"{r['C_union']:>6.3f} {int(r['n_union']):>5d}  "
              f"{r['dC_raw']:>+7.4f} {r['dC_pen']:>+7.4f}")
    print(f"\nBottom 10 by ΔC_pen (most coherence-hurting if merged):")
    for _, r in pairs.sort_values("dC_pen").head(10).iterrows():
        print(f"  {r['A']:>22s}  {r['B']:>22s}  {r['C_A']:>6.3f}  {r['C_B']:>6.3f}  "
              f"{int(r['nA']):>3d} {int(r['nB']):>3d}  "
              f"{r['C_union']:>6.3f} {int(r['n_union']):>5d}  "
              f"{r['dC_raw']:>+7.4f} {r['dC_pen']:>+7.4f}")
    print(f"\n  pairs with ΔC_raw ≥ +0.03: "
          f"{int((pairs['dC_raw'] >= 0.03).sum())} / {len(pairs)}")
    print(f"  pairs with ΔC_pen ≥ +0.03 (would pass production Stitch): "
          f"{int((pairs['dC_pen'] >= 0.03).sum())} / {len(pairs)}")
    print(f"  pairs with ΔC_raw ≤ 0: "
          f"{int((pairs['dC_raw'] <= 0).sum())} / {len(pairs)}")
    print(f"  pairs with ΔC_pen ≤ 0: "
          f"{int((pairs['dC_pen'] <= 0).sum())} / {len(pairs)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
