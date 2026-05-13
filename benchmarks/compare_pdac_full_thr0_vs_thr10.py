#!/usr/bin/env python3
"""Compare full-PDAC SEG outputs between two PMI panels: thr=0 vs thr=10.

Reports:
  - runtime / peak RSS / entity counts (from each summary.json)
  - per-tx label agreement (ARI / RI / NMI / h / c / V) on assigned-in-both
  - per-tx label agreement under singleton encoding
  - per-entity overlap: how many thr10 entities map cleanly to thr0?
  - assignment-status crosstab (assigned in only-thr0, only-thr10, both, neither)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score, rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)

REPO = Path(__file__).resolve().parents[1]
DIR_A = REPO / "benchmarks" / "pdac_full_seq"        # thr=10
DIR_B = REPO / "benchmarks" / "pdac_full_seq_thr0"    # thr=0
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}


def _is_un(s: pd.Series) -> np.ndarray:
    return (s.isin(SENTINELS) | s.str.endswith("_rejected", na=False)).to_numpy()


def _codes(labels: pd.Series, singletons: bool = True) -> np.ndarray:
    is_un = _is_un(labels)
    codes, _ = pd.factorize(labels.to_numpy(), sort=False)
    codes = codes.astype(np.int64)
    if singletons:
        tx_idx = np.arange(labels.size, dtype=np.int64)
        codes[is_un] = -2 - tx_idx[is_un]
    else:
        codes[is_un] = -1
    return codes


def main() -> int:
    # Summaries
    s_a = json.loads((DIR_A / "summary.json").read_text())
    s_b = json.loads((DIR_B / "summary.json").read_text())
    print("=== runtime / entity counts ===")
    print(f"  {'metric':24s}  {'thr=10':>12s}  {'thr=0':>12s}  {'Δ (thr0 − thr10)':>18s}")
    for k in [
        "panel_rows", "wall_seconds", "peak_rss_gb",
        "n_cells_out", "cells_lost", "retention_pct",
        "n_partials", "n_components",
        "n_assigned_tx", "n_unassigned_tx", "coverage_pct",
    ]:
        va = s_a.get(k); vb = s_b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            print(f"  {k:24s}  {va:>12}  {vb:>12}  {vb-va:>+18}")
        else:
            print(f"  {k:24s}  {str(va):>12s}  {str(vb):>12s}")

    # Per-tx partitions
    print("\nloading partitions ...", flush=True)
    pa = pd.read_parquet(DIR_A / "partition_sequential.parquet")
    pb = pd.read_parquet(DIR_B / "partition_sequential.parquet")
    assert (pa["transcript_id"].to_numpy() == pb["transcript_id"].to_numpy()).all(), \
        "tx order mismatch"
    lab_a = pa["label"].astype(str)
    lab_b = pb["label"].astype(str)
    un_a = _is_un(lab_a); un_b = _is_un(lab_b)

    # Assignment-status crosstab
    print("\n=== assignment-status crosstab (tx-level) ===")
    n_tx = len(pa)
    n_both = int((~un_a & ~un_b).sum())
    n_only_a = int((~un_a & un_b).sum())
    n_only_b = int((un_a & ~un_b).sum())
    n_neither = int((un_a & un_b).sum())
    print(f"  total tx:               {n_tx:>12,d}")
    print(f"  assigned in BOTH:       {n_both:>12,d}  ({100*n_both/n_tx:.2f}%)")
    print(f"  assigned only in thr10: {n_only_a:>12,d}  ({100*n_only_a/n_tx:.2f}%)")
    print(f"  assigned only in thr0:  {n_only_b:>12,d}  ({100*n_only_b/n_tx:.2f}%)")
    print(f"  unassigned in BOTH:     {n_neither:>12,d}  ({100*n_neither/n_tx:.2f}%)")

    # Per-tx ARI / h / c / V
    print("\n=== per-tx agreement (thr0 vs thr10) ===")
    print(f"  {'encoding':24s}  {'n':>12s}  {'ARI':>7s}  {'RI':>7s}  "
          f"{'NMI':>7s}  {'h':>7s}  {'c':>7s}  {'V':>7s}")
    sing_a = _codes(lab_a, True); sing_b = _codes(lab_b, True)
    mega_a = _codes(lab_a, False); mega_b = _codes(lab_b, False)
    both = (~un_a) & (~un_b)
    for set_label, sl, tl, n in [
        ("singletons (all tx)", sing_a, sing_b, len(pa)),
        ("assigned-in-both",    mega_a[both], mega_b[both], int(both.sum())),
    ]:
        ari = float(adjusted_rand_score(sl, tl))
        ri = float(rand_score(sl, tl))
        nmi = float(normalized_mutual_info_score(sl, tl))
        h, c, v = homogeneity_completeness_v_measure(sl, tl)
        print(f"  {set_label:24s}  {n:>12,d}  {ari:>7.4f}  {ri:>7.4f}  "
              f"{nmi:>7.4f}  {h:>7.4f}  {c:>7.4f}  {v:>7.4f}")

    # Per-entity mapping: for each thr10 entity (assigned in thr10), what fraction
    # of its tx ended up in a SINGLE thr0 entity? (= purity)
    print("\n=== per-entity purity (thr10 → thr0 mode) ===")
    sub = pd.DataFrame({"a": lab_a[~un_a].to_numpy(),
                         "b": lab_b[~un_a].to_numpy()})
    by_a = sub.groupby("a")["b"]
    sizes = by_a.size()
    mode_count = by_a.apply(lambda x: x.value_counts().iloc[0])
    purity = mode_count / sizes
    # Treat "assigned in thr10 but unassigned in thr0" as a special bin:
    # how many thr10 entities completely vanished in thr0?
    unassigned_in_b = sub["b"].isin(SENTINELS) | sub["b"].str.endswith("_rejected", na=False)
    frac_b_unassigned = sub.assign(unb=unassigned_in_b.to_numpy()).groupby("a")["unb"].mean()
    n_thr10_ent = len(sizes)
    n_pure = int((purity >= 0.99).sum())
    n_lost_in_b = int((frac_b_unassigned >= 0.5).sum())
    n_split = int((purity < 0.99).sum())
    print(f"  thr10 entities (assigned): {n_thr10_ent:,}")
    print(f"  high purity in thr0 (>=99%): {n_pure:,}  ({100*n_pure/n_thr10_ent:.1f}%)")
    print(f"  >=50% of tx unassigned in thr0: {n_lost_in_b:,}  ({100*n_lost_in_b/n_thr10_ent:.1f}%)")
    print(f"  weighted-mean purity (by size):  "
          f"{float(np.average(purity, weights=sizes)):.4f}")

    # Also reverse: thr0 → thr10
    print("\n=== per-entity purity (thr0 → thr10 mode) ===")
    sub2 = pd.DataFrame({"a": lab_a[~un_b].to_numpy(),
                          "b": lab_b[~un_b].to_numpy()})
    by_b = sub2.groupby("b")["a"]
    sizes_b = by_b.size()
    mode_count_b = by_b.apply(lambda x: x.value_counts().iloc[0])
    purity_b = mode_count_b / sizes_b
    n_thr0_ent = len(sizes_b)
    n_pure_b = int((purity_b >= 0.99).sum())
    print(f"  thr0 entities (assigned): {n_thr0_ent:,}")
    print(f"  high purity in thr10 (>=99%): {n_pure_b:,}  ({100*n_pure_b/n_thr0_ent:.1f}%)")
    print(f"  weighted-mean purity (by size):  "
          f"{float(np.average(purity_b, weights=sizes_b)):.4f}")

    # Per-tx etype crosstab
    if "_etype" in pa.columns and "_etype" in pb.columns:
        print("\n=== _etype crosstab (per-tx, both panels assigned) ===")
        ea = pa["_etype"].astype(str)
        eb = pb["_etype"].astype(str)
        mask = (~un_a) & (~un_b)
        ct = pd.crosstab(ea[mask], eb[mask], rownames=["thr10"], colnames=["thr0"])
        print(ct.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
