#!/usr/bin/env python3
"""Pairwise Mahalanobis-D for every multi-label D-cluster on the PDAC 2 mm ROI.

Reads the two saved snapshots at the Stitch boundary, identifies D-clusters
that absorbed >=2 C-labels, computes pairwise D between each C-label pair, and
saves both the per-pair table (parquet) and a histogram PNG.

The maha-rescue threshold is mahalanobis_d_rescue = 1.0 (default). Pairs with
D <= 1.0 in this output are direct rescues; pairs with D > 1.0 are cascading
second-order merges (one of the upstream merges was Maha-rescued, then a
subsequent normal-DeltaC merge admitted a further C-lab whose pairwise D
against the original is > 1.0).
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("analysis/phase1_vs_stitch_maha/pdac_2mm")


def maha_d(ca: np.ndarray, cb: np.ndarray) -> float:
    if ca.shape[0] < 2 or cb.shape[0] < 2:
        return float("nan")
    mu = ca.mean(axis=0) - cb.mean(axis=0)
    cov_a = np.atleast_2d(np.cov(ca, rowvar=False, ddof=1))
    cov_b = np.atleast_2d(np.cov(cb, rowvar=False, ddof=1))
    denom = ca.shape[0] + cb.shape[0] - 2
    cov = ((ca.shape[0] - 1) * cov_a + (cb.shape[0] - 1) * cov_b) / denom
    try:
        if not np.isfinite(np.linalg.cond(cov)) or np.linalg.cond(cov) > 1e12:
            return float("nan")
        sol = np.linalg.solve(cov, mu.reshape(-1, 1))
        d2 = float((mu.reshape(1, -1) @ sol).item())
    except np.linalg.LinAlgError:
        return float("nan")
    if not np.isfinite(d2) or d2 < 0:
        return float("nan")
    return float(np.sqrt(d2))


def main() -> int:
    c = pd.read_parquet(OUT / "snap_stitch_C.parquet")
    d = pd.read_parquet(OUT / "snap_stitch_D.parquet")
    c["lab"] = c.lab.astype(str)
    d["lab"] = d.lab.astype(str)
    m = c.rename(columns={"lab": "lab_C"}).merge(
        d[["transcript_id", "lab"]].rename(columns={"lab": "lab_D"}),
        on="transcript_id", how="inner")

    # D-clusters with >=2 distinct C-labs.
    g = m.groupby("lab_D")["lab_C"].nunique()
    multi = g[g >= 2].index
    print(f"multi-C-lab D-clusters: {len(multi)}")
    sub = m[m.lab_D.isin(multi)][["lab_D", "lab_C", "x", "y", "z"]]

    rows: list[tuple[str, str, str, int, int, float]] = []
    for lab_d, g_d in sub.groupby("lab_D"):
        clouds = {lc: g_c[["x", "y", "z"]].to_numpy() for lc, g_c in g_d.groupby("lab_C")}
        for a, b in combinations(sorted(clouds), 2):
            ca, cb = clouds[a], clouds[b]
            rows.append((lab_d, a, b, ca.shape[0], cb.shape[0], maha_d(ca, cb)))
    pairs = pd.DataFrame(rows, columns=["lab_D", "lab_a", "lab_b", "n_a", "n_b", "D"])
    pairs.to_parquet(OUT / "maha_d_pairs.parquet", index=False)
    print(f"saved {len(pairs)} pairs -> maha_d_pairs.parquet")

    f = pairs.D.dropna()
    print(f"D summary: n={len(f)} median={f.median():.3f} mean={f.mean():.3f} "
          f"p25={f.quantile(.25):.3f} p75={f.quantile(.75):.3f} max={f.max():.3f}")
    le1 = int((f <= 1.0).sum())
    print(f"D <= 1.0 (direct rescue regime): {le1} ({le1/len(f)*100:.1f}%)")
    print(f"D >  1.0 (cascading): {len(f)-le1} ({(len(f)-le1)/len(f)*100:.1f}%)")

    # Histogram (no matplotlib styling - just the bars).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=140)
    bins = np.arange(0.0, max(8.0, float(f.max()) + 0.5), 0.25)
    ax.hist(f, bins=bins, color="#4477aa", edgecolor="white", linewidth=0.4)
    ax.axvline(1.0, color="#cc3311", linestyle="--", linewidth=1.4,
               label="mahalanobis_d_rescue = 1.0")
    ax.set_xlabel("Mahalanobis D (pooled-cov, 3D)")
    ax.set_ylabel("# C-label pairs in same D-cluster")
    ax.set_title(f"PDAC 2 mm ROI — pairwise D within Maha-affected D-clusters "
                 f"(n={len(f)} pairs, {len(multi)} clusters)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0, bins[-1])
    fig.tight_layout()
    fig.savefig(OUT / "maha_d_hist.png", dpi=140)
    print(f"saved histogram -> {OUT / 'maha_d_hist.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
