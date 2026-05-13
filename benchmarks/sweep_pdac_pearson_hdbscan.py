#!/usr/bin/env python3
"""HDBSCAN clustering on the saved Pearson-UMAP embedding.

Loads scanpy_pearson_state/{input,seg_cells,seg_partials}.h5ad and runs
HDBSCAN on the UMAP coordinates. HDBSCAN finds clusters at each point's
natural density scale — no resolution parameter, and points that don't
fit any dense region get labeled -1 ("noise"). Useful for spotting
artifact entities (small-tx clusters tend to land in noise).

Usage:
    PYTHONPATH=src:. python sweep_pdac_pearson_hdbscan.py [min_cluster_size]

Default min_cluster_size = 200.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import hdbscan

warnings.filterwarnings("ignore", category=FutureWarning)
sc.settings.verbosity = 1

REPO = Path(__file__).resolve().parents[1]
STATE_DIR = REPO / "benchmarks" / "pdac_full_seq" / "scanpy_pearson_state"
RNG = 42


def _plot_pair(adata, name, min_cluster_size, out_path):
    emb = adata.obsm["X_umap"]
    sizes = adata.obs["n_tx"].to_numpy()
    labels = adata.obs["hdbscan"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=120)

    color = np.log10(sizes)
    lo, hi = np.percentile(color, [1, 99])
    sc_ax = axes[0]
    s = sc_ax.scatter(emb[:, 0], emb[:, 1], c=np.clip(color, lo, hi),
                       s=1.5, alpha=0.6, cmap="viridis", linewidths=0)
    plt.colorbar(s, ax=sc_ax, fraction=0.04, pad=0.02, label="log10(n_tx)")
    sc_ax.set_title(f"{name} — log10(n_tx)", fontsize=11)
    sc_ax.set_xticks([]); sc_ax.set_yticks([])
    sc_ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: HDBSCAN clusters
    unique = np.unique(labels)
    n_clust = int((unique >= 0).sum())
    n_noise = int((labels == -1).sum())
    cmap = plt.get_cmap("tab20" if n_clust <= 20 else "gist_ncar", max(n_clust, 1))
    ax2 = axes[1]
    # Plot noise first (gray, low alpha), then clusters on top
    noise_sel = labels == -1
    if noise_sel.any():
        ax2.scatter(emb[noise_sel, 0], emb[noise_sel, 1], s=1.0, alpha=0.3,
                     c="#aaaaaa", label=f"noise ({int(noise_sel.sum()):,})",
                     linewidths=0)
    for rank, k in enumerate(sorted(unique[unique >= 0],
                                       key=lambda c: -(labels == c).sum())):
        sel = labels == k
        ax2.scatter(emb[sel, 0], emb[sel, 1], s=1.2, alpha=0.6,
                     c=[cmap(rank % cmap.N)],
                     label=f"{k} ({int(sel.sum()):,})",
                     linewidths=0)
    leg = ax2.legend(loc="upper right", fontsize=6, markerscale=4,
                      framealpha=0.85, ncol=2 if n_clust > 10 else 1)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    ax2.set_title(f"{name} — HDBSCAN min_cluster_size={min_cluster_size}  "
                   f"({n_clust} clusters, {n_noise:,} noise)",
                   fontsize=11)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"PDAC Pearson residuals + HDBSCAN on UMAP", fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}", flush=True)


def main(min_cluster_size: int) -> int:
    out_dir = REPO / "benchmarks" / "pdac_full_seq" / f"scanpy_pearson_hdbscan_mcs{min_cluster_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    adatas = {}
    for name in ("input", "seg_cells", "seg_partials"):
        h5 = STATE_DIR / f"{name}.h5ad"
        if not h5.exists():
            print(f"[{name}] no state at {h5}", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        t = time.time()
        adata = sc.read_h5ad(h5)
        print(f"  loaded {adata.shape} in {time.time()-t:.1f}s", flush=True)

        emb = adata.obsm["X_umap"].astype(np.float32)
        t = time.time()
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(10, min_cluster_size // 10),
            cluster_selection_method="eom",
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(emb)
        n_clust = int(len(set(labels)) - (1 if -1 in labels else 0))
        n_noise = int((labels == -1).sum())
        print(f"  HDBSCAN: {n_clust} clusters, {n_noise:,} noise points "
              f"({100*n_noise/len(labels):.1f}%)  in {time.time()-t:.1f}s",
              flush=True)
        adata.obs["hdbscan"] = labels.astype(int)

        _plot_pair(adata, name, min_cluster_size,
                    out_dir / f"umap_hdbscan_{name}.png")
        pd.DataFrame({
            "entity": adata.obs["entity"].to_numpy(),
            "umap_1": emb[:, 0], "umap_2": emb[:, 1],
            "n_tx": adata.obs["n_tx"].to_numpy(),
            "hdbscan": labels.astype(int),
        }).to_parquet(out_dir / f"embeddings_{name}.parquet", index=False)
        adatas[name] = adata

    print(f"\ntotal wall: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    mcs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    raise SystemExit(main(mcs))
