#!/usr/bin/env python3
"""Fast Leiden sweep on the saved Pearson-residual AnnData state.

Loads the .h5ad files produced by build_pdac_pearson_state.py and runs
sc.tl.leiden at a user-specified resolution for each subset. Plots
+ saves embeddings each time. Re-runs at any new resolution in seconds.

Usage:
    PYTHONPATH=src:. python sweep_pdac_pearson_leiden.py 0.3
    PYTHONPATH=src:. python sweep_pdac_pearson_leiden.py 0.15
    PYTHONPATH=src:. python sweep_pdac_pearson_leiden.py 0.5
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
import anndata as ad

warnings.filterwarnings("ignore", category=FutureWarning)
sc.settings.verbosity = 1

REPO = Path(__file__).resolve().parents[1]
STATE_DIR = REPO / "benchmarks" / "pdac_full_seq" / "scanpy_pearson_state"
RNG = 42


def _plot_pair(adata, name, leiden_res, out_path):
    emb = adata.obsm["X_umap"]
    sizes = adata.obs["n_tx"].to_numpy()
    leiden = adata.obs["leiden"].astype("category")
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

    cats = leiden.cat.categories
    n_cats = len(cats)
    cmap = plt.get_cmap("tab20" if n_cats <= 20 else "gist_ncar", n_cats)
    inv = leiden.cat.codes.to_numpy()
    order = np.argsort(np.bincount(inv))[::-1]
    ax2 = axes[1]
    for rank, k in enumerate(order):
        sel = inv == k
        ax2.scatter(emb[sel, 0], emb[sel, 1], s=1.2, alpha=0.6,
                     c=[cmap(rank % cmap.N)],
                     label=f"{cats[k]} ({int(sel.sum()):,})",
                     linewidths=0)
    leg = ax2.legend(loc="upper right", fontsize=6, markerscale=4,
                      framealpha=0.85, ncol=2 if n_cats > 10 else 1)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    ax2.set_title(f"{name} — leiden res={leiden_res}  ({n_cats} clusters)",
                   fontsize=11)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"PDAC Pearson residuals + leiden res={leiden_res}",
                  fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}", flush=True)


def main(leiden_res: float) -> int:
    out_dir = REPO / "benchmarks" / "pdac_full_seq" / f"scanpy_pearson_res{int(leiden_res*100):03d}_partial10"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    adatas = {}
    for name in ("input", "seg_cells", "seg_partials"):
        h5 = STATE_DIR / f"{name}.h5ad"
        if not h5.exists():
            print(f"[{name}] no state at {h5} — run build_pdac_pearson_state.py first",
                  flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        t = time.time()
        adata = sc.read_h5ad(h5)
        print(f"  loaded {adata.shape} in {time.time()-t:.1f}s", flush=True)
        t = time.time()
        sc.tl.leiden(
            adata, resolution=leiden_res, key_added="leiden",
            flavor="igraph", n_iterations=2, directed=False, random_state=RNG,
        )
        n_clusters = adata.obs["leiden"].nunique()
        print(f"  leiden res={leiden_res} → {n_clusters} clusters "
              f"({time.time()-t:.1f}s)", flush=True)

        _plot_pair(adata, name, leiden_res, out_dir / f"umap_scanpy_{name}.png")
        pd.DataFrame({
            "entity": adata.obs["entity"].to_numpy(),
            "umap_1": adata.obsm["X_umap"][:, 0],
            "umap_2": adata.obsm["X_umap"][:, 1],
            "n_tx": adata.obs["n_tx"].to_numpy(),
            "leiden": adata.obs["leiden"].astype(str).to_numpy(),
        }).to_parquet(out_dir / f"embeddings_{name}.parquet", index=False)
        adatas[name] = adata

    # Combined 3-panel
    if len(adatas) >= 2:
        fig, axes = plt.subplots(1, len(adatas),
                                  figsize=(7.5 * len(adatas), 7.5), dpi=110)
        if len(adatas) == 1:
            axes = [axes]
        for ax, (name, adata) in zip(axes, adatas.items()):
            emb = adata.obsm["X_umap"]
            leiden = adata.obs["leiden"].astype("category")
            cats = leiden.cat.categories
            n_cats = len(cats)
            cmap = plt.get_cmap("tab20" if n_cats <= 20 else "gist_ncar", n_cats)
            inv = leiden.cat.codes.to_numpy()
            order = np.argsort(np.bincount(inv))[::-1]
            for rank, k in enumerate(order):
                sel = inv == k
                ax.scatter(emb[sel, 0], emb[sel, 1], s=0.8, alpha=0.55,
                            c=[cmap(rank % cmap.N)],
                            label=f"{cats[k]} ({int(sel.sum()):,})",
                            linewidths=0)
            leg = ax.legend(loc="upper right", fontsize=6, markerscale=4,
                             framealpha=0.85, ncol=2 if n_cats > 10 else 1)
            for h in leg.legend_handles:
                h.set_alpha(1.0)
            ax.set_title(f"{name}  (n={adata.n_obs:,}, {n_cats} clusters)",
                          fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            ax.spines[["top", "right"]].set_visible(False)
        plt.suptitle(f"PDAC Pearson residuals — Leiden res={leiden_res}",
                      fontsize=12, y=1.0)
        plt.tight_layout()
        combined = out_dir / "umap_scanpy_leiden_combined.png"
        plt.savefig(combined, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  -> {combined}", flush=True)

    print(f"\ntotal wall: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    res = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    raise SystemExit(main(res))
