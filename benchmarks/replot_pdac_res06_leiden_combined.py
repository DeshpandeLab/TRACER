#!/usr/bin/env python3
"""Re-plot lung scanpy_recipe_res06_partial10 as a 3-panel combined Leiden-colored view.

Reads the saved embeddings parquets, no recompute. Produces:
  umap_scanpy_leiden_combined.png  — 3 panels (input / cells / partials),
                                       each colored by its own Leiden cluster.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DIR = REPO / "benchmarks" / "pdac_full_seq" / "scanpy_recipe_res06_partial10"


def main() -> int:
    panels = [
        ("input",        DIR / "embeddings_input.parquet"),
        ("seg_cells",    DIR / "embeddings_seg_cells.parquet"),
        ("seg_partials", DIR / "embeddings_seg_partials.parquet"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(22.5, 7.5), dpi=110)
    for ax, (name, p) in zip(axes, panels):
        df = pd.read_parquet(p)
        emb = df[["umap_1", "umap_2"]].to_numpy()
        leiden = df["leiden"].astype(str)
        cats = sorted(leiden.unique(), key=lambda s: (len(s), s))
        n_cats = len(cats)
        cmap = plt.get_cmap("tab20" if n_cats <= 20 else "gist_ncar", n_cats)
        inv = pd.Categorical(leiden, categories=cats).codes
        order = np.argsort(np.bincount(inv))[::-1]
        for rank, k in enumerate(order):
            sel = inv == k
            ax.scatter(emb[sel, 0], emb[sel, 1], s=1.2, alpha=0.6,
                        c=[cmap(rank % cmap.N)],
                        label=f"{cats[k]} ({int(sel.sum()):,})",
                        linewidths=0)
        leg = ax.legend(loc="upper right", fontsize=6, markerscale=4,
                         framealpha=0.85,
                         ncol=2 if n_cats > 10 else 1)
        for h in leg.legend_handles:
            h.set_alpha(1.0)
        ax.set_title(f"{name}  (n={len(df):,}, {n_cats} clusters)", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("PDAC scanpy recipe — Leiden res=0.6 clusters per subset  "
                  "(normalize_total + log1p + PCA(300) + n_neighbors=30 + "
                  "UMAP min_dist=0.1)",
                  fontsize=12, y=1.0)
    plt.tight_layout()
    out = DIR / "umap_scanpy_leiden_combined.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
