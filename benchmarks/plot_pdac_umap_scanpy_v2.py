#!/usr/bin/env python3
"""PDAC UMAP via the canonical scanpy recipe from metrics_umap.ipynb.

Mirrors the notebook's run_umap_pipeline:
  sc.pp.normalize_total(adata)          # library-size normalize
  sc.pp.log1p(adata)
  sc.pp.pca(adata, n_comps=300)
  sc.pp.neighbors(adata, n_neighbors=30)
  sc.tl.umap(adata)                      # default min_dist=0.5, metric=euclidean
  sc.tl.leiden(adata, resolution=0.3)    # ← user-requested resolution

Plots each subset (input cell_id / SEG cells / SEG partials) colored by:
  - log10(n_tx) (same coloring as before, for direct comparison)
  - leiden cluster id at resolution=0.3

Outputs in benchmarks/lung_full_seq/scanpy_recipe_v2/.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
import scanpy as sc
import anndata as ad

warnings.filterwarnings("ignore", category=FutureWarning)
sc.settings.verbosity = 1

PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
REPO = Path(__file__).resolve().parents[1]
PART_PATH = REPO / "benchmarks" / "pdac_full_seq" / "partition_sequential.parquet"
OUT_DIR = REPO / "benchmarks" / "pdac_full_seq" / "scanpy_recipe_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TX = 20
N_PCS = 300
N_NEIGHBORS = 30
LEIDEN_RES = 1.0
RNG = 42
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan", "0"}


def _build_adata(df, ent_col, gene_col, min_tx):
    """Build an AnnData of (entity × gene) counts, filtered to entities
    with >= min_tx total transcripts."""
    ent_idx, ents = pd.factorize(df[ent_col].astype(str), sort=False)
    gene_codes, genes = pd.factorize(df[gene_col].astype(str), sort=False)
    n_gene = len(genes)
    data = np.ones(len(df), dtype=np.float32)
    m = sparse.coo_matrix(
        (data, (ent_idx, gene_codes)),
        shape=(len(ents), n_gene), dtype=np.float32,
    ).tocsr()
    sizes = np.asarray(m.sum(axis=1)).ravel()
    keep = sizes >= min_tx
    adata = ad.AnnData(
        X=m[keep], obs=pd.DataFrame({"entity": ents[keep], "n_tx": sizes[keep]}),
        var=pd.DataFrame(index=genes.astype(str)),
    )
    adata.obs_names = adata.obs["entity"].astype(str).to_numpy()
    return adata


def _run_pipeline(adata, label):
    print(f"  [{label}] normalize_total + log1p + PCA + neighbors + UMAP + leiden ...",
          flush=True)
    t = time.time()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    n_pcs_use = min(N_PCS, adata.n_vars - 1, adata.n_obs - 1)
    sc.pp.pca(adata, n_comps=n_pcs_use)
    sc.pp.neighbors(
        adata, n_neighbors=N_NEIGHBORS,
        n_pcs=min(n_pcs_use, adata.obsm["X_pca"].shape[1]),
    )
    sc.tl.umap(adata, min_dist=0.05, random_state=RNG)
    sc.tl.leiden(
        adata, resolution=LEIDEN_RES, key_added="leiden",
        flavor="igraph", n_iterations=2, directed=False, random_state=RNG,
    )
    n_clusters = adata.obs["leiden"].nunique()
    print(f"    {label} done in {time.time()-t:.1f}s  "
          f"({n_clusters} leiden clusters @ res={LEIDEN_RES})", flush=True)
    return adata


def _plot_pair(adata, name, out_path):
    """2-panel: log10(n_tx) | leiden clusters."""
    emb = adata.obsm["X_umap"]
    sizes = adata.obs["n_tx"].to_numpy()
    leiden = adata.obs["leiden"].astype("category")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=120)

    # Panel 1: log10(n_tx)
    color = np.log10(sizes)
    lo, hi = np.percentile(color, [1, 99])
    sc_ax = axes[0]
    s = sc_ax.scatter(emb[:, 0], emb[:, 1], c=np.clip(color, lo, hi),
                       s=1.5, alpha=0.6, cmap="viridis", linewidths=0)
    plt.colorbar(s, ax=sc_ax, fraction=0.04, pad=0.02, label="log10(n_tx)")
    sc_ax.set_title(f"{name} — log10(n_tx)", fontsize=11)
    sc_ax.set_xticks([]); sc_ax.set_yticks([])
    sc_ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: leiden
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
    ax2.set_title(f"{name} — leiden res={LEIDEN_RES}  ({n_cats} clusters)",
                   fontsize=11)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"PDAC scanpy recipe — normalize_total + log1p + PCA(300) + "
                  f"n_neighbors={N_NEIGHBORS} + UMAP default + leiden res={LEIDEN_RES}",
                  fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}", flush=True)


def main() -> int:
    t0 = time.time()
    print("loading partition + feature_name ...", flush=True)
    part = pd.read_parquet(PART_PATH)
    feats = pd.read_parquet(PDAC_PARQUET, columns=["transcript_id", "feature_name"])
    feats = feats.set_index("transcript_id").reindex(part["transcript_id"]).reset_index()
    df = part.copy()
    df["feature_name"] = feats["feature_name"].astype(str).to_numpy()
    df["cell_id"] = df["cell_id"].astype(str)
    df["label"] = df["label"].astype(str)
    df["_etype"] = df["_etype"].astype(str)

    inp_mask = ~df["cell_id"].isin(SENTINELS) & ~df["cell_id"].str.endswith("_rejected", na=False)
    seg_mask = ~df["label"].isin(SENTINELS) & ~df["label"].str.endswith("_rejected", na=False)
    cell_mask = seg_mask & (df["_etype"] == "cell")
    part_mask = seg_mask & (df["_etype"] == "partial")
    print(f"  input-assigned tx: {int(inp_mask.sum()):,}", flush=True)
    print(f"  SEG cell tx:       {int(cell_mask.sum()):,}", flush=True)
    print(f"  SEG partial tx:    {int(part_mask.sum()):,}", flush=True)

    adatas = {}
    for name, mask, ent_col in [
        ("input",        inp_mask,  "cell_id"),
        ("seg_cells",    cell_mask, "label"),
        ("seg_partials", part_mask, "label"),
    ]:
        print(f"\n=== {name} ===", flush=True)
        adata = _build_adata(df.loc[mask], ent_col, "feature_name", MIN_TX)
        print(f"  AnnData: {adata.shape}", flush=True)
        if adata.n_obs < N_NEIGHBORS + 1:
            print(f"  too few entities to UMAP, skipping", flush=True)
            continue
        adata = _run_pipeline(adata, name)
        _plot_pair(adata, name, OUT_DIR / f"umap_scanpy_{name}.png")
        # Persist embedding + leiden assignments
        out_df = pd.DataFrame({
            "entity": adata.obs["entity"].to_numpy(),
            "umap_1": adata.obsm["X_umap"][:, 0],
            "umap_2": adata.obsm["X_umap"][:, 1],
            "n_tx": adata.obs["n_tx"].to_numpy(),
            "leiden": adata.obs["leiden"].astype(str).to_numpy(),
        })
        out_df.to_parquet(OUT_DIR / f"embeddings_{name}.parquet", index=False)
        adatas[name] = adata

    # 3-panel combined: just the log10(n_tx) view for direct compare to prior v*
    if adatas:
        fig, axes = plt.subplots(1, len(adatas),
                                  figsize=(7.5 * len(adatas), 7.5), dpi=110)
        for ax, (name, adata) in zip(axes, adatas.items()):
            emb = adata.obsm["X_umap"]
            color = np.log10(adata.obs["n_tx"].to_numpy())
            lo, hi = np.percentile(color, [1, 99])
            s = ax.scatter(emb[:, 0], emb[:, 1], c=np.clip(color, lo, hi),
                            s=1.2, alpha=0.6, cmap="viridis", linewidths=0)
            plt.colorbar(s, ax=ax, fraction=0.04, pad=0.02, label="log10(n_tx)")
            ax.set_title(f"{name}  (n={adata.n_obs:,})", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            ax.spines[["top", "right"]].set_visible(False)
        plt.suptitle(f"PDAC scanpy recipe (normalize_total + log1p + PCA(300) + "
                      f"n_neighbors={N_NEIGHBORS} + UMAP min_dist=0.05)",
                      fontsize=12, y=1.0)
        plt.tight_layout()
        combined = OUT_DIR / "umap_scanpy_combined.png"
        plt.savefig(combined, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  -> {combined}", flush=True)

    print(f"\ntotal wall: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
