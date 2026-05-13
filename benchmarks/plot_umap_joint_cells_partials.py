#!/usr/bin/env python3
"""Joint UMAP of SEG cells + partials in a shared embedding.

Pools both subsets into one AnnData, runs the canonical scanpy pipeline
once, and colors by entity-origin (cell vs partial). Answers: do
partials live IN the cell manifold (real fragments), or off in their
own region (different biology / noise)?

Two-panel output for each dataset:
  - colored by entity-origin (cell/partial)
  - colored by leiden cluster on joint embedding

Run as:
    PYTHONPATH=src:. python plot_umap_joint_cells_partials.py lung
    PYTHONPATH=src:. python plot_umap_joint_cells_partials.py pdac
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
from scipy import sparse
import scanpy as sc
import anndata as ad

warnings.filterwarnings("ignore", category=FutureWarning)
sc.settings.verbosity = 1

REPO = Path(__file__).resolve().parents[1]

DATASETS = {
    "lung": {
        "parquet": Path("/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/"
                          "tutorials/lung_cancer/data/lung_cancer_df.parquet"),
        "partition": REPO / "benchmarks/lung_full_seq/partition_sequential.parquet",
        "out_dir": REPO / "benchmarks/lung_full_seq/joint_scanpy",
    },
    "pdac": {
        "parquet": Path("/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/"
                          "tutorials/pdac_io/data/outs/transcripts.parquet"),
        "partition": REPO / "benchmarks/pdac_full_seq/partition_sequential.parquet",
        "out_dir": REPO / "benchmarks/pdac_full_seq/joint_scanpy",
    },
}

MIN_TX = 20
N_PCS = 300
N_NEIGHBORS = 30
LEIDEN_RES = 1.0
UMAP_MIN_DIST = 0.05
RNG = 42
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan", "0"}


def _build_joint_adata(df, ent_col, gene_col, min_tx, origin):
    """Build AnnData with rows for entities, carrying an `origin` annotation."""
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
    return ents[keep], m[keep], sizes[keep], genes.astype(str), origin


def _run_pipeline(adata, label):
    print(f"  [{label}] normalize_total + log1p + PCA + neighbors + UMAP + leiden ...",
          flush=True)
    t = time.time()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    n_pcs_use = min(N_PCS, adata.n_vars - 1, adata.n_obs - 1)
    sc.pp.pca(adata, n_comps=n_pcs_use)
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS,
                    n_pcs=min(n_pcs_use, adata.obsm["X_pca"].shape[1]))
    sc.tl.umap(adata, min_dist=UMAP_MIN_DIST, random_state=RNG)
    sc.tl.leiden(
        adata, resolution=LEIDEN_RES, key_added="leiden",
        flavor="igraph", n_iterations=2, directed=False, random_state=RNG,
    )
    n_clusters = adata.obs["leiden"].nunique()
    print(f"    {label} done in {time.time()-t:.1f}s  "
          f"({n_clusters} leiden clusters @ res={LEIDEN_RES})", flush=True)
    return adata


def _plot_joint(adata, title_prefix, out_path):
    emb = adata.obsm["X_umap"]
    origin = adata.obs["origin"].to_numpy()
    leiden = adata.obs["leiden"].astype("category")

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), dpi=120)

    # Panel 1: origin
    ax = axes[0]
    origin_cats = ["cell", "partial"]
    colors = {"cell": "#1f77b4", "partial": "#ff7f0e"}
    # Plot the more numerous on bottom
    counts = {c: int((origin == c).sum()) for c in origin_cats}
    order = sorted(origin_cats, key=lambda c: -counts[c])
    for c in order:
        sel = (origin == c)
        ax.scatter(emb[sel, 0], emb[sel, 1], s=1.0, alpha=0.45,
                    c=colors[c], label=f"{c} (n={counts[c]:,})", linewidths=0)
    leg = ax.legend(loc="upper right", fontsize=9, markerscale=5,
                     framealpha=0.85)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    ax.set_title(f"{title_prefix} — entity origin (joint UMAP)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: leiden on joint
    ax = axes[1]
    cats = leiden.cat.categories
    n_cats = len(cats)
    cmap = plt.get_cmap("tab20" if n_cats <= 20 else "gist_ncar", n_cats)
    inv = leiden.cat.codes.to_numpy()
    cluster_order = np.argsort(np.bincount(inv))[::-1]
    for rank, k in enumerate(cluster_order):
        sel = inv == k
        ax.scatter(emb[sel, 0], emb[sel, 1], s=1.0, alpha=0.5,
                    c=[cmap(rank % cmap.N)],
                    label=f"{cats[k]} ({int(sel.sum()):,})", linewidths=0)
    leg = ax.legend(loc="upper right", fontsize=6, markerscale=4,
                     framealpha=0.85, ncol=2 if n_cats > 10 else 1)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    ax.set_title(f"{title_prefix} — leiden res={LEIDEN_RES}  "
                  f"({n_cats} clusters on joint)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"{title_prefix}: joint UMAP of SEG cells + partials  "
                  f"(normalize_total + log1p + PCA(300) + n_neighbors={N_NEIGHBORS} + "
                  f"UMAP min_dist={UMAP_MIN_DIST})",
                  fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}", flush=True)

    # Compute per-cluster origin composition (% partials in each cluster)
    by_clust = pd.DataFrame({"leiden": leiden, "origin": origin})
    comp = (by_clust.groupby("leiden")["origin"]
            .value_counts(normalize=True).unstack(fill_value=0))
    sizes = by_clust.groupby("leiden").size()
    comp = comp.assign(n=sizes)
    print(f"\n  per-leiden-cluster origin composition (joint embedding):")
    print(comp.to_string())
    return comp


def main(dataset: str) -> int:
    cfg = DATASETS[dataset]
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"=== joint UMAP for {dataset} ===", flush=True)
    print("loading partition + feature_name ...", flush=True)
    part = pd.read_parquet(cfg["partition"])
    if dataset == "pdac":
        feats = pd.read_parquet(cfg["parquet"],
                                  columns=["transcript_id", "feature_name"])
    else:
        feats = pd.read_parquet(cfg["parquet"],
                                  columns=["transcript_id", "feature_name"])
    feats = feats.set_index("transcript_id").reindex(part["transcript_id"]).reset_index()
    df = part.copy()
    df["feature_name"] = feats["feature_name"].astype(str).to_numpy()
    df["label"] = df["label"].astype(str)
    df["_etype"] = df["_etype"].astype(str)

    seg_mask = ~df["label"].isin(SENTINELS) & ~df["label"].str.endswith("_rejected", na=False)
    cell_mask = seg_mask & (df["_etype"] == "cell")
    part_mask = seg_mask & (df["_etype"] == "partial")
    print(f"  SEG cell tx:    {int(cell_mask.sum()):,}", flush=True)
    print(f"  SEG partial tx: {int(part_mask.sum()):,}", flush=True)

    # Build separate AnnDatas first to compute on a common gene index
    # Easier: build joint via combined (label-namespaced) entity column
    cells_df = df.loc[cell_mask].copy()
    cells_df["entity_full"] = "cell::" + cells_df["label"]
    cells_df["origin"] = "cell"
    parts_df = df.loc[part_mask].copy()
    parts_df["entity_full"] = "partial::" + parts_df["label"]
    parts_df["origin"] = "partial"
    pooled = pd.concat([cells_df, parts_df], ignore_index=True)
    print(f"  pooled tx: {len(pooled):,}", flush=True)

    # Build joint AnnData
    ent_idx, ents = pd.factorize(pooled["entity_full"].astype(str), sort=False)
    gene_codes, genes = pd.factorize(pooled["feature_name"].astype(str), sort=False)
    n_gene = len(genes)
    data = np.ones(len(pooled), dtype=np.float32)
    m = sparse.coo_matrix(
        (data, (ent_idx, gene_codes)),
        shape=(len(ents), n_gene), dtype=np.float32,
    ).tocsr()
    sizes = np.asarray(m.sum(axis=1)).ravel()
    # origin lookup per entity_full (one-to-one)
    ef_origin = (pooled.drop_duplicates("entity_full")
                  .set_index("entity_full")["origin"])
    origin = ef_origin.reindex(ents).to_numpy()
    keep = sizes >= MIN_TX
    obs = pd.DataFrame({
        "entity": ents[keep], "n_tx": sizes[keep], "origin": origin[keep],
    }, index=ents[keep])
    var = pd.DataFrame(index=genes.astype(str))
    adata = ad.AnnData(X=m[keep], obs=obs, var=var)
    print(f"  joint AnnData: {adata.shape}", flush=True)
    print(f"    cell origin:    {int((adata.obs['origin'] == 'cell').sum()):,}",
          flush=True)
    print(f"    partial origin: {int((adata.obs['origin'] == 'partial').sum()):,}",
          flush=True)

    adata = _run_pipeline(adata, dataset)
    comp = _plot_joint(adata, dataset.upper(),
                        cfg["out_dir"] / "umap_joint_cells_partials.png")
    # Persist embedding + leiden + origin
    pd.DataFrame({
        "entity": adata.obs["entity"].to_numpy(),
        "origin": adata.obs["origin"].to_numpy(),
        "umap_1": adata.obsm["X_umap"][:, 0],
        "umap_2": adata.obsm["X_umap"][:, 1],
        "n_tx": adata.obs["n_tx"].to_numpy(),
        "leiden": adata.obs["leiden"].astype(str).to_numpy(),
    }).to_parquet(cfg["out_dir"] / "embeddings_joint.parquet", index=False)
    comp.to_csv(cfg["out_dir"] / "cluster_origin_composition.csv")

    print(f"\ntotal wall: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "lung"
    raise SystemExit(main(ds))
