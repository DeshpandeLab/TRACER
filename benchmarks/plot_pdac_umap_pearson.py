#!/usr/bin/env python3
"""PDAC scanpy recipe with Pearson residuals instead of normalize_total+log1p.

Pipeline:
  sc.experimental.pp.normalize_pearson_residuals(adata)   # NB-residuals
  sc.pp.pca(adata, n_comps=300)
  sc.pp.neighbors(adata, n_neighbors=30)                   # default euclidean
  sc.tl.umap(adata, min_dist=0.1)
  sc.tl.leiden(adata, resolution=0.6)

Pearson residuals stabilize variance across n_tx and normalize for
total count implicitly — better-grounded for sparse count data than
normalize_total + log1p.

Outputs in benchmarks/pdac_full_seq/scanpy_pearson_res06_partial10/.
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
OUT_DIR = REPO / "benchmarks" / "pdac_full_seq" / "scanpy_pearson_res06_partial10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TX = 20
MIN_TX_PARTIAL = 10
N_PCS = 300
N_NEIGHBORS = 30
LEIDEN_RES = 0.6
UMAP_MIN_DIST = 0.1
RNG = 42
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan", "0"}


def _build_adata(df, ent_col, gene_col, min_tx):
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
    print(f"  [{label}] pearson_residuals + PCA + neighbors + UMAP + leiden ...",
          flush=True)
    t = time.time()
    adata.layers["counts"] = adata.X.copy()
    sc.experimental.pp.normalize_pearson_residuals(adata)
    n_pcs_use = min(N_PCS, adata.n_vars - 1, adata.n_obs - 1)
    sc.pp.pca(adata, n_comps=n_pcs_use)
    sc.pp.neighbors(
        adata, n_neighbors=N_NEIGHBORS,
        n_pcs=min(n_pcs_use, adata.obsm["X_pca"].shape[1]),
    )
    sc.tl.umap(adata, min_dist=UMAP_MIN_DIST, random_state=RNG)
    sc.tl.leiden(
        adata, resolution=LEIDEN_RES, key_added="leiden",
        flavor="igraph", n_iterations=2, directed=False, random_state=RNG,
    )
    n_clusters = adata.obs["leiden"].nunique()
    print(f"    {label} done in {time.time()-t:.1f}s  "
          f"({n_clusters} leiden clusters @ res={LEIDEN_RES})", flush=True)
    return adata


def _plot_pair(adata, name, out_path):
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
    ax2.set_title(f"{name} — leiden res={LEIDEN_RES}  ({n_cats} clusters)",
                   fontsize=11)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"PDAC Pearson residuals + PCA(300) + n_neighbors={N_NEIGHBORS} "
                  f"+ UMAP min_dist={UMAP_MIN_DIST} + leiden res={LEIDEN_RES}",
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
        min_tx_use = MIN_TX_PARTIAL if name == "seg_partials" else MIN_TX
        adata = _build_adata(df.loc[mask], ent_col, "feature_name", min_tx_use)
        print(f"  AnnData: {adata.shape}  (min_tx={min_tx_use})", flush=True)
        if adata.n_obs < N_NEIGHBORS + 1:
            print(f"  too few entities to UMAP, skipping", flush=True)
            continue
        adata = _run_pipeline(adata, name)
        _plot_pair(adata, name, OUT_DIR / f"umap_scanpy_{name}.png")
        out_df = pd.DataFrame({
            "entity": adata.obs["entity"].to_numpy(),
            "umap_1": adata.obsm["X_umap"][:, 0],
            "umap_2": adata.obsm["X_umap"][:, 1],
            "n_tx": adata.obs["n_tx"].to_numpy(),
            "leiden": adata.obs["leiden"].astype(str).to_numpy(),
        })
        out_df.to_parquet(OUT_DIR / f"embeddings_{name}.parquet", index=False)
        adatas[name] = adata

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
        plt.suptitle(f"PDAC Pearson residuals — PCA(300) + n_neighbors={N_NEIGHBORS}"
                      f" + UMAP min_dist={UMAP_MIN_DIST} + leiden res={LEIDEN_RES}",
                      fontsize=12, y=1.0)
        plt.tight_layout()
        combined = OUT_DIR / "umap_scanpy_combined.png"
        plt.savefig(combined, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  -> {combined}", flush=True)

    # Cross-tab comparison: Pearson-residual leiden vs standard-recipe leiden
    print(f"\n=== leiden cross-tab vs scanpy_recipe_res06_partial10 ===",
          flush=True)
    STANDARD_DIR = REPO / "benchmarks" / "pdac_full_seq" / "scanpy_recipe_res06_partial10"
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    for name in adatas.keys():
        std_path = STANDARD_DIR / f"embeddings_{name}.parquet"
        if not std_path.exists():
            print(f"  [{name}] no standard run at {std_path}, skipping", flush=True)
            continue
        std = pd.read_parquet(std_path)[["entity", "leiden"]].rename(
            columns={"leiden": "leiden_std"})
        pr = pd.read_parquet(OUT_DIR / f"embeddings_{name}.parquet"
                              )[["entity", "leiden"]].rename(
                                  columns={"leiden": "leiden_pearson"})
        m = pr.merge(std, on="entity", how="inner")
        print(f"\n  [{name}] joint n={len(m):,}", flush=True)
        print(f"    n_clusters: pearson={m['leiden_pearson'].nunique()}, "
              f"standard={m['leiden_std'].nunique()}", flush=True)
        ari = adjusted_rand_score(m["leiden_pearson"], m["leiden_std"])
        nmi = normalized_mutual_info_score(m["leiden_pearson"], m["leiden_std"])
        print(f"    ARI = {ari:.4f},  NMI = {nmi:.4f}", flush=True)
        # Crosstab (rows = pearson clusters, cols = standard clusters)
        ct = pd.crosstab(m["leiden_pearson"], m["leiden_std"])
        print(f"    crosstab (pearson rows × standard cols):")
        print(ct.to_string())

    print(f"\ntotal wall: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
