#!/usr/bin/env python3
"""Marker-gene heatmap per Leiden cluster for PDAC scanpy outputs.

For each subset (input, seg_cells, seg_partials):
  1. Rebuild AnnData from partition + feature_name
  2. Apply same preprocessing as the UMAP run (normalize_total + log1p)
  3. Inject saved Leiden cluster assignments
  4. Run sc.tl.rank_genes_groups (wilcoxon) to identify per-cluster markers
  5. Plot a matrix plot with top-N genes per cluster (~7 per cluster ≈ 40-50 total)

By default uses the res=0.3 standard scanpy recipe outputs. Pass a
different SOURCE_DIR to switch (e.g. cosine_no_pca_res04).
"""
from __future__ import annotations

import os
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

PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
REPO = Path(__file__).resolve().parents[1]
PART_PATH = REPO / "benchmarks" / "pdac_full_seq" / "partition_sequential.parquet"
DEFAULT_SOURCE = "scanpy_recipe_res03_partial10"
SOURCE_DIR = REPO / "benchmarks" / "pdac_full_seq" / os.environ.get(
    "SOURCE_DIR", DEFAULT_SOURCE
)
N_GENES_PER_CLUSTER = 7  # top-N per cluster

SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan", "0"}


def _build_adata_with_leiden(df, ent_col, gene_col, leiden_df):
    """Build AnnData, restrict to entities present in leiden_df, attach leiden."""
    ent_idx, ents = pd.factorize(df[ent_col].astype(str), sort=False)
    gene_codes, genes = pd.factorize(df[gene_col].astype(str), sort=False)
    n_gene = len(genes)
    data = np.ones(len(df), dtype=np.float32)
    m = sparse.coo_matrix(
        (data, (ent_idx, gene_codes)),
        shape=(len(ents), n_gene), dtype=np.float32,
    ).tocsr()
    sizes = np.asarray(m.sum(axis=1)).ravel()
    # Filter to entities in leiden_df
    leiden_map = leiden_df.set_index("entity")["leiden"]
    ent_str = ents.astype(str)
    mask = pd.Series(ent_str).isin(leiden_map.index).to_numpy()
    adata = ad.AnnData(
        X=m[mask],
        obs=pd.DataFrame({
            "entity": ent_str[mask],
            "n_tx": sizes[mask],
            "leiden": leiden_map.reindex(ent_str[mask]).astype(str).to_numpy(),
        }),
        var=pd.DataFrame(index=genes.astype(str)),
    )
    adata.obs_names = adata.obs["entity"].astype(str).to_numpy()
    return adata


def _process_subset(name, mask, ent_col, df, source_dir):
    """For one subset: build adata, find markers, plot heatmap."""
    emb_path = source_dir / f"embeddings_{name}.parquet"
    if not emb_path.exists():
        print(f"  [{name}] no embeddings at {emb_path}, skipping", flush=True)
        return
    leiden_df = pd.read_parquet(emb_path)
    print(f"\n=== {name}  (n_entities w/ leiden: {len(leiden_df):,}) ===", flush=True)
    t = time.time()
    adata = _build_adata_with_leiden(df.loc[mask], ent_col, "feature_name", leiden_df)
    print(f"  AnnData: {adata.shape}  (after leiden alignment)", flush=True)
    if adata.n_obs == 0:
        print(f"  empty after alignment, skipping", flush=True)
        return

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    adata.obs["leiden"] = adata.obs["leiden"].astype("category")

    print(f"  rank_genes_groups (wilcoxon) ...", flush=True)
    sc.tl.rank_genes_groups(
        adata, groupby="leiden", method="wilcoxon",
        n_genes=max(N_GENES_PER_CLUSTER * 2, 20),
    )

    # Build a deduplicated marker list: top N per cluster, in cluster order
    rgg = adata.uns["rank_genes_groups"]
    names = pd.DataFrame(rgg["names"])  # columns = cluster ids, rows = rank
    clusters = list(names.columns)
    marker_list = []
    seen = set()
    for c in clusters:
        for g in names[c].head(N_GENES_PER_CLUSTER):
            if g not in seen:
                marker_list.append(g)
                seen.add(g)
    print(f"  found {len(marker_list)} unique markers across {len(clusters)} clusters",
          flush=True)

    # Matrixplot: mean expression per cluster, dot size = % cells expressing
    sc.set_figure_params(dpi=120, frameon=False)
    fig = sc.pl.dotplot(
        adata, var_names=marker_list, groupby="leiden",
        standard_scale="var",
        title=f"PDAC {name} — top {N_GENES_PER_CLUSTER}/cluster markers "
              f"(n={adata.n_obs:,}, {len(clusters)} clusters)",
        show=False, return_fig=True,
        figsize=(max(8, len(marker_list) * 0.22), max(3, len(clusters) * 0.4)),
    )
    out = source_dir / f"marker_dotplot_{name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close("all")
    print(f"  -> {out}", flush=True)

    # Also save the raw markers table
    rows = []
    pvals = pd.DataFrame(rgg["pvals_adj"])
    logfc = pd.DataFrame(rgg["logfoldchanges"])
    scores = pd.DataFrame(rgg["scores"])
    for c in clusters:
        for rank, g in enumerate(names[c].head(N_GENES_PER_CLUSTER * 2)):
            rows.append({
                "cluster": c, "rank": rank + 1, "gene": g,
                "score": float(scores[c].iloc[rank]),
                "logfc": float(logfc[c].iloc[rank]),
                "pval_adj": float(pvals[c].iloc[rank]),
            })
    pd.DataFrame(rows).to_csv(source_dir / f"markers_{name}.csv", index=False)
    print(f"  wall: {time.time()-t:.1f}s", flush=True)


def main() -> int:
    print(f"SOURCE_DIR: {SOURCE_DIR}", flush=True)
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

    for name, mask, ent_col in [
        ("input",        inp_mask,  "cell_id"),
        ("seg_cells",    cell_mask, "label"),
        ("seg_partials", part_mask, "label"),
    ]:
        _process_subset(name, mask, ent_col, df, SOURCE_DIR)

    print(f"\ntotal wall: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
