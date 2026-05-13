#!/usr/bin/env python3
"""One-time build of the Pearson-residual scanpy state for PDAC subsets.

Runs the expensive steps once and saves the full AnnData as .h5ad
(including X_pca, neighbor graph, and X_umap). After this, Leiden can
be re-run at any resolution in seconds via sweep_pdac_pearson_leiden.py.

Output: benchmarks/pdac_full_seq/scanpy_pearson_state/{input,seg_cells,seg_partials}.h5ad
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
STATE_DIR = REPO / "benchmarks" / "pdac_full_seq" / "scanpy_pearson_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

MIN_TX = 20
MIN_TX_PARTIAL = 10
N_PCS = 300
N_NEIGHBORS = 30
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

    for name, mask, ent_col in [
        ("input",        inp_mask,  "cell_id"),
        ("seg_cells",    cell_mask, "label"),
        ("seg_partials", part_mask, "label"),
    ]:
        out_h5 = STATE_DIR / f"{name}.h5ad"
        if out_h5.exists():
            print(f"\n[{name}] state already exists at {out_h5}, skipping",
                  flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        t = time.time()
        min_tx_use = MIN_TX_PARTIAL if name == "seg_partials" else MIN_TX
        adata = _build_adata(df.loc[mask], ent_col, "feature_name", min_tx_use)
        print(f"  AnnData: {adata.shape}  (min_tx={min_tx_use})", flush=True)

        adata.layers["counts"] = adata.X.copy()
        print(f"  pearson_residuals ...", flush=True)
        sc.experimental.pp.normalize_pearson_residuals(adata)
        print(f"  PCA ...", flush=True)
        n_pcs_use = min(N_PCS, adata.n_vars - 1, adata.n_obs - 1)
        sc.pp.pca(adata, n_comps=n_pcs_use)
        print(f"  neighbors ...", flush=True)
        sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS,
                        n_pcs=min(n_pcs_use, adata.obsm["X_pca"].shape[1]))
        print(f"  UMAP ...", flush=True)
        sc.tl.umap(adata, min_dist=UMAP_MIN_DIST, random_state=RNG)

        # Drop the dense Pearson-residual X to save disk — we keep counts,
        # X_pca, neighbor graph, and X_umap which is what leiden needs.
        adata.X = adata.layers["counts"].copy()  # restore counts as X
        del adata.layers["counts"]

        print(f"  saving {out_h5} ...", flush=True)
        adata.write_h5ad(out_h5, compression="gzip")
        print(f"  [{name}] state built in {time.time()-t:.1f}s", flush=True)

    print(f"\ntotal wall: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
