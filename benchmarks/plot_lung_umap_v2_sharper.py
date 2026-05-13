#!/usr/bin/env python3
"""UMAP v2 for lung: same settings as PDAC v2.

Reuses bench_lung_full_seq.py's partition. Settings match
plot_pdac_umap_v2_sharper.py:
  - log1p, no row normalization
  - n_neighbors=15, min_dist=0.1, metric=cosine
  - min_tx=20, n_pcs=30

Outputs land in benchmarks/lung_full_seq/v2/
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import umap

LUNG_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/lung_cancer/"
    "data/lung_cancer_df.parquet"
)
REPO = Path(__file__).resolve().parents[1]
PART_PATH = REPO / "benchmarks" / "lung_full_seq" / "partition_sequential.parquet"
OUT_DIR = REPO / "benchmarks" / "lung_full_seq" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TX = 20
N_PCS = 30
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
RNG = 42
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan", "0"}


def _build_count_matrix(df: pd.DataFrame, ent_col: str, gene_col: str
                         ) -> tuple[sparse.csr_matrix, np.ndarray]:
    ent_idx, ents = pd.factorize(df[ent_col].astype(str), sort=False)
    gene_idx, _ = pd.factorize(df[gene_col].astype(str), sort=False)
    data = np.ones(len(df), dtype=np.float32)
    n_gene = gene_idx.max() + 1
    m = sparse.coo_matrix(
        (data, (ent_idx, gene_idx)),
        shape=(len(ents), int(n_gene)),
        dtype=np.float32,
    ).tocsr()
    return m, np.asarray(ents)


def _filter_min_tx(m: sparse.csr_matrix, ents: np.ndarray, min_tx: int):
    sizes = np.asarray(m.sum(axis=1)).ravel()
    keep = sizes >= min_tx
    return m[keep], ents[keep], sizes[keep]


def _preprocess(m: sparse.csr_matrix) -> sparse.csr_matrix:
    m = m.copy()
    m.data = np.log1p(m.data)
    return m


def _pca_umap(mat: sparse.csr_matrix, label: str) -> np.ndarray:
    n_components = min(N_PCS, mat.shape[1] - 1, mat.shape[0] - 1)
    print(f"  [{label}] TruncatedSVD → {n_components} PCs ...", flush=True)
    t = time.time()
    svd = TruncatedSVD(n_components=n_components, random_state=RNG)
    pcs = svd.fit_transform(mat)
    print(f"    svd: {time.time()-t:.1f}s   var_explained: "
          f"{svd.explained_variance_ratio_.sum():.3f}", flush=True)
    print(f"  [{label}] UMAP (n_neighbors={UMAP_NEIGHBORS}, "
          f"min_dist={UMAP_MIN_DIST}) ...", flush=True)
    t = time.time()
    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
        metric="cosine", random_state=RNG, verbose=False,
    )
    emb = reducer.fit_transform(pcs)
    print(f"    umap: {time.time()-t:.1f}s", flush=True)
    return emb


def _plot(emb: np.ndarray, color: np.ndarray, cmap_label: str,
           title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=130)
    lo, hi = np.percentile(color, [1, 99])
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=np.clip(color, lo, hi),
                     s=1.5, alpha=0.6, cmap="viridis", linewidths=0)
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label=cmap_label)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP-1", fontsize=10); ax.set_ylabel("UMAP-2", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}", flush=True)


def main() -> int:
    t0 = time.time()
    print("loading partition + feature_name ...", flush=True)
    part = pd.read_parquet(PART_PATH)
    feats = pd.read_parquet(
        LUNG_PARQUET, columns=["transcript_id", "feature_name"],
    )
    feats = feats.set_index("transcript_id").reindex(part["transcript_id"]).reset_index()
    df = part.copy()
    df["feature_name"] = feats["feature_name"].astype(str).to_numpy()
    df["cell_id"] = df["cell_id"].astype(str)
    df["label"] = df["label"].astype(str)
    df["_etype"] = df["_etype"].astype(str)
    print(f"  tx: {len(df):,}; features: {df['feature_name'].nunique():,}",
          flush=True)

    inp_mask = ~df["cell_id"].isin(SENTINELS) \
        & ~df["cell_id"].str.endswith("_rejected", na=False)
    seg_mask = ~df["label"].isin(SENTINELS) \
        & ~df["label"].str.endswith("_rejected", na=False)
    cell_mask = seg_mask & (df["_etype"] == "cell")
    part_mask = seg_mask & (df["_etype"] == "partial")
    print(f"  input-assigned tx: {int(inp_mask.sum()):,}", flush=True)
    print(f"  SEG cell tx:       {int(cell_mask.sum()):,}", flush=True)
    print(f"  SEG partial tx:    {int(part_mask.sum()):,}", flush=True)

    out_records = []
    for name, mask, ent_col in [
        ("input",        inp_mask,  "cell_id"),
        ("seg_cells",    cell_mask, "label"),
        ("seg_partials", part_mask, "label"),
    ]:
        print(f"\n=== {name} ===", flush=True)
        if int(mask.sum()) < 100:
            print(f"  too few tx ({int(mask.sum())}), skipping", flush=True)
            continue
        M, ents = _build_count_matrix(df.loc[mask], ent_col, "feature_name")
        print(f"  matrix: {M.shape}", flush=True)
        M, ents, sizes = _filter_min_tx(M, ents, MIN_TX)
        print(f"  filtered (min_tx>={MIN_TX}): {M.shape}", flush=True)
        if M.shape[0] < UMAP_NEIGHBORS + 1:
            print(f"  too few entities to UMAP, skipping", flush=True)
            continue
        Mp = _preprocess(M)
        emb = _pca_umap(Mp, name)
        _plot(emb, np.log10(sizes), "log10(n_tx)",
               f"Lung {name} — n={len(ents):,}  "
               f"(min_tx={MIN_TX}, n_neighbors={UMAP_NEIGHBORS}, "
               f"min_dist={UMAP_MIN_DIST}, no row-norm)",
               OUT_DIR / f"umap_v2_{name}.png")
        out_records.append(pd.DataFrame({
            "kind": name,
            "entity": ents,
            "umap_1": emb[:, 0], "umap_2": emb[:, 1],
            "n_tx": sizes,
        }))

    if len(out_records) >= 2:
        fig, axes = plt.subplots(1, len(out_records),
                                  figsize=(7.5 * len(out_records), 7.5),
                                  dpi=110)
        if len(out_records) == 1:
            axes = [axes]
        for ax, rec in zip(axes, out_records):
            emb = rec[["umap_1", "umap_2"]].to_numpy()
            sizes = rec["n_tx"].to_numpy()
            color = np.log10(sizes)
            lo, hi = np.percentile(color, [1, 99])
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=np.clip(color, lo, hi),
                             s=1.2, alpha=0.6, cmap="viridis", linewidths=0)
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="log10(n_tx)")
            ax.set_title(f"{rec['kind'].iloc[0]}  (n={len(rec):,})",
                          fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            ax.spines[["top", "right"]].set_visible(False)
        plt.suptitle(f"Lung UMAP v2 — log1p only, "
                      f"n_neighbors={UMAP_NEIGHBORS}, min_dist={UMAP_MIN_DIST}",
                      fontsize=12, y=1.0)
        plt.tight_layout()
        combined = OUT_DIR / "umap_v2_combined.png"
        plt.savefig(combined, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  -> {combined}", flush=True)

    emb_path = OUT_DIR / "umap_v2_embeddings.parquet"
    pd.concat(out_records, ignore_index=True).to_parquet(emb_path, index=False)
    print(f"  -> {emb_path}", flush=True)
    print(f"\ntotal wall: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
