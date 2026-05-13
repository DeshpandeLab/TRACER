#!/usr/bin/env python3
"""Lung UMAP v6: no SVD, full-dim Hellinger.

vs v5 (Hellinger with 30 PCs):
  - skip TruncatedSVD entirely
  - UMAP on all 300 features (the √p representation)

Rationale: with only ~300 features, dimensionality reduction isn't
needed for computational tractability, and the bottom PCs carry real
biological signal (rare-cell-type markers, subtle gene programs). Also
metric choice (Hellinger) matters more without the lossy PC step,
because all 300 dims contribute to neighbor distances.
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
from sklearn.preprocessing import normalize
import umap

LUNG_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/lung_cancer/"
    "data/lung_cancer_df.parquet"
)
REPO = Path(__file__).resolve().parents[1]
PART_PATH = REPO / "benchmarks" / "lung_full_seq" / "partition_sequential.parquet"
OUT_DIR = REPO / "benchmarks" / "lung_full_seq" / "v6_no_svd_hellinger"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TX = 20
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
RNG = 42
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan", "0"}


def _build_count_matrix(df, ent_col, gene_col):
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


def _filter_min_tx(m, ents, min_tx):
    sizes = np.asarray(m.sum(axis=1)).ravel()
    keep = sizes >= min_tx
    return m[keep], ents[keep], sizes[keep]


def _preprocess_hellinger(m):
    m = normalize(m, norm="l1", axis=1).copy()
    m.data = np.sqrt(m.data, dtype=np.float32)
    return m


def _umap_direct(mat, label):
    """UMAP on the full-dim matrix (no SVD). UMAP handles sparse input."""
    print(f"  [{label}] UMAP no-SVD (d={mat.shape[1]}, "
          f"n_neighbors={UMAP_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, "
          f"metric=euclidean on √p) ...", flush=True)
    t = time.time()
    # Dense float32 for UMAP — for d=300 and n~30k, dense is ~36 MB. Cheap.
    dense = np.asarray(mat.todense(), dtype=np.float32)
    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
        metric="euclidean", random_state=RNG, verbose=False,
    )
    emb = reducer.fit_transform(dense)
    print(f"    umap: {time.time()-t:.1f}s", flush=True)
    return emb


def _plot(emb, color, cmap_label, title, out_path):
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
    feats = pd.read_parquet(LUNG_PARQUET, columns=["transcript_id", "feature_name"])
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

    out_records = []
    for name, mask, ent_col in [
        ("input",        inp_mask,  "cell_id"),
        ("seg_cells",    cell_mask, "label"),
        ("seg_partials", part_mask, "label"),
    ]:
        print(f"\n=== {name} ===", flush=True)
        M, ents = _build_count_matrix(df.loc[mask], ent_col, "feature_name")
        print(f"  matrix: {M.shape}", flush=True)
        M, ents, sizes = _filter_min_tx(M, ents, MIN_TX)
        print(f"  filtered (min_tx>={MIN_TX}): {M.shape}", flush=True)
        if M.shape[0] < UMAP_NEIGHBORS + 1:
            print(f"  too few entities to UMAP, skipping", flush=True)
            continue
        Mp = _preprocess_hellinger(M)
        emb = _umap_direct(Mp, name)
        _plot(emb, np.log10(sizes), "log10(n_tx)",
               f"Lung {name} — n={len(ents):,}, d=300  "
               f"(min_tx={MIN_TX}, n_neighbors={UMAP_NEIGHBORS}, "
               f"min_dist={UMAP_MIN_DIST}, Hellinger no-SVD)",
               OUT_DIR / f"umap_v6_{name}.png")
        out_records.append(pd.DataFrame({
            "kind": name, "entity": ents,
            "umap_1": emb[:, 0], "umap_2": emb[:, 1],
            "n_tx": sizes,
        }))

    if len(out_records) >= 2:
        fig, axes = plt.subplots(1, len(out_records),
                                  figsize=(7.5 * len(out_records), 7.5), dpi=110)
        for ax, rec in zip(axes, out_records):
            emb = rec[["umap_1", "umap_2"]].to_numpy()
            color = np.log10(rec["n_tx"].to_numpy())
            lo, hi = np.percentile(color, [1, 99])
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=np.clip(color, lo, hi),
                             s=1.2, alpha=0.6, cmap="viridis", linewidths=0)
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="log10(n_tx)")
            ax.set_title(f"{rec['kind'].iloc[0]}  (n={len(rec):,})", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            ax.spines[["top", "right"]].set_visible(False)
        plt.suptitle(f"Lung UMAP v6 — Hellinger, NO SVD (d=300), "
                      f"n_neighbors={UMAP_NEIGHBORS}, min_dist={UMAP_MIN_DIST}",
                      fontsize=12, y=1.0)
        plt.tight_layout()
        combined = OUT_DIR / "umap_v6_combined.png"
        plt.savefig(combined, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  -> {combined}", flush=True)

    pd.concat(out_records, ignore_index=True).to_parquet(
        OUT_DIR / "umap_v6_embeddings.parquet", index=False)
    print(f"\ntotal wall: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
