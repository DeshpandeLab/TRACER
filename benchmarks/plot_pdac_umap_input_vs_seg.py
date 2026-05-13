#!/usr/bin/env python3
"""UMAP comparison: PDAC Xenium input cell_id vs sequential SEG output.

For each partition (input cell_id, SEG output label), build the
entity × gene count matrix from the original transcripts.parquet,
filter to entities with >= MIN_TX transcripts to keep UMAP tractable
and reduce noise, log1p-normalize + L2 row-normalize, PCA→UMAP, and
plot side-by-side. Color by:
  - entity size (n_tx)
  - entity-type (_etype) for SEG side; nucleus-overlap fraction for input

Inputs:
  partitions: benchmarks/pdac_full_seq/partition_sequential.parquet
  features:   tutorials/pdac_io/data/outs/transcripts.parquet (feature_name)

Outputs:
  benchmarks/pdac_full_seq/umap_input.png
  benchmarks/pdac_full_seq/umap_seg.png
  benchmarks/pdac_full_seq/umap_combined.png
  benchmarks/pdac_full_seq/umap_embeddings.parquet
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
from sklearn.preprocessing import normalize
import umap

PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/"
    "data/outs/transcripts.parquet"
)
REPO = Path(__file__).resolve().parents[1]
PART_PATH = REPO / "benchmarks" / "pdac_full_seq" / "partition_sequential.parquet"
OUT_DIR = REPO / "benchmarks" / "pdac_full_seq"

MIN_TX = 20  # min transcripts per entity to include in UMAP
N_PCS = 30
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.3
RNG = 42

SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan", "0"}


def _build_count_matrix(df: pd.DataFrame, ent_col: str, gene_col: str
                         ) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Return (counts CSR [n_ent x n_gene], ent_index, gene_index)."""
    ent_idx, ents = pd.factorize(df[ent_col].astype(str), sort=False)
    gene_idx, genes = pd.factorize(df[gene_col].astype(str), sort=False)
    data = np.ones(len(df), dtype=np.float32)
    m = sparse.coo_matrix(
        (data, (ent_idx, gene_idx)),
        shape=(len(ents), len(genes)),
        dtype=np.float32,
    ).tocsr()
    return m, np.asarray(ents), np.asarray(genes)


def _filter_min_tx(m: sparse.csr_matrix, ents: np.ndarray, min_tx: int
                    ) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    sizes = np.asarray(m.sum(axis=1)).ravel()
    keep = sizes >= min_tx
    return m[keep], ents[keep], sizes[keep]


def _normalize_for_umap(m: sparse.csr_matrix) -> np.ndarray:
    # log1p of counts, then L2-normalize per row
    m = m.copy()
    m.data = np.log1p(m.data)
    m = normalize(m, norm="l2", axis=1)
    return m


def _pca_umap(mat_norm: sparse.csr_matrix, label: str) -> np.ndarray:
    n_components = min(N_PCS, mat_norm.shape[1] - 1, mat_norm.shape[0] - 1)
    print(f"  [{label}] TruncatedSVD → {n_components} PCs ...", flush=True)
    t = time.time()
    svd = TruncatedSVD(n_components=n_components, random_state=RNG)
    pcs = svd.fit_transform(mat_norm)
    print(f"    svd wall: {time.time()-t:.1f}s  "
          f"var_explained: {svd.explained_variance_ratio_.sum():.3f}", flush=True)

    print(f"  [{label}] UMAP fit ...", flush=True)
    t = time.time()
    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RNG,
        verbose=False,
    )
    emb = reducer.fit_transform(pcs)
    print(f"    umap wall: {time.time()-t:.1f}s", flush=True)
    return emb


def _plot(emb: np.ndarray, sizes: np.ndarray, color: np.ndarray | None,
           color_label: str, title: str, out_path: Path,
           categorical: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 7.5), dpi=130)
    if categorical:
        cats, inv = np.unique(color, return_inverse=True)
        cmap = plt.get_cmap("tab10", len(cats))
        # Plot largest-count categories first (under), small last (over)
        order = np.argsort(np.bincount(inv))[::-1]
        for k_pos, k in enumerate(order):
            sel = inv == k
            ax.scatter(emb[sel, 0], emb[sel, 1],
                        s=1.5, alpha=0.55, c=[cmap(k_pos)],
                        label=f"{cats[k]} (n={sel.sum():,})",
                        linewidths=0)
        leg = ax.legend(loc="upper right", fontsize=8, markerscale=4,
                         framealpha=0.85)
        for h in leg.legend_handles:
            h.set_alpha(1.0)
    else:
        # Continuous: clip to 1-99th percentile for visual contrast
        c = color if color is not None else np.log1p(sizes)
        lo, hi = np.percentile(c, [1, 99])
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=np.clip(c, lo, hi),
                         s=1.2, alpha=0.6, cmap="viridis", linewidths=0)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(color_label, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP-1", fontsize=10)
    ax.set_ylabel("UMAP-2", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}", flush=True)


def main() -> int:
    t0 = time.time()
    print(f"loading partition ...", flush=True)
    part = pd.read_parquet(PART_PATH)
    print(f"  {len(part):,} tx [{time.time()-t0:.1f}s]", flush=True)
    print(f"loading transcripts.parquet (feature_name + overlaps_nucleus) ...", flush=True)
    feats = pd.read_parquet(
        PDAC_PARQUET, columns=["transcript_id", "feature_name", "overlaps_nucleus"],
    )
    # Align feature info to partition order
    feats = feats.set_index("transcript_id").reindex(part["transcript_id"]).reset_index()
    df = pd.concat([part[["cell_id", "label", "_etype"]].reset_index(drop=True),
                     feats[["feature_name", "overlaps_nucleus"]].reset_index(drop=True)],
                    axis=1)
    df["feature_name"] = df["feature_name"].astype(str)
    print(f"  feature_name unique: {df['feature_name'].nunique():,}", flush=True)

    # ---------------------------------------------------------------
    # Filter to assigned tx in each partition for that partition's matrix
    # ---------------------------------------------------------------
    # Input side: drop UNASSIGNED / 0 / sentinel cell_ids
    df["cell_id"] = df["cell_id"].astype(str)
    inp_mask = ~df["cell_id"].isin(SENTINELS) & ~df["cell_id"].str.endswith("_rejected", na=False)
    seg_mask = ~df["label"].isin(SENTINELS) & ~df["label"].str.endswith("_rejected", na=False)
    print(f"\n  input assigned tx:  {int(inp_mask.sum()):,}  "
          f"({100*inp_mask.mean():.2f}%)", flush=True)
    print(f"  SEG assigned tx:    {int(seg_mask.sum()):,}  "
          f"({100*seg_mask.mean():.2f}%)", flush=True)

    # ---------------------------------------------------------------
    # Build matrices, filter min_tx, normalize, run UMAP
    # ---------------------------------------------------------------
    print(f"\nbuilding INPUT (cell_id) entity × gene matrix ...", flush=True)
    t = time.time()
    M_in, ents_in, _ = _build_count_matrix(df.loc[inp_mask], "cell_id", "feature_name")
    print(f"  {M_in.shape}  [{time.time()-t:.1f}s]", flush=True)
    M_in, ents_in, sizes_in = _filter_min_tx(M_in, ents_in, MIN_TX)
    print(f"  filtered (n_tx>={MIN_TX}): {M_in.shape}", flush=True)

    print(f"\nbuilding SEG (label) entity × gene matrix ...", flush=True)
    t = time.time()
    M_seg, ents_seg, _ = _build_count_matrix(df.loc[seg_mask], "label", "feature_name")
    print(f"  {M_seg.shape}  [{time.time()-t:.1f}s]", flush=True)
    M_seg, ents_seg, sizes_seg = _filter_min_tx(M_seg, ents_seg, MIN_TX)
    print(f"  filtered (n_tx>={MIN_TX}): {M_seg.shape}", flush=True)

    # Annotations
    # For SEG side: get _etype per entity (cell vs partial)
    etype_map = (df.loc[seg_mask, ["label", "_etype"]]
                   .drop_duplicates("label").set_index("label")["_etype"])
    seg_etype = etype_map.reindex(ents_seg).fillna("unknown").astype(str).to_numpy()
    # For INPUT side: fraction of tx with overlaps_nucleus == True
    df_in = df.loc[inp_mask, ["cell_id", "overlaps_nucleus"]]
    nuc_frac = df_in.groupby("cell_id")["overlaps_nucleus"].mean()
    nuc_frac_in = nuc_frac.reindex(ents_in).fillna(0.0).to_numpy()

    print(f"\nnormalizing matrices ...", flush=True)
    Mn_in = _normalize_for_umap(M_in)
    Mn_seg = _normalize_for_umap(M_seg)

    print(f"\nUMAP on INPUT ({M_in.shape[0]:,} entities) ...", flush=True)
    emb_in = _pca_umap(Mn_in, "input")
    print(f"\nUMAP on SEG ({M_seg.shape[0]:,} entities) ...", flush=True)
    emb_seg = _pca_umap(Mn_seg, "SEG")

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------
    print(f"\nplotting ...", flush=True)
    _plot(emb_in, sizes_in, np.log10(sizes_in),
          "log10(n_tx)",
          f"PDAC input cell_id (n={M_in.shape[0]:,} entities, min_tx={MIN_TX})",
          OUT_DIR / "umap_input_size.png")
    _plot(emb_in, sizes_in, nuc_frac_in,
          "nucleus-overlap fraction",
          f"PDAC input cell_id — nucleus overlap (n={M_in.shape[0]:,})",
          OUT_DIR / "umap_input_nuc.png")
    _plot(emb_seg, sizes_seg, np.log10(sizes_seg),
          "log10(n_tx)",
          f"PDAC SEG output (n={M_seg.shape[0]:,} entities, min_tx={MIN_TX})",
          OUT_DIR / "umap_seg_size.png")
    _plot(emb_seg, sizes_seg, seg_etype, "_etype",
          f"PDAC SEG output — entity type (n={M_seg.shape[0]:,})",
          OUT_DIR / "umap_seg_etype.png", categorical=True)

    # Combined 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 14), dpi=120)
    for ax, emb, sizes, ann, ann_lab, ttl, categorical in [
        (axes[0, 0], emb_in, sizes_in, np.log10(sizes_in), "log10(n_tx)",
         f"Input cell_id — size  (n={M_in.shape[0]:,})", False),
        (axes[0, 1], emb_in, sizes_in, nuc_frac_in, "nuc-overlap frac",
         f"Input cell_id — nucleus overlap", False),
        (axes[1, 0], emb_seg, sizes_seg, np.log10(sizes_seg), "log10(n_tx)",
         f"SEG output — size  (n={M_seg.shape[0]:,})", False),
        (axes[1, 1], emb_seg, sizes_seg, seg_etype, "_etype",
         f"SEG output — entity type", True),
    ]:
        if categorical:
            cats, inv = np.unique(ann, return_inverse=True)
            cmap = plt.get_cmap("tab10", len(cats))
            order = np.argsort(np.bincount(inv))[::-1]
            for k_pos, k in enumerate(order):
                sel = inv == k
                ax.scatter(emb[sel, 0], emb[sel, 1], s=1.0, alpha=0.55,
                           c=[cmap(k_pos)], label=f"{cats[k]} (n={sel.sum():,})",
                           linewidths=0)
            leg = ax.legend(loc="upper right", fontsize=7, markerscale=4,
                             framealpha=0.85)
            for h in leg.legend_handles:
                h.set_alpha(1.0)
        else:
            lo, hi = np.percentile(ann, [1, 99])
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=np.clip(ann, lo, hi),
                            s=1.0, alpha=0.55, cmap="viridis", linewidths=0)
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label=ann_lab)
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle(f"PDAC: Xenium input cell_id vs sequential SEG output  "
                  f"(min_tx={MIN_TX}, {N_PCS} PCs, cosine UMAP)",
                  fontsize=12, y=0.995)
    plt.tight_layout()
    combined = OUT_DIR / "umap_combined.png"
    plt.savefig(combined, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {combined}", flush=True)

    # Save embeddings for later re-plotting
    emb_path = OUT_DIR / "umap_embeddings.parquet"
    out_records = []
    for ent_arr, emb, sizes, kind in [
        (ents_in, emb_in, sizes_in, "input"),
        (ents_seg, emb_seg, sizes_seg, "seg"),
    ]:
        out_records.append(pd.DataFrame({
            "kind": kind,
            "entity": ent_arr,
            "umap_1": emb[:, 0],
            "umap_2": emb[:, 1],
            "n_tx": sizes,
        }))
    pd.concat(out_records, ignore_index=True).to_parquet(emb_path, index=False)
    print(f"  -> {emb_path}", flush=True)

    print(f"\ndone in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
