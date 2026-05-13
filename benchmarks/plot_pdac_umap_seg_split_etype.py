#!/usr/bin/env python3
"""Replot SEG UMAP split by _etype (cell vs partial) into separate panels.

Reuses the saved umap_embeddings.parquet from plot_pdac_umap_input_vs_seg.py
so we don't recompute UMAP. Joins each SEG entity to its _etype via the
partition file. Saves:

  pdac_full_seq/umap_seg_cells.png      — only cell entities
  pdac_full_seq/umap_seg_partials.png   — only partial entities
  pdac_full_seq/umap_seg_split.png      — side-by-side cells | partials
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DIR_OUT = REPO / "benchmarks" / "pdac_full_seq"
EMB_PATH = DIR_OUT / "umap_embeddings.parquet"
PART_PATH = DIR_OUT / "partition_sequential.parquet"


def _scatter(ax, emb, ann, title, cmap_label):
    lo, hi = np.percentile(ann, [1, 99])
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=np.clip(ann, lo, hi),
                     s=1.0, alpha=0.55, cmap="viridis", linewidths=0)
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label=cmap_label)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP-1", fontsize=10)
    ax.set_ylabel("UMAP-2", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    print(f"loading embeddings ...", flush=True)
    emb = pd.read_parquet(EMB_PATH)
    seg = emb[emb["kind"] == "seg"].reset_index(drop=True)
    print(f"  SEG entities in embedding: {len(seg):,}", flush=True)

    print(f"loading partition (for _etype) ...", flush=True)
    part = pd.read_parquet(PART_PATH, columns=["label", "_etype"])
    etype = (part.assign(label=part["label"].astype(str))
                  .drop_duplicates("label")
                  .set_index("label")["_etype"].astype(str))
    seg["_etype"] = seg["entity"].astype(str).map(etype).fillna("unknown")
    counts = seg["_etype"].value_counts().to_dict()
    print(f"  _etype counts: {counts}", flush=True)

    # Compute global plot extents so cells / partials are on the same axes
    pad_x = 0.04 * (seg["umap_1"].max() - seg["umap_1"].min())
    pad_y = 0.04 * (seg["umap_2"].max() - seg["umap_2"].min())
    xlim = (seg["umap_1"].min() - pad_x, seg["umap_1"].max() + pad_x)
    ylim = (seg["umap_2"].min() - pad_y, seg["umap_2"].max() + pad_y)

    for et in ("cell", "partial"):
        sub = seg[seg["_etype"] == et].reset_index(drop=True)
        if len(sub) == 0:
            print(f"  skipping {et} (no entities)", flush=True)
            continue
        fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=130)
        _scatter(ax, sub[["umap_1", "umap_2"]].to_numpy(),
                  np.log10(sub["n_tx"].to_numpy()),
                  f"PDAC SEG output — {et}s only  (n={len(sub):,}, min_tx=20)",
                  "log10(n_tx)")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        out = DIR_OUT / f"umap_seg_{et}s.png"
        plt.tight_layout()
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out}", flush=True)

    # Side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=120)
    for ax, et in zip(axes, ("cell", "partial")):
        sub = seg[seg["_etype"] == et].reset_index(drop=True)
        _scatter(ax, sub[["umap_1", "umap_2"]].to_numpy(),
                  np.log10(sub["n_tx"].to_numpy()),
                  f"{et}s  (n={len(sub):,})", "log10(n_tx)")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    plt.suptitle("PDAC SEG output — split by entity type",
                  fontsize=12, y=1.0)
    plt.tight_layout()
    out = DIR_OUT / "umap_seg_split.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
