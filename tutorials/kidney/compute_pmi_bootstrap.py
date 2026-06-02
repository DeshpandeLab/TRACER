#!/usr/bin/env python
"""Compute the gene-gene PMI matrix from the kidney scRNA-seq reference
using GENESIS's bootstrap builder (`tracer.metrics.compute_pmi_bootstrap`).

This reference is single-cell RNA-seq, so co-occurrence "contexts" are
*cells* and "features" are *genes*: a gene is present in a cell when its
raw count >= `min_occurrences_per_context`. Pairwise PMI of presence is
then estimated with a spatial/active-sampling bootstrap that emits only
the pairs whose 95% CI clears +/- tau (sparse output).

Loader note: we import `tracer.metrics` in isolation (without running
`tracer/__init__.py`, which pulls torch/open3d/geopandas) by registering a
minimal package namespace and stubbing `geopandas` (imported at module top
in metrics.py but unused by `compute_pmi_bootstrap`). `_kernels` is pure
numba, so no Cython build is needed.

Outputs (under tutorials/kidney/output/, gitignored by convention):
  - kidney_pmi_bootstrap_long.csv   : gene_i, gene_j, PMI  (settled pairs)
  - kidney_pmi_bootstrap_W.npz      : G x G upper-triangle CSR PMI matrix
  - kidney_pmi_bootstrap_genes.txt  : gene order for the matrix
  - kidney_pmi_bootstrap_diagnostics.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import scipy.sparse as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC = os.path.join(REPO_ROOT, "src")
H5AD = os.path.join(SCRIPT_DIR, "kidney_reference_harmonized.h5ad")
OUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Tunables (mirror the GENESIS tutorial defaults; PMI metric per request).
METRIC = "pmi"
MIN_OCCURRENCES_PER_CONTEXT = 2     # gene present in a cell when count >= 2
MIN_EXPECTED_COOCCUR_FOR_EVIDENCE = 10.0
SEED = 0
# Bootstrap iteration ceiling + block size (gene-row mode). Empirically (A/B on a
# 2,500-gene subset of this reference): ~97% of candidate pairs settle within the
# FIRST 100 samples, +0.1% by 700, and the ~2.7% that reach the cap almost never
# settle (genuinely near-tau / low-support). So max=300 with a 100-iter block
# (settle checks at 100/200/300) is ~3.3x faster than max=1000/block=200 while the
# settled set differs by only ~0.1% (boundary churn at the ~0.13-PMI resolution
# floor). max=10000 (library default) was far past the knee.
MAX_BOOTSTRAPS = 300
BOOTSTRAP_BLOCK = 100      # coarse_block == refine_block; settle checks at 100/200/300
# Output filename prefix is chosen per --mode in main() and passed to
# _write_outputs(): "legacy_pmi" (legacy) or "bootstrap_pmi" (gene-row bootstrap).

# Legacy-only sparse mode: the full bootstrap resampling slices the presence
# matrix for all candidate pairs at once, which at whole-transcriptome scale
# (~24.9M pairs -> ~290 GB/iter) segfaults. Setting the bootstrap-eligibility
# threshold to +inf makes `can_bootstrap` empty, so every high-evidence pair is
# routed to `legacy_only` and gets its full-data population PMI written to
# W_sparse (Stage 2/3 of compute_pmi_bootstrap), and the function early-returns
# before the resampling loop. We keep the evidence filtering + mutual-exclusion
# (-1 / -log E) sentinels; we lose only the bootstrap CIs.
MIN_EXPECTED_COOCCUR_FOR_BOOTSTRAP = float("inf")


def _load_compute_pmi_bootstrap():
    """Import tracer.metrics without triggering the heavy package __init__."""
    pkg = types.ModuleType("tracer")
    pkg.__path__ = [os.path.join(SRC, "tracer")]
    sys.modules["tracer"] = pkg
    # geopandas is imported at the top of metrics.py but unused by the
    # bootstrap path; stub it so we don't need the full geo stack.
    sys.modules.setdefault("geopandas", types.ModuleType("geopandas"))
    metrics = importlib.import_module("tracer.metrics")
    return metrics


def build_long_df(h5ad_path: str) -> tuple[pd.DataFrame, int, int]:
    """Long-format (cell_id, feature_name, count) from the raw count matrix.

    Uses categoricals built from integer codes so we never materialize
    16.5M python strings. Returns (df, n_cells, n_genes_total).
    """
    import anndata as ad

    a = ad.read_h5ad(h5ad_path)
    # Prefer the explicit raw-count layer; fall back to X (identical here).
    if "counts" in a.layers:
        X = a.layers["counts"]
    else:
        X = a.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocoo()

    # Guard: PMI on duplicate gene labels would be ambiguous.
    var_names = a.var_names
    if not var_names.is_unique:
        n_dup = var_names.duplicated().sum()
        print(f"[warn] {n_dup} duplicate var_names; calling var_names_make_unique()")
        a.var_names_make_unique()
        var_names = a.var_names

    obs_cats = pd.Index(a.obs_names.astype(str))
    var_cats = pd.Index(var_names.astype(str))

    df = pd.DataFrame({
        "cell_id": pd.Categorical.from_codes(X.row.astype(np.int64), categories=obs_cats),
        "feature_name": pd.Categorical.from_codes(X.col.astype(np.int64), categories=var_cats),
        "count": np.rint(X.data).astype(np.int32),
    })
    return df, a.n_obs, a.n_vars


def _write_outputs(result, out_prefix: str, extra_meta: dict) -> None:
    """Write the four standard artifacts (W.npz, long.csv, genes.txt,
    diagnostics.json) for either mode. Shared by legacy and bootstrap paths."""
    W = result.W_sparse
    genes = np.asarray(result.genes, dtype=str)
    print(f"[info] W: {W.shape[0]}x{W.shape[1]} upper-tri CSR, "
          f"settled pairs (nnz)={W.nnz:,}")

    # 1) sparse matrix + gene order
    sp.save_npz(os.path.join(OUT_DIR, f"{out_prefix}_W.npz"), W.tocsr())
    np.savetxt(os.path.join(OUT_DIR, f"{out_prefix}_genes.txt"),
               genes, fmt="%s")

    # 2) long-format settled pairs
    coo = W.tocoo()
    long_df = pd.DataFrame({
        "gene_i": genes[coo.row],
        "gene_j": genes[coo.col],
        "PMI": coo.data.astype(np.float64),
    }).sort_values("PMI", ascending=False, ignore_index=True)
    long_df.to_csv(os.path.join(OUT_DIR, f"{out_prefix}_long.csv"), index=False)

    # 3) diagnostics
    diag = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in result.diagnostics.items()}
    meta = {
        "n_genes_in_matrix": int(len(genes)),
        "metric": METRIC,
        "min_occurrences_per_context": MIN_OCCURRENCES_PER_CONTEXT,
        "min_expected_cooccur_for_evidence": MIN_EXPECTED_COOCCUR_FOR_EVIDENCE,
        "seed": SEED,
        "nonzero_entries": int(W.nnz),
    }
    meta.update(extra_meta)
    diag["_meta"] = meta
    with open(os.path.join(OUT_DIR, f"{out_prefix}_diagnostics.json"), "w") as fh:
        json.dump(diag, fh, indent=2, default=str)

    print(f"\n[done] outputs in {OUT_DIR}")
    print(long_df.head(15).to_string(index=False))


def run_legacy(metrics) -> None:
    """LEGACY-only sparse path: long-form df + min_expected_cooccur_for_bootstrap=inf."""
    print(f"[info] loading {H5AD}")
    df, n_cells, n_genes_total = build_long_df(H5AD)
    print(f"[info] cells={n_cells:,}  genes(total)={n_genes_total:,}  "
          f"nonzero (cell,gene) rows={len(df):,}")

    print(f"[info] running compute_pmi_bootstrap (LEGACY-ONLY sparse) "
          f"metric={METRIC!r}, min_occ={MIN_OCCURRENCES_PER_CONTEXT}, "
          f"min_expected_cooccur={MIN_EXPECTED_COOCCUR_FOR_EVIDENCE}, "
          f"min_expected_cooccur_for_bootstrap={MIN_EXPECTED_COOCCUR_FOR_BOOTSTRAP}, "
          f"seed={SEED}")
    result = metrics.compute_pmi_bootstrap(
        df,
        group_key="cell_id",
        feature_col="feature_name",
        count_col="count",
        metric=METRIC,
        min_occurrences_per_context=MIN_OCCURRENCES_PER_CONTEXT,
        min_expected_cooccur_for_evidence=MIN_EXPECTED_COOCCUR_FOR_EVIDENCE,
        min_expected_cooccur_for_bootstrap=MIN_EXPECTED_COOCCUR_FOR_BOOTSTRAP,
        set_neg_one=True,
        seed=SEED,
        show_progress=True,
    )
    _write_outputs(result, "legacy_pmi", {
        "n_cells": int(n_cells),
        "n_genes_total": int(n_genes_total),
        "min_expected_cooccur_for_bootstrap": MIN_EXPECTED_COOCCUR_FOR_BOOTSTRAP,
        "mode": "legacy_only_sparse",
    })


def run_bootstrap(metrics, no_checkpoint: bool) -> None:
    """Gene-row bootstrap path: pass the counts matrix directly (no build_long_df)."""
    import anndata as ad

    print(f"[info] loading {H5AD}")
    a = ad.read_h5ad(H5AD)
    X = a.layers["counts"] if "counts" in a.layers else a.X
    print(f"[info] cells={a.n_obs:,}  genes(total)={a.n_vars:,}")

    ckpt = None if no_checkpoint else os.path.join(OUT_DIR, "bootstrap_pmi.ckpt.npz")
    print(f"[info] running compute_pmi_bootstrap (gene-row BOOTSTRAP) "
          f"metric={METRIC!r}, min_occ={MIN_OCCURRENCES_PER_CONTEXT}, "
          f"min_expected_cooccur={MIN_EXPECTED_COOCCUR_FOR_EVIDENCE}, "
          f"max_bootstraps={MAX_BOOTSTRAPS}, block={BOOTSTRAP_BLOCK}, "
          f"checkpoint={ckpt!r}, seed={SEED}")
    result = metrics.compute_pmi_bootstrap(
        None,
        counts=(X, a.var_names.astype(str).to_numpy(), a.obs_names.astype(str).to_numpy()),
        metric="pmi",
        min_occurrences_per_context=MIN_OCCURRENCES_PER_CONTEXT,
        min_expected_cooccur_for_evidence=MIN_EXPECTED_COOCCUR_FOR_EVIDENCE,
        bootstrap_kernel="gene_row", gene_order="prob_ascending",
        gene_batch_peak_gb=16.0, checkpoint_path=ckpt,
        max_bootstraps=MAX_BOOTSTRAPS,
        coarse_block=BOOTSTRAP_BLOCK, refine_block=BOOTSTRAP_BLOCK,
        seed=SEED, show_progress=True,
    )
    _write_outputs(result, "bootstrap_pmi", {
        "n_cells": int(a.n_obs),
        "n_genes_total": int(a.n_vars),
        "bootstrap_kernel": "gene_row",
        "gene_order": "prob_ascending",
        "max_bootstraps": MAX_BOOTSTRAPS,
        "bootstrap_block": BOOTSTRAP_BLOCK,
        "checkpoint_path": ckpt,
        "mode": "gene_row_bootstrap",
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["legacy", "bootstrap"], default="legacy")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="bootstrap mode only: disable checkpoint/resume")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    metrics = _load_compute_pmi_bootstrap()
    print(f"[info] tracer.metrics: {metrics.__file__}")
    print(f"[info] mode={args.mode}")

    if args.mode == "bootstrap":
        run_bootstrap(metrics, no_checkpoint=args.no_checkpoint)
    else:
        run_legacy(metrics)


if __name__ == "__main__":
    main()
