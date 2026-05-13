#!/usr/bin/env python3
"""Coherence quartiles for each config in the PDAC PMI sweep.

For each sweep config's saved partition_{config}.parquet:
  1. Rebuild entity gene-sets from partition + original transcripts
  2. Compute coherence (C = purity - conflict) per entity using W_thr0
  3. Split by _etype (cell vs partial), report Q1/median/Q3

Outputs: coherence_summary.csv with rows = configs, columns =
  cell_n, cell_C_Q1, cell_C_med, cell_C_Q3,
  partial_n, partial_C_Q1, partial_C_med, partial_C_Q3
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SWEEP_DIR = REPO / "benchmarks" / "pdac_pmi_sweep"
PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
ROI_X = (6255.0, 6921.67)
ROI_Y = (2023.7, 2690.37)
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan", "0"}
TAU = 0.05  # coherence threshold (in panel value units — PMI in this case)


def _etype(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    is_cascade = s.str.startswith("cascade_") | s.str.startswith("UNASSIGNED_")
    n_dashes = s.str.count("-")
    is_partial = is_cascade | (n_dashes >= 2)
    is_un = s.isin(SENTINELS) | s.str.endswith("_rejected", na=False)
    out = pd.Series("cell", index=s.index)
    out[is_partial] = "partial"
    out[is_un] = "unassigned"
    return out


def _build_W(panel_path, all_genes):
    panel = pd.read_parquet(panel_path).rename(columns={"value": "NPMI"})
    panel["gene_i"] = panel["gene_i"].astype(str)
    panel["gene_j"] = panel["gene_j"].astype(str)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    G = len(all_genes)
    W = np.full((G, G), np.nan, dtype=np.float32)
    gi = panel["gene_i"].map(gene_to_idx)
    gj = panel["gene_j"].map(gene_to_idx)
    have = gi.notna() & gj.notna()
    gi = gi[have].to_numpy(dtype=np.int64)
    gj = gj[have].to_numpy(dtype=np.int64)
    v = panel.loc[have, "NPMI"].to_numpy(dtype=np.float32)
    W[gi, gj] = v
    W[gj, gi] = v
    return W, gene_to_idx


def _coherence_per_entity(entity_arr, gene_arr, W, gene_to_idx, tau=TAU):
    ent_codes, ent_index = pd.factorize(entity_arr, sort=False)
    gene_codes = np.array([gene_to_idx.get(g, -1) for g in gene_arr], dtype=np.int64)
    keep = gene_codes >= 0
    pair_df = (pd.DataFrame({"e": ent_codes[keep], "g": gene_codes[keep]})
                 .drop_duplicates()
                 .sort_values(["e", "g"]).reset_index(drop=True))
    sizes = pair_df.groupby("e").size()
    starts = sizes.cumsum().shift(fill_value=0).astype(np.int64).to_numpy()
    counts = sizes.to_numpy()
    n_ent = len(sizes)
    e_codes = sizes.index.to_numpy()
    g_arr = pair_df["g"].to_numpy()

    try:
        from tracer import _cy_prune
        use_cy = True
    except Exception:
        use_cy = False

    C = np.zeros(n_ent, dtype=np.float32)
    k_genes = np.zeros(n_ent, dtype=np.int32)
    for i in range(n_ent):
        a, b = starts[i], starts[i] + counts[i]
        gids = g_arr[a:b]
        k = gids.size
        k_genes[i] = k
        if k < 2:
            continue
        if use_cy:
            gids32 = np.ascontiguousarray(gids, dtype=np.int32)
            C_val, _pu, _cf = _cy_prune.coherence_count_kernel(
                gids32, W, float(tau)
            )
            C[i] = C_val
        else:
            sub = W[np.ix_(gids, gids)]
            iu = np.triu_indices(k, k=1)
            w = sub[iu]
            w = w[~np.isnan(w)]
            if w.size > 0:
                pu = float((w > tau).mean()); cf = float((w < -tau).mean())
                C[i] = pu - cf
    return pd.DataFrame({"entity": ent_index[e_codes], "C": C, "k_genes": k_genes})


def main() -> int:
    t0 = time.time()
    print("loading ROI tx + feature_name ...", flush=True)
    df = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "x_location", "y_location", "feature_name"],
    ).rename(columns={"x_location": "x", "y_location": "y"})
    mask = df["x"].between(*ROI_X) & df["y"].between(*ROI_Y)
    df = df.loc[mask].reset_index(drop=True)
    df["feature_name"] = df["feature_name"].astype(str)
    print(f"  {len(df):,} tx", flush=True)

    # Gene universe (panel + ROI)
    panel_raw = pd.read_parquet(PANEL)
    all_genes = sorted(set(panel_raw["gene_i"].astype(str))
                        | set(panel_raw["gene_j"].astype(str))
                        | set(df["feature_name"].unique()))
    W, gene_to_idx = _build_W(PANEL, all_genes)
    print(f"  W: {W.shape}, panel non-NaN: {int((~np.isnan(W)).sum() // 2):,}", flush=True)

    # Load sweep summary to iterate configs in order
    summary = pd.read_csv(SWEEP_DIR / "sweep_summary.csv")
    summary = summary[summary["ok"] == True].reset_index(drop=True)
    print(f"  {len(summary)} configs to analyze", flush=True)

    rows = []
    for i, row in summary.iterrows():
        cfg = row["config"]
        p = SWEEP_DIR / f"partition_{cfg}.parquet"
        if not p.exists():
            continue
        t = time.time()
        part = pd.read_parquet(p)
        # Align by transcript_id
        m = (df.merge(part, on="transcript_id", how="inner"))
        m["_etype"] = _etype(m["label"])
        cell_df = m[m["_etype"] == "cell"]
        part_df = m[m["_etype"] == "partial"]
        # Coherence per cell entity
        coh_cell = _coherence_per_entity(
            cell_df["label"].to_numpy(),
            cell_df["feature_name"].to_numpy(),
            W, gene_to_idx,
        )
        coh_cell = coh_cell[coh_cell["k_genes"] >= 2]
        coh_part = _coherence_per_entity(
            part_df["label"].to_numpy(),
            part_df["feature_name"].to_numpy(),
            W, gene_to_idx,
        )
        coh_part = coh_part[coh_part["k_genes"] >= 2]

        def Q(s, q): return float(np.percentile(s, q)) if len(s) else float("nan")
        rec = {
            "config": cfg,
            "pmi_thr": row["pmi_thr"], "mean_admit": row["mean_admit"],
            "percentile": row["percentile"],
            "cell_n":      len(coh_cell),
            "cell_Q1":     Q(coh_cell["C"], 25),
            "cell_med":    Q(coh_cell["C"], 50),
            "cell_Q3":     Q(coh_cell["C"], 75),
            "cell_mean":   float(coh_cell["C"].mean()) if len(coh_cell) else float("nan"),
            "partial_n":   len(coh_part),
            "partial_Q1":  Q(coh_part["C"], 25),
            "partial_med": Q(coh_part["C"], 50),
            "partial_Q3":  Q(coh_part["C"], 75),
            "partial_mean":float(coh_part["C"].mean()) if len(coh_part) else float("nan"),
        }
        rows.append(rec)
        print(f"  [{i+1}/{len(summary)}] {cfg}: "
              f"cell C med={rec['cell_med']:.3f}  "
              f"partial C med={rec['partial_med']:.3f}  "
              f"({time.time()-t:.1f}s)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(SWEEP_DIR / "coherence_summary.csv", index=False)
    print(f"\nsummary -> {SWEEP_DIR / 'coherence_summary.csv'}")
    print(f"total wall: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
