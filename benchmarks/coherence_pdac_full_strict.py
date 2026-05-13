#!/usr/bin/env python3
"""Coherence deciles for the 4 full-tissue PDAC partitions.

Computes per-entity coherence (C = purity − conflict) over the entity's
unique gene set, using the partition's OWN panel. Reports deciles
p10, p20, ..., p90 + min/max/mean for cells and partials separately.

Inputs:
  - pdac_full_seq/partition_sequential.parquet           (thr=10 default)
  - pdac_full_seq_thr0/partition_sequential.parquet      (thr=0 default)
  - pdac_full_seq_thr10_strict/partition_sequential.parquet  (thr=10 strict)
  - pdac_full_seq_thr0_strict/partition_sequential.parquet   (thr=0 strict)

Each scored against its own panel.

Output: pdac_full_seq/coherence_strict_compare.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
PANEL_THR10 = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr10.parquet"
)
PANEL_THR0 = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
REPO = Path(__file__).resolve().parents[1]

VARIANTS = [
    ("thr10_default", REPO / "benchmarks" / "pdac_full_seq" / "partition_sequential.parquet", PANEL_THR10),
    ("thr10_strict",  REPO / "benchmarks" / "pdac_full_seq_thr10_strict" / "partition_sequential.parquet", PANEL_THR10),
    ("thr0_default",  REPO / "benchmarks" / "pdac_full_seq_thr0" / "partition_sequential.parquet", PANEL_THR0),
    ("thr0_strict",   REPO / "benchmarks" / "pdac_full_seq_thr0_strict" / "partition_sequential.parquet", PANEL_THR0),
]

SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan", "0"}
TAU = 0.2  # PMI-scale coherence threshold (1.22× chance) — used by strict params


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
            C_val, _pu, _cf = _cy_prune.coherence_count_kernel(gids32, W, float(tau))
            C[i] = C_val
    return pd.DataFrame({"entity": ent_index[e_codes], "C": C, "k_genes": k_genes})


def _etype_from_label(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    is_cascade = s.str.startswith("cascade_") | s.str.startswith("UNASSIGNED_")
    n_dashes = s.str.count("-")
    is_partial = is_cascade | (n_dashes >= 2)
    is_un = s.isin(SENTINELS) | s.str.endswith("_rejected", na=False)
    out = pd.Series("cell", index=s.index)
    out[is_partial] = "partial"
    out[is_un] = "unassigned"
    return out


def main() -> int:
    t0 = time.time()
    print("loading transcripts feature_name (full PDAC) ...", flush=True)
    feats = pd.read_parquet(
        PDAC_PARQUET, columns=["transcript_id", "feature_name"],
    )
    feats["feature_name"] = feats["feature_name"].astype(str)
    print(f"  {len(feats):,} tx", flush=True)

    # Union gene universe across both panels + features
    panel_thr10 = pd.read_parquet(PANEL_THR10)
    panel_thr0 = pd.read_parquet(PANEL_THR0)
    all_genes = sorted(set(panel_thr10["gene_i"].astype(str))
                        | set(panel_thr10["gene_j"].astype(str))
                        | set(panel_thr0["gene_i"].astype(str))
                        | set(panel_thr0["gene_j"].astype(str))
                        | set(feats["feature_name"].unique()))
    W_thr10, gene_to_idx = _build_W(PANEL_THR10, all_genes)
    W_thr0, _ = _build_W(PANEL_THR0, all_genes)
    print(f"  W_thr10: {int((~np.isnan(W_thr10)).sum() // 2):,} non-NaN pairs", flush=True)
    print(f"  W_thr0:  {int((~np.isnan(W_thr0)).sum() // 2):,} non-NaN pairs", flush=True)

    rows = []
    for name, part_path, _panel_path in VARIANTS:
        t = time.time()
        print(f"\n=== {name} ===", flush=True)
        if not part_path.exists():
            print(f"  no partition at {part_path}, skipping", flush=True)
            continue
        part = pd.read_parquet(part_path, columns=["transcript_id", "label"])
        # Merge feature_name
        m = part.merge(feats, on="transcript_id", how="left")
        m["feature_name"] = m["feature_name"].astype(str)
        m["_etype"] = _etype_from_label(m["label"])

        # Compute per-entity n_tx (tx count per label, all assigned tx)
        n_tx_per_ent = m.loc[m["_etype"] != "unassigned"].groupby("label").size()
        cell_df = m[m["_etype"] == "cell"]
        part_df = m[m["_etype"] == "partial"]
        n_cell_ent = cell_df["label"].astype(str).nunique()
        n_part_ent = part_df["label"].astype(str).nunique()
        print(f"  cell tx: {len(cell_df):,}, entities: {n_cell_ent:,}", flush=True)
        print(f"  partial tx: {len(part_df):,}, entities: {n_part_ent:,}", flush=True)

        W = W_thr10 if "thr10" in name else W_thr0
        coh_c = _coherence_per_entity(
            cell_df["label"].astype(str).to_numpy(),
            cell_df["feature_name"].to_numpy(),
            W, gene_to_idx,
        )
        coh_c = coh_c[coh_c["k_genes"] >= 2]
        coh_p = _coherence_per_entity(
            part_df["label"].astype(str).to_numpy(),
            part_df["feature_name"].to_numpy(),
            W, gene_to_idx,
        )
        coh_p = coh_p[coh_p["k_genes"] >= 2]

        # Decile summary — coherence AND n_tx
        for etype_label, sub in [("cell", coh_c), ("partial", coh_p)]:
            v = sub["C"].to_numpy()
            qs = {f"C_p{p}": float(np.percentile(v, p)) for p in range(10, 100, 10)}
            # Attach n_tx per entity
            sub_ntx = n_tx_per_ent.reindex(sub["entity"]).fillna(0).to_numpy()
            qs_ntx = {f"ntx_p{p}": float(np.percentile(sub_ntx, p))
                      for p in range(10, 100, 10)}
            rows.append({
                "variant": name, "etype": etype_label,
                "n": len(sub),
                "C_min": float(v.min()) if len(v) else float("nan"),
                "C_max": float(v.max()) if len(v) else float("nan"),
                "C_mean": float(v.mean()) if len(v) else float("nan"),
                "ntx_min": float(sub_ntx.min()) if len(sub_ntx) else float("nan"),
                "ntx_max": float(sub_ntx.max()) if len(sub_ntx) else float("nan"),
                "ntx_mean": float(sub_ntx.mean()) if len(sub_ntx) else float("nan"),
                **qs, **qs_ntx,
            })
        print(f"  wall: {time.time()-t:.1f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(REPO / "benchmarks" / "pdac_full_seq" / "coherence_strict_compare.csv",
               index=False)
    print(f"\nsummary -> benchmarks/pdac_full_seq/coherence_strict_compare.csv")
    print(f"total wall: {time.time()-t0:.1f}s")

    # Pretty-print
    print("\n" + "=" * 130)
    print("Coherence (C) deciles:")
    cols = ["variant", "etype", "n",
            "C_p10", "C_p20", "C_p30", "C_p40", "C_p50", "C_p60", "C_p70", "C_p80", "C_p90", "C_mean"]
    print(out[cols].to_string(index=False))
    print()
    print("n_tx deciles:")
    cols2 = ["variant", "etype", "n",
             "ntx_p10", "ntx_p20", "ntx_p30", "ntx_p40", "ntx_p50",
             "ntx_p60", "ntx_p70", "ntx_p80", "ntx_p90", "ntx_mean"]
    print(out[cols2].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
