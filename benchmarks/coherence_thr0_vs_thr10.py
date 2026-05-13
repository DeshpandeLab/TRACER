#!/usr/bin/env python3
"""Coherence-quartile comparison of thr=0 vs thr=10 SEG outputs.

For each entity in each partition, compute its NPMI coherence C
(purity − conflict at τ=0.05) using the corresponding panel. Report
quartiles + mean by entity-type (cell vs partial vs all).

Also cross-evaluate: score thr=0 entities against the thr=10 panel
and vice versa, to see if entities a panel rejected look qualitatively
worse to the other panel.

Inputs: partition_sequential.parquet from each of pdac_full_seq{,_thr0}.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DIR_A = REPO / "benchmarks" / "pdac_full_seq"        # thr=10
DIR_B = REPO / "benchmarks" / "pdac_full_seq_thr0"    # thr=0
PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/"
    "data/outs/transcripts.parquet"
)
PANEL_A = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr10.parquet"
)
PANEL_B = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}
TAU = 0.05  # NPMI coherence threshold (matches pipeline default)


def _build_W(panel_path: Path, all_genes: list[str]) -> np.ndarray:
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
    return W


def _coherence_per_entity(
    df: pd.DataFrame, ent_col: str, W: np.ndarray,
    gene_to_idx: dict[str, int], tau: float = TAU,
) -> pd.DataFrame:
    """For each entity, compute C = purity − conflict over the unique
    gene set, where purity = #(pair > tau) / |P| and conflict =
    #(pair < -tau) / |P|. Returns a DataFrame indexed by entity.
    """
    # Get unique (entity, gene) pairs
    ent_arr = df[ent_col].to_numpy()
    gene_arr = df["feature_name"].to_numpy()
    ent_codes, ent_index = pd.factorize(ent_arr, sort=False)
    gene_codes = np.array([gene_to_idx.get(g, -1) for g in gene_arr],
                           dtype=np.int64)
    # Drop rows with genes not in panel
    keep = gene_codes >= 0
    ent_codes = ent_codes[keep]
    gene_codes = gene_codes[keep]
    # Dedup (entity, gene)
    pair_df = pd.DataFrame({"e": ent_codes, "g": gene_codes}).drop_duplicates()
    # Sort by entity
    pair_df = pair_df.sort_values(["e", "g"]).reset_index(drop=True)
    # For each entity, slice the gene set and compute coherence
    # We do this vectorized via groupby with apply — slow-ish but ok for ~470k
    # entities at this scale.
    sizes = pair_df.groupby("e").size()
    starts = sizes.cumsum().shift(fill_value=0).astype(np.int64).to_numpy()
    counts = sizes.to_numpy()
    n_ent = len(sizes)
    e_codes = sizes.index.to_numpy()
    g_arr = pair_df["g"].to_numpy()

    C = np.full(n_ent, np.nan, dtype=np.float32)
    purity = np.full(n_ent, np.nan, dtype=np.float32)
    conflict = np.full(n_ent, np.nan, dtype=np.float32)
    k_genes = np.zeros(n_ent, dtype=np.int32)
    n_pairs_total = np.zeros(n_ent, dtype=np.int64)

    # Try to use the Cython kernel for the per-entity coherence
    try:
        from tracer import _cy_prune  # noqa
        use_cy = True
    except Exception:
        use_cy = False

    for i in range(n_ent):
        a, b = starts[i], starts[i] + counts[i]
        gids = g_arr[a:b]
        k = gids.size
        k_genes[i] = k
        if k < 2:
            C[i], purity[i], conflict[i] = 0.0, 0.0, 0.0
            n_pairs_total[i] = 0
            continue
        if use_cy:
            gids32 = np.ascontiguousarray(gids, dtype=np.int32)
            c_val, pu, cf = _cy_prune.coherence_count_kernel(
                gids32, W, float(tau)
            )
            C[i] = c_val; purity[i] = pu; conflict[i] = cf
        else:
            sub = W[np.ix_(gids, gids)]
            # upper triangle (k choose 2)
            iu = np.triu_indices(k, k=1)
            w = sub[iu]
            w = w[~np.isnan(w)]
            if w.size == 0:
                C[i], purity[i], conflict[i] = 0.0, 0.0, 0.0
                continue
            pu = float((w > tau).mean())
            cf = float((w < -tau).mean())
            C[i] = pu - cf; purity[i] = pu; conflict[i] = cf
        n_pairs_total[i] = k * (k - 1) // 2

    return pd.DataFrame({
        "entity": ent_index[e_codes],
        "C": C, "purity": purity, "conflict": conflict,
        "k_genes": k_genes, "n_pairs": n_pairs_total,
    })


def _quartiles(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0}
    q = np.percentile(x, [0, 25, 50, 75, 100])
    return {
        "n": int(x.size),
        "min": float(q[0]), "Q1": float(q[1]), "median": float(q[2]),
        "Q3": float(q[3]), "max": float(q[4]),
        "mean": float(x.mean()), "std": float(x.std()),
        "iqr": float(q[3] - q[1]),
    }


def _print_quartile_table(label: str, df: pd.DataFrame, etype_col: str | None,
                            min_tx: int = 0, tx_count: pd.Series | None = None
                            ) -> None:
    print(f"\n=== {label}  (n_entities={len(df):,}) ===")
    # Compute weighted-by-tx and unweighted summaries
    cols = ["C", "purity", "conflict"]
    print(f"  {'group':18s}  {'n':>10s}  {'min':>7s}  {'Q1':>7s}  "
          f"{'median':>7s}  {'Q3':>7s}  {'max':>7s}  {'mean':>7s}  {'std':>7s}")
    df_use = df.copy()
    if min_tx > 0 and tx_count is not None:
        df_use = df_use.assign(n_tx=tx_count.reindex(df_use["entity"]).to_numpy())
        df_use = df_use[df_use["n_tx"] >= min_tx]
    for grp, sub in [("all", df_use)] + (
        [(et, df_use[df_use[etype_col] == et]) for et in sorted(df_use[etype_col].unique())]
        if etype_col else []):
        for col in cols:
            x = sub[col].dropna().to_numpy()
            q = _quartiles(x)
            tag = f"{grp} • {col}" if col != "C" else f"{grp:>5s} • {col}"
            if "n" not in q or q["n"] == 0:
                print(f"  {tag:18s}  {'0':>10s}  <empty>")
                continue
            print(f"  {tag:18s}  {q['n']:>10,}  "
                  f"{q['min']:>+7.3f}  {q['Q1']:>+7.3f}  "
                  f"{q['median']:>+7.3f}  {q['Q3']:>+7.3f}  "
                  f"{q['max']:>+7.3f}  {q['mean']:>+7.3f}  {q['std']:>7.3f}")
        print()


def main() -> int:
    t0 = time.time()
    print("loading partitions ...", flush=True)
    pa = pd.read_parquet(DIR_A / "partition_sequential.parquet")
    pb = pd.read_parquet(DIR_B / "partition_sequential.parquet")
    assert (pa["transcript_id"].to_numpy() == pb["transcript_id"].to_numpy()).all()
    print(f"  thr=10: {len(pa):,} tx;   thr=0: {len(pb):,} tx", flush=True)

    print("loading transcripts.parquet (feature_name) ...", flush=True)
    feats = pd.read_parquet(PDAC_PARQUET,
                            columns=["transcript_id", "feature_name"])
    feats = feats.set_index("transcript_id").reindex(pa["transcript_id"]).reset_index()
    pa = pa.assign(feature_name=feats["feature_name"].astype(str).to_numpy())
    pb = pb.assign(feature_name=feats["feature_name"].astype(str).to_numpy())

    # Gene universe (union of all genes mentioned in either panel + transcripts)
    panel_a = pd.read_parquet(PANEL_A)
    panel_b = pd.read_parquet(PANEL_B)
    all_genes = sorted(set(panel_a["gene_i"].astype(str))
                         | set(panel_a["gene_j"].astype(str))
                         | set(panel_b["gene_i"].astype(str))
                         | set(panel_b["gene_j"].astype(str))
                         | set(pa["feature_name"].unique()))
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    print(f"  gene universe: {len(all_genes):,}", flush=True)

    print("building panel matrices ...", flush=True)
    W_a = _build_W(PANEL_A, all_genes)
    W_b = _build_W(PANEL_B, all_genes)
    n_in_a = int((~np.isnan(W_a)).sum() // 2)
    n_in_b = int((~np.isnan(W_b)).sum() // 2)
    print(f"  W_thr10: {n_in_a:,} non-NaN pairs", flush=True)
    print(f"  W_thr0:  {n_in_b:,} non-NaN pairs", flush=True)

    # Filter to assigned tx in each partition
    un_a = pa["label"].astype(str).isin(SENTINELS) | \
           pa["label"].astype(str).str.endswith("_rejected", na=False)
    un_b = pb["label"].astype(str).isin(SENTINELS) | \
           pb["label"].astype(str).str.endswith("_rejected", na=False)
    pa_assigned = pa.loc[~un_a, ["label", "_etype", "feature_name"]].rename(
        columns={"label": "entity"})
    pb_assigned = pb.loc[~un_b, ["label", "_etype", "feature_name"]].rename(
        columns={"label": "entity"})
    # tx-count per entity
    n_tx_a = pa_assigned.groupby("entity").size()
    n_tx_b = pb_assigned.groupby("entity").size()
    # etype per entity
    etype_a = pa_assigned.drop_duplicates("entity").set_index("entity")["_etype"]
    etype_b = pb_assigned.drop_duplicates("entity").set_index("entity")["_etype"]

    print(f"\ncomputing coherence (own panel) ...", flush=True)
    t = time.time()
    coh_a = _coherence_per_entity(pa_assigned, "entity", W_a, gene_to_idx)
    print(f"  thr=10 entities scored against thr=10 panel: "
          f"{time.time()-t:.1f}s, n={len(coh_a):,}", flush=True)
    t = time.time()
    coh_b = _coherence_per_entity(pb_assigned, "entity", W_b, gene_to_idx)
    print(f"  thr=0  entities scored against thr=0  panel: "
          f"{time.time()-t:.1f}s, n={len(coh_b):,}", flush=True)

    # Cross-panel scoring
    print(f"\ncomputing coherence (cross panel) ...", flush=True)
    t = time.time()
    coh_a_on_b = _coherence_per_entity(pa_assigned, "entity", W_b, gene_to_idx)
    print(f"  thr=10 entities scored against thr=0 panel:  "
          f"{time.time()-t:.1f}s", flush=True)
    t = time.time()
    coh_b_on_a = _coherence_per_entity(pb_assigned, "entity", W_a, gene_to_idx)
    print(f"  thr=0  entities scored against thr=10 panel: "
          f"{time.time()-t:.1f}s", flush=True)

    # Attach _etype
    coh_a["_etype"] = coh_a["entity"].map(etype_a).fillna("unknown")
    coh_b["_etype"] = coh_b["entity"].map(etype_b).fillna("unknown")
    coh_a_on_b["_etype"] = coh_a_on_b["entity"].map(etype_a).fillna("unknown")
    coh_b_on_a["_etype"] = coh_b_on_a["entity"].map(etype_b).fillna("unknown")

    # Report — unfiltered (all assigned entities), and filtered to k_genes >= 2
    # (entities with only 1 gene can't have any pair → C undefined). We always
    # set C=0 for those, so they bias the lower tail. Report both.

    print("\n" + "=" * 78)
    print("OWN-PANEL coherence (each method scored against the panel it used)")
    print("=" * 78)
    _print_quartile_table("thr=10 (own panel)", coh_a[coh_a["k_genes"] >= 2], "_etype")
    _print_quartile_table("thr=0  (own panel)", coh_b[coh_b["k_genes"] >= 2], "_etype")

    print("\n" + "=" * 78)
    print("CROSS-PANEL coherence")
    print("=" * 78)
    _print_quartile_table("thr=10 scored on thr=0 panel",
                            coh_a_on_b[coh_a_on_b["k_genes"] >= 2], "_etype")
    _print_quartile_table("thr=0  scored on thr=10 panel",
                            coh_b_on_a[coh_b_on_a["k_genes"] >= 2], "_etype")

    # Persist
    coh_a.to_parquet(DIR_A / "entity_coherence.parquet", index=False)
    coh_b.to_parquet(DIR_B / "entity_coherence.parquet", index=False)
    coh_a_on_b.to_parquet(DIR_A / "entity_coherence_cross.parquet", index=False)
    coh_b_on_a.to_parquet(DIR_B / "entity_coherence_cross.parquet", index=False)
    print(f"\nentity coherence parquets saved to both dirs.")
    print(f"\ntotal wall: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
