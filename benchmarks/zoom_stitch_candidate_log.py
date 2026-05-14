#!/usr/bin/env python3
"""Emulate production Stitch's candidate-enumeration + gating on the NOSEG
worst-case ROI (cell `nloapcgp-1`, 14 cascade fragments) and dump a full
per-pair decision log.

Production Stitch parameters (from tests/_pipeline_runner.py:1489):
    candidate_source = "grid"
    G                = 2.0   µm
    G_z              = 1.0   µm
    stitch_neighborhood = "8"   (xy 8-Moore)
    z_neighbor_depth   = 1     (±1 z-bin)
    min_local_tx_per_entity = 3
    deltaC_min       = 0.03
    penalize_simplicity = True
    threshold        = 0.2     (PMI scale)
    mode             = "count", metric = "pmi"

For each ordered pair (A, B) of cascade entities, we log:
    n_A, n_B           — tx counts
    bins_A, bins_B     — unique grid bins occupied
    witness_A, witness_B — unique tx of each that share a bin with the other
                          (within the 8-Moore + z-window neighborhood)
    co_bins            — count of bins where pair co-occurs
    pass_witness       — both witness counts ≥ 3
    C_A, C_B, C_union  — coherence values
    dC_raw, dC_pen     — ΔC under both formulations
    pass_deltaC_min    — dC_pen ≥ 0.03
    final_verdict      — accepted iff pass_witness AND pass_deltaC_min

Output: benchmarks/stitch_zoom_seg_vs_noseg/zoom_worst_stitch_candidates.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
ZOOM_DIR = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg"

# Production Stitch knobs
G_XY = 2.0
G_Z = 1.0
Z_DEPTH = 1
MIN_LOCAL_TX = 3
DELTAC_MIN = -0.01
C_UNION_BYPASS = 0.9  # bypass ΔC gate if C(union) ≥ this
TAU = 0.2
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}


def _build_W(panel_path, all_genes):
    panel = pd.read_parquet(panel_path).rename(columns={"value": "NPMI"})
    panel["gene_i"] = panel["gene_i"].astype(str)
    panel["gene_j"] = panel["gene_j"].astype(str)
    g2i = {g: i for i, g in enumerate(all_genes)}
    G = len(all_genes)
    W = np.full((G, G), np.nan, dtype=np.float32)
    gi = panel["gene_i"].map(g2i)
    gj = panel["gene_j"].map(g2i)
    have = gi.notna() & gj.notna()
    gi = gi[have].to_numpy(dtype=np.int64)
    gj = gj[have].to_numpy(dtype=np.int64)
    v = panel.loc[have, "NPMI"].to_numpy(dtype=np.float32)
    W[gi, gj] = v; W[gj, gi] = v
    return W, g2i


def coherence(gene_set, W, g2i, tau=TAU):
    """C = purity − conflict over unique gene pairs."""
    gids = [g2i[g] for g in gene_set if g in g2i]
    if len(gids) < 2:
        return float("nan"), 0
    gids = np.asarray(sorted(set(gids)), dtype=np.int64)
    k = len(gids)
    sub = W[np.ix_(gids, gids)]
    iu = np.triu_indices(k, k=1)
    w = sub[iu]
    w = w[~np.isnan(w)]
    if w.size == 0:
        return float("nan"), k
    purity = float((w > tau).mean())
    conflict = float((w < -tau).mean())
    return purity - conflict, k


def main() -> int:
    zoom = pd.read_parquet(ZOOM_DIR / "zoom_worst_tx.parquet")
    feats = pd.read_parquet(
        PDAC_PARQUET, columns=["transcript_id", "feature_name"]
    )
    m = zoom.merge(feats, on="transcript_id", how="left")
    m["feature_name"] = m["feature_name"].astype(str)
    m["noseg_lab"] = m["noseg_lab"].astype(str)

    # Subset to the 14 cascade fragments belonging to cell nloapcgp-1
    sub = m[
        (m["cell_id"].astype(str) == "nloapcgp-1")
        & (~m["noseg_lab"].isin(SENTINELS))
    ].copy()
    print(f"nloapcgp-1 fragments: {sub['noseg_lab'].nunique()} entities, "
          f"{len(sub)} tx", flush=True)

    # Need z coordinate — pull from raw
    raw_z = pd.read_parquet(
        PDAC_PARQUET, columns=["transcript_id", "z_location"]
    ).rename(columns={"z_location": "z"})
    sub = sub.merge(raw_z, on="transcript_id", how="left")
    print(f"  z range: [{sub['z'].min():.2f}, {sub['z'].max():.2f}]",
          flush=True)

    # Build W
    panel_raw = pd.read_parquet(PANEL)
    all_genes = sorted(set(panel_raw["gene_i"].astype(str))
                       | set(panel_raw["gene_j"].astype(str))
                       | set(m["feature_name"].unique()))
    W, g2i = _build_W(PANEL, all_genes)

    # Bin each tx into (xb, yb, zb)
    xb = np.floor(sub["x"].to_numpy() / G_XY).astype(np.int64)
    yb = np.floor(sub["y"].to_numpy() / G_XY).astype(np.int64)
    zb = np.floor(sub["z"].to_numpy() / G_Z).astype(np.int64)
    sub["xb"] = xb; sub["yb"] = yb; sub["zb"] = zb

    # Per entity: tx ids, gene set, bin set, n_tx
    by_ent = {}
    for ent, grp in sub.groupby("noseg_lab"):
        by_ent[ent] = {
            "tx_ids": grp["transcript_id"].to_numpy(),
            "genes": set(grp["feature_name"].unique()),
            "bins": set(zip(grp["xb"].tolist(),
                            grp["yb"].tolist(),
                            grp["zb"].tolist())),
            "n_tx": len(grp),
            "tx_bins": list(zip(grp["transcript_id"].tolist(),
                                grp["xb"].tolist(),
                                grp["yb"].tolist(),
                                grp["zb"].tolist())),
        }

    ents = sorted(by_ent.keys())
    print(f"  entities: {ents}", flush=True)

    # Precompute coherence per entity
    C_ent = {}
    n_g_ent = {}
    for e in ents:
        C, k = coherence(by_ent[e]["genes"], W, g2i)
        C_ent[e] = C; n_g_ent[e] = k

    # 8-Moore xy offsets + ±Z_DEPTH z offsets
    nbrs = [(dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in range(-Z_DEPTH, Z_DEPTH + 1)]

    records = []
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            a, b = ents[i], ents[j]
            A = by_ent[a]; B = by_ent[b]

            # Witness count: how many tx of A live in a bin that some tx of
            # B occupies within the (8-Moore xy + ±Z_DEPTH z) window?
            # Build expanded set of bins reachable from A's bins.
            B_bins = B["bins"]
            # For each tx of A, check if any of (xb+dx, yb+dy, zb+dz) is in B_bins
            wit_A_ids = set()
            for (tid, xa, ya, za) in A["tx_bins"]:
                for dx, dy, dz in nbrs:
                    if (xa + dx, ya + dy, za + dz) in B_bins:
                        wit_A_ids.add(tid)
                        break
            A_bins = A["bins"]
            wit_B_ids = set()
            for (tid, xb_, yb_, zb_) in B["tx_bins"]:
                for dx, dy, dz in nbrs:
                    if (xb_ + dx, yb_ + dy, zb_ + dz) in A_bins:
                        wit_B_ids.add(tid)
                        break

            # Count of A-bins that overlap any B-bin (expanded)
            co_bins = 0
            for (xa, ya, za) in A_bins:
                for dx, dy, dz in nbrs:
                    if (xa + dx, ya + dy, za + dz) in B_bins:
                        co_bins += 1
                        break

            # Effective threshold per side: min(MIN_LOCAL_TX, n_tx).
            # A 2-tx entity can never produce 3 witnesses; the gate would
            # block every merge involving it. Cap the requirement at the
            # entity's own size.
            eff_A = min(MIN_LOCAL_TX, A["n_tx"])
            eff_B = min(MIN_LOCAL_TX, B["n_tx"])
            pass_wit = (len(wit_A_ids) >= eff_A
                        and len(wit_B_ids) >= eff_B)

            # Coherence values
            Ca, Cb = C_ent[a], C_ent[b]
            na, nb = max(n_g_ent[a], 1), max(n_g_ent[b], 1)
            Cu, n_union = coherence(A["genes"] | B["genes"], W, g2i)
            n_union = max(n_union, 1)
            dC_raw = Cu - max(Ca, Cb)
            C_sep_pen = max(Ca - 1.0 / na, Cb - 1.0 / nb)
            dC_pen = (Cu - 1.0 / n_union) - C_sep_pen
            pass_dc = dC_pen >= DELTAC_MIN
            # Bypass: accept if union is still highly coherent even
            # when ΔC says reject. Spatial witness gate still applies.
            pass_bypass = (np.isfinite(Cu) and Cu >= C_UNION_BYPASS)
            accepted = pass_wit and (pass_dc or pass_bypass)

            records.append({
                "A": a, "B": b,
                "n_A": A["n_tx"], "n_B": B["n_tx"],
                "bins_A": len(A["bins"]), "bins_B": len(B["bins"]),
                "co_bins": co_bins,
                "witness_A": len(wit_A_ids),
                "witness_B": len(wit_B_ids),
                "eff_min_A": eff_A, "eff_min_B": eff_B,
                "pass_witness": pass_wit,
                "C_A": Ca, "C_B": Cb,
                "n_gA": na, "n_gB": nb,
                "C_union": Cu, "n_union": n_union,
                "dC_raw": dC_raw, "dC_pen": dC_pen,
                "pass_deltaC": pass_dc,
                "pass_bypass": pass_bypass,
                "accepted": accepted,
            })

    df = pd.DataFrame(records)
    out = ZOOM_DIR / "zoom_worst_stitch_candidates.csv"
    df.to_csv(out, index=False)
    print(f"\nFull candidate-edge log → {out}  ({len(df)} pairs)", flush=True)

    # Headline numbers
    n_total = len(df)
    n_co_bins = int((df["co_bins"] > 0).sum())
    n_pass_wit = int(df["pass_witness"].sum())
    n_pass_dc = int(df["pass_deltaC"].sum())
    n_accepted = int(df["accepted"].sum())
    print(f"\n  pairs with co-occurring grid bins (8-Moore + ±{Z_DEPTH}z): "
          f"{n_co_bins} / {n_total}")
    print(f"  pairs passing min_local_tx_per_entity=3:                "
          f"{n_pass_wit} / {n_total}")
    print(f"  pairs passing deltaC_min=0.03 (penalize_simplicity):    "
          f"{n_pass_dc} / {n_total}")
    n_pass_bypass = int(df["pass_bypass"].sum())
    n_bypass_only = int((df["pass_bypass"] & ~df["pass_deltaC"]
                         & df["pass_witness"]).sum())
    print(f"  pairs passing C(union) ≥ {C_UNION_BYPASS} bypass:        "
          f"{n_pass_bypass} / {n_total}")
    print(f"  pairs accepted (witness AND (deltaC OR bypass)):       "
          f"{n_accepted} / {n_total}")
    print(f"    of which gained from bypass (ΔC fail, bypass pass):  "
          f"{n_bypass_only}")

    # Diagnostics: pairs that pass deltaC but fail witness (would be merged if not for witness)
    blocked_by_wit = df[df["pass_deltaC"] & ~df["pass_witness"]]
    print(f"\nPairs passing ΔC_pen but blocked by witness ({len(blocked_by_wit)}):")
    if len(blocked_by_wit):
        for _, r in blocked_by_wit.iterrows():
            print(f"  {r['A']:>22s}  {r['B']:>22s}  n=({int(r['n_A'])},{int(r['n_B'])})  "
                  f"wit=({int(r['witness_A'])},{int(r['witness_B'])})  "
                  f"co_bins={int(r['co_bins']):>2d}  "
                  f"ΔC_pen={r['dC_pen']:+.4f}")

    # Pairs that pass both
    print(f"\nPairs ACCEPTED (witness ≥3 AND ΔC_pen ≥0.03):")
    acc = df[df["accepted"]]
    if len(acc):
        for _, r in acc.iterrows():
            print(f"  {r['A']:>22s}  {r['B']:>22s}  n=({int(r['n_A'])},{int(r['n_B'])})  "
                  f"wit=({int(r['witness_A'])},{int(r['witness_B'])})  "
                  f"co_bins={int(r['co_bins']):>2d}  "
                  f"ΔC_pen={r['dC_pen']:+.4f}")
    else:
        print("  NONE — no merger survives both gates.")

    # Pairs that pass witness but fail deltaC
    blocked_by_dc = df[df["pass_witness"] & ~df["pass_deltaC"]]
    print(f"\nPairs spatially adjacent (witness ≥3) but blocked by ΔC:")
    print(f"  {'A':>22s}  {'B':>22s}  n=    "
          f"wit=    co_bins  C_A     C_B    C_uni    ΔC_pen")
    for _, r in blocked_by_dc.sort_values("dC_pen", ascending=False).iterrows():
        print(f"  {r['A']:>22s}  {r['B']:>22s}  "
              f"n=({int(r['n_A']):>2d},{int(r['n_B']):>2d}) "
              f"wit=({int(r['witness_A']):>2d},{int(r['witness_B']):>2d}) "
              f"{int(r['co_bins']):>3d}      "
              f"{r['C_A']:>6.3f}  {r['C_B']:>6.3f}  {r['C_union']:>6.3f}  "
              f"{r['dC_pen']:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
