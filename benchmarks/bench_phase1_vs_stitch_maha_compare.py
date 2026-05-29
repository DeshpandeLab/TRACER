#!/usr/bin/env python3
"""Four-arm head-to-head: baseline vs Stitch-only Maha rescue vs
Phase-1-only Maha rescue vs BOTH, on two ROIs (50µm EMT + 2mm PDAC).

Arms (all SEG pipeline)
-----------------------
  A baseline       — neither rescue
  B Stitch-only    — cfg.stitch.mahalanobis_d_rescue=1.0
  C Phase-1-only   — cfg.phase1.maha_remerge_d=1.0
  D Both           — Stitch and Phase-1 rescue both on

Decision criterion
------------------
If arm D == arm C in entity / partition counts, Stitch-time rescue is
redundant once Phase-1 rescue runs — recommend deprecating Stitch's
rescue. If D differs from C, Stitch catches edge cases Phase 1 misses
(retain).

Run
---
    PYTHONPATH=src LIBOMP_PREFIX=$(brew --prefix libomp) \
        /opt/homebrew/Caskroom/miniconda/base/envs/genesis_env/bin/python \
        benchmarks/bench_phase1_vs_stitch_maha_compare.py
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/"
    "data/outs/transcripts.parquet"
)
PANEL_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr10.parquet"
)
OUT_BASE = Path(__file__).resolve().parents[1] / "analysis/phase1_vs_stitch_maha"

# ROI definitions
EMT_CENTER = (10525.0, 1775.0)
EMT_HALF = 25.0
PDAC_CENTER = (7255.0, 3023.7)
PDAC_HALF = 1000.0

SENT = {"-1", "DROP", "UNASSIGNED", "nan", "__GUARD_SKIP__"}


def _codes(labels: np.ndarray) -> np.ndarray:
    is_un = pd.Series(labels).isin(SENT).to_numpy()
    c, _ = pd.factorize(labels, sort=False)
    c = c.astype(np.int64)
    c[is_un] = -1
    return c


def _metrics(a: np.ndarray, b: np.ndarray) -> dict:
    keep = (a >= 0) & (b >= 0)
    n = int(keep.sum())
    if n == 0:
        return dict(n=0, ARI=float("nan"), h=float("nan"),
                    c=float("nan"), V=float("nan"))
    a_d, _ = pd.factorize(a[keep], sort=False)
    b_d, _ = pd.factorize(b[keep], sort=False)
    C = sp.coo_matrix(
        (np.ones(n, np.int64), (a_d, b_d)),
        shape=(a_d.max() + 1, b_d.max() + 1),
    ).tocsr()
    nij = C.data.astype(np.float64)
    ai = np.asarray(C.sum(1)).ravel().astype(np.float64)
    bj = np.asarray(C.sum(0)).ravel().astype(np.float64)
    idx = (nij @ nij - n) / 2.0
    at = (ai @ ai - n) / 2.0
    bt = (bj @ bj - n) / 2.0
    npairs = n * (n - 1) / 2.0
    ex = at * bt / npairs if npairs else 0.0
    mx = (at + bt) / 2.0
    ari = (idx - ex) / (mx - ex) if mx != ex else 1.0

    def ent(p):
        p = p[p > 0] / n
        return -np.sum(p * np.log(p))

    Ha, Hb = ent(ai), ent(bj)
    pj = nij[nij > 0] / n
    Hj = -np.sum(pj * np.log(pj))
    MI = max(0.0, Ha + Hb - Hj)
    h = MI / Ha if Ha > 0 else 1.0
    cc = MI / Hb if Hb > 0 else 1.0
    V = 2 * h * cc / (h + cc) if (h + cc) else 0.0
    return dict(n=n, ARI=ari, h=h, c=cc, V=V)


def _build_W(panel_path: Path):
    panel = pd.read_parquet(panel_path).rename(columns={"value": "PMI"})
    panel["gene_i"] = panel["gene_i"].astype(str)
    panel["gene_j"] = panel["gene_j"].astype(str)
    genes = sorted(set(panel.gene_i) | set(panel.gene_j))
    g2i = {g: i for i, g in enumerate(genes)}
    W = np.full((len(genes), len(genes)), np.nan, np.float32)
    gi = panel.gene_i.map(g2i).to_numpy(np.int64)
    gj = panel.gene_j.map(g2i).to_numpy(np.int64)
    v = panel.PMI.to_numpy(np.float32)
    W[gi, gj] = v
    W[gj, gi] = v
    np.fill_diagonal(W, np.nan)
    return W, g2i


def _coh_summary(part: pd.DataFrame, W, g2i) -> dict:
    from tracer.stitching import coherence
    cs = []
    for lbl, sub in part.groupby("label"):
        if str(lbl) in SENT:
            continue
        ids = np.array(
            [g2i[g] for g in sub.feature_name.astype(str).unique()
             if g in g2i], dtype=np.int64,
        )
        if ids.size < 2:
            continue
        C, _, _ = coherence(ids, W, mode="count", threshold=0.2, metric="pmi")
        cs.append(C)
    if not cs:
        return dict(n=0, mean=float("nan"), p10=float("nan"), p50=float("nan"))
    s = pd.Series(cs)
    return dict(
        n=len(s), mean=float(s.mean()),
        p10=float(s.quantile(0.1)),
        p50=float(s.quantile(0.5)),
    )


def _load_roi(center, half) -> pd.DataFrame:
    xlo, xhi = center[0] - half, center[0] + half
    ylo, yhi = center[1] - half, center[1] + half
    df = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "overlaps_nucleus",
                 "feature_name", "x_location", "y_location", "z_location"],
    ).rename(columns={"x_location": "x", "y_location": "y", "z_location": "z"})
    df = df.loc[df.x.between(xlo, xhi) & df.y.between(ylo, yhi)].reset_index(drop=True)
    df["cell_id"] = df["cell_id"].astype(str)
    df["feature_name"] = df["feature_name"].astype(str)
    return df


def _cfg_with(maha_phase1=None, maha_stitch=None):
    from tracer.config import load_config
    c = load_config()
    new_phase1 = dataclasses.replace(
        c.phase1, maha_remerge_d=maha_phase1,
    )
    new_stitch = dataclasses.replace(
        c.stitch, mahalanobis_d_rescue=maha_stitch,
    )
    return dataclasses.replace(c, phase1=new_phase1, stitch=new_stitch)


def _run_arm(tag: str, df: pd.DataFrame, panel: pd.DataFrame, cfg,
             W, g2i, out_dir: Path) -> dict:
    import tests._pipeline_runner as runner
    from tests._pipeline_runner import run_segmented_pipeline
    import tracer.stitching as stitching
    if hasattr(stitching, "_LAST_GATE_STATS"):
        stitching._LAST_GATE_STATS.clear()

    runner.PHASE1_RERANK_ENABLED = True
    runner.PHASE1_REASSIGN_AFTER_1C = True

    t = time.time()
    df_out, progression = run_segmented_pipeline(df.copy(), panel, cfg=cfg)
    wall = time.time() - t

    col = "stitched" if "stitched" in df_out.columns else "tracer_id"
    labels = (
        df_out.set_index("transcript_id")[col].astype(str)
        .reindex(df.transcript_id).to_numpy()
    )
    n_ent = int(pd.Series(labels)[~pd.Series(labels).isin(SENT)].nunique())
    n_un = int(pd.Series(labels).isin(SENT).sum())
    gt = df.cell_id.astype(str).to_numpy()
    m = _metrics(_codes(gt), _codes(labels))

    part = pd.DataFrame({
        "transcript_id": df.transcript_id.to_numpy(),
        "cell_id": gt,
        "label": labels,
        "feature_name": df.feature_name.to_numpy(),
        "x": df.x.to_numpy(), "y": df.y.to_numpy(),
    })
    part.to_parquet(out_dir / f"partition_{tag}.parquet", index=False)
    coh = _coh_summary(part, W, g2i)
    gate = dict(getattr(stitching, "_LAST_GATE_STATS", {}))

    # Phase-1 rescue counts come from the progression stage.
    n_phase1_resc = 0
    for stg in progression:
        if stg.get("stage") == "Phase1-Maha-Remerge":
            # Compare entity count delta vs the previous stage.
            # Use the stage_seconds field for sanity; entity drop = rescues.
            prev = None
            for s in progression:
                if s.get("stage") == "Phase1-QC":
                    prev = s
                    break
            if prev is not None:
                # entity count is "n_cells + n_partials + n_components"
                # in the recorded state dict.
                cur_n = (stg.get("n_cells", 0) + stg.get("n_partials", 0)
                         + stg.get("n_components", 0))
                prev_n = (prev.get("n_cells", 0) + prev.get("n_partials", 0)
                          + prev.get("n_components", 0))
                n_phase1_resc = max(0, prev_n - cur_n)
            break

    return dict(
        wall=wall, n_ent=n_ent, n_un=n_un, m=m,
        coh=coh, gate=gate,
        n_phase1_resc=int(n_phase1_resc),
        labels=labels,
    )


def _print_table(roi_tag: str, results: dict) -> None:
    print("\n" + "=" * 130, flush=True)
    print(f"COMPARISON SUMMARY — {roi_tag}", flush=True)
    print("=" * 130, flush=True)
    hdr = (f"{'arm':14s}  {'ent':>5s}  {'unas':>6s}  "
           f"{'ARI':>6s}  {'h':>6s}  {'c':>6s}  {'V':>6s}  "
           f"{'coh.mean':>8s}  {'coh.p10':>7s}  "
           f"{'st_resc':>7s}  {'p1_resc':>7s}  "
           f"{'wall':>7s}")
    print(hdr, flush=True)
    print("-" * 130, flush=True)
    for tag in ("A_baseline", "B_stitch", "C_phase1", "D_both"):
        if tag not in results:
            continue
        r = results[tag]
        m, coh, gate = r["m"], r["coh"], r["gate"]
        print(f"{tag:14s}  {r['n_ent']:>5d}  {r['n_un']:>6d}  "
              f"{m['ARI']:>6.3f}  {m['h']:>6.3f}  {m['c']:>6.3f}  {m['V']:>6.3f}  "
              f"{coh['mean']:>8.3f}  {coh['p10']:>7.3f}  "
              f"{gate.get('mahalanobis_rescues', 0):>7d}  "
              f"{r['n_phase1_resc']:>7d}  "
              f"{r['wall']:>6.1f}s", flush=True)


def _compare_C_vs_D(results: dict) -> dict:
    """Entity-set diff between arm C (Phase-1-only) and arm D (both).
    If identical, Stitch-time rescue adds nothing."""
    if "C_phase1" not in results or "D_both" not in results:
        return {}
    lc = results["C_phase1"]["labels"]
    ld = results["D_both"]["labels"]
    ents_c = set(pd.Series(lc)[~pd.Series(lc).isin(SENT)].unique())
    ents_d = set(pd.Series(ld)[~pd.Series(ld).isin(SENT)].unique())
    identical = (lc == ld).all()
    same_entities = ents_c == ents_d
    n_diff = int((lc != ld).sum())
    return dict(
        identical_per_tx=bool(identical),
        same_entity_set=bool(same_entities),
        n_tx_disagree=n_diff,
        n_ent_C=len(ents_c),
        n_ent_D=len(ents_d),
        # arm D additional rescues = mahalanobis_rescues count of D
        # if D == C → those rescues did nothing new (would have been
        # caught in Phase 1).
        n_stitch_rescues_D=int(
            results["D_both"]["gate"].get("mahalanobis_rescues", 0)
        ),
    )


def main() -> int:
    if not PDAC_PARQUET.exists() or not PANEL_PARQUET.exists():
        print(f"SKIP: missing data\n  PDAC={PDAC_PARQUET}\n  "
              f"PANEL={PANEL_PARQUET}", flush=True)
        return 0

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    panel = (pd.read_parquet(PANEL_PARQUET).rename(columns={"value": "NPMI"})
             [["gene_i", "gene_j", "NPMI"]])
    W, g2i = _build_W(PANEL_PARQUET)

    # ----------------------------------------------------------------
    # ROI 1 — 50µm EMT.
    # ----------------------------------------------------------------
    t0 = time.time()
    out_emt = OUT_BASE / "emt_50um"
    out_emt.mkdir(parents=True, exist_ok=True)
    df_emt = _load_roi(EMT_CENTER, EMT_HALF)
    print(f"\n[EMT ROI] {len(df_emt):,} tx, "
          f"{df_emt.cell_id.nunique()} cell_ids "
          f"[loaded {time.time()-t0:.1f}s]", flush=True)

    res_emt: dict = {}
    arms = [
        ("A_baseline", None, None),
        ("B_stitch",   None, 1.0),
        ("C_phase1",   1.0,  None),
        ("D_both",     1.0,  1.0),
    ]
    for tag, m_p1, m_st in arms:
        cfg = _cfg_with(maha_phase1=m_p1, maha_stitch=m_st)
        print(f"\n>>> EMT  {tag}  (phase1={m_p1}  stitch={m_st}) <<<",
              flush=True)
        res_emt[tag] = _run_arm(tag, df_emt, panel, cfg, W, g2i, out_emt)
    _print_table("EMT 50µm ROI", res_emt)
    diff_emt = _compare_C_vs_D(res_emt)
    print("\nEMT  C-vs-D entity-set comparison:", flush=True)
    for k, v in diff_emt.items():
        print(f"   {k:24s} = {v}", flush=True)

    # ----------------------------------------------------------------
    # ROI 2 — 2 mm PDAC.
    # ----------------------------------------------------------------
    t1 = time.time()
    out_pdac = OUT_BASE / "pdac_2mm"
    out_pdac.mkdir(parents=True, exist_ok=True)
    df_pdac = _load_roi(PDAC_CENTER, PDAC_HALF)
    print(f"\n[PDAC ROI] {len(df_pdac):,} tx, "
          f"{df_pdac.cell_id.nunique()} cell_ids "
          f"[loaded {time.time()-t1:.1f}s]", flush=True)

    res_pdac: dict = {}
    for tag, m_p1, m_st in arms:
        cfg = _cfg_with(maha_phase1=m_p1, maha_stitch=m_st)
        print(f"\n>>> PDAC {tag}  (phase1={m_p1}  stitch={m_st}) <<<",
              flush=True)
        res_pdac[tag] = _run_arm(tag, df_pdac, panel, cfg, W, g2i, out_pdac)
    _print_table("PDAC 2mm ROI", res_pdac)
    diff_pdac = _compare_C_vs_D(res_pdac)
    print("\nPDAC C-vs-D entity-set comparison:", flush=True)
    for k, v in diff_pdac.items():
        print(f"   {k:24s} = {v}", flush=True)

    # ----------------------------------------------------------------
    # Decision summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 130, flush=True)
    print("DECISION CRITERION  (Stitch-time rescue deprecable when D ≡ C)",
          flush=True)
    print("=" * 130, flush=True)
    for roi_tag, diff in (("EMT", diff_emt), ("PDAC", diff_pdac)):
        verdict = (
            "DEPRECATE Stitch-time rescue  (no add'l catches over Phase 1)"
            if diff.get("identical_per_tx", False)
            else f"KEEP Stitch-time rescue   ({diff.get('n_tx_disagree', 0)} "
                 f"tx disagree;  D adds something C misses)"
        )
        print(f"  {roi_tag:6s} → {verdict}", flush=True)

    # Persist a concise summary CSV.
    summary_rows = []
    for roi, res in (("EMT", res_emt), ("PDAC", res_pdac)):
        for tag, r in res.items():
            summary_rows.append({
                "roi": roi, "arm": tag,
                "n_ent": r["n_ent"], "n_un": r["n_un"],
                "ARI": r["m"]["ARI"], "V": r["m"]["V"],
                "coh_mean": r["coh"]["mean"],
                "stitch_rescues": r["gate"].get("mahalanobis_rescues", 0),
                "phase1_rescues": r["n_phase1_resc"],
                "wall_s": round(r["wall"], 2),
            })
    pd.DataFrame(summary_rows).to_csv(
        OUT_BASE / "summary.csv", index=False
    )
    print(f"\nsummary CSV: {OUT_BASE / 'summary.csv'}", flush=True)
    print(f"partitions in {OUT_BASE}", flush=True)
    print(f"total wall: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
