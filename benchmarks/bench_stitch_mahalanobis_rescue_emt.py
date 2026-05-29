#!/usr/bin/env python3
"""Stitch Mahalanobis-D RESCUE bench on an EMT-interface 50µm ROI.

ROI: 50µm window centered at (10525, 1775) — the densest EMT-interface
patch in the PDAC tissue (mes=558 / epi=585 balanced; 16 Xenium cells).
Epithelial cells (EPCAM/CEACAM6) interdigitate with mesenchymal
partials (ACTA2/SPARC/FN1) — the "two-program anti-correlation that
drags ΔC slightly negative on a legitimate single-cell merge"
geometry the Maha rescue targets.

Four arms (full SEG / NOSEG pipeline run per arm; pre-Stitch stages
are deterministic so all divergence isolates to Stitch):

  baseline                       no rescue                       (production)
  baseline + rescue_1.0          mahalanobis_d_rescue=1.0,
                                  rescue_delta_c_floor=-0.2

Expected outcomes:
  • jiecahje stays split — its CAF↔TAM pair has D≈1.59 > 1.0 AND
    ΔC likely < -0.2.
  • jikammne stays split — ΔC ≈ -0.49 < -0.2 floor (geometry-low D
    cannot rescue this engulfment doublet — that's the floor's job).
  • Any EMT-like fragmentation in baseline (ΔC borderline-negative,
    geometrically enmeshed) gets consolidated by rescue → entity
    count drops slightly; per-entity sizes shift.

Run::

    PYTHONPATH=src python benchmarks/bench_stitch_mahalanobis_rescue_emt.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

PDAC = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/"
    "data/outs/transcripts.parquet"
)
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr10.parquet"
)
OUT = Path(__file__).resolve().parents[1] / "analysis/stitch-maha-rescue/emt_roi"
OUT.mkdir(parents=True, exist_ok=True)

# EMT-interface 50µm ROI
ROI_CENTER = (10525.0, 1775.0)
ROI_HALF = 25.0
ROI_X = (ROI_CENTER[0] - ROI_HALF, ROI_CENTER[0] + ROI_HALF)
ROI_Y = (ROI_CENTER[1] - ROI_HALF, ROI_CENTER[1] + ROI_HALF)

SENT = {"-1", "DROP", "UNASSIGNED", "nan", "__GUARD_SKIP__"}
MES = {"FN1", "SPARC", "ACTA2", "VCAN", "LUM", "DCN", "PDGFRA", "TGFB1", "MMP9"}
EPI = {"EPCAM", "CEACAM6", "CEACAM1"}


def _etype(label: str) -> str:
    if label in SENT:
        return "drop"
    if label.startswith("cascade_"):
        return "component" if label.count("-") == 1 else "cascade_partial"
    n = label.count("-")
    if n == 1:
        return "cell"
    if n == 2:
        return "partial"
    return "unknown"


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
        return dict(n=0, ARI=float("nan"), NMI=float("nan"),
                    h=float("nan"), c=float("nan"), V=float("nan"))
    a_d, _ = pd.factorize(a[keep], sort=False)
    b_d, _ = pd.factorize(b[keep], sort=False)
    C = sp.coo_matrix((np.ones(n, np.int64), (a_d, b_d)),
                      shape=(a_d.max() + 1, b_d.max() + 1)).tocsr()
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
    nmi = MI / np.sqrt(Ha * Hb) if Ha > 0 and Hb > 0 else 0.0
    return dict(n=n, ARI=ari, NMI=nmi, h=h, c=cc, V=V)


def _build_W(panel_path):
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


def _emt_counts(part: pd.DataFrame, min_tx: int = 3) -> dict:
    """Classify entities Epi / Mes / EMT / Neither by marker-tx count."""
    out = {"Epi": 0, "Mes": 0, "EMT": 0, "Neither": 0}
    fp = part[~part.label.astype(str).isin(SENT)]
    for _lbl, sub in fp.groupby("label"):
        ne = int(sub.feature_name.isin(EPI).sum())
        nm = int(sub.feature_name.isin(MES).sum())
        epi_p, mes_p = ne >= min_tx, nm >= min_tx
        cls = ("EMT" if (epi_p and mes_p)
               else "Epi" if epi_p else "Mes" if mes_p else "Neither")
        out[cls] += 1
    return out


def _coh_summary(part: pd.DataFrame, W, g2i) -> dict:
    from tracer.stitching import coherence
    cs = []
    for lbl, sub in part.groupby("label"):
        if str(lbl) in SENT:
            continue
        ids = np.array([g2i[g] for g in sub.feature_name.astype(str).unique()
                        if g in g2i], dtype=np.int64)
        if ids.size < 2:
            continue
        C, _, _ = coherence(ids, W, mode="count", threshold=0.2, metric="pmi")
        cs.append(C)
    if not cs:
        return dict(n=0)
    s = pd.Series(cs)
    return dict(n=len(s), mean=float(s.mean()), p10=float(s.quantile(0.1)),
                p50=float(s.quantile(0.5)))


def main() -> int:
    # Bail out gracefully if integration data isn't on disk.
    if not PDAC.exists() or not PANEL.exists():
        print(
            f"SKIP: tutorial data not on disk\n  PDAC={PDAC} (exists={PDAC.exists()})\n"
            f"  PANEL={PANEL} (exists={PANEL.exists()})\n"
            f"This bench requires the PDAC IO tutorial dataset. Synthetic "
            f"unit tests (tests/test_stitch_mahalanobis_rescue.py) cover the "
            f"mechanism end-to-end.",
            flush=True,
        )
        return 0

    import tests._pipeline_runner as runner
    from tests._pipeline_runner import run_segmented_pipeline, run_noseg_pipeline
    from tracer.config import load_config
    import tracer.stitching as stitching

    t0 = time.time()
    df = pd.read_parquet(
        PDAC,
        columns=["transcript_id", "cell_id", "overlaps_nucleus",
                 "feature_name", "x_location", "y_location", "z_location"],
    ).rename(columns={"x_location": "x", "y_location": "y", "z_location": "z"})
    df = df.loc[df.x.between(*ROI_X) & df.y.between(*ROI_Y)].reset_index(drop=True)
    df["cell_id"] = df["cell_id"].astype(str)
    df["feature_name"] = df["feature_name"].astype(str)
    print(f"ROI: {len(df):,} tx, {df.cell_id.nunique()} cell_ids, "
          f"mes={int(df.feature_name.isin(MES).sum())} "
          f"epi={int(df.feature_name.isin(EPI).sum())}  "
          f"[{time.time()-t0:.1f}s]", flush=True)

    panel = (pd.read_parquet(PANEL).rename(columns={"value": "NPMI"})
             [["gene_i", "gene_j", "NPMI"]])
    W, g2i = _build_W(PANEL)

    runner.PHASE1_RERANK_ENABLED = True
    runner.PHASE1_REASSIGN_AFTER_1C = True
    cfg_seg = load_config()                      # SEG: witness default
    cfg_noseg = load_config(platform="noseg")    # NOSEG: 5-pass rescue preset

    # Monkey-patch the Stitch call to inject the rescue knobs.
    _orig = runner.apply_stitching_to_transcripts_memory_efficient
    inject: dict = {}

    def patched(*args, **kwargs):
        kwargs.update(inject)
        return _orig(*args, **kwargs)
    runner.apply_stitching_to_transcripts_memory_efficient = patched

    pipelines = [
        ("seg", run_segmented_pipeline, cfg_seg),
        ("noseg", run_noseg_pipeline, cfg_noseg),
    ]
    rescue_arms = [
        ("baseline", {}),
        ("rescue_1.0", {"mahalanobis_d_rescue": 1.0,
                        "rescue_delta_c_floor": -0.2}),
    ]

    gt = df.cell_id.astype(str).to_numpy()
    gt_c = _codes(gt)
    results = {}
    for pl_tag, fn, cfg in pipelines:
        for arm_tag, kw in rescue_arms:
            tag = f"{pl_tag}_{arm_tag}"
            inject.clear()
            inject.update(kw)
            if hasattr(stitching, "_LAST_GATE_STATS"):
                stitching._LAST_GATE_STATS.clear()
            t = time.time()
            df_out, _prog = fn(df.copy(), panel, cfg=cfg)
            wall = time.time() - t
            col = "stitched" if "stitched" in df_out.columns else "tracer_id"
            labels = (df_out.set_index("transcript_id")[col].astype(str)
                      .reindex(df.transcript_id).to_numpy())
            n_ent = int(pd.Series(labels)[~pd.Series(labels).isin(SENT)].nunique())
            n_un = int(pd.Series(labels).isin(SENT).sum())
            m = _metrics(gt_c, _codes(labels))
            part = pd.DataFrame({"transcript_id": df.transcript_id.to_numpy(),
                                 "cell_id": gt, "label": labels,
                                 "feature_name": df.feature_name.to_numpy(),
                                 "x": df.x.to_numpy(), "y": df.y.to_numpy()})
            part.to_parquet(OUT / f"partition_{tag}.parquet", index=False)
            coh = _coh_summary(part, W, g2i)
            emt = _emt_counts(part)
            gate = dict(getattr(stitching, "_LAST_GATE_STATS", {}))
            results[tag] = dict(wall=wall, n_ent=n_ent, n_un=n_un, m=m,
                                coh=coh, emt=emt, gate=gate, labels=labels)
            print(f"\n=== {tag} ===  [{wall:.1f}s]", flush=True)
            print(f"  entities={n_ent}  unassigned={n_un}", flush=True)
            print(f"  vs cell_id: ARI={m['ARI']:.4f} h={m['h']:.4f} "
                  f"c={m['c']:.4f} V={m['V']:.4f} (n={m['n']})", flush=True)
            print(f"  Epi/Mes/EMT/Neither (>=3 each, any balance): "
                  f"{emt['Epi']}/{emt['Mes']}/{emt['EMT']}/{emt['Neither']}", flush=True)
            print(f"  coherence: n={coh.get('n')} "
                  f"mean={coh.get('mean', float('nan')):.3f} "
                  f"p10={coh.get('p10', float('nan')):.3f}", flush=True)
            print(f"  gate stats: {gate}", flush=True)

    # Summary table.
    print("\n" + "=" * 90, flush=True)
    print(f"{'run':22s}  {'ent':>4s}  {'unas':>5s}  {'ARI':>6s}  {'h':>6s}  "
          f"{'c':>6s}  {'Epi':>3s}  {'Mes':>3s}  {'EMT':>3s}  {'merges':>6s}  "
          f"{'resc':>4s}", flush=True)
    print("=" * 90, flush=True)
    for pl_tag, _fn, _cfg in pipelines:
        for arm_tag, _kw in rescue_arms:
            tag = f"{pl_tag}_{arm_tag}"
            r = results[tag]
            print(f"{tag:22s}  {r['n_ent']:>4d}  {r['n_un']:>5d}  "
                  f"{r['m']['ARI']:>6.3f}  {r['m']['h']:>6.3f}  {r['m']['c']:>6.3f}  "
                  f"{r['emt']['Epi']:>3d}  {r['emt']['Mes']:>3d}  {r['emt']['EMT']:>3d}  "
                  f"{r['gate'].get('merges_total', 0):>6d}  "
                  f"{r['gate'].get('mahalanobis_rescues', 0):>4d}", flush=True)

    # Sanity-check the canonical doublets: jiecahje and jikammne must
    # remain split under rescue arms (per design — floor protects them).
    for pl_tag in ("seg", "noseg"):
        base_lbl = results[f"{pl_tag}_baseline"]["labels"]
        rescue_lbl = results[f"{pl_tag}_rescue_1.0"]["labels"]
        # Per-tx-set check: for each Xenium cell_id of interest, see
        # how many distinct labels the rescue produces.
        for cell_key in ("jiecahje", "jikammne"):
            mask = pd.Series(gt).str.contains(cell_key, na=False).to_numpy()
            if mask.sum() == 0:
                continue
            n_base = int(pd.Series(base_lbl[mask]).nunique())
            n_resc = int(pd.Series(rescue_lbl[mask]).nunique())
            print(f"  {pl_tag} cells matching '{cell_key}': "
                  f"baseline {n_base} labels → rescue {n_resc} labels "
                  f"(tx={int(mask.sum())})", flush=True)

    print(f"\nall outputs in {OUT}", flush=True)
    print(f"total wall: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
