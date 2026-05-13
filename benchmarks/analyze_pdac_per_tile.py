#!/usr/bin/env python3
"""Per-tile breakdown of tile-parallel vs sequential agreement.

For each of the 9 tiles in the 3x3 grid, restrict to tx whose cell-centroid
falls in that tile and compute the metric battery against the sequential
reference. Identifies whether any single tile is dragging the global score.

Uses the partitions saved by ``bench_pdac_roi_tile_postprocess.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score, rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)

PDAC_PARQUET = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/"
    "data/outs/transcripts.parquet"
)
ROI_CENTER = (7255.0, 3023.7)
ROI_HALF_SIDE = 1000.0
N_TILES_XY = (3, 3)
REPO = Path(__file__).resolve().parents[1]
IN_DIR = REPO / "benchmarks" / "pdac_roi_tile_postprocess"
SENTINELS = {"-1", "DROP", "UNASSIGNED", "nan"}


def _is_un(s: pd.Series) -> np.ndarray:
    return (s.isin(SENTINELS) | s.str.endswith("_rejected", na=False)).to_numpy()


def _codes(labels: pd.Series, singletons: bool = True) -> np.ndarray:
    is_un = _is_un(labels)
    codes, _ = pd.factorize(labels.to_numpy(), sort=False)
    codes = codes.astype(np.int64)
    if singletons:
        tx_idx = np.arange(labels.size, dtype=np.int64)
        codes[is_un] = -2 - tx_idx[is_un]
    else:
        codes[is_un] = -1
    return codes


def main() -> int:
    # Load tx coords + sequential + post-rescue partitions
    df_raw = pd.read_parquet(
        PDAC_PARQUET,
        columns=["transcript_id", "cell_id", "x_location", "y_location"],
    ).rename(columns={"x_location": "x", "y_location": "y"})
    xc, yc = ROI_CENTER; h = ROI_HALF_SIDE
    mask = df_raw["x"].between(xc - h, xc + h) & df_raw["y"].between(yc - h, yc + h)
    df_raw = df_raw.loc[mask].reset_index(drop=True)

    seq = pd.read_parquet(IN_DIR / "partition_sequential.parquet")
    naive = pd.read_parquet(IN_DIR / "partition_tiled_naive.parquet")
    dis = pd.read_parquet(IN_DIR / "partition_tiled_disambig.parquet")
    resc = pd.read_parquet(IN_DIR / "partition_tiled_postrescue.parquet")
    # All four are in the same tx order (the bench writes them aligned).
    assert (seq["transcript_id"].to_numpy() == df_raw["transcript_id"].to_numpy()).all()

    # Compute per-cell tile id (same logic as orchestrator: cell centroid → tile)
    cent = df_raw.groupby("cell_id")[["x", "y"]].mean()
    x_min, x_max = float(cent["x"].min()), float(cent["x"].max())
    y_min, y_max = float(cent["y"].min()), float(cent["y"].max())
    n_x, n_y = N_TILES_XY
    x_edges = np.linspace(x_min, x_max + 1e-9, n_x + 1)
    y_edges = np.linspace(y_min, y_max + 1e-9, n_y + 1)
    ix = np.clip(np.searchsorted(x_edges, cent["x"].to_numpy(), side="right") - 1, 0, n_x - 1)
    iy = np.clip(np.searchsorted(y_edges, cent["y"].to_numpy(), side="right") - 1, 0, n_y - 1)
    tile_of_cell = pd.Series(ix * n_y + iy, index=cent.index, name="tile")
    tx_tile = df_raw["cell_id"].map(tile_of_cell).to_numpy()

    def _metric_subset(seq_lab: pd.Series, t_lab: pd.Series, sel: np.ndarray) -> dict:
        if sel.sum() < 2:
            return {}
        s_sub = seq_lab.iloc[sel].reset_index(drop=True)
        t_sub = t_lab.iloc[sel].reset_index(drop=True)
        seq_un = _is_un(s_sub); t_un = _is_un(t_sub)
        sing_s = _codes(s_sub, True); sing_t = _codes(t_sub, True)
        both = (~seq_un) & (~t_un)
        if both.sum() < 2:
            return {}
        mega_s = _codes(s_sub, False); mega_t = _codes(t_sub, False)
        ari = float(adjusted_rand_score(sing_s, sing_t))
        ari_b = float(adjusted_rand_score(mega_s[both], mega_t[both]))
        h_, c_, v_ = homogeneity_completeness_v_measure(sing_s, sing_t)
        # purity (seq → tiled)
        sub = pd.DataFrame({"seq": s_sub[~seq_un].to_numpy(),
                             "til": t_sub[~seq_un].to_numpy()})
        if len(sub) == 0:
            purity = float("nan")
        else:
            by_seq = sub.groupby("seq")["til"]
            sizes = by_seq.size().to_numpy()
            mode_count = by_seq.apply(lambda x: x.value_counts().iloc[0]).to_numpy()
            purity = float(np.average(mode_count / sizes, weights=sizes))
        return {
            "n_tx": int(sel.sum()),
            "n_seq_assigned": int((~seq_un).sum()),
            "n_til_assigned": int((~t_un).sum()),
            "n_both": int(both.sum()),
            "ARI_sing": ari,
            "ARI_both": ari_b,
            "h": float(h_), "c": float(c_), "V": float(v_),
            "purity": purity,
        }

    seq_lab = seq["label"].astype(str)
    naive_lab = naive["label"].astype(str)
    dis_lab = dis["label"].astype(str)
    resc_lab = resc["label"].astype(str)

    print(f"3x3 tile grid layout (ix*3+iy):  cell coord range x={x_min:.0f}..{x_max:.0f}, y={y_min:.0f}..{y_max:.0f}")
    print()

    for name, t_lab in [("naive", naive_lab),
                          ("disambig", dis_lab),
                          ("rescue", resc_lab)]:
        print(f"\n=== {name} per-tile ===")
        print(f"  {'tile':>5s} {'(ix,iy)':>8s} {'n_tx':>8s} {'n_seq':>8s} {'n_til':>8s} "
              f"{'ARI_s':>7s} {'ARI_b':>7s} {'h':>7s} {'c':>7s} {'V':>7s} {'purity':>7s}")
        for tile_id in range(n_x * n_y):
            sel = (tx_tile == tile_id)
            r = _metric_subset(seq_lab, t_lab, sel)
            if not r:
                print(f"  {tile_id:>5d}    <empty/insufficient>")
                continue
            ixx, iyy = tile_id // n_y, tile_id % n_y
            print(f"  {tile_id:>5d} {f'({ixx},{iyy})':>8s} {r['n_tx']:>8,} "
                  f"{r['n_seq_assigned']:>8,} {r['n_til_assigned']:>8,} "
                  f"{r['ARI_sing']:>7.4f} {r['ARI_both']:>7.4f} "
                  f"{r['h']:>7.4f} {r['c']:>7.4f} {r['V']:>7.4f} "
                  f"{r['purity']:>7.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
