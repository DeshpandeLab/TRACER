#!/usr/bin/env python3
"""Cross-tile pair analysis: how much of the global disagreement is
intra-tile vs cross-tile?

We can't easily ARI a pair-subset, but we CAN decompose Rand-Index by
pair type. For each pair-class (same-tile vs different-tile), count:
  - agree (both partitions co-cluster, or both split)
  - disagree
This identifies whether the global metric drop comes from intra-tile
losses (tile 4 density) or cross-tile losses (boundary splits +
cascade-label collisions).

Also, for each sequential entity, report:
  - n_tiles_spanned
  - n_tiled_labels (in naive / disambig / rescue)
  - fraction with n_tiles_spanned > 1
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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


def main() -> int:
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

    # =====================================================================
    # Part 1: For each seq entity, how many tiles does it span?
    # And how many distinct tiled labels does it map to (split count)?
    # =====================================================================
    seq_lab = seq["label"].astype(str)
    seq_assigned = ~_is_un(seq_lab)
    df_seq = pd.DataFrame({
        "seq": seq_lab.to_numpy(),
        "tile": tx_tile,
        "naive": naive["label"].astype(str).to_numpy(),
        "dis": dis["label"].astype(str).to_numpy(),
        "resc": resc["label"].astype(str).to_numpy(),
        "assigned": seq_assigned,
    })
    df_a = df_seq[df_seq["assigned"]].reset_index(drop=True)

    print(f"Sequential assigned entities (excluding sentinels):")
    n_total_seq = df_a["seq"].nunique()
    print(f"  total seq entities: {n_total_seq:,}")
    print(f"  total assigned tx:  {len(df_a):,}")

    by_seq = df_a.groupby("seq", observed=True)
    spans = by_seq["tile"].nunique()
    print()
    print("Tile-span distribution of seq entities:")
    for k in (1, 2, 3, 4, 5):
        n = int((spans == k).sum())
        pct = 100 * n / n_total_seq
        print(f"  spans exactly {k} tile{'s' if k>1 else ''}: {n:>7,}  ({pct:5.2f}%)")
    n_multi = int((spans >= 2).sum())
    print(f"  spans >=2 tiles:           {n_multi:>7,}  ({100*n_multi/n_total_seq:.2f}%)")
    # Tx that LIVE in a multi-tile seq entity:
    multi_seqs = set(spans[spans >= 2].index)
    multi_tx_mask = df_a["seq"].isin(multi_seqs).to_numpy()
    n_multi_tx = int(multi_tx_mask.sum())
    print(f"  tx in multi-tile entities: {n_multi_tx:>7,}  "
          f"({100*n_multi_tx/len(df_a):.2f}% of assigned tx)")

    # For each tiled variant: how many distinct tiled labels does each
    # multi-tile seq entity map to?
    print()
    print("Among multi-tile seq entities, distribution of tiled-label-count "
          "(how many distinct tiled labels the seq entity is split into):")
    df_multi = df_a[multi_tx_mask].reset_index(drop=True)
    for var_name in ("naive", "dis", "resc"):
        splits = df_multi.groupby("seq", observed=True)[var_name].nunique()
        n_perfect = int((splits == 1).sum())  # all tx in same tiled label
        n_split = int((splits >= 2).sum())
        avg_split = float(splits.mean())
        max_split = int(splits.max())
        print(f"  {var_name:>5s}: avg n_til/seq = {avg_split:.2f}, "
              f"max = {max_split}, "
              f"perfect (1 label) = {n_perfect:,}/{n_multi:,} ({100*n_perfect/n_multi:.1f}%), "
              f"split (>=2) = {n_split:,} ({100*n_split/n_multi:.1f}%)")

    # =====================================================================
    # Part 2: pair-level decomposition (Rand-Index by pair-class)
    # For each pair (i, j):
    #   pair_class: same_tile or cross_tile
    #   agree: (seq_same iff tiled_same)
    #   disagree: otherwise
    # Computing this exactly is O(N^2). We approximate via subsample of
    # 50k tx (=> 1.25e9 pairs, still too much). Instead, we use the
    # contingency-style trick: for each (seq, til) pair-class, count tx,
    # then compute pairs from counts.
    # =====================================================================
    # For pair counts, use:
    #   pairs_same_seq = sum_s (n_s choose 2)
    #   pairs_same_til = sum_t (n_t choose 2)
    #   pairs_same_both = sum_{s,t} (n_{s,t} choose 2)
    #   pairs_same_tile = sum_k (n_k choose 2) where k = tile id
    # We want intra-tile pair-class:
    #   pairs_same_tile, pairs_same_tile & same_seq, etc.
    # Easiest: do the contingency over (seq, tile) and (til, tile) and
    # (seq, til, tile), then aggregate.
    def _ri_components(seq_arr, til_arr, tile_arr):
        """Return dict of pair counts decomposed by pair-class."""
        # Build (seq, til, tile) 3-way contingency
        df_ = pd.DataFrame({"s": seq_arr, "t": til_arr, "k": tile_arr})
        # Counts for each (s, t, k)
        cnt_stk = df_.groupby(["s", "t", "k"], observed=True).size()
        # And (s, k), (t, k), (k,)
        cnt_sk = df_.groupby(["s", "k"], observed=True).size()
        cnt_tk = df_.groupby(["t", "k"], observed=True).size()
        cnt_k = df_.groupby(["k"], observed=True).size()
        # Pair counts within each cell of contingency
        def pairs(x):
            x = x.astype(np.int64)
            return int((x * (x - 1) // 2).sum())
        # Same-tile pairs decomposed by (same_seq, same_til):
        #   same_tile_same_both = sum_{s,t,k} (n_{s,t,k} choose 2)
        same_tile_same_both = pairs(cnt_stk)
        #   same_tile_same_seq  = sum_{s,k}   (n_{s,k} choose 2)
        same_tile_same_seq  = pairs(cnt_sk)
        same_tile_same_til  = pairs(cnt_tk)
        same_tile_total     = pairs(cnt_k)
        # Same-tile same_seq only (not same_til): same_tile_same_seq - same_tile_same_both
        same_tile_same_seq_diff_til = same_tile_same_seq - same_tile_same_both
        same_tile_same_til_diff_seq = same_tile_same_til - same_tile_same_both
        same_tile_diff_both = same_tile_total - same_tile_same_seq - same_tile_same_til + same_tile_same_both
        # Now total counts (ignoring tile):
        cnt_st = df_.groupby(["s", "t"], observed=True).size()
        cnt_s = df_.groupby(["s"], observed=True).size()
        cnt_t = df_.groupby(["t"], observed=True).size()
        N = int(df_.shape[0])
        total_pairs = N * (N - 1) // 2
        total_same_both = pairs(cnt_st)
        total_same_seq = pairs(cnt_s)
        total_same_til = pairs(cnt_t)
        # Cross-tile pairs:
        cross_total = total_pairs - same_tile_total
        cross_same_seq = total_same_seq - same_tile_same_seq
        cross_same_til = total_same_til - same_tile_same_til
        cross_same_both = total_same_both - same_tile_same_both
        cross_same_seq_diff_til = cross_same_seq - cross_same_both
        cross_same_til_diff_seq = cross_same_til - cross_same_both
        cross_diff_both = cross_total - cross_same_seq - cross_same_til + cross_same_both
        return {
            "same_tile": {
                "total": same_tile_total,
                "agree_both_same": same_tile_same_both,
                "agree_both_diff": same_tile_diff_both,
                "disagree_seq_only": same_tile_same_seq_diff_til,
                "disagree_til_only": same_tile_same_til_diff_seq,
            },
            "cross_tile": {
                "total": cross_total,
                "agree_both_same": cross_same_both,
                "agree_both_diff": cross_diff_both,
                "disagree_seq_only": cross_same_seq_diff_til,
                "disagree_til_only": cross_same_til_diff_seq,
            },
            "all": {
                "total": total_pairs,
                "agree_both_same": total_same_both,
                "agree_both_diff": total_pairs - total_same_seq - total_same_til + total_same_both,
                "disagree_seq_only": total_same_seq - total_same_both,
                "disagree_til_only": total_same_til - total_same_both,
            }
        }

    # NOTE: For tractability, drop unassigned (sentinel) tx from this
    # decomposition — sentinel-singleton encoding would make every
    # sentinel-pair "same_seq_diff_til" or vice versa, which we already
    # know dominates intuitively. Restrict to assigned-in-both.
    naive_lab = naive["label"].astype(str).to_numpy()
    dis_lab = dis["label"].astype(str).to_numpy()
    resc_lab = resc["label"].astype(str).to_numpy()
    seq_arr = seq_lab.to_numpy()

    for var_name, til_arr in [("naive", naive_lab), ("dis", dis_lab), ("resc", resc_lab)]:
        t_un = _is_un(pd.Series(til_arr))
        keep = seq_assigned & (~t_un)
        comps = _ri_components(seq_arr[keep], til_arr[keep], tx_tile[keep])
        print()
        print(f"=== {var_name} pair-decomposition (assigned-in-both, "
              f"{int(keep.sum()):,} tx) ===")
        for cls in ("same_tile", "cross_tile", "all"):
            d = comps[cls]
            tot = d["total"]
            if tot == 0:
                continue
            agree = d["agree_both_same"] + d["agree_both_diff"]
            disagree = d["disagree_seq_only"] + d["disagree_til_only"]
            print(f"  {cls:>10s}: total={tot:>14,d}  agree={agree:>14,d} "
                  f"({100*agree/tot:5.2f}%)  disagree={disagree:>14,d} "
                  f"({100*disagree/tot:5.2f}%)")
            print(f"             agree_both_same = {d['agree_both_same']:>14,d}  "
                  f"agree_both_diff = {d['agree_both_diff']:>14,d}")
            print(f"             disagree (split-by-tile, same in seq): "
                  f"{d['disagree_seq_only']:>14,d}  "
                  f"(merge-across-til, diff in seq): {d['disagree_til_only']:>14,d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
