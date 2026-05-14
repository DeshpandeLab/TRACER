#!/usr/bin/env python3
"""Adjust the merger footprint metric to account for fragmentation
and internal holes.

For each merger entity:
    raw_tiles   = occupied 1µm tiles
    bridge_tiles = minimum tiles to connect all fragments via straight-line
                   bridges between closest connected-component pairs
    hole_tiles  = tiles inside the connected footprint that have no tx
                  (interior holes — biologically implausible for one entity)
    filled_tiles = raw + bridges + holes
    effective_density = n_tx / filled_tiles

A biologically-plausible single-cell entity should have:
    filled_tiles ≈ raw_tiles (few fragments, few holes)
    density at natural tissue level (~1.3 tx/tile)

An over-merger spanning multiple compartments will have many bridges
and/or holes, inflating filled_tiles substantially.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tracer.stitching import apply_stitching_to_transcripts_memory_efficient

ZOOM = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg" / "zoom_worst_tx.parquet"
PDAC = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
SENT = {"-1", "DROP", "UNASSIGNED", "nan"}
G = 1.0


def connected_components(bins: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """4-connectivity connected components of a tile set."""
    seen = set()
    comps = []
    for start in bins:
        if start in seen:
            continue
        stack = [start]
        comp = set()
        while stack:
            b = stack.pop()
            if b in seen:
                continue
            seen.add(b)
            comp.add(b)
            x, y = b
            for nb in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                if nb in bins and nb not in seen:
                    stack.append(nb)
        comps.append(comp)
    return comps


def line_tiles(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Bresenham-ish straight line of tile coords from a to b inclusive."""
    x0, y0 = a; x1, y1 = b
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1; sy = 1 if y0 < y1 else -1
    err = dx - dy
    out = []
    x, y = x0, y0
    while True:
        out.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x += sx
        if e2 < dx:
            err += dx; y += sy
    return out


def closest_pair(c1: set, c2: set) -> tuple[tuple[int, int], tuple[int, int], int]:
    """Min L1 distance pair between two tile sets. Returns (p1, p2, dist)."""
    best = None
    for a in c1:
        for b in c2:
            d = abs(a[0]-b[0]) + abs(a[1]-b[1])
            if best is None or d < best[2]:
                best = (a, b, d)
    return best


def bridge_components(comps: list[set]) -> set[tuple[int, int]]:
    """Greedily connect components via straight-line bridges. Returns the
    set of tiles added (not including the original component tiles)."""
    if len(comps) <= 1:
        return set()
    bridges: set[tuple[int, int]] = set()
    all_tiles = set().union(*comps)
    # Greedy: repeatedly merge the closest two components
    comps = [set(c) for c in comps]
    while len(comps) > 1:
        # find closest pair across all current components
        best_i, best_j, best_pair, best_dist = None, None, None, None
        for i in range(len(comps)):
            for j in range(i+1, len(comps)):
                p1, p2, d = closest_pair(comps[i], comps[j])
                if best_dist is None or d < best_dist:
                    best_i, best_j, best_pair, best_dist = i, j, (p1, p2), d
        # bridge them with a straight line
        line = line_tiles(*best_pair)
        new_bridge = {t for t in line if t not in all_tiles}
        bridges |= new_bridge
        all_tiles |= new_bridge
        # merge the two components
        merged = comps[best_i] | comps[best_j] | new_bridge
        comps = [c for k, c in enumerate(comps) if k not in (best_i, best_j)] + [merged]
    return bridges


def fill_holes(bins: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Return any 'hole' tiles inside the bbox of `bins` that aren't reachable
    from the bbox boundary via 4-connectivity through non-bin tiles."""
    if not bins:
        return set()
    xs = [b[0] for b in bins]; ys = [b[1] for b in bins]
    xmin, xmax = min(xs)-1, max(xs)+1
    ymin, ymax = min(ys)-1, max(ys)+1
    # Flood fill from outside (start at bbox corners), 4-conn through ~bins
    outside = set()
    stack = [(xmin, ymin)]
    bounds = lambda p: xmin <= p[0] <= xmax and ymin <= p[1] <= ymax
    while stack:
        p = stack.pop()
        if p in outside or p in bins or not bounds(p):
            continue
        outside.add(p)
        for d in [(1,0),(-1,0),(0,1),(0,-1)]:
            nb = (p[0]+d[0], p[1]+d[1])
            if bounds(nb) and nb not in outside and nb not in bins:
                stack.append(nb)
    # holes = bbox interior tiles not in bins and not outside
    holes = set()
    for x in range(xmin, xmax+1):
        for y in range(ymin, ymax+1):
            if (x, y) not in bins and (x, y) not in outside:
                holes.add((x, y))
    return holes


def main() -> int:
    zoom = pd.read_parquet(ZOOM)
    feats = pd.read_parquet(PDAC, columns=["transcript_id", "feature_name"])
    zcol = pd.read_parquet(PDAC, columns=["transcript_id", "z_location"]).rename(
        columns={"z_location": "z"})
    df = zoom.merge(feats, on="transcript_id", how="left").merge(
        zcol, on="transcript_id", how="left")
    df["feature_name"] = df["feature_name"].astype(str)
    df["seg_lab"] = df["seg_lab"].astype(str)
    df["cell_id"] = df["cell_id"].astype(str)
    df = df.reset_index(drop=True)
    all_cell_ids = set(df.loc[~df["cell_id"].isin(SENT), "cell_id"].unique())
    def lab_to_etype(lab):
        if lab in SENT: return "drop"
        if lab.startswith("cascade_"): return "component"
        if lab in all_cell_ids: return "cell"
        return "partial"
    df["_etype"] = df["seg_lab"].map(lab_to_etype)
    df["tracer_id"] = df["seg_lab"]
    df.loc[df["tracer_id"].isin(SENT), "tracer_id"] = "-1"
    df["xb"] = np.floor(df["x"].to_numpy()/G).astype(int)
    df["yb"] = np.floor(df["y"].to_numpy()/G).astype(int)

    panel_raw = pd.read_parquet(PANEL)
    all_genes = sorted(set(panel_raw["gene_i"].astype(str))
                       | set(panel_raw["gene_j"].astype(str))
                       | set(df["feature_name"].unique()))
    g2i = {g: i for i, g in enumerate(all_genes)}
    Gn = len(all_genes)
    W = np.full((Gn, Gn), np.nan, dtype=np.float32)
    gi = panel_raw["gene_i"].astype(str).map(g2i)
    gj = panel_raw["gene_j"].astype(str).map(g2i)
    have = gi.notna() & gj.notna()
    gi = gi[have].to_numpy(np.int64); gj = gj[have].to_numpy(np.int64)
    v = panel_raw.loc[have, "value"].to_numpy(np.float32)
    W[gi, gj] = v; W[gj, gi] = v
    np.fill_diagonal(W, np.nan)

    df_s, _ = apply_stitching_to_transcripts_memory_efficient(
        df_final=df, aux={"W": W, "gene_to_idx": g2i},
        entity_col="tracer_id", gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        mode="count", threshold=0.2, metric="pmi",
        penalize_simplicity=True, deltaC_min=0.03, c_union_bypass=0.9,
        dist_threshold=5.0, out_col="stitched", show_progress=False,
        candidate_source="grid", G=2.0, stitch_neighborhood="8",
        G_z=1.0, z_neighbor_depth=1, min_local_tx_per_entity=3,
    )
    df_s["stitched"] = df_s["stitched"].astype(str)
    df_s["xb"] = df["xb"]; df_s["yb"] = df["yb"]

    # Per-stitched-entity: raw_tiles, n_components, bridges, holes, filled
    rows = []
    for sl, grp in df_s[~df_s["stitched"].isin(SENT)].groupby("stitched"):
        bins = set(zip(grp["xb"].tolist(), grp["yb"].tolist()))
        comps = connected_components(bins)
        bridges = bridge_components(comps)
        holes = fill_holes(bins | bridges)
        rows.append({
            "stitched": sl,
            "n_tx": len(grp),
            "raw_tiles": len(bins),
            "n_components": len(comps),
            "bridge_tiles": len(bridges),
            "hole_tiles": len(holes),
            "filled_tiles": len(bins) + len(bridges) + len(holes),
        })
    out = pd.DataFrame(rows)
    out["density_raw"] = out["n_tx"] / out["raw_tiles"]
    out["density_filled"] = out["n_tx"] / out["filled_tiles"]

    # Same for SEG cells (reference)
    seg_rows = []
    for sl, grp in df[df["_etype"]=="cell"].groupby("seg_lab"):
        bins = set(zip(grp["xb"].tolist(), grp["yb"].tolist()))
        comps = connected_components(bins)
        bridges = bridge_components(comps)
        holes = fill_holes(bins | bridges)
        seg_rows.append({
            "n_tx": len(grp), "raw_tiles": len(bins),
            "n_components": len(comps),
            "bridge_tiles": len(bridges), "hole_tiles": len(holes),
            "filled_tiles": len(bins)+len(bridges)+len(holes),
        })
    seg_df = pd.DataFrame(seg_rows)
    seg_df["density_raw"] = seg_df["n_tx"] / seg_df["raw_tiles"]
    seg_df["density_filled"] = seg_df["n_tx"] / seg_df["filled_tiles"]

    print(f"SEG cells (n={len(seg_df)}) reference distribution:")
    print(seg_df[["raw_tiles","n_components","bridge_tiles","hole_tiles",
                   "filled_tiles","density_filled"]]
          .describe(percentiles=[0.5,0.75,0.9,0.95,0.99,1.0]).to_string())
    print()

    # Show mergers (re-stitched entities with ≥2 pre-entities) sorted by n_tx
    mergers_pre = (df_s[~df_s["stitched"].isin(SENT)]
                    .groupby("stitched")
                    .agg(n_pre=("tracer_id", lambda s: len(set(s)-SENT)))
                    .reset_index())
    out_m = out.merge(mergers_pre, on="stitched")
    out_m = out_m[out_m["n_pre"] >= 2].sort_values("n_tx", ascending=False)
    print(f"Mergers (with ≥2 pre-stitch entities), sorted by n_tx:")
    print(f"  {'stitched':>22s}  {'n_tx':>4s}  {'raw':>4s}  "
          f"{'comps':>5s}  {'bridge':>6s}  {'hole':>4s}  "
          f"{'filled':>6s}  {'d_raw':>6s}  {'d_filled':>8s}")
    for _, r in out_m.iterrows():
        flag = ""
        if r["filled_tiles"] > seg_df["filled_tiles"].max():
            flag = "  ← OVER max SEG filled"
        elif r["filled_tiles"] > seg_df["filled_tiles"].quantile(0.95):
            flag = "  ← >95% SEG filled"
        print(f"  {r['stitched']:>22s}  {int(r['n_tx']):>4d}  "
              f"{int(r['raw_tiles']):>4d}  {int(r['n_components']):>5d}  "
              f"{int(r['bridge_tiles']):>6d}  {int(r['hole_tiles']):>4d}  "
              f"{int(r['filled_tiles']):>6d}  "
              f"{r['density_raw']:>6.2f}  {r['density_filled']:>8.2f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
