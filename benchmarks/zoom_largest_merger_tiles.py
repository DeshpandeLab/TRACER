#!/usr/bin/env python3
"""Tile footprint of the largest re-stitched merger entity, colored by
the pre-stitch entity each tile came from.

Identifies the biggest merger after running Stitch with bypass=0.9 on
SEG's seg_lab (with proper _etype reconstructed). For that merger:
    - Plot its tile footprint at G=1 µm
    - Color each tile by the pre-stitch entity (seg_lab) of its tx
    - If a tile contains tx from multiple pre-stitch entities, color
      it as a translucent overlay of all contributors
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib import patheffects as _patheffects

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tracer.stitching import apply_stitching_to_transcripts_memory_efficient

PANEL = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/"
    "bootstrap-only-flavor/benchmarks/bootstrap_thr_pdac/W_thr0.parquet"
)
PDAC = Path(
    "/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/tutorials/pdac_io/data/outs/"
    "transcripts.parquet"
)
ZOOM = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg" / "zoom_worst_tx.parquet"
ZOOM_DIR = REPO / "benchmarks" / "stitch_zoom_seg_vs_noseg"
SENT = {"-1", "DROP", "UNASSIGNED", "nan"}
G_TILE = 1.0


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
    # Reconstruct _etype per entity (matches what production pipeline emits)
    all_cell_ids = set(df.loc[~df["cell_id"].isin(SENT), "cell_id"].unique())
    def lab_to_etype(lab):
        if lab in SENT: return "drop"
        if lab.startswith("cascade_"): return "component"
        if lab in all_cell_ids: return "cell"
        return "partial"
    df["_etype"] = df["seg_lab"].map(lab_to_etype)
    df["tracer_id"] = df["seg_lab"]
    df.loc[df["tracer_id"].isin(SENT), "tracer_id"] = "-1"

    panel_raw = pd.read_parquet(PANEL)
    all_genes = sorted(set(panel_raw["gene_i"].astype(str))
                       | set(panel_raw["gene_j"].astype(str))
                       | set(df["feature_name"].unique()))
    g2i = {g: i for i, g in enumerate(all_genes)}
    G = len(all_genes)
    W = np.full((G, G), np.nan, dtype=np.float32)
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

    # Find biggest merger (≥ 2 pre-stitch entities)
    mergers = (df_s[~df_s["stitched"].isin(SENT)]
                .groupby("stitched")
                .agg(pre_ents=("tracer_id", lambda s: sorted(set(s) - SENT)),
                     n_tx=("stitched", "size"))
                .reset_index())
    mergers["n_pre"] = mergers["pre_ents"].apply(len)
    real = mergers[mergers["n_pre"] >= 2].sort_values("n_tx", ascending=False)
    target = real.iloc[0]
    print(f"Biggest merger: {target['stitched']}  n_tx={int(target['n_tx'])}  "
          f"pre={target['pre_ents']}")

    # Subset to this merger's tx
    sub = df_s[df_s["stitched"] == target["stitched"]].copy()
    sub["xb"] = np.floor(sub["x"].to_numpy() / G_TILE).astype(int)
    sub["yb"] = np.floor(sub["y"].to_numpy() / G_TILE).astype(int)

    pre_ents = target["pre_ents"]
    cmap = plt.get_cmap("tab10", max(len(pre_ents), 3))
    color_for = {p: cmap(i % cmap.N) for i, p in enumerate(pre_ents)}

    # Per (xb,yb) tile, list the pre-stitch entities contributing tx and their counts.
    tile_grp = (sub.groupby(["xb", "yb", "tracer_id"]).size()
                .reset_index(name="n_tx"))
    # For each tile, sum across pre-entities so we know total tx; build a contributor list
    tile_unique = tile_grp.groupby(["xb", "yb"]).agg(
        contributors=("tracer_id", list),
        counts=("n_tx", list),
        total=("n_tx", "sum"),
    ).reset_index()

    pad = 1.0
    xmin = sub["x"].min() - pad; xmax = sub["x"].max() + pad
    ymin = sub["y"].min() - pad; ymax = sub["y"].max() + pad

    fig, ax = plt.subplots(figsize=(13, 13), dpi=140)
    # Faint grid
    x0 = np.floor(xmin / G_TILE) * G_TILE; x1 = np.ceil(xmax / G_TILE) * G_TILE
    y0 = np.floor(ymin / G_TILE) * G_TILE; y1 = np.ceil(ymax / G_TILE) * G_TILE
    for x in np.arange(x0, x1 + G_TILE * 0.5, G_TILE):
        ax.axvline(x, color="#eeeeee", linewidth=0.3, zorder=0)
    for y in np.arange(y0, y1 + G_TILE * 0.5, G_TILE):
        ax.axhline(y, color="#eeeeee", linewidth=0.3, zorder=0)

    # Each tile: if single-contributor, solid fill in that contributor's color.
    # If multi-contributor, stack translucent layers per contributor.
    for _, t in tile_unique.iterrows():
        xb, yb = int(t["xb"]), int(t["yb"])
        contributors = list(t["contributors"])
        counts = list(t["counts"])
        # Layer per contributor (translucent so overlapping shows)
        for cname, cnt in zip(contributors, counts):
            c = color_for[cname]
            ax.add_patch(Rectangle(
                (xb * G_TILE, yb * G_TILE), G_TILE, G_TILE,
                facecolor=c, alpha=0.45 if len(contributors)>1 else 0.7,
                edgecolor="none", zorder=1,
            ))
        # Border around the tile if it has any tx
        ax.add_patch(Rectangle(
            (xb * G_TILE, yb * G_TILE), G_TILE, G_TILE,
            facecolor="none", edgecolor="#444444", linewidth=0.4,
            zorder=2,
        ))

    # tx dots colored by contributor entity, on top
    for pe in pre_ents:
        ent = sub[sub["tracer_id"] == pe]
        c = color_for[pe]
        ax.scatter(ent["x"], ent["y"], s=22, c=[c], alpha=0.95,
                    edgecolor="black", linewidths=0.4, zorder=4,
                    label=f"{pe} (n={len(ent)})")

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    n_tiles = len(tile_unique)
    n_multi = int((tile_unique["contributors"].apply(len) >= 2).sum())
    ax.set_title(
        f"Largest re-stitched merger: {target['stitched']}\n"
        f"n_tx={int(target['n_tx'])}, n_tiles={n_tiles}, "
        f"multi-contributor tiles={n_multi}\n"
        f"Each tile colored by pre-stitch entity (translucent stack if shared)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    plt.tight_layout()
    out = ZOOM_DIR / "zoom_largest_merger_tiles.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
