"""Simulate Xenium-style DAPI + Voronoi cell segmentation on synthetic
transcripts.

Used by ``benchmarks/pr_benchmark.py`` to feed TRACER a realistic
noisy-segmentation input rather than ground-truth ``cell_id``. The
recovery ARI then measures TRACER's value-add over the upstream
segmenter, not just "does the pipeline pass good input through cleanly."

Algorithm
---------

1. **DAPI ellipse fit** (xy plane, z-blind):
   per ground-truth cell, gather the nuclear transcripts that survived
   sectioning. Cells with at least ``dapi_min_tx`` nuclear tx register
   a DAPI signal — their xy centroid becomes a Voronoi seed. Cells
   with fewer nuclear tx (typically clipped or those whose nucleus
   sits outside the slab) produce no DAPI and are excluded.

2. **Voronoi assignment** (xy plane, z-blind):
   each transcript is reassigned to the cell-id of the nearest
   DAPI-positive centroid (xy distance only — mirrors how a
   2D segmenter handles a 3D tissue section). Tx that originally
   belonged to a cell without DAPI are absorbed into the nearest
   DAPI-positive neighbor — a realistic mis-assignment error mode.

The function preserves the ground-truth cell_id in a new column
``cell_id_truth`` so downstream code can compute ARI vs the original
partition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAPI_MIN_NUCLEAR_TX = 3
# Real Xenium's fallback expands a cell up to ~5 µm beyond its
# nucleus when no membrane stain is available; tx farther than that
# are left unassigned. We mirror that cap here.
DEFAULT_MAX_DIST_FROM_NUCLEUS_UM = 5.0


def simulate_voronoi_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """Voronoi-by-cell-centroid segmentation with **no DAPI threshold**.

    Every ground-truth cell that has at least one transcript surviving
    sectioning becomes a Voronoi seed (centroid = xy-mean of that cell's
    surviving tx). All transcripts are then reassigned to the cell of
    the nearest seed in xy. This isolates the **z-projection / xy-only
    assignment error** from the DAPI-loss error: a tx in cell A at low
    z can be reassigned to cell B if cell B's xy-centroid is closer,
    even though their z values differ.

    Use as an intermediate scenario between
    ``section + ground-truth`` (no segmentation noise at all) and
    ``simulate_dapi_voronoi_segmentation`` (also drops cells below the
    DAPI threshold).
    """
    df = df.copy()
    df["cell_id_truth"] = df["cell_id"].astype(str)
    cells = sorted({c for c in df["cell_id_truth"] if c != "-1"})
    if not cells:
        df["cell_id"] = "-1"
        return df

    centers = np.array([
        df.loc[df["cell_id_truth"] == c, ["x", "y"]].mean(axis=0).to_numpy()
        for c in cells
    ], dtype=np.float64)
    pts = df[["x", "y"]].to_numpy(dtype=np.float64)
    d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    nearest = d2.argmin(axis=1)
    df["cell_id"] = pd.Series(
        [cells[i] for i in nearest], index=df.index, dtype=str,
    )
    return df


def simulate_dapi_voronoi_segmentation(
    df: pd.DataFrame,
    *,
    dapi_min_tx: int = DAPI_MIN_NUCLEAR_TX,
    max_dist_from_nucleus_um: float | None = DEFAULT_MAX_DIST_FROM_NUCLEUS_UM,
) -> pd.DataFrame:
    """Apply a simplified Xenium-style DAPI + Voronoi segmentation.

    Parameters
    ----------
    df : pd.DataFrame
        Synthetic transcript df with at least the columns
        ``cell_id, x, y, is_nuclear``.
    dapi_min_tx : int, default 3
        Minimum nuclear-tx count for a cell to register a DAPI signal.
        Cells below this threshold are dropped from the segmentation.
    max_dist_from_nucleus_um : float or None, default 5.0
        Real Xenium's fallback rule: cells extend up to this distance
        beyond their nucleus when no membrane stain is available; tx
        farther than that are left unassigned. Pass ``None`` to disable
        the cap (pure Voronoi tessellation, every tx assigned to nearest
        DAPI cell regardless of distance).

    Returns
    -------
    df_segmented : pd.DataFrame
        Copy of ``df`` where ``cell_id`` is overwritten by the
        simulated segmentation. The original ground-truth cell_id is
        preserved as ``cell_id_truth``. Tx whose nearest DAPI centroid
        is further than ``max_dist_from_nucleus_um`` get
        ``cell_id == "-1"``. If no cell has enough nuclear tx to
        register DAPI, every transcript becomes unassigned.
    """
    df = df.copy()
    df["cell_id_truth"] = df["cell_id"].astype(str)

    # 1. DAPI: count nuclear tx per ground-truth cell; keep those above threshold
    nuc = df[df["is_nuclear"]]
    nuc_counts = nuc["cell_id_truth"].value_counts()
    dapi_cells: list[str] = nuc_counts[nuc_counts >= dapi_min_tx].index.tolist()
    if not dapi_cells:
        df["cell_id"] = "-1"
        return df

    # Centroid (xy only) of nuclear tx per DAPI-positive cell
    centers = np.array([
        nuc.loc[nuc["cell_id_truth"] == c, ["x", "y"]].mean(axis=0).to_numpy()
        for c in dapi_cells
    ], dtype=np.float64)

    # 2. Voronoi assignment: each tx → nearest DAPI centroid in xy
    pts = df[["x", "y"]].to_numpy(dtype=np.float64)
    # Pairwise xy distances; n_tx and n_centers are both small (~200, ~8)
    d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    nearest = d2.argmin(axis=1)
    assigned = np.array([dapi_cells[i] for i in nearest], dtype=object)

    # 3. Apply Xenium-style "5 µm beyond nucleus" cap: tx farther than
    # ``max_dist_from_nucleus_um`` from their nearest DAPI centroid are
    # left unassigned (extracellular / orphaned).
    if max_dist_from_nucleus_um is not None:
        nearest_d = np.sqrt(d2[np.arange(len(pts)), nearest])
        too_far = nearest_d > max_dist_from_nucleus_um
        assigned[too_far] = "-1"

    df["cell_id"] = pd.Series(assigned, index=df.index, dtype=str)
    return df
