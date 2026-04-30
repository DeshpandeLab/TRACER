"""Synthetic data generators shared across tests (and reusable from
``benchmarks/``).

Three generators:

- :func:`make_synthetic_npmi_panel` — plants 5 known gene-pair structures
  (positive, negative, independent, dropout-high, dropout-low) for testing
  the bootstrap NPMI panel builder. Returns a long-format
  ``(cell_id, feature_name)`` DataFrame.

- :func:`make_synthetic_transcripts` — plants ``n_cells`` cells arranged
  on a coarse xy grid, each cell drawn from one of ``n_types`` archetype
  gene-panel templates with strong within-type co-occurrence.
  Returns a transcript-level DataFrame matching the schema TRACER pipeline
  stages expect (``transcript_id``, ``feature_name``, ``cell_id``, ``x``,
  ``y``, ``z``) plus a ``ground_truth`` dict.

- :func:`make_synthetic_npmi_panel_for_transcripts` — builds a
  long-format NPMI / PMI panel consistent with the planted gene-coherence
  structure of :func:`make_synthetic_transcripts`. Suitable as the
  ``npmi_df`` argument to ``prune_transcripts_fast`` and downstream
  pipeline stages.

All three are deterministic given a seed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# 1. NPMI bootstrap correctness fixture
# -----------------------------------------------------------------------

def make_synthetic_npmi_panel(N: int = 1000, G: int = 20, seed: int = 42
                              ) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a synthetic transcript table with planted gene-pair structures.

    Returns
    -------
    df : pd.DataFrame
        Long-format with columns ``cell_id``, ``feature_name``. Each cell is
        repeated ``min_occurrences_per_context=2`` so the bootstrap has
        enough observations.
    M : np.ndarray of shape ``(N, G)``
        The cell × gene presence matrix (0/1) used to construct ``df``.

    Planted structures (gene indices):
      - 0, 1: strong positive co-occurrence (~40% jointly present).
      - 2, 3: strong mutual exclusivity (~80% have exactly one).
      - 4, 5: independent at rate 0.3 each.
      - 6, 7: rare and independent (~0.02 each), zero observed cooccur,
        E[cooccur] ≈ 0.4 (< 10) → indeterminate under the bootstrap rule.
      - 8, 9: high marginal (~0.4 each), zero observed cooccur,
        E[cooccur] ≈ 160 (≫ 10) → ``neg_one`` sentinel.
      - 10..G-1: noise at 10%.
    """
    rng = np.random.default_rng(seed)

    M = np.zeros((N, G), dtype=np.int8)

    # Genes 0,1: strong positive — both in 40% of cells.
    coexp = rng.random(N) < 0.4
    M[coexp, 0] = 1
    M[coexp, 1] = 1
    M[rng.random(N) < 0.02, 0] = 1
    M[rng.random(N) < 0.02, 1] = 1

    # Genes 2,3: mutual exclusivity, exactly one of them in 80% of cells.
    which = rng.random(N) < 0.4
    in_pair = rng.random(N) < 0.8
    M[in_pair & which, 2] = 1
    M[in_pair & ~which, 3] = 1

    # Genes 4,5: independent at rate 0.3.
    M[rng.random(N) < 0.3, 4] = 1
    M[rng.random(N) < 0.3, 5] = 1

    # Genes 6,7: rare and independent (~0.02 each).
    M[rng.random(N) < 0.02, 6] = 1
    M[rng.random(N) < 0.02, 7] = 1
    both = (M[:, 6] == 1) & (M[:, 7] == 1)
    if both.any():
        idxs = np.flatnonzero(both)
        half = len(idxs) // 2
        M[idxs[:half], 7] = 0
        M[idxs[half:], 6] = 0

    # Genes 8,9: high marginal (~0.4 each), zero cooccur.
    p89 = rng.random(N)
    g8 = p89 < 0.4
    g9 = (p89 >= 0.4) & (p89 < 0.8)
    M[g8, 8] = 1
    M[g9, 9] = 1

    # Other genes: noise at 10%.
    for g in range(10, G):
        M[rng.random(N) < 0.1, g] = 1

    rows = []
    for cid in range(N):
        for g in range(G):
            if M[cid, g]:
                rows.append((str(cid), f"gene_{g:02d}"))
                rows.append((str(cid), f"gene_{g:02d}"))
    df = pd.DataFrame(rows, columns=["cell_id", "feature_name"])
    return df, M


# -----------------------------------------------------------------------
# 2. Synthetic transcripts (for end-to-end pipeline tests)
# -----------------------------------------------------------------------

def make_synthetic_transcripts(
    n_cells: int = 8,
    tx_per_cell: int = 25,
    n_genes: int = 12,
    n_types: int = 3,
    cell_spacing_um: float = 50.0,
    cell_radius_um: float = 3.0,
    z_range_um: tuple[float, float] = (0.0, 5.0),
    cross_type_noise_pct: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a transcript-level DataFrame with planted cell structure.

    Layout: cells on a ``ceil(sqrt(n_cells)) × ceil(sqrt(n_cells))`` xy
    grid at ``cell_spacing_um`` µm pitch. Each cell is assigned one of
    ``n_types`` archetype types (cycling); each type has its own
    archetype gene panel of ``n_genes / n_types`` genes. ``tx_per_cell``
    transcripts per cell are sampled — ``(1 - cross_type_noise_pct)``
    from the archetype panel, rest from any other gene.

    Returns
    -------
    df : pd.DataFrame
        Columns ``transcript_id`` (str), ``feature_name`` (str),
        ``cell_id`` (str — ground truth), ``x``, ``y``, ``z`` (float32).
        ``feature_name`` is named ``"gene_00"``, ``"gene_01"``, etc.

    ground_truth : dict
        ``n_cells`` (int)
        ``n_types`` (int)
        ``cell_centers`` (list[(x, y)])
        ``cell_to_type`` (dict[str_cell_id -> int])
        ``type_to_genes`` (dict[int -> list[str_gene_name]])
        ``gene_to_type`` (dict[str_gene_name -> int])
    """
    rng = np.random.default_rng(seed)

    if n_genes % n_types != 0:
        raise ValueError(
            f"n_genes={n_genes} must be divisible by n_types={n_types} "
            "for clean archetype panels."
        )
    genes_per_type = n_genes // n_types
    gene_names = [f"gene_{i:02d}" for i in range(n_genes)]
    type_to_genes: dict[int, list[str]] = {
        t: gene_names[t * genes_per_type: (t + 1) * genes_per_type]
        for t in range(n_types)
    }
    gene_to_type = {g: t for t, gs in type_to_genes.items() for g in gs}

    # Layout: roughly square grid
    grid_n = int(np.ceil(np.sqrt(n_cells)))
    cell_centers: list[tuple[float, float]] = []
    cell_to_type: dict[str, int] = {}
    rows = []
    tx_id = 0
    for c in range(n_cells):
        gx, gy = c % grid_n, c // grid_n
        cx = (gx + 0.5) * cell_spacing_um
        cy = (gy + 0.5) * cell_spacing_um
        cell_centers.append((cx, cy))
        ctype = c % n_types
        cell_to_type[str(c)] = ctype

        archetype_genes = type_to_genes[ctype]
        other_genes = [g for g in gene_names if g not in archetype_genes]

        for _ in range(tx_per_cell):
            # Gene draw
            if rng.random() < cross_type_noise_pct:
                gene = other_genes[int(rng.integers(len(other_genes)))]
            else:
                gene = archetype_genes[int(rng.integers(len(archetype_genes)))]
            # Coord draw within cell radius
            r = rng.uniform(0, cell_radius_um)
            theta = rng.uniform(0, 2 * np.pi)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            z = rng.uniform(*z_range_um)
            rows.append((str(tx_id), gene, str(c), float(x), float(y), float(z)))
            tx_id += 1

    df = pd.DataFrame(
        rows,
        columns=["transcript_id", "feature_name", "cell_id", "x", "y", "z"],
    )
    df = df.astype({"x": np.float32, "y": np.float32, "z": np.float32})

    ground_truth: dict[str, Any] = {
        "n_cells": n_cells,
        "n_types": n_types,
        "cell_centers": cell_centers,
        "cell_to_type": cell_to_type,
        "type_to_genes": type_to_genes,
        "gene_to_type": gene_to_type,
    }
    return df, ground_truth


def make_synthetic_npmi_panel_for_transcripts(
    df: pd.DataFrame,
    ground_truth: dict[str, Any],
    *,
    same_type_pmi: float = 1.0,
    cross_type_pmi: float = -1.0,
    metric: str = "pmi",
) -> pd.DataFrame:
    """Build a long-format NPMI / PMI panel matching the planted structure.

    For every (g_i, g_j) pair in the transcript dataframe's gene
    vocabulary, emit one row:

      - within the same archetype panel → ``same_type_pmi``
      - across panels → ``cross_type_pmi``

    The column name ``NPMI`` is used (matching the legacy CSV convention
    that ``prune_transcripts_fast`` and other downstream stages expect).
    The values are interpreted in whatever metric the pipeline is configured
    for (``metric`` is for documentation only — the column name is fixed).

    Defaults (``+1.0`` / ``-1.0``) are calibrated for ``metric="pmi"``:
    ``+1.0 > ln(1.5) ≈ 0.405`` and ``-1.0 < ln(1/3) ≈ -1.099`` — clearly on
    the positive / negative side of the modern thresholds.
    """
    _ = metric  # for documentation; column name is fixed.
    gene_to_type = ground_truth["gene_to_type"]
    genes = sorted(set(df["feature_name"].astype(str)))

    rows = []
    for i, gi in enumerate(genes):
        for gj in genes[i + 1:]:
            ti = gene_to_type.get(gi)
            tj = gene_to_type.get(gj)
            if ti is not None and tj is not None and ti == tj:
                v = same_type_pmi
            else:
                v = cross_type_pmi
            rows.append((gi, gj, float(v)))
    return pd.DataFrame(rows, columns=["gene_i", "gene_j", "NPMI"])
