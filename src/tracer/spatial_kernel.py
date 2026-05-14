"""Shared candidate-enumeration kernel for Rescue and Stitch.

Both stages enumerate spatial neighborhoods over a tile-binned tx
population with entity labels, but with different access patterns:

  - **Stitch** (bulk): for every (entity_A, entity_B) pair whose tx share
    a neighborhood, return the witness count from each side.
  - **Rescue** (per-tx): for one query tx, return the entities with tx
    in adjacent bins (so per-entity nearest-tx distance can be computed).

This module factors out the common machinery so both stages share:

  - the xy-neighborhood definition (``parse_xy_offsets`` supports the
    same ``"0"`` / ``"4"`` / ``"8"`` / ``"R<N>"`` strings as Stitch);
  - the bin encoding (``tracer.graph.bin_xy``-packed int64 keys);
  - the witness-count semantics (per-side unique tx, optionally capped
    at the entity's own ``n_tx`` so tiny entities aren't unfairly
    blocked by a fixed threshold).

API
---
- :class:`GridIndex` — frozen dataclass wrapping the inverted indices.
- :func:`build_grid_index` — factory from coords + entity codes.
- :func:`enumerate_pair_witnesses` — bulk per-pair emission (Stitch's
  candidate-enumeration phase).
- :func:`neighbor_entities` — per-bin entity lookup (Rescue's per-tx
  candidate lookup).
- :func:`parse_xy_offsets` — translate neighborhood spec → (dx, dy)
  tuple.

Behavior contract
-----------------
- ``enumerate_pair_witnesses`` is byte-equivalent to the inline
  enumeration block in :func:`tracer.stitching.stitch_entities_hierarchical`
  (lines ~1280-1455 of stitching.py prior to the refactor).
- ``neighbor_entities`` mirrors the per-bin lookup in
  :func:`tracer.spatial.reassign_unassigned_grid_pool`.

Both behaviors are verified by ``tests/test_spatial_kernel.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .graph import _BIN_BIAS, bin_xy

__all__ = [
    "GridIndex",
    "build_grid_index",
    "enumerate_pair_witnesses",
    "neighbor_entities",
    "parse_xy_offsets",
]


# ---------------------------------------------------------------------
# Neighborhood spec parsing
# ---------------------------------------------------------------------
def parse_xy_offsets(neighborhood: str) -> tuple[tuple[int, int], ...]:
    """Translate a neighborhood spec into a full list of (dx, dy) offsets.

    Returns offsets EXCLUDING (0, 0). The caller adds the same-bin self
    case separately when needed.

    Supported:
      ``"0"``   — no offsets (same-bin only)
      ``"4"``   — 4-Moore (orthogonal ±1 only)
      ``"8"``   — 8-Moore (±1 in both axes)
      ``"R<N>"`` — generalised Moore-N (5×5 for R2, 7×7 for R3, ...)
    """
    if neighborhood == "0":
        return ()
    if neighborhood == "4":
        return ((1, 0), (-1, 0), (0, 1), (0, -1))
    if neighborhood == "8":
        return tuple(
            (dx, dy)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if not (dx == 0 and dy == 0)
        )
    if isinstance(neighborhood, str) and neighborhood.startswith("R"):
        try:
            R = int(neighborhood[1:])
        except ValueError as exc:
            raise ValueError(
                f"neighborhood='{neighborhood}': not a valid 'R<N>' spec"
            ) from exc
        if R < 1:
            raise ValueError(
                f"neighborhood='R{R}' must have R>=1; use '0' for same-bin-only"
            )
        return tuple(
            (dx, dy)
            for dx in range(-R, R + 1) for dy in range(-R, R + 1)
            if not (dx == 0 and dy == 0)
        )
    raise ValueError(
        f"neighborhood must be one of '0', '4', '8', or 'R<N>' (got {neighborhood!r})"
    )


def parse_xy_half_offsets(neighborhood: str) -> tuple[tuple[int, int], ...]:
    """Half-plane subset of :func:`parse_xy_offsets` — for symmetric pair
    enumeration where unordered (bin_a, bin_b) should appear once.

    The half-plane rule: keep ``(dx, dy)`` iff ``dx > 0`` OR
    ``(dx == 0 AND dy > 0)``. Matches the convention of the existing
    ``stitch_neighborhood = "8"`` hardcoded list in
    :mod:`tracer.stitching` so the kernel can be a drop-in.

    For ``"8"`` this yields ``((0, 1), (1, -1), (1, 0), (1, 1))``.
    """
    full = parse_xy_offsets(neighborhood)
    return tuple(
        (dx, dy) for (dx, dy) in full
        if dx > 0 or (dx == 0 and dy > 0)
    )


# ---------------------------------------------------------------------
# Grid index
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class GridIndex:
    """Tile-binned inverted index over a tx population with entity labels.

    Attributes
    ----------
    G_xy, G_z : bin sizes (G_z = None → 2D-only index).
    n_tx : total indexed tx count (post-filter for valid entity codes).
    n_entities : highest entity code + 1.
    entity_n_tx : ndarray[int64], shape (n_entities,) — total tx per entity.
    bc_grouped : pd.DataFrame with columns (``bin_xy``, ``bin_z``,
        ``comp``, ``n_tx``) — one row per (bin, entity) pair with the
        tx count of that entity in that bin. ``bin_z`` is 0 when
        ``G_z`` is None.
    transcript_bin_keys : ndarray[int64], shape (n_tx,) — xy bin key
        for each indexed tx, in input order (used by per-tx Rescue
        lookups).
    transcript_bin_z : ndarray[int64], shape (n_tx,) — z bin index per
        indexed tx (0 when ``G_z`` is None).
    transcript_entity : ndarray[int64], shape (n_tx,) — entity code
        per indexed tx.
    transcript_valid_mask : ndarray[bool], shape (n_tx_input,) — mask
        of which input tx were indexed (excludes entity_code < 0).
    """
    G_xy: float
    G_z: float | None
    n_tx: int
    n_entities: int
    entity_n_tx: np.ndarray
    bc_grouped: pd.DataFrame
    transcript_bin_keys: np.ndarray
    transcript_bin_z: np.ndarray
    transcript_entity: np.ndarray
    transcript_valid_mask: np.ndarray


def build_grid_index(
    coords: np.ndarray,
    entity_codes: np.ndarray,
    *,
    G_xy: float,
    G_z: float | None = None,
) -> GridIndex:
    """Build a :class:`GridIndex` from coords + entity codes.

    Parameters
    ----------
    coords : (n, 2|3) ndarray
        Spatial coordinates. Columns 0/1 are x, y. Column 2 is z
        (consulted only when ``G_z`` is not None).
    entity_codes : (n,) ndarray, int-like
        Entity code per tx (typically ``pd.factorize`` output). Entries
        ``< 0`` are dropped (e.g., unassigned tx).
    G_xy : float
        xy bin size in µm.
    G_z : float, optional
        z bin size in µm. ``None`` → 2D-only index.
    """
    coords = np.asarray(coords)
    entity_codes = np.asarray(entity_codes, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coords must be (n, 2+) ndarray, got {coords.shape}")
    if coords.shape[0] != entity_codes.shape[0]:
        raise ValueError(
            f"coords and entity_codes length mismatch: "
            f"{coords.shape[0]} vs {entity_codes.shape[0]}"
        )
    if G_xy <= 0:
        raise ValueError(f"G_xy must be > 0; got {G_xy}")
    if G_z is not None and G_z <= 0:
        raise ValueError(f"G_z must be > 0 or None; got {G_z}")
    if G_z is not None and coords.shape[1] < 3:
        raise ValueError("G_z is set but coords has no z column")

    valid = entity_codes >= 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        empty_bc = pd.DataFrame(
            {"bin_xy": pd.Series(dtype=np.int64),
             "bin_z": pd.Series(dtype=np.int64),
             "comp": pd.Series(dtype=np.int64),
             "n_tx": pd.Series(dtype=np.int64)}
        )
        return GridIndex(
            G_xy=G_xy, G_z=G_z, n_tx=0, n_entities=0,
            entity_n_tx=np.zeros(0, dtype=np.int64),
            bc_grouped=empty_bc,
            transcript_bin_keys=np.zeros(0, dtype=np.int64),
            transcript_bin_z=np.zeros(0, dtype=np.int64),
            transcript_entity=np.zeros(0, dtype=np.int64),
            transcript_valid_mask=valid,
        )

    xy_keys = bin_xy(coords[valid, :2], G_xy)
    if G_z is None:
        bz = np.zeros(n_valid, dtype=np.int64)
    else:
        bz = np.floor(coords[valid, 2] / G_z).astype(np.int64)

    comp_codes = entity_codes[valid]

    bc_df = pd.DataFrame({
        "bin_xy": xy_keys.astype(np.int64),
        "bin_z": bz,
        "comp": comp_codes.astype(np.int64),
    })
    bc_grouped = (
        bc_df.groupby(["bin_xy", "bin_z", "comp"], sort=False, as_index=False)
            .size().rename(columns={"size": "n_tx"})
    )
    bc_grouped["n_tx"] = bc_grouped["n_tx"].astype(np.int64)

    n_entities = int(comp_codes.max() + 1)
    entity_n_tx = np.zeros(n_entities, dtype=np.int64)
    uniq, counts = np.unique(comp_codes, return_counts=True)
    entity_n_tx[uniq] = counts

    return GridIndex(
        G_xy=G_xy,
        G_z=G_z,
        n_tx=n_valid,
        n_entities=n_entities,
        entity_n_tx=entity_n_tx,
        bc_grouped=bc_grouped,
        transcript_bin_keys=xy_keys.astype(np.int64),
        transcript_bin_z=bz,
        transcript_entity=comp_codes.astype(np.int64),
        transcript_valid_mask=valid,
    )


# ---------------------------------------------------------------------
# Bulk entity-pair witness enumeration (Stitch)
# ---------------------------------------------------------------------
def _shift_bin_xy(xy: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift a packed bin_xy int64 array by (dx, dy). Mirror of the
    helper in stitching.py."""
    bx = (xy >> np.int64(32)) - _BIN_BIAS
    by = (xy & np.int64(0xFFFFFFFF)) - _BIN_BIAS
    return (
        ((bx + dx + _BIN_BIAS).astype(np.int64) << np.int64(32))
        | (by + dy + _BIN_BIAS).astype(np.int64)
    )


def enumerate_pair_witnesses(
    idx: GridIndex,
    *,
    neighborhood: str = "8",
    z_depth: int = 0,
    witness_min: int = 0,
    cap_at_n_tx: bool = True,
) -> pd.DataFrame:
    """Bulk emission of all entity-pair candidates with witness counts.

    For each unordered entity pair (lo, hi) with ``lo < hi`` whose tx
    share a neighborhood (xy reach defined by ``neighborhood`` + ``±z_depth``
    z bins), compute:

      - ``n_lo``: unique tx of ``lo`` across all (lo, hi)-shared bins;
      - ``n_hi``: same for ``hi``;
      - ``n_records``: total cross-product count
        ``Σ n_tx(lo, bin_a) * n_tx(hi, bin_b)`` over all neighborhood-
        adjacent (bin_a, bin_b) pairs (matches Stitch's existing
        ``min_candidate_edges`` semantic).

    Half-plane enumeration is used to ensure each unordered pair appears
    once. ``lo`` and ``hi`` are the lower / higher entity code.

    Parameters
    ----------
    idx : GridIndex
        Tile-binned index. Must have been built with the same ``G_xy``
        and ``G_z`` that you want the neighborhood computed at.
    neighborhood : str
        Same semantics as Stitch's ``stitch_neighborhood`` kwarg:
        ``"0"`` | ``"4"`` | ``"8"`` | ``"R<N>"``.
    z_depth : int
        Maximum ``|Δz|`` (in z-bins) for adjacency. Only meaningful
        when the index was built with ``G_z`` set.
    witness_min : int
        If > 0, drop pairs where ``n_lo`` or ``n_hi`` is below the
        effective minimum (see ``cap_at_n_tx``). 0 = no filter.
    cap_at_n_tx : bool
        When applying ``witness_min``, the effective minimum per side
        is ``min(witness_min, entity_n_tx[e])`` so tiny entities aren't
        unfairly blocked by a fixed threshold.

    Returns
    -------
    pd.DataFrame with columns ``lo``, ``hi``, ``n_lo``, ``n_hi``,
    ``n_records``. Empty if no pairs survive.
    """
    if idx.n_tx == 0 or idx.bc_grouped.empty:
        return pd.DataFrame(columns=["lo", "hi", "n_lo", "n_hi", "n_records"])

    bc_grouped = idx.bc_grouped

    xy_half = parse_xy_half_offsets(neighborhood)
    if idx.G_z is None:
        z_with_dz0: list[int] = [0]
        z_strict_pos: list[int] = []
    else:
        z_with_dz0 = list(range(-z_depth, z_depth + 1))
        z_strict_pos = list(range(1, z_depth + 1))

    # Build the (offset) iteration. The (0, 0, 0) case enumerates same-
    # bin entity pairs; positive offsets enumerate inter-bin pairs.
    offsets_iter: list[tuple[int, int, int]] = [(0, 0, 0)]
    if idx.G_z is None:
        offsets_iter += [(dx, dy, 0) for (dx, dy) in xy_half]
    else:
        offsets_iter += [
            (dx, dy, dz)
            for (dx, dy) in xy_half
            for dz in z_with_dz0
        ] + [(0, 0, dz) for dz in z_strict_pos]

    records: list[pd.DataFrame] = []
    for dx, dy, dz in offsets_iter:
        if dx == 0 and dy == 0 and dz == 0:
            # Same-bin: self-join with comp_a < comp_b
            merged = bc_grouped.merge(
                bc_grouped, on=["bin_xy", "bin_z"], suffixes=("_a", "_b"),
            )
            merged = merged[merged["comp_a"] < merged["comp_b"]]
            if len(merged) == 0:
                continue
            lo_arr = merged["comp_a"].to_numpy()
            hi_arr = merged["comp_b"].to_numpy()
            count_arr = (
                merged["n_tx_a"].to_numpy() * merged["n_tx_b"].to_numpy()
            )
            bin_lo_xy = merged["bin_xy"].to_numpy()
            bin_lo_z = merged["bin_z"].to_numpy()
            bin_hi_xy = bin_lo_xy
            bin_hi_z = bin_lo_z
        else:
            # Different-bin: shift `bc_grouped` and join.
            right = bc_grouped.copy()
            right["bin_xy_join"] = _shift_bin_xy(
                right["bin_xy"].to_numpy(), -dx, -dy
            )
            right["bin_z_join"] = right["bin_z"] - dz
            merged = bc_grouped.merge(
                right,
                left_on=["bin_xy", "bin_z"],
                right_on=["bin_xy_join", "bin_z_join"],
                suffixes=("_a", "_b"),
            )
            merged = merged[merged["comp_a"] != merged["comp_b"]]
            if len(merged) == 0:
                continue
            comp_a = merged["comp_a"].to_numpy()
            comp_b = merged["comp_b"].to_numpy()
            count_arr = (
                merged["n_tx_a"].to_numpy() * merged["n_tx_b"].to_numpy()
            )
            bin_a_xy = merged["bin_xy_a"].to_numpy()
            bin_a_z = merged["bin_z_a"].to_numpy()
            bin_b_xy = merged["bin_xy_b"].to_numpy()
            bin_b_z = merged["bin_z_b"].to_numpy()
            swap_mask = comp_a > comp_b
            lo_arr = np.where(swap_mask, comp_b, comp_a)
            hi_arr = np.where(swap_mask, comp_a, comp_b)
            bin_lo_xy = np.where(swap_mask, bin_b_xy, bin_a_xy)
            bin_lo_z = np.where(swap_mask, bin_b_z, bin_a_z)
            bin_hi_xy = np.where(swap_mask, bin_a_xy, bin_b_xy)
            bin_hi_z = np.where(swap_mask, bin_a_z, bin_b_z)

        records.append(pd.DataFrame({
            "lo": lo_arr.astype(np.int64),
            "hi": hi_arr.astype(np.int64),
            "count": count_arr.astype(np.int64),
            "blxy": bin_lo_xy.astype(np.int64),
            "blz": bin_lo_z.astype(np.int64),
            "bhxy": bin_hi_xy.astype(np.int64),
            "bhz": bin_hi_z.astype(np.int64),
        }))

    if not records:
        return pd.DataFrame(columns=["lo", "hi", "n_lo", "n_hi", "n_records"])

    all_records = pd.concat(records, ignore_index=True, copy=False)

    # n_records: sum of cross-product counts per pair (matches Stitch's
    # min_candidate_edges aggregation).
    n_records = (
        all_records.groupby(["lo", "hi"], sort=False, as_index=False)
            ["count"].sum().rename(columns={"count": "n_records"})
    )

    # Per-side witness counts: dedup at unique (entity-pair, side-bin)
    # level, look up n_tx via bc_grouped, sum per pair.
    ar_u = all_records[
        ["lo", "hi", "blxy", "blz", "bhxy", "bhz"]
    ].drop_duplicates()

    lo_merged = (
        ar_u[["lo", "hi", "blxy", "blz"]]
            .drop_duplicates()
            .merge(
                bc_grouped,
                left_on=["blxy", "blz", "lo"],
                right_on=["bin_xy", "bin_z", "comp"],
                how="left",
            )
    )
    lo_merged["n_tx"] = lo_merged["n_tx"].fillna(0).astype(np.int64)
    lo_summed = (
        lo_merged.groupby(["lo", "hi"], sort=False, as_index=False)
            ["n_tx"].sum().rename(columns={"n_tx": "n_lo"})
    )

    hi_merged = (
        ar_u[["lo", "hi", "bhxy", "bhz"]]
            .drop_duplicates()
            .merge(
                bc_grouped,
                left_on=["bhxy", "bhz", "hi"],
                right_on=["bin_xy", "bin_z", "comp"],
                how="left",
            )
    )
    hi_merged["n_tx"] = hi_merged["n_tx"].fillna(0).astype(np.int64)
    hi_summed = (
        hi_merged.groupby(["lo", "hi"], sort=False, as_index=False)
            ["n_tx"].sum().rename(columns={"n_tx": "n_hi"})
    )

    out = (
        n_records.merge(lo_summed, on=["lo", "hi"], how="left")
                 .merge(hi_summed, on=["lo", "hi"], how="left")
    )
    out["n_lo"] = out["n_lo"].fillna(0).astype(np.int64)
    out["n_hi"] = out["n_hi"].fillna(0).astype(np.int64)
    out = out[["lo", "hi", "n_lo", "n_hi", "n_records"]]

    if witness_min > 0:
        mlt = int(witness_min)
        if cap_at_n_tx:
            eff_lo = np.minimum(mlt, idx.entity_n_tx[out["lo"].to_numpy()])
            eff_hi = np.minimum(mlt, idx.entity_n_tx[out["hi"].to_numpy()])
        else:
            eff_lo = mlt
            eff_hi = mlt
        keep = (out["n_lo"].to_numpy() >= eff_lo) & (
            out["n_hi"].to_numpy() >= eff_hi
        )
        out = out.loc[keep].reset_index(drop=True)

    return out


# ---------------------------------------------------------------------
# Per-bin entity lookup (Rescue)
# ---------------------------------------------------------------------
def neighbor_entities(
    idx: GridIndex,
    bin_xy_key: int,
    bin_z: int = 0,
    *,
    neighborhood: str = "8",
    z_depth: int = 0,
) -> dict[int, list[tuple[int, int]]]:
    """For a query bin, return the entities present in any neighborhood
    bin and the (bin_xy, bin_z) tuples they occupy.

    Useful for Rescue's per-tx candidate lookup: pass the query tx's
    own bin, get back the entities with tx in reach.

    Parameters
    ----------
    idx : GridIndex
    bin_xy_key : int
        Packed int64 xy bin key from :func:`tracer.graph.bin_xy`.
    bin_z : int
        z bin index (ignored if the index was built without ``G_z``).
    neighborhood : str
        Same semantics as Stitch.
    z_depth : int
        Maximum ``|Δz|`` reach in z bins.

    Returns
    -------
    dict[entity_code, list[(bin_xy, bin_z)]]
        Mapping from entity codes present in any reachable bin to the
        list of bins (in idx) where they have tx. Includes the query
        bin itself (offset (0, 0, 0)).
    """
    if idx.n_tx == 0:
        return {}

    xy_full = parse_xy_offsets(neighborhood)
    # Full neighborhood including (0, 0): same bin, then xy offsets.
    xy_all: list[tuple[int, int]] = [(0, 0)] + list(xy_full)

    if idx.G_z is None:
        z_offsets = [0]
    else:
        z_offsets = list(range(-z_depth, z_depth + 1))

    # Resolve each candidate bin key via the bc_grouped index. Building
    # a tiny query-DataFrame and merging is more idiomatic than per-key
    # dict iteration when bc_grouped is large.
    bc = idx.bc_grouped
    if bc.empty:
        return {}

    # Compute the set of reachable bins as (bin_xy, bin_z) tuples.
    cand_xy = np.fromiter(
        (
            ((bin_xy_key >> np.int64(32)) - _BIN_BIAS + dx + _BIN_BIAS) << np.int64(32)
            | ((bin_xy_key & np.int64(0xFFFFFFFF)) - _BIN_BIAS + dy + _BIN_BIAS)
            for (dx, dy) in xy_all for _ in z_offsets
        ),
        dtype=np.int64,
        count=len(xy_all) * len(z_offsets),
    )
    cand_z = np.fromiter(
        (bin_z + dz for _ in xy_all for dz in z_offsets),
        dtype=np.int64,
        count=len(xy_all) * len(z_offsets),
    )
    q = pd.DataFrame({"bin_xy": cand_xy, "bin_z": cand_z}).drop_duplicates()
    found = q.merge(bc, on=["bin_xy", "bin_z"], how="inner")
    if found.empty:
        return {}
    out: dict[int, list[tuple[int, int]]] = {}
    for ent, sub in found.groupby("comp", sort=False):
        out[int(ent)] = list(
            zip(sub["bin_xy"].astype(np.int64).tolist(),
                sub["bin_z"].astype(np.int64).tolist())
        )
    return out
