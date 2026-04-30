"""Phase 4: Hierarchical entity stitching."""

import heapq
import itertools
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ._repro import _ensure_reproducibility_seed
from ._utils import relu_symmetric
from .graph import bin_xy, delaunay_edges, neighbor_bins


# ---------- Phase 4: Hierarchical Stitching ----------
# ----------------------------
# Helpers: entity type / parse
# ----------------------------
def infer_entity_type(entity_id: str) -> str:
    """
    Returns one of: 'cell', 'partial', 'component', 'drop', 'unknown'
    """
    if entity_id is None or (isinstance(entity_id, float) and np.isnan(entity_id)):
        return "unknown"
    s = str(entity_id)
    if s == "DROP":
        return "drop"
    if s.startswith("UNASSIGNED_"):
        return "component"
    if "-" in s:
        return "partial"
    # otherwise treat as cell (original)
    return "cell"


def build_entity_table(
    df_final: pd.DataFrame,
    *,
    entity_col: str,
    gene_col: str = "feature_name",
    coord_cols=("x", "y", "z"),
):
    """
    Build per-entity summary:
      - centroid (x,y,z)
      - unique genes list
      - type: cell/partial/component
    """
    # Read-only view of the two columns we need — no full-df copy.
    # Previously called `.astype(str).str.strip()` on `df_final[gene_col]`
    # which forced an O(100M)-row Python string op; assume gene names
    # arrive normalised (use `prepare_transcript_df` upstream).
    ent = df_final[entity_col].astype(str)
    keep = ent.notna() & (ent != "DROP") & (ent != "nan")

    # Slice to the keep rows. `.loc` is a view when the mask is boolean.
    df = df_final.loc[keep, [entity_col, gene_col, *coord_cols]].copy()
    df[entity_col] = df[entity_col].astype(str)

    # entity type
    df["_etype"] = df[entity_col].map(infer_entity_type)
    df = df[df["_etype"].isin(["cell", "partial", "component"])]

    # centroid (`observed=True` avoids processing empty categorical groups
    # when entity_col is categorical).
    cent = df.groupby(entity_col, sort=True, observed=True)[list(coord_cols)].mean()

    # unique genes per entity (sorted for deterministic downstream mapping)
    genes = df.groupby(entity_col, sort=True, observed=True)[gene_col].unique()
    genes = genes.apply(lambda arr: np.sort(arr.astype(str)))

    etype = df.groupby(entity_col, observed=True)["_etype"].first()

    summary = cent.join(genes.rename("genes")).join(etype.rename("etype"))
    summary = summary.reset_index().rename(columns={entity_col: "entity_id"})
    return summary


# -------------------------------------------
# Coherence C(gene-set) using NPMI
# -------------------------------------------

_VALID_COHERENCE_MODES = ("count", "magnitude")


def _slice_npmi_submatrix(npmi_mat, gene_ids):
    """Return a dense float submatrix for the given gene indices.

    Handles both dense ``np.ndarray`` and ``scipy.sparse`` inputs. For
    sparse inputs we slice to a sparse submatrix and densify only the
    small per-entity block; absent entries become exact zeros — by
    design (see :func:`compute_npmi_bootstrap` docs).
    """
    try:
        from scipy import sparse
    except ImportError:  # pragma: no cover
        sparse = None

    if sparse is not None and sparse.issparse(npmi_mat):
        # Slice rows then cols (CSR/CSC). The bootstrap stores only the
        # upper triangle of W_sparse, but `gene_ids` may reorder genes
        # so the upper triangle of `sub` no longer covers the same cells
        # as the upper triangle of the original. Symmetrise via sub+sub.T
        # — exactly one of (sub[a,b], sub[b,a]) is nonzero by
        # construction, so the sum is just the value, not doubled.
        sub = npmi_mat[gene_ids, :][:, gene_ids]
        dense = np.asarray(sub.todense())
        return dense + dense.T
    return npmi_mat[np.ix_(gene_ids, gene_ids)]


def coherence(
    gene_ids: np.ndarray,
    npmi_mat: np.ndarray,
    *,
    mode: str = "count",
    threshold: float = 0.05,
    metric: str = "npmi",
) -> tuple[float, float, float]:
    """Unified coherence — returns ``(C, purity, conflict)``.

    The function operates on the values stored in ``npmi_mat`` and is
    metric-agnostic in its math. The ``metric`` kwarg is purely an
    advisory parameter that:
      (a) validates the caller's intent against the chosen ``mode``, and
      (b) documents the threshold's interpretation.

    Parameters
    ----------
    gene_ids : np.ndarray
        Indices into ``npmi_mat`` for the gene set under consideration.
    npmi_mat : np.ndarray or scipy.sparse
        Square matrix of pairwise association values. Caller's
        responsibility to ensure entries are NPMI (bounded [-1,+1]) or
        PMI (unbounded log-fold-enrichment) consistent with ``metric``.
    mode : {"count", "magnitude"}
        ``"count"`` — purity = #(w > threshold) / |P|;
        conflict = #(w < -threshold) / |P|. Threshold-based fraction.

        ``"magnitude"`` — purity = Σmax(w, 0) / Σ|w|;
        conflict = Σmax(-w, 0) / Σ|w|. **Only valid with metric="npmi"**
        because PMI's unbounded magnitude lets a single rare-strong
        pair dominate the sum.
    threshold : float
        Dead-zone threshold τ. Used directly in ``"count"`` mode. The
        natural calibration depends on ``metric``: NPMI thresholds are
        typically in [0.01, 0.1]; PMI thresholds reflect log-fold
        enrichment (e.g., 0.4 ≈ "50% above independence").
    metric : {"npmi", "pmi"}
        Advisory parameter naming the metric in ``npmi_mat``. Raises
        ``ValueError`` if ``metric="pmi"`` is paired with
        ``mode="magnitude"``.

    Returns
    -------
    C : float
        ``purity - conflict``.
    purity : float
    conflict : float
    """
    k = int(gene_ids.size)
    if k < 2:
        return 0.0, 0.0, 0.0
    if mode not in _VALID_COHERENCE_MODES:
        raise ValueError(
            f"mode must be one of {_VALID_COHERENCE_MODES!r} (got {mode!r})"
        )
    if metric not in ("npmi", "pmi"):
        raise ValueError(f"metric must be 'npmi' or 'pmi' (got {metric!r})")
    if metric == "pmi" and mode == "magnitude":
        raise ValueError(
            "metric='pmi' is incompatible with mode='magnitude' — PMI's "
            "unbounded magnitude lets rare-strong pairs dominate the sum. "
            "Use metric='pmi' with mode='count' instead."
        )

    sub = _slice_npmi_submatrix(npmi_mat, gene_ids)
    iu = np.triu_indices(k, k=1)
    vals = sub[iu]
    vals = vals[np.isfinite(vals)]
    P = vals.size
    if P == 0:
        return 0.0, 0.0, 0.0

    if mode == "count":
        purity = float(np.sum(vals > threshold)) / P
        conflict = float(np.sum(vals < -threshold)) / P
    else:  # magnitude
        denom = float(np.sum(np.abs(vals)))
        if denom == 0.0:
            return 0.0, 0.0, 0.0
        purity = float(np.sum(np.maximum(vals, 0.0))) / denom
        conflict = float(np.sum(np.maximum(-vals, 0.0))) / denom

    return float(purity - conflict), float(purity), float(conflict)


def signal_strength(gene_ids: np.ndarray, npmi_mat: np.ndarray) -> float:
    """``S(G) = Σ|w_ij|`` over finite (i, j) pairs (manuscript Eq 22).

    Diagnostic — not folded into ΔC. Returns 0.0 for sets of <2 genes
    or sets with no observed pairs.
    """
    k = int(gene_ids.size)
    if k < 2:
        return 0.0
    sub = _slice_npmi_submatrix(npmi_mat, gene_ids)
    iu = np.triu_indices(k, k=1)
    vals = sub[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.sum(np.abs(vals)))


def deltaC(
    genes_u: np.ndarray,
    genes_v: np.ndarray,
    npmi_mat: np.ndarray,
    *,
    mode: str = "count",
    threshold: float = 0.05,
    penalize_simplicity: bool = True,
    metric: str = "npmi",
) -> float:
    """Unified ΔC across coherence modes.

    Without ``penalize_simplicity``::

        ΔC = C(union) - max(C(u), C(v))

    With ``penalize_simplicity`` (default), each per-cluster C is
    adjusted by ``-1/n`` and the union by ``-1/(n_u + n_v)`` so a
    larger merged set must produce strictly higher coherence to win
    over the simpler-to-explain split.

    ``metric`` is forwarded to :func:`coherence`; see its docstring.
    """
    C_u, _, _ = coherence(genes_u, npmi_mat, mode=mode, threshold=threshold, metric=metric)
    C_v, _, _ = coherence(genes_v, npmi_mat, mode=mode, threshold=threshold, metric=metric)
    union = np.unique(np.concatenate([genes_u, genes_v]))
    C_union, _, _ = coherence(union, npmi_mat, mode=mode, threshold=threshold, metric=metric)

    if not penalize_simplicity:
        return float(C_union - max(C_u, C_v))

    nu = max(int(genes_u.size), 1)
    nv = max(int(genes_v.size), 1)
    n_union = nu + nv
    C_sep = max(C_u - 1.0 / nu, C_v - 1.0 / nv)
    return float(C_union - (1.0 / n_union) - C_sep)


def compute_housekeeping_mask(
    W,
    *,
    pos_thresh: float = 0.05,
    neg_thresh: float = -0.05,
    min_strong_count: int = 5,
) -> np.ndarray:
    """Bool array of length ``G``. ``True`` = keep gene, ``False`` = drop.

    A gene is flagged as housekeeping if it has fewer than
    ``min_strong_count`` strong-positive (NPMI > ``pos_thresh``) AND
    fewer than ``min_strong_count`` strong-negative (NPMI < ``neg_thresh``)
    neighbors. The diagonal is ignored. NaN entries don't count toward
    either tally. Accepts dense or sparse ``W``.
    """
    try:
        from scipy import sparse
    except ImportError:  # pragma: no cover
        sparse = None

    if W.shape[0] != W.shape[1]:
        raise ValueError("W must be square")
    G = int(W.shape[0])
    if G == 0:
        return np.empty((0,), dtype=bool)

    if sparse is not None and sparse.issparse(W):
        # The bootstrap CSR stores only the upper triangle.
        # Symmetrise virtually by counting both rows and columns.
        Wcsr = W.tocsr().astype(np.float32)
        # Boolean masks as sparse — diagonal entries shouldn't be stored
        # (NpmiBootstrapResult never stores i==j) but be defensive.
        Wcsr.setdiag(0.0)
        Wcsr.eliminate_zeros()
        pos_mask = (Wcsr > pos_thresh)
        neg_mask = (Wcsr < neg_thresh)
        # Counts: per-row + per-column (since only upper-tri stored,
        # row i counts the j>i neighbors and col i counts the j<i ones).
        pos_counts = (
            np.asarray(pos_mask.sum(axis=1)).ravel()
            + np.asarray(pos_mask.sum(axis=0)).ravel()
        )
        neg_counts = (
            np.asarray(neg_mask.sum(axis=1)).ravel()
            + np.asarray(neg_mask.sum(axis=0)).ravel()
        )
    else:
        W_arr = np.asarray(W, dtype=np.float32)
        diag_mask = np.eye(G, dtype=bool)
        pos = (W_arr > pos_thresh) & ~diag_mask
        neg = (W_arr < neg_thresh) & ~diag_mask
        pos_counts = pos.sum(axis=1)
        neg_counts = neg.sum(axis=1)

    return (pos_counts >= min_strong_count) | (neg_counts >= min_strong_count)


# -------------------------------------------
# Legacy coherence wrappers (deprecated)
# -------------------------------------------
# The four functions below are thin wrappers around `coherence` and
# `deltaC` that preserve the public names but emit DeprecationWarning.
# Mappings — and behavior shifts — are documented per-function.


def coherence_C_from_genes(
    gene_ids: np.ndarray,
    npmi_mat: np.ndarray,
    *,
    purity_threshold: float = 0.05,
):
    """Deprecated. Use ``coherence(..., mode="count", threshold=purity_threshold)``.

    Behavior is unchanged for this wrapper — both the legacy
    implementation and ``coherence(mode="count")`` count pairs above the
    threshold and divide by |P|.
    """
    warnings.warn(
        "coherence_C_from_genes is deprecated; use "
        "coherence(gene_ids, npmi_mat, mode='count', threshold=purity_threshold).",
        DeprecationWarning,
        stacklevel=2,
    )
    return coherence(gene_ids, npmi_mat, mode="count", threshold=purity_threshold)


def coherence_C_from_genes_relu(
    gene_ids: np.ndarray,
    npmi_mat: np.ndarray,
    *,
    tau: float = 0.05,
    use_relative: bool = False,
):
    """Deprecated. Use ``coherence(..., mode='count'|'magnitude')``.

    Mapping:
      - ``use_relative=False`` → ``coherence(mode='count', threshold=tau)``
      - ``use_relative=True``  → ``coherence(mode='magnitude', threshold=tau)``

    Behavior has changed. The legacy implementation used post-ReLU
    magnitude sums (``Σmax(w-τ, 0) / |P|`` when ``use_relative=False``;
    ``Σmax(w-τ, 0) / Σ|ReLU(w)|`` when ``use_relative=True``). The new
    modes use raw NPMI and the count-fraction (``mode='count'``) or
    raw-magnitude ratio (``mode='magnitude'``). See release notes.
    """
    target_mode = "magnitude" if use_relative else "count"
    warnings.warn(
        "coherence_C_from_genes_relu is deprecated; use "
        f"coherence(gene_ids, npmi_mat, mode={target_mode!r}, threshold=tau). "
        "Behavior has changed (raw NPMI; no ReLU pre-step). See release notes.",
        DeprecationWarning,
        stacklevel=2,
    )
    return coherence(gene_ids, npmi_mat, mode=target_mode, threshold=tau)


def deltaC_between_clusters(
    genes_u: np.ndarray,
    genes_v: np.ndarray,
    npmi_mat: np.ndarray,
    *,
    purity_threshold: float = 0.05,
    penalize_simplicity: bool = True,
):
    """Deprecated. Use ``deltaC(..., mode='count', threshold=purity_threshold)``.

    Behavior is unchanged for this wrapper.
    """
    warnings.warn(
        "deltaC_between_clusters is deprecated; use "
        "deltaC(genes_u, genes_v, npmi_mat, mode='count', threshold=purity_threshold).",
        DeprecationWarning,
        stacklevel=2,
    )
    return deltaC(
        genes_u,
        genes_v,
        npmi_mat,
        mode="count",
        threshold=purity_threshold,
        penalize_simplicity=penalize_simplicity,
    )


def deltaC_between_clusters_relu(
    genes_u: np.ndarray,
    genes_v: np.ndarray,
    npmi_mat: np.ndarray,
    *,
    tau: float = 0.05,
    use_relative: bool = False,
    penalize_simplicity: bool = True,
):
    """Deprecated. Use ``deltaC(..., mode='count'|'magnitude')``.

    Mapping:
      - ``use_relative=False`` → ``deltaC(mode='count', threshold=tau)``
      - ``use_relative=True``  → ``deltaC(mode='magnitude', threshold=tau)``

    Behavior has changed (same shift as ``coherence_C_from_genes_relu``).
    """
    target_mode = "magnitude" if use_relative else "count"
    warnings.warn(
        "deltaC_between_clusters_relu is deprecated; use "
        f"deltaC(genes_u, genes_v, npmi_mat, mode={target_mode!r}, threshold=tau). "
        "Behavior has changed (raw NPMI; no ReLU pre-step). See release notes.",
        DeprecationWarning,
        stacklevel=2,
    )
    return deltaC(
        genes_u,
        genes_v,
        npmi_mat,
        mode=target_mode,
        threshold=tau,
        penalize_simplicity=penalize_simplicity,
    )


# ----------------------------
# Union-Find (Disjoint Set Union)
# ----------------------------
class DSU:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


# --------------------------------------
# Constrained hierarchical ΔC stitching
# --------------------------------------
_LEGACY_STITCH_KWARG_SENTINEL = object()


def stitch_entities_hierarchical(
    summary_df: pd.DataFrame,
    aux: dict,
    *,
    mode: str = "count",
    threshold: float = 0.05,
    metric: str = "npmi",
    penalize_simplicity=True,
    deltaC_min=0.0,
    use_3d=True,
    dist_threshold: float | None = None,
    candidate_source: str = "delaunay",
    G: float | None = None,
    stitch_neighborhood: str = "8",
    transcript_coords: np.ndarray | None = None,
    transcript_entity_codes: np.ndarray | None = None,
    # Deprecated kwargs — translated to mode/threshold below.
    purity_threshold=_LEGACY_STITCH_KWARG_SENTINEL,
    tau=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relu=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relative=_LEGACY_STITCH_KWARG_SENTINEL,
):
    """Hierarchical entity stitching driven by ΔC.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Required columns: ``entity_id``, ``x``, ``y``, ``z`` (or just
        ``x``, ``y`` if ``use_3d=False``), ``genes`` (np.ndarray[str]),
        ``etype`` in ``{'cell', 'partial', 'component'}``.
    aux : dict
        Must contain ``"W"`` (NPMI matrix) and ``"gene_to_idx"``. May
        contain ``"housekeeping_mask"`` (bool array of length G); when
        present, gene indices flagged ``False`` are removed from each
        entity's gene set before ΔC is computed.
    mode : {"count", "magnitude"}
        Coherence semantics. See :func:`coherence`.
    threshold : float
        Dead-zone threshold τ used by :func:`coherence` / :func:`deltaC`.
    penalize_simplicity : bool
        If True, ΔC penalizes smaller gene sets; see :func:`deltaC`.
    deltaC_min : float
        Minimum ΔC required to merge two clusters.
    use_3d : bool
        Use 3D or 2D coordinates for centroid distance.

    Other Parameters
    ----------------
    purity_threshold, tau, use_relu, use_relative : deprecated
        Legacy kwargs from before the coherence consolidation. Passing
        any of them emits ``DeprecationWarning`` and translates to
        ``mode``/``threshold``. See release notes for the behavior
        shift.

    Returns
    -------
    entity_to_stitched : dict[str, str]
    info : dict
        Cluster bookkeeping; currently just ``{"root_to_label": ...}``.
    """
    legacy_passed = {
        name: value
        for name, value in (
            ("purity_threshold", purity_threshold),
            ("tau", tau),
            ("use_relu", use_relu),
            ("use_relative", use_relative),
        )
        if value is not _LEGACY_STITCH_KWARG_SENTINEL
    }
    if legacy_passed:
        warnings.warn(
            "stitch_entities_hierarchical: legacy kwargs "
            f"{sorted(legacy_passed)} are deprecated; pass mode='count'|"
            "'magnitude' and threshold instead. Translating with the same "
            "behavior shift as the coherence wrappers; see release notes.",
            DeprecationWarning,
            stacklevel=2,
        )
        eff_use_relu = legacy_passed.get("use_relu", True)
        eff_use_relative = legacy_passed.get("use_relative", False)
        eff_tau = legacy_passed.get("tau", _LEGACY_STITCH_KWARG_SENTINEL)
        eff_pt = legacy_passed.get("purity_threshold", _LEGACY_STITCH_KWARG_SENTINEL)

        if not eff_use_relu:
            mode = "count"
        elif eff_use_relative:
            mode = "magnitude"
        else:
            mode = "count"

        eff_tau_set = eff_tau is not _LEGACY_STITCH_KWARG_SENTINEL
        eff_pt_set = eff_pt is not _LEGACY_STITCH_KWARG_SENTINEL
        if eff_tau_set and eff_pt_set and eff_tau != eff_pt:
            warnings.warn(
                "stitch_entities_hierarchical: both tau and purity_threshold "
                f"passed with different values ({eff_tau!r} vs {eff_pt!r}); "
                "using tau.",
                DeprecationWarning,
                stacklevel=2,
            )
        if eff_tau_set:
            threshold = eff_tau
        elif eff_pt_set:
            threshold = eff_pt

    npmi_mat = aux["W"]
    gene_to_idx = aux["gene_to_idx"]
    housekeeping_mask = aux.get("housekeeping_mask")

    # map entity -> gene indices
    entity_ids = summary_df["entity_id"].astype(str).to_numpy()
    etypes = summary_df["etype"].astype(str).to_numpy()

    gene_id_lists = []
    for genes in summary_df["genes"].values:
        g = pd.Index(np.asarray(genes, dtype=str)).map(gene_to_idx)
        g = np.sort(g[~pd.isna(g)].astype(int).unique())
        g = np.asarray(g, dtype=np.int32)
        if housekeeping_mask is not None and g.size > 0:
            g = g[housekeeping_mask[g]]
        gene_id_lists.append(g)

    # points
    if use_3d:
        pts = summary_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    else:
        pts = summary_df[["x", "y"]].to_numpy(dtype=np.float64)

    N = len(entity_ids)
    if N <= 1:
        return {entity_ids[0]: entity_ids[0]}, {}

    # ----------------------------------------------------------------
    # Candidate edge enumeration: Delaunay over centroids OR bin-grid
    # ----------------------------------------------------------------
    if candidate_source not in ("delaunay", "grid"):
        raise ValueError(
            f"candidate_source must be 'delaunay' or 'grid' (got {candidate_source!r})"
        )

    adj: list[list[int]] | None = None

    if candidate_source == "delaunay":
        # Delaunay edges (use SciPy by default)
        edges = delaunay_edges(pts)

        # Optionally filter edges by geometric length to reduce candidate merges
        if dist_threshold is not None:
            if len(edges) > 0:
                ei = np.asarray(edges, dtype=np.int64)
                p0 = pts[ei[:, 0]]
                p1 = pts[ei[:, 1]]
                dists = np.linalg.norm(p0 - p1, axis=1)
                keep = dists <= float(dist_threshold)
                edges = [tuple(x) for x in ei[keep]]

        # adjacency on original nodes
        adj = [[] for _ in range(N)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

    else:  # candidate_source == "grid"
        if G is None:
            raise ValueError("G must be provided when candidate_source='grid'")
        if stitch_neighborhood not in ("4", "8"):
            raise ValueError(
                f"stitch_neighborhood must be '4' or '8' (got {stitch_neighborhood!r})"
            )
        if transcript_coords is None or transcript_entity_codes is None:
            raise ValueError(
                "transcript_coords and transcript_entity_codes must be "
                "provided when candidate_source='grid'"
            )
        if transcript_coords.shape[0] != transcript_entity_codes.shape[0]:
            raise ValueError(
                "transcript_coords and transcript_entity_codes must have "
                "equal length"
            )

        # Map transcripts to (bin_key, entity_idx). Skip transcripts whose
        # entity_idx is < 0 (e.g., DROP / unmapped labels).
        bin_keys_all = bin_xy(transcript_coords[:, :2], G)
        ec = np.asarray(transcript_entity_codes, dtype=np.int64)
        valid = ec >= 0
        bin_keys = bin_keys_all[valid].tolist()
        comp_codes = ec[valid].tolist()

        bin_to_comps = defaultdict(set)
        comp_to_bins = defaultdict(set)
        for bk, c in zip(bin_keys, comp_codes):
            bin_to_comps[bk].add(c)
            comp_to_bins[c].add(bk)

        # Initial candidate enumeration:
        # - within-bin: all unordered pairs of distinct components
        # - cross-bin (half-neighborhood): all (a, b) with a in bin, b in
        #   half-neighbor bin, a != b. Half-neighborhood ensures every
        #   unordered (bin_a, bin_b) pair is enumerated exactly once.
        half_topology = "half-4" if stitch_neighborhood == "4" else "half-8"
        candidate_pairs: set[tuple[int, int]] = set()
        for bk, comps in bin_to_comps.items():
            if len(comps) > 1:
                comps_sorted = sorted(comps)
                for a, b in itertools.combinations(comps_sorted, 2):
                    candidate_pairs.add((a, b))
            for nb in neighbor_bins(bk, topology=half_topology):
                nb_comps = bin_to_comps.get(nb)
                if not nb_comps:
                    continue
                for a in comps:
                    for b in nb_comps:
                        if a != b:
                            lo, hi = (a, b) if a < b else (b, a)
                            candidate_pairs.add((lo, hi))

        edges = list(candidate_pairs)

        # Indices are no longer needed after initial enumeration; release memory.
        del bin_to_comps, comp_to_bins

    # cluster metadata tracked at DSU roots
    dsu = DSU(N)

    # track whether a cluster contains a real cell (constraint)
    has_cell = np.array([t == "cell" for t in etypes], dtype=bool)

    # For label preference
    # store lists of member entity_ids by type at roots (kept as python sets for simplicity)
    cell_ids = [set([entity_ids[i]]) if etypes[i] == "cell" else set() for i in range(N)]
    partial_ids = [set([entity_ids[i]]) if etypes[i] == "partial" else set() for i in range(N)]
    comp_ids = [set([entity_ids[i]]) if etypes[i] == "component" else set() for i in range(N)]

    # store gene_id union at roots (as sorted unique arrays)
    root_genes = gene_id_lists[:]  # list of np arrays

    # constraint: can we merge clusters A and B?
    def can_merge(ra, rb):
        # never merge two clusters that both contain a cell
        if has_cell[ra] and has_cell[rb]:
            return False
        return True

    # ----------------------------------------------------------------
    # Per-root coherence cache.
    #
    # The heap loop pops O(N · avg_neighbours) candidate edges and each
    # call to deltaC needs C(ra), C(rb), and C(ra ∪ rb). Without a
    # cache, C(ra) and C(rb) get recomputed once for every neighbour
    # they appear with — a 2–3× redundant cost on the rejected pops
    # that dominate the loop. We cache (C, purity, conflict) per root
    # and invalidate the entry immediately after dsu.union (when the
    # root's gene set changes). C(ra ∪ rb) is genuinely new each merge
    # and is not cached.
    # ----------------------------------------------------------------
    root_C_cache: dict[int, tuple[float, float, float]] = {}
    cache_hits = 0
    cache_misses = 0

    def C_of_root(root_idx: int) -> tuple[float, float, float]:
        nonlocal cache_hits, cache_misses
        cached = root_C_cache.get(root_idx)
        if cached is not None:
            cache_hits += 1
            return cached
        cache_misses += 1
        triple = coherence(
            root_genes[root_idx], npmi_mat,
            mode=mode, threshold=threshold, metric=metric,
        )
        root_C_cache[root_idx] = triple
        return triple

    # compute deltaC between current roots
    def compute_deltaC_roots(ra, rb):
        Cu, _, _ = C_of_root(ra)
        Cv, _, _ = C_of_root(rb)
        union = np.unique(np.concatenate([root_genes[ra], root_genes[rb]]))
        Cunion, _, _ = coherence(
            union, npmi_mat, mode=mode, threshold=threshold, metric=metric,
        )
        if not penalize_simplicity:
            return float(Cunion - max(Cu, Cv))
        nu = max(int(root_genes[ra].size), 1)
        nv = max(int(root_genes[rb].size), 1)
        n_union = nu + nv
        C_sep = max(Cu - 1.0 / nu, Cv - 1.0 / nv)
        return float(Cunion - (1.0 / n_union) - C_sep)

    # max-heap of candidate edges by deltaC (lazy updates)
    def _heap_item(dc, a, b):
        # Deterministic tie-breaking: enforce ordered endpoints
        if a > b:
            a, b = b, a
        return (-dc, a, b)

    heap = []
    for i, j in edges:
        di = compute_deltaC_roots(i, j)
        if np.isfinite(di) and di >= deltaC_min:
            heapq.heappush(heap, _heap_item(di, i, j))

    # greedy merging
    while heap:
        neg_dc, a, b = heapq.heappop(heap)
        dc = -neg_dc

        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            continue
        if not can_merge(ra, rb):
            continue

        # recompute deltaC for current clusters (because a,b may have merged)
        dc_now = compute_deltaC_roots(ra, rb)
        if not (np.isfinite(dc_now) and dc_now >= deltaC_min):
            continue

        # merge (choose new root)
        rnew = dsu.union(ra, rb)
        rold = rb if rnew == ra else ra

        # update cluster metadata onto rnew
        has_cell[rnew] = has_cell[rnew] or has_cell[rold]
        cell_ids[rnew] |= cell_ids[rold]
        partial_ids[rnew] |= partial_ids[rold]
        comp_ids[rnew] |= comp_ids[rold]

        # union genes
        if root_genes[rnew].size == 0:
            root_genes[rnew] = root_genes[rold]
        elif root_genes[rold].size == 0:
            pass
        else:
            root_genes[rnew] = np.unique(np.concatenate([root_genes[rnew], root_genes[rold]])).astype(np.int32)

        # clear old to save memory
        cell_ids[rold].clear()
        partial_ids[rold].clear()
        comp_ids[rold].clear()
        root_genes[rold] = np.empty((0,), dtype=np.int32)

        # Invalidate cached coherence for both old roots — rnew's gene
        # set just changed; rold is now empty. They'll be recomputed on
        # next access.
        root_C_cache.pop(ra, None)
        root_C_cache.pop(rb, None)

        # Boundary expansion: push new candidate edges around rnew.
        # Lazy DSU revalidation at pop handles any duplicate pushes.
        if candidate_source == "delaunay":
            # Reuse original node adjacency via a and b endpoints.
            for nbr in (adj[a] + adj[b]):
                rn = dsu.find(nbr)
                rr = dsu.find(rnew)
                if rn == rr:
                    continue
                if not can_merge(rr, rn):
                    continue
                dtry = compute_deltaC_roots(rr, rn)
                if np.isfinite(dtry) and dtry >= deltaC_min:
                    heapq.heappush(heap, _heap_item(dtry, rr, rn))
        else:  # candidate_source == "grid"
            # No explicit boundary expansion needed: in grid mode, the
            # initial candidate enumeration is comprehensive (every
            # spatially-adjacent component pair is in the heap), and
            # lazy DSU revalidation at pop handles merges. Skipping
            # expansion + index maintenance is both correct and avoids
            # an O(|bins(rnew)| * 9) scan per merge.
            pass

    # choose stitched label per final root with priority: cell > partial > component
    root_to_label = {}
    for i in range(N):
        r = dsu.find(i)
        if r in root_to_label:
            continue
        if cell_ids[r]:
            label = sorted(cell_ids[r])[0]          # deterministic
        elif partial_ids[r]:
            label = sorted(partial_ids[r])[0]
        else:
            label = sorted(comp_ids[r])[0]
        root_to_label[r] = label

    entity_to_stitched = {entity_ids[i]: root_to_label[dsu.find(i)] for i in range(N)}
    info = {
        "root_to_label": root_to_label,
        "coherence_cache_hits": cache_hits,
        "coherence_cache_misses": cache_misses,
    }
    return entity_to_stitched, info

def apply_stitching_to_transcripts(
    df_final: pd.DataFrame,
    aux: dict,
    *,
    entity_col="cell_id_final",   # final id column
    gene_col="feature_name",
    coord_cols=("x", "y", "z"),
    mode: str = "count",
    threshold: float = 0.05,
    penalize_simplicity=True,
    deltaC_min=0.0,
    use_3d=True,
    out_col="cell_id_stitched",
    purity_threshold=_LEGACY_STITCH_KWARG_SENTINEL,
    tau=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relu=_LEGACY_STITCH_KWARG_SENTINEL,
):
    _ensure_reproducibility_seed()
    # build entity table (centroids + genes)
    summary = build_entity_table(
        df_final,
        entity_col=entity_col,
        gene_col=gene_col,
        coord_cols=coord_cols,
    )

    # rename centroid cols to x,y,z expected by stitching function
    # (build_entity_table keeps original names)
    if tuple(coord_cols) == ("x", "y", "z"):
        summary = summary.rename(columns={"x": "x", "y": "y", "z": "z"})
    else:
        # if different coordinate column names used, map them:
        summary = summary.rename(columns={coord_cols[0]: "x", coord_cols[1]: "y", coord_cols[2]: "z"})

    legacy_kwargs = {}
    if purity_threshold is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["purity_threshold"] = purity_threshold
    if tau is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["tau"] = tau
    if use_relu is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["use_relu"] = use_relu

    # stitch entities
    entity_to_stitched, info = stitch_entities_hierarchical(
        summary_df=summary.rename(columns={"entity_id": "entity_id"}),
        aux=aux,
        mode=mode,
        threshold=threshold,
        penalize_simplicity=penalize_simplicity,
        deltaC_min=deltaC_min,
        use_3d=use_3d,
        dist_threshold=None,
        **legacy_kwargs,
    )

    # map back to transcripts
    df_out = df_final.copy()
    ent = df_out[entity_col].astype(str)

    # default: keep original entity label (DROP stays DROP)
    df_out[out_col] = ent

    # apply stitched labels to non-drop entities
    mask = ent.notna() & (ent != "DROP") & (ent != "nan")
    df_out.loc[mask, out_col] = ent[mask].map(entity_to_stitched).fillna(ent[mask])

    return df_out, entity_to_stitched


def apply_stitching_to_transcripts_fast(
    df_final: pd.DataFrame,
    aux: dict,
    *,
    entity_col="cell_id_final",
    gene_col="feature_name",
    coord_cols=("x", "y", "z"),
    mode: str = "count",
    threshold: float = 0.05,
    penalize_simplicity=True,
    deltaC_min=0.0,
    use_3d=True,
    out_col="cell_id_stitched",
    show_progress: bool = True,
    purity_threshold=_LEGACY_STITCH_KWARG_SENTINEL,
    tau=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relu=_LEGACY_STITCH_KWARG_SENTINEL,
):
    """
    Fast wrapper around `apply_stitching_to_transcripts`.
    - Builds entity table and runs hierarchical stitching, with optional progress bars.
    - Uses ReLU-based coherence scoring by default for robust cluster merging.
    - Returns same outputs as original function.

    Parameters
    ----------
    df_final : pd.DataFrame
        Transcript-level data with entity assignments
    aux : dict
        Contains NPMI matrix ("W") and gene mapping ("gene_to_idx")
    entity_col : str
        Column with current entity labels
    gene_col : str
        Column with gene names
    coord_cols : tuple
        Coordinate column names
    purity_threshold : float
        Threshold for original scoring (used if use_relu=False)
    tau : float
        Dead-zone threshold for ReLU (used if use_relu=True, default)
    use_relu : bool
        If True, use ReLU-based coherence (default, faster and more robust)
    penalize_simplicity : bool
        Penalize smaller gene sets in deltaC
    deltaC_min : float
        Minimum deltaC for merging
    use_3d : bool
        Use 3D coordinates
    out_col : str
        Output column name
    show_progress : bool
        Show progress bar

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with stitched labels
    entity_to_stitched : dict
        Mapping from original to stitched entity IDs
    """
    _ensure_reproducibility_seed()
    # build entity table (centroids + genes)
    if show_progress:
        # small progress step for entity build
        pbar = tqdm(total=2, desc="stitching")
    else:
        pbar = None

    summary = build_entity_table(
        df_final,
        entity_col=entity_col,
        gene_col=gene_col,
        coord_cols=coord_cols,
    )
    if pbar is not None:
        pbar.update(1)

    # rename centroid cols if necessary
    if tuple(coord_cols) == ("x", "y", "z"):
        summary = summary.rename(columns={"x": "x", "y": "y", "z": "z"})
    else:
        summary = summary.rename(columns={coord_cols[0]: "x", coord_cols[1]: "y", coord_cols[2]: "z"})

    legacy_kwargs = {}
    if purity_threshold is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["purity_threshold"] = purity_threshold
    if tau is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["tau"] = tau
    if use_relu is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["use_relu"] = use_relu

    # stitch entities (this is the heavy op)
    entity_to_stitched, info = stitch_entities_hierarchical(
        summary_df=summary.rename(columns={"entity_id": "entity_id"}),
        aux=aux,
        mode=mode,
        threshold=threshold,
        penalize_simplicity=penalize_simplicity,
        deltaC_min=deltaC_min,
        use_3d=use_3d,
        dist_threshold=None,
        **legacy_kwargs,
    )

    if pbar is not None:
        pbar.update(1)
        pbar.close()

    # map back to transcripts using vectorized numpy lookup (much faster than pandas.map())
    df_out = df_final.copy()
    ent = df_out[entity_col].astype(str)
    df_out[out_col] = ent

    mask = ent.notna() & (ent != "DROP") & (ent != "nan")

    if mask.sum() > 0:
        # Fully vectorized mapping using pandas.Series.map() (much faster than loop)
        ent_values = ent[mask]

        # Convert dict to pandas Series for vectorized .map()
        mapping_series = pd.Series(entity_to_stitched)

        # Vectorized map with fillna for unmapped values (keeps original)
        stitched_values = ent_values.map(mapping_series).fillna(ent_values)

        # Single assignment
        df_out.loc[mask, out_col] = stitched_values

    return df_out, entity_to_stitched


def apply_stitching_to_transcripts_memory_efficient(
    df_final: pd.DataFrame,
    aux: dict,
    *,
    entity_col: str = "tracer_id",
    gene_col: str = "feature_name",
    coord_cols=("x", "y", "z"),
    mode: str = "count",
    threshold: float = 0.05,
    metric: str = "npmi",
    penalize_simplicity: bool = True,
    deltaC_min: float = 0.0,
    use_3d: bool = True,
    dist_threshold: float | None = 15.0,
    out_col: str = "tracer_id",
    debug_stages: bool = False,
    debug_legacy_col: str = "cell_id_stitched",
    show_progress: bool = True,
    in_place: bool = False,
    map_mode: str = "categorical",
    chunk_size: int | None = 2_000_000,
    candidate_source: str = "delaunay",
    G: float | None = None,
    stitch_neighborhood: str = "8",
    purity_threshold=_LEGACY_STITCH_KWARG_SENTINEL,
    tau=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relu=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relative=_LEGACY_STITCH_KWARG_SENTINEL,
):
    """
    Memory-efficient stitching wrapper optimized for very large datasets (10M+ rows).

    This function mirrors `apply_stitching_to_transcripts_fast` but minimizes
    temporary allocations when mapping stitched labels back to transcripts.

    Parameters
    ----------
    df_final : pd.DataFrame
        Transcript-level data with entity assignments
    aux : dict
        Contains NPMI matrix ("W") and gene mapping ("gene_to_idx")
    entity_col : str
        Column with current entity labels
    gene_col : str
        Column with gene names
    coord_cols : tuple
        Coordinate column names
    purity_threshold : float
        Threshold for original scoring (used if use_relu=False)
    tau : float
        Dead-zone threshold for ReLU (used if use_relu=True, default)
    use_relu : bool
        If True, use ReLU-based coherence (default, faster and more robust)
    use_relative : bool
        If True (and use_relu=True), use relative_purity and
        relative_conflict for stitching.
    penalize_simplicity : bool
        Penalize smaller gene sets in deltaC
    deltaC_min : float
        Minimum deltaC for merging
    use_3d : bool
        Use 3D coordinates
    out_col : str
        Output column name
    show_progress : bool
        Show progress bar
    in_place : bool
        If True, write output to the input DataFrame without copying
    map_mode : {"categorical", "chunked"}
        Mapping strategy to minimize memory use.
        - "categorical": map category codes (fast, low memory)
        - "chunked": map in chunks using pandas Series.map()
    chunk_size : int or None
        Chunk size for "chunked" mapping. None maps all at once.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with stitched labels
    entity_to_stitched : dict
        Mapping from original to stitched entity IDs
    """
    _ensure_reproducibility_seed()
    if show_progress:
        pbar = tqdm(total=2, desc="stitching")
    else:
        pbar = None

    summary = build_entity_table(
        df_final,
        entity_col=entity_col,
        gene_col=gene_col,
        coord_cols=coord_cols,
    )
    if pbar is not None:
        pbar.update(1)

    if tuple(coord_cols) == ("x", "y", "z"):
        summary = summary.rename(columns={"x": "x", "y": "y", "z": "z"})
    else:
        summary = summary.rename(columns={coord_cols[0]: "x", coord_cols[1]: "y", coord_cols[2]: "z"})

    # Build per-transcript inputs for grid candidate enumeration if requested.
    transcript_coords = None
    transcript_entity_codes = None
    if candidate_source == "grid":
        # Map each transcript's entity string to its row index in summary_df.
        entity_id_arr = summary["entity_id"].astype(str).to_numpy()
        entity_to_idx = {eid: i for i, eid in enumerate(entity_id_arr)}
        ent_str = df_final[entity_col].astype(str).to_numpy()
        transcript_entity_codes = np.fromiter(
            (entity_to_idx.get(e, -1) for e in ent_str),
            dtype=np.int64,
            count=len(ent_str),
        )
        transcript_coords = df_final[[coord_cols[0], coord_cols[1]]].to_numpy(dtype=np.float64)

    legacy_kwargs = {}
    if purity_threshold is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["purity_threshold"] = purity_threshold
    if tau is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["tau"] = tau
    if use_relu is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["use_relu"] = use_relu
    if use_relative is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["use_relative"] = use_relative

    entity_to_stitched, info = stitch_entities_hierarchical(
        summary_df=summary.rename(columns={"entity_id": "entity_id"}),
        aux=aux,
        mode=mode,
        threshold=threshold,
        metric=metric,
        penalize_simplicity=penalize_simplicity,
        deltaC_min=deltaC_min,
        use_3d=use_3d,
        dist_threshold=dist_threshold,
        candidate_source=candidate_source,
        G=G,
        stitch_neighborhood=stitch_neighborhood,
        transcript_coords=transcript_coords,
        transcript_entity_codes=transcript_entity_codes,
        **legacy_kwargs,
    )

    if pbar is not None:
        pbar.update(1)
        pbar.close()

    df_out = df_final if in_place else df_final.copy()
    ent = df_out[entity_col]

    if map_mode == "categorical":
        ent_cat = ent.astype("category")
        categories = ent_cat.cat.categories.astype(str)
        mapped_categories = pd.Index(categories).map(lambda x: entity_to_stitched.get(x, x))

        # Fast path: one-to-one mapping (no merges) -> just rename categories
        if mapped_categories.is_unique:
            df_out[out_col] = ent_cat.cat.rename_categories(mapped_categories)
        else:
            # Slow path: merges exist, recode via factorization
            new_cat_codes, new_categories = pd.factorize(mapped_categories, sort=False)
            ent_codes = ent_cat.cat.codes.to_numpy(copy=False)

            out_codes = np.full_like(ent_codes, -1)
            valid = ent_codes >= 0
            if valid.any():
                out_codes[valid] = new_cat_codes[ent_codes[valid]]

            df_out[out_col] = pd.Categorical.from_codes(out_codes, categories=new_categories)
        if debug_stages and debug_legacy_col != out_col:
            df_out[debug_legacy_col] = df_out[out_col].copy()
        return df_out, entity_to_stitched
    elif map_mode == "chunked":
        ent_str = ent.astype(str)
        df_out[out_col] = ent_str

        mask = ent_str.notna() & (ent_str != "DROP") & (ent_str != "nan")
        if mask.any():
            idx = np.flatnonzero(mask.to_numpy())
            mapping_series = pd.Series(entity_to_stitched)

            if chunk_size is None:
                vals = ent_str.iloc[idx]
                mapped = vals.map(mapping_series).fillna(vals)
                df_out.iloc[idx, df_out.columns.get_loc(out_col)] = mapped.to_numpy()
            else:
                for start in range(0, len(idx), chunk_size):
                    end = start + chunk_size
                    sel = idx[start:end]
                    vals = ent_str.iloc[sel]
                    mapped = vals.map(mapping_series).fillna(vals)
                    df_out.iloc[sel, df_out.columns.get_loc(out_col)] = mapped.to_numpy()
    else:
        raise ValueError("map_mode must be 'categorical' or 'chunked'")

    if debug_stages and debug_legacy_col != out_col:
        df_out[debug_legacy_col] = df_out[out_col].copy()
    return df_out, entity_to_stitched
