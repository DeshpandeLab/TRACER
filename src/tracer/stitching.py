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
from .graph import _BIN_BIAS, bin_xy, delaunay_edges, neighbor_bins, unpack_bin


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
    # Unassigned sentinels — must be checked BEFORE the partial-by-hyphen
    # rule, otherwise "-1" gets misclassified as a partial (because it
    # contains a hyphen) and Stitch tries to merge a phantom "-1
    # partial" entity with real cells.
    #
    # The unassigned class includes:
    #   - fixed sentinels: -1, DROP, UNASSIGNED, nan
    #   - stage-rejected diagnostics: *_rejected (prune_rejected,
    #     group_rejected, demote_rejected — see spatial.UNASSIGNED_LABELS)
    # All map to "unknown" so the cell/partial/component whitelist at the
    # call sites uniformly excludes them — no label-specific filter
    # downstream needs updating when a new stage-rejection sentinel is
    # added.
    if s in ("-1", "DROP", "UNASSIGNED", "nan") or s.endswith("_rejected"):
        return "unknown"
    if s.startswith("UNASSIGNED_"):
        return "component"
    if "-" in s:
        return "partial"
    # otherwise treat as cell (original)
    return "cell"


def estimate_within_cell_dz_threshold(
    df: pd.DataFrame,
    *,
    entity_col: str,
    z_col: str = "z",
    n_sample: int = 50,
    min_entity_size: int = 5,
    cohens_d_threshold: float = 3.0,
    target_percentile: float = 90.0,
    unimodal_percentile: float = 50.0,
    min_recommended_G_z: float = 1.0,
    seed: int = 42,
) -> dict:
    """Estimate a within-cell |Δz| threshold from segmented input.

    Pools pairwise |Δz| values across a sample of segmented entities,
    fits a 2-component Gaussian mixture model, tests for bimodality
    via Cohen's d on the fitted component means, and returns the
    target percentile of the **smaller-mean component** when bimodal
    (otherwise the percentile of the full distribution).

    The intent: in a noisy DAPI/Voronoi segmentation that merges
    stacked stratum cells, the within-entity |Δz| distribution is
    bimodal — a low-Δz mode (within-stratum tx pairs) and a high-Δz
    mode (cross-stratum tx pairs from the merged column). The smaller-
    mean mode reflects within-cell scale; its right tail (90 %ile by
    default) is a robust upper bound on legitimate within-cell |Δz|
    that downstream stitching can use as a Δz filter threshold.

    On clean, unimodal data (e.g. ground-truth labels), the GMM
    collapses, Cohen's d falls below ``cohens_d_threshold``, and the
    percentile of the full pooled distribution is returned instead.

    Parameters
    ----------
    df : pd.DataFrame
        Transcript-level table with at least ``entity_col`` and
        ``z_col``.
    entity_col : str
        Column whose distinct non-``"-1"`` values define entities.
    z_col : str, default ``"z"``
        Column holding the z coordinate (µm).
    n_sample : int, default 50
        Number of entities to randomly sample. If fewer eligible
        entities exist, all are used.
    min_entity_size : int, default 5
        Skip entities below this transcript count (no meaningful
        pairwise statistic).
    cohens_d_threshold : float, default 3.0
        Cohen's d cutoff between the two GMM components for
        declaring bimodality. d ≥ 3 corresponds to nearly-disjoint
        modes (the means are 3+ pooled std-deviations apart). The
        default is intentionally strict because a unimodal
        triangular distribution (e.g., within-cell |Δz| pairs from
        clean ground-truth cells) trivially splits into two GMM
        components with d ≈ 2 — strict cutoff prevents that
        spurious bimodality from misleading the threshold.
    target_percentile : float in [0, 100], default 90
        Percentile of the **smaller-mean GMM mode** to report as the
        threshold when the data is bimodal. The smaller mode is the
        within-cell distribution (cross-stratum pairs go in the larger
        mode), so its right tail is a robust upper bound on legitimate
        within-cell |Δz|.
    unimodal_percentile : float in [0, 100], default 50
        Percentile of the **full pooled distribution** to report when
        the data is unimodal (Cohen's d below cutoff). The unimodal
        case typically arises from clean segmentation — the right
        tail then includes pathologically z-elongated entities and
        isn't a reliable scale, so the median is a more robust
        within-cell-scale estimate than higher percentiles.
    min_recommended_G_z : float, default 1.0
        Floor for the ``recommended_G_z`` output (in µm). Useful when
        downstream tooling assumes integer-µm bins.
    seed : int, default 42
        Random seed for entity sampling and GMM initialization.

    Returns
    -------
    result : dict with keys
        - ``threshold`` (float, µm): the recommended |Δz| threshold
        - ``bimodal`` (bool): whether Cohen's d ≥ threshold
        - ``cohens_d`` (float): effect size between the two modes
        - ``gmm_means`` (list of 2 floats): fitted component means
        - ``gmm_stds`` (list of 2 floats): fitted component stds
        - ``gmm_weights`` (list of 2 floats): mixing proportions
        - ``smaller_mode_mean`` (float): mean of the smaller-mean mode
        - ``smaller_mode_std`` (float): std of the smaller-mean mode
        - ``smaller_mode_weight`` (float): mixing weight of that mode
        - ``n_sampled_entities`` (int)
        - ``n_pairs`` (int): total within-entity pairs pooled
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as e:
        raise ImportError(
            "estimate_within_cell_dz_threshold requires scikit-learn"
        ) from e

    rng = np.random.default_rng(seed)
    s = df[entity_col].astype(str)
    sizes = df[s != "-1"].groupby(entity_col).size()
    eligible = sizes[sizes >= int(min_entity_size)].index.tolist()
    if not eligible:
        return {
            "threshold": float("nan"), "bimodal": False, "cohens_d": 0.0,
            "gmm_means": [float("nan"), float("nan")],
            "gmm_stds":  [float("nan"), float("nan")],
            "gmm_weights": [float("nan"), float("nan")],
            "smaller_mode_mean": float("nan"),
            "smaller_mode_std":  float("nan"),
            "smaller_mode_weight": float("nan"),
            "recommended_G_z": float("nan"),
            "n_sampled_entities": 0, "n_pairs": 0,
        }

    if len(eligible) > n_sample:
        sampled = rng.choice(eligible, size=n_sample, replace=False).tolist()
    else:
        sampled = eligible

    pooled = []
    for e in sampled:
        z = df.loc[df[entity_col] == e, z_col].to_numpy(dtype=float)
        if len(z) < 2:
            continue
        ii, jj = np.triu_indices(len(z), k=1)
        pooled.append(np.abs(z[ii] - z[jj]))
    arr = np.concatenate(pooled) if pooled else np.empty(0)

    if arr.size < 10:
        return {
            "threshold": float("nan"), "bimodal": False, "cohens_d": 0.0,
            "gmm_means": [float("nan"), float("nan")],
            "gmm_stds":  [float("nan"), float("nan")],
            "gmm_weights": [float("nan"), float("nan")],
            "smaller_mode_mean": float("nan"),
            "smaller_mode_std":  float("nan"),
            "smaller_mode_weight": float("nan"),
            "recommended_G_z": float("nan"),
            "n_sampled_entities": len(sampled), "n_pairs": int(arr.size),
        }

    X = arr.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=int(seed),
                          max_iter=200, n_init=4)
    gmm.fit(X)
    means = gmm.means_.flatten()
    stds = np.sqrt(np.maximum(gmm.covariances_.flatten(), 1e-12))
    weights = gmm.weights_

    pooled_std = float(np.sqrt((stds[0] ** 2 + stds[1] ** 2) / 2))
    cohens_d = float(abs(means[0] - means[1]) / pooled_std) if pooled_std > 0 else 0.0
    bimodal = bool(cohens_d >= float(cohens_d_threshold))

    smaller_idx = int(np.argmin(means))
    if bimodal:
        # Soft-assign every pair to its most likely component, then
        # compute the percentile of pairs assigned to the smaller mode.
        resp = gmm.predict_proba(X)
        in_smaller = resp[:, smaller_idx] >= 0.5
        smaller_arr = arr[in_smaller]
        if smaller_arr.size == 0:
            smaller_arr = arr  # safety
        threshold = float(np.percentile(smaller_arr, float(target_percentile)))
    else:
        threshold = float(np.percentile(arr, float(unimodal_percentile)))

    # Recommended G_z is bimodality-aware:
    #   - unimodal: ceil(threshold), the smallest 1-µm bin still above
    #     within-cell scale. Wide enough to admit cell-spanning merges
    #     at depth=1, narrow enough to bound them.
    #   - bimodal: floor(threshold). The threshold is the smaller-mode
    #     90 %ile (within-cell upper bound); a bin smaller than that
    #     guarantees an empty-bin moat between the within-cell mode
    #     and the cross-stratum mode, which Split & Stitch can refuse
    #     to bridge at depth=1.
    if bimodal:
        recommended_G_z = float(max(float(min_recommended_G_z),
                                    np.floor(threshold)))
    else:
        recommended_G_z = float(max(float(min_recommended_G_z),
                                    np.ceil(threshold)))

    return {
        "threshold": threshold,
        "bimodal": bimodal,
        "cohens_d": cohens_d,
        "gmm_means": means.tolist(),
        "gmm_stds":  stds.tolist(),
        "gmm_weights": weights.tolist(),
        "smaller_mode_mean":   float(means[smaller_idx]),
        "smaller_mode_std":    float(stds[smaller_idx]),
        "smaller_mode_weight": float(weights[smaller_idx]),
        "recommended_G_z":     recommended_G_z,
        "n_sampled_entities":  int(len(sampled)),
        "n_pairs":             int(arr.size),
    }


def compute_within_entity_dz_stats(
    df: pd.DataFrame,
    *,
    entity_col: str,
    z_col: str = "z",
    etype_filter: tuple[str, ...] | None = ("cell",),
    min_entity_size: int = 5,
    percentiles: tuple[float, ...] = (50, 75, 90, 95, 99),
) -> dict[str, float]:
    """Pool within-entity pairwise |Δz| across all entities, return stats.

    Used to derive a data-driven Δz threshold for stitching's
    ``min_close_edges_dz`` guard: any cross-component candidate pair whose
    z-spread exceeds the within-cell scale is geometrically unlikely to
    be same-cell and can be filtered before agglomerative scoring.

    Parameters
    ----------
    df : pd.DataFrame
        Transcript-level table with at least ``entity_col`` and ``z_col``.
    entity_col : str
        Column whose distinct non-``"-1"`` values define entities.
    z_col : str, default ``"z"``
        Column holding the z coordinate (µm).
    etype_filter : tuple of {"cell", "partial", "component"} or None
        Restrict the pool to entities of these types (per
        :func:`infer_entity_type`). Pass ``None`` to include all
        non-``"-1"`` entities. Default ``("cell",)`` — cells are the
        most representative reference scale.
    min_entity_size : int
        Skip entities with fewer than this many transcripts (no
        pairwise statistic).
    percentiles : tuple of float
        Percentiles to report alongside the median. Values in [0, 100].

    Returns
    -------
    stats : dict
        Keys: ``n_entities`` (int), ``n_pairs`` (int), ``median`` (float),
        ``mean`` (float), ``max`` (float), and one entry per requested
        percentile, e.g. ``"p75"``. All distances in same units as
        ``z_col`` (typically µm). Returns NaN-filled dict if no data.
    """
    s = df[entity_col].astype(str)
    keep = s != "-1"
    if etype_filter is not None:
        # Prefer the upstream-emitted _etype column when present
        # (correct on FFPE cell_ids). Fall back to label-string parsing
        # for back-compat.
        if "_etype" in df.columns:
            types = df["_etype"].astype(str)
        else:
            types = s.map(infer_entity_type)
        keep = keep & types.isin(etype_filter)
    sub = df[keep]
    pooled: list[np.ndarray] = []
    n_kept = 0
    for _, g in sub.groupby(entity_col, sort=False):
        if len(g) < max(2, int(min_entity_size)):
            continue
        z = g[z_col].to_numpy(dtype=float)
        ii, jj = np.triu_indices(len(z), k=1)
        pooled.append(np.abs(z[ii] - z[jj]))
        n_kept += 1
    if not pooled:
        out = {"n_entities": 0, "n_pairs": 0,
               "median": float("nan"), "mean": float("nan"),
               "max": float("nan")}
        for p in percentiles:
            out[f"p{int(p)}"] = float("nan")
        return out
    arr = np.concatenate(pooled)
    out = {
        "n_entities": int(n_kept),
        "n_pairs": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }
    for p in percentiles:
        out[f"p{int(p)}"] = float(np.percentile(arr, p))
    return out


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

    # entity type — prefer the upstream-emitted `_etype` column when
    # present (correct on Xenium FFPE / IO cell_ids). Fall back to the
    # label-string parser for back-compat on input frames without _etype.
    if "_etype" in df_final.columns:
        df["_etype"] = df_final.loc[keep, "_etype"].astype(str).to_numpy()
    else:
        df["_etype"] = df[entity_col].map(infer_entity_type)
    df = df[df["_etype"].isin(["cell", "partial", "component"])]

    # centroid (`observed=True` avoids processing empty categorical groups
    # when entity_col is categorical).
    grouped_coords = df.groupby(entity_col, sort=True, observed=True)[list(coord_cols)]
    cent = grouped_coords.mean()
    # Per-axis min/max — used by the spatial centroid-in-bbox gate at
    # Stitch time. Cheap O(N_tx) extra pass.
    bbox_min = grouped_coords.min().rename(columns={c: f"{c}_min" for c in coord_cols})
    bbox_max = grouped_coords.max().rename(columns={c: f"{c}_max" for c in coord_cols})

    # unique genes per entity (sorted for deterministic downstream mapping)
    genes = df.groupby(entity_col, sort=True, observed=True)[gene_col].unique()
    genes = genes.apply(lambda arr: np.sort(arr.astype(str)))

    etype = df.groupby(entity_col, observed=True)["_etype"].first()

    # Per-entity tx count — used as the "size" in the asymmetric
    # smaller-inside-larger spatial test.
    n_tx = df.groupby(entity_col, observed=True)[gene_col].size().rename("n_tx")

    summary = (
        cent.join(bbox_min).join(bbox_max)
            .join(genes.rename("genes"))
            .join(etype.rename("etype"))
            .join(n_tx)
    )
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

    # Fast path: count-mode + dense float32 W → Cython kernel.
    # ~5-10× per-call speedup vs numpy at ROI/full scale.
    if mode == "count":
        try:
            import scipy.sparse as _sp
            if not _sp.issparse(npmi_mat) and isinstance(npmi_mat, np.ndarray) \
               and npmi_mat.dtype == np.float32:
                from . import _cy_prune
                gids32 = np.ascontiguousarray(gene_ids, dtype=np.int32)
                C, purity, conflict = _cy_prune.coherence_count_kernel(
                    gids32, npmi_mat, float(threshold)
                )
                return float(C), float(purity), float(conflict)
        except (ImportError, AttributeError):
            pass  # fall through to numpy path

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


# Diagnostic counters populated by stitch_entities_hierarchical when the
# spatial-centroid bypass gate is active. Reset at the start of each
# call. Read by callers after the call returns (e.g. CLI / sweep tooling
# that wants to log how many pairs the gate captured).
_LAST_GATE_STATS: dict[str, int] = {}


def _stitch_entities_hierarchical_decomposable(
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
    G_z: float | None = None,
    z_neighbor_depth: int = 0,
    transcript_coords: np.ndarray | None = None,
    transcript_entity_codes: np.ndarray | None = None,
    min_candidate_edges: int | str = 0,
    # Optional per-entity-witness count: drop candidate pair (E1, E2)
    # unless EACH entity contributes at least `min_local_tx_per_entity`
    # unique tx in the shared bin neighborhood (xy 8-Moore + ±depth z
    # bins). Catches single-bridging-tx candidates that sneak through
    # the diagonal-Moore reach (~5.66 µm at G=2). Symmetric in (E1, E2)
    # — resistant to mass-dominated cross-product counts.
    # Default 0 = off (current behavior unchanged).
    min_local_tx_per_entity: int = 0,
    max_pair_median_dz: float | None = None,
    min_close_edges_dz: float | None = None,
    min_close_edges_n: int = 0,
):
    """**EXPERIMENTAL — opt-in via `use_decomposable_stitch=True`.**

    Lazy DSU + max-heap greedy with decomposable coherence primitives.
    Algorithmic complexity: O(M log N) instead of the eager path's
    O(rounds × candidate_pairs). Designed for tissue-scale (200k+
    entities) where the eager path becomes the dominant runtime.

    Strategy summary (full design + math validation in
    `/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/TODO.md`):
      1. Pre-compute per-original `(n_above, n_below, n_finite)` and
         per-spatial-pair cross primitives.
      2. DSU groups + max-heap of candidate-pair ΔC values.
      3. On heap pop: check DSU root staleness; if stale, recompute
         from current primitives and reinsert.
      4. On merge: combine running primitive sums + cross-sums to all
         neighbour groups; push fresh ΔC entries for new candidate
         pairs to the heap.

    Bit-match expectation:
      Per-call ΔC values are bit-equivalent to the eager path
      (validated on 1000 µm ROI: 71k calls, 0 mismatches). The merge
      sequence may differ on exact ties due to FP rounding in the
      cross-segment arithmetic, but final entity-to-stitched parity
      matches the eager output to within ~0.001 ARI in practice.

    Implementation strategy:
      1. Reuse the eager path's setup (candidate-pair build, filters,
         centroids, gene-id mapping) — call `stitch_entities_hierarchical`
         in a flag-disambiguated mode that returns just the prepared
         state. Implementation here re-does the prep inline to avoid
         a refactor of the existing function.
      2. Maintain per-DSU-root primitive sums (n_above, n_below,
         n_finite) accumulated across all merges in that root.
      3. For each candidate pair: compute ΔC by combining roots'
         current primitive sums plus a fresh cross-segment computation
         for the merge boundary. No re-iteration of the union's full
         gene-pair set.
      4. On merge: update primitive sums by adding the cross
         contribution; gene set is the union of the two roots.

    The cross computation uses the 6-segment decomposition validated
    in `/tmp/validate_decomp_coh.py`:
        triu(A∪B) = triu(A−B) + triu(B−A) + triu(A∩B)
                   + cross(A−B, B−A) + cross(A−B, A∩B) + cross(B−A, A∩B)

    Implementation is integrated into `stitch_entities_hierarchical`
    directly (see the `if use_decomposable_stitch …` branches inside
    `C_of_root`, `compute_deltaC_roots`, and the merge step). This
    helper is a thin wrapper that simply forwards with the flag set.
    """
    return stitch_entities_hierarchical(
        summary_df=summary_df, aux=aux,
        mode=mode, threshold=threshold, metric=metric,
        penalize_simplicity=penalize_simplicity, deltaC_min=deltaC_min,
        use_3d=use_3d, dist_threshold=dist_threshold,
        candidate_source=candidate_source, G=G,
        stitch_neighborhood=stitch_neighborhood,
        G_z=G_z, z_neighbor_depth=z_neighbor_depth,
        transcript_coords=transcript_coords,
        transcript_entity_codes=transcript_entity_codes,
        min_candidate_edges=min_candidate_edges,
        min_local_tx_per_entity=min_local_tx_per_entity,
        max_pair_median_dz=max_pair_median_dz,
        min_close_edges_dz=min_close_edges_dz,
        min_close_edges_n=min_close_edges_n,
        use_decomposable_stitch=True,  # actually invoke the primitive path
    )


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
    G_z: float | None = None,
    z_neighbor_depth: int = 0,
    transcript_coords: np.ndarray | None = None,
    transcript_entity_codes: np.ndarray | None = None,
    min_candidate_edges: int | str = 0,
    # Optional per-entity-witness count: drop candidate pair (E1, E2)
    # unless EACH entity contributes at least `min_local_tx_per_entity`
    # unique tx in the shared bin neighborhood (xy 8-Moore + ±depth z
    # bins). Catches single-bridging-tx candidates that sneak through
    # the diagonal-Moore reach (~5.66 µm at G=2). Symmetric in (E1, E2)
    # — resistant to mass-dominated cross-product counts.
    # Default 0 = off (current behavior unchanged).
    min_local_tx_per_entity: int = 0,
    max_pair_median_dz: float | None = None,
    min_close_edges_dz: float | None = None,
    min_close_edges_n: int = 0,
    # Deprecated kwargs — translated to mode/threshold below.
    purity_threshold=_LEGACY_STITCH_KWARG_SENTINEL,
    tau=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relu=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relative=_LEGACY_STITCH_KWARG_SENTINEL,
    # Experimental: lazy DSU+max-heap merge with decomposable coherence
    # primitives. Validated bit-match on 1000 µm ROI (71k coh calls,
    # 894 merges, 0 mismatches) but not yet on full-tissue scale.
    # Default False (use the existing eager-recompute greedy).
    use_decomposable_stitch: bool = False,
    # Experimental: top-K positive-clique fast-gate for candidate pairs
    # at heap-init. For each entity, precompute its K signature genes
    # (highest sum of positive PMI to others in same entity). For each
    # candidate pair (i, j), scan the K×K cross-PMI block: if ANY entry
    # is < neg_npmi_threshold, REJECT the pair without computing its
    # full ΔC. Cuts heap-init Python-loop overhead by skipping the
    # majority of cell-pair candidates (most are biologically
    # incompatible). 0 = disabled (no behavior change). Recommended
    # K = 3-5 for empirical bit-match.
    fast_gate_top_k: int = 0,
    fast_gate_mean_threshold: float = 0.0,
    # Optional: per-entity tx counts. Used when multi-partial mergers
    # tie on suffix → majority-tx-count winner gets the merged label.
    # If None, falls back to lexicographic tiebreak.
    entity_n_tx: dict[str, int] | None = None,
    # Optional spatial centroid-in-bbox bypass at merge time. When True,
    # candidate pairs where the SMALLER entity's centroid lies inside
    # the LARGER entity's per-axis tx-coord range are MERGED without
    # PMI evaluation (positive override / Tier-1 in the 3-tier cascade).
    # Default False (no spatial bypass; standard ΔC-driven merging).
    spatial_centroid_gate: bool = False,
    # Tightness of the spatial-overlap test. K=1 → bbox check; K≥2 →
    # require K tx coords above AND K below smaller's centroid per
    # axis. Higher K → stricter (more interior).
    spatial_centroid_k: int = 1,
    # Per-entity tx-coord arrays. Required for K≥2.
    # dict[entity_id -> (n_tx, n_dim) ndarray].
    entity_tx_coords: dict | None = None,
    # Spatial gate mode (only when spatial_centroid_gate=True):
    #   "pre"  — current behavior: spatial bypass returns sentinel ΔC,
    #            so spatial-overlap pairs MERGE FIRST regardless of ΔC.
    #            Spatial overrides any ΔC verdict, including rejections.
    #   "post" — spatial gate fires only as a tiebreaker on ΔC-rejected
    #            pairs. ΔC takes priority for accepting and ranking; if
    #            a pair fails the ΔC test (dc < deltaC_min), THEN check
    #            the spatial gate; if centroids match, merge anyway.
    #            More conservative — lets ΔC do its job, only uses
    #            spatial as a fallback for marginal-but-co-located pairs.
    spatial_gate_mode: str = "pre",
    # Flipped spatial test: instead of "smaller's centroid inside larger's
    # tx cloud", check "larger's centroid inside smaller's tx cloud".
    # Effective K is dynamically capped at floor(n_smaller / 3) so small
    # partials use a lighter K. This is the right test for detecting
    # whether the cell's tx are arranged AROUND the partial (i.e., the
    # partial is a real fragment of the cell), as opposed to the partial
    # being embedded INSIDE the cell (the contamination case).
    spatial_gate_flipped: bool = False,
):
    """Hierarchical entity stitching driven by ΔC.

    The optional ``min_candidate_edges`` kwarg filters candidate pairs
    by the number of supporting transcript-level cross-bin connections.
    A pair (A, B) is admitted only when at least
    ``min_candidate_edges`` transcript pairs (tx_a in A, tx_b in B) lie
    in candidate bin neighborhoods. Pass an integer for a fixed
    threshold or the string ``"min"`` for a per-pair adaptive
    threshold of ``min(n_A, n_B)`` where n_X is the entity tx count.
    Only meaningful when ``candidate_source='grid'``.

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

    # When use_decomposable_stitch=True, the merge loop below uses the
    # `_compute_deltaC_via_primitives` helper instead of recomputing
    # coherence(union) from scratch on every ΔC eval. All other setup
    # (candidate-pair build, filters, DSU, max-heap, lazy stale-pop)
    # is shared with the eager path. See `_stitch_entities_hierarchical_decomposable`
    # docstring for the algorithm rationale + bit-match expectation.
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
        if stitch_neighborhood not in ("0", "4", "8"):
            raise ValueError(
                f"stitch_neighborhood must be '0', '4' or '8' (got {stitch_neighborhood!r})"
            )
        if z_neighbor_depth < 0:
            raise ValueError(f"z_neighbor_depth must be ≥ 0 (got {z_neighbor_depth})")
        if z_neighbor_depth > 0 and G_z is None:
            raise ValueError(
                "z_neighbor_depth > 0 requires G_z to be set"
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
        # Bin keys are either int64 (xy-only, packed) or (xy_int64, bz_int)
        # tuples (xyz). Tuple keys cost a small dict-overhead penalty but
        # the entity counts at stitch time are moderate.
        ec = np.asarray(transcript_entity_codes, dtype=np.int64)
        valid = ec >= 0
        xy_keys = bin_xy(transcript_coords[:, :2], G)[valid]
        comp_codes = ec[valid]
        if G_z is not None:
            if transcript_coords.shape[1] < 3:
                raise ValueError(
                    "G_z requires transcript_coords to have a z column"
                )
            bz_arr = np.floor(
                transcript_coords[valid, 2] / float(G_z)
            ).astype(np.int64)
            bin_keys = list(zip(xy_keys.tolist(), bz_arr.tolist()))
        else:
            bin_keys = xy_keys.tolist()

        bin_to_comps = defaultdict(set)
        # Per-(bin, entity) tx counts so we can weight candidate edges
        # by the supporting tx-tx pair count for the optional
        # min_candidate_edges filter. Same memory order as bin_to_comps.
        bin_to_comp_counts: dict = defaultdict(lambda: defaultdict(int))
        for bk, c in zip(bin_keys, comp_codes.tolist()):
            bin_to_comps[bk].add(c)
            bin_to_comp_counts[bk][c] += 1
        # Total tx per entity (for min_candidate_edges='min' mode)
        entity_tx_total: dict[int, int] = defaultdict(int)
        for c in comp_codes.tolist():
            entity_tx_total[c] += 1

        # Half-neighborhood directions in xy. "0" → empty (same-bin pairs only).
        if stitch_neighborhood == "0":
            xy_half_offsets: tuple[tuple[int, int], ...] = ()
        elif stitch_neighborhood == "4":
            xy_half_offsets = ((0, 1), (1, 0))
        else:  # "8"
            xy_half_offsets = ((0, 1), (1, -1), (1, 0), (1, 1))

        # z-offsets for the candidate-enumeration window. We use a half-
        # window in z to avoid enumerating each unordered (bin_a, bin_b)
        # pair twice: positive dz only, plus dz=0 for in-plane neighbors.
        # For z_neighbor_depth=0, z is a "same-bin only" partition.
        if G_z is None:
            z_offsets_with_dz0: list[int] = [0]
            z_offsets_strict_pos: list[int] = []
        else:
            z_offsets_with_dz0 = list(range(-z_neighbor_depth, z_neighbor_depth + 1))
            z_offsets_strict_pos = list(range(1, z_neighbor_depth + 1))

        candidate_pairs: set[tuple[int, int]] = set()
        # Tx-tx supporting-edge count per candidate pair, used to apply
        # the optional min_candidate_edges filter below.
        pair_tx_edges: dict[tuple[int, int], int] = defaultdict(int)

        # Per-entity unique-bin tracking for the optional
        # `min_local_tx_per_entity` filter. Maps (lo, hi) → set of bin
        # keys where lo (resp. hi) contributed. Empty when the filter
        # is off (no memory cost).
        track_local = int(min_local_tx_per_entity) > 0
        pair_lo_bins: dict[tuple[int, int], set] = (
            defaultdict(set) if track_local else {}
        )
        pair_hi_bins: dict[tuple[int, int], set] = (
            defaultdict(set) if track_local else {}
        )

        def _record(a, b, n_tx, bk_a=None, bk_b=None):
            if a == b or n_tx <= 0:
                return
            if a < b:
                lo, hi = a, b
                bk_lo, bk_hi = bk_a, bk_b
            else:
                lo, hi = b, a
                bk_lo, bk_hi = bk_b, bk_a
            candidate_pairs.add((lo, hi))
            pair_tx_edges[(lo, hi)] += int(n_tx)
            if track_local and bk_lo is not None and bk_hi is not None:
                pair_lo_bins[(lo, hi)].add(bk_lo)
                pair_hi_bins[(lo, hi)].add(bk_hi)

        for bk, comps in bin_to_comps.items():
            cc_a = bin_to_comp_counts[bk]
            # within-bin: all unordered pairs of distinct components
            if len(comps) > 1:
                for a, b in itertools.combinations(sorted(comps), 2):
                    _record(a, b, cc_a[a] * cc_a[b], bk_a=bk, bk_b=bk)

            # Cross-bin: emit each unordered (bin_a, bin_b) exactly once.
            # In the 2D path we use the "half" xy offsets at dz=0.
            # In the 3D path we use:
            #   - half xy offsets at every dz in [-depth..+depth]
            #   - full xy (offset 0,0) at strictly-positive dz only
            # so each unordered xy/z bin pair is enumerated once.
            if G_z is None:
                xy_packed = bk
                bx, by = unpack_bin(xy_packed)
                for dx, dy in xy_half_offsets:
                    nb_xy_int = int(
                        (np.int64(bx + dx + _BIN_BIAS) << np.int64(32))
                        | np.int64(by + dy + _BIN_BIAS)
                    )
                    nb_comps = bin_to_comps.get(nb_xy_int)
                    if not nb_comps:
                        continue
                    cc_b = bin_to_comp_counts[nb_xy_int]
                    for a in comps:
                        for b in nb_comps:
                            _record(a, b, cc_a[a] * cc_b[b],
                                    bk_a=bk, bk_b=nb_xy_int)
            else:
                xy_packed, bz = bk
                bx, by = unpack_bin(xy_packed)
                # (a) half xy offsets across all dz (incl. dz=0)
                for dx, dy in xy_half_offsets:
                    for dz in z_offsets_with_dz0:
                        nb_xy_int = int(
                        (np.int64(bx + dx + _BIN_BIAS) << np.int64(32))
                        | np.int64(by + dy + _BIN_BIAS)
                    )
                        nb_key = (nb_xy_int, bz + dz)
                        nb_comps = bin_to_comps.get(nb_key)
                        if not nb_comps:
                            continue
                        cc_b = bin_to_comp_counts[nb_key]
                        for a in comps:
                            for b in nb_comps:
                                _record(a, b, cc_a[a] * cc_b[b],
                                        bk_a=bk, bk_b=nb_key)
                # (b) same xy bin (dx=dy=0), strictly-positive dz
                for dz in z_offsets_strict_pos:
                    nb_key = (xy_packed, bz + dz)
                    nb_comps = bin_to_comps.get(nb_key)
                    if not nb_comps:
                        continue
                    cc_b = bin_to_comp_counts[nb_key]
                    for a in comps:
                        for b in nb_comps:
                            _record(a, b, cc_a[a] * cc_b[b],
                                    bk_a=bk, bk_b=nb_key)

        # Optional minimum-supporting-edges filter.
        if min_candidate_edges:
            if isinstance(min_candidate_edges, str):
                if min_candidate_edges != "min":
                    raise ValueError(
                        f"min_candidate_edges string mode must be 'min' "
                        f"(got {min_candidate_edges!r})"
                    )
                kept = {
                    p for p, n in pair_tx_edges.items()
                    if n >= min(entity_tx_total[p[0]], entity_tx_total[p[1]])
                }
            else:
                thr = int(min_candidate_edges)
                kept = {p for p, n in pair_tx_edges.items() if n >= thr}
            candidate_pairs = kept

        # Optional per-entity-witness count filter. Drop a candidate
        # pair (E1, E2) unless EACH entity contributes at least
        # `min_local_tx_per_entity` UNIQUE tx in the bins where they
        # co-occur (xy 8-Moore + z window). Symmetric in (E1, E2) —
        # not fooled by a 1-tx × N-tx bridging pair where the cross-
        # product count alone would pass `min_candidate_edges`.
        if track_local:
            mlt = int(min_local_tx_per_entity)
            kept_local = set()
            for (lo, hi) in candidate_pairs:
                lo_bins = pair_lo_bins.get((lo, hi), ())
                hi_bins = pair_hi_bins.get((lo, hi), ())
                # Each tx is in exactly one bin (we floored coords), so
                # summing per-bin per-entity tx counts over UNIQUE bins
                # = the unique-tx witness count for that entity.
                n_lo = sum(bin_to_comp_counts[b][lo] for b in lo_bins)
                n_hi = sum(bin_to_comp_counts[b][hi] for b in hi_bins)
                if n_lo >= mlt and n_hi >= mlt:
                    kept_local.add((lo, hi))
            candidate_pairs = kept_local

        # Optional per-pair median |Δz| guard. Reject candidate pairs
        # whose member tx have a median pairwise |Δz| larger than the
        # threshold. Useful when the bin filter under-discriminates due
        # to grid-alignment artefacts (a 1.5 µm physical gap can hide
        # inside one G_z=2 bin if the bin boundary aligns badly).
        if max_pair_median_dz is not None and G_z is not None:
            # Build per-entity z-coord array once
            ent_to_z: dict[int, np.ndarray] = defaultdict(list)
            for c, z in zip(comp_codes.tolist(),
                            transcript_coords[valid, 2].tolist()):
                ent_to_z[c].append(z)
            ent_to_z_arr = {c: np.asarray(zs) for c, zs in ent_to_z.items()}
            kept2 = set()
            for (a, b) in candidate_pairs:
                za = ent_to_z_arr.get(a)
                zb = ent_to_z_arr.get(b)
                if za is None or zb is None:
                    continue
                dz = np.abs(za[:, None] - zb[None, :]).ravel()
                if float(np.median(dz)) <= float(max_pair_median_dz):
                    kept2.add((a, b))
            candidate_pairs = kept2

        # Count-based Δz guard: admit only if at least
        # ``min_close_edges_n`` tx-tx pairs across (A, B) have
        # |Δz| < ``min_close_edges_dz``. Picks up the asymmetry between
        # within-cell pairs (where some edges are very tight) and
        # cross-stratum pairs (where every edge clears the gap).
        if (min_close_edges_dz is not None and min_close_edges_n > 0
                and G_z is not None):
            ent_to_z3: dict[int, np.ndarray] = defaultdict(list)
            for c, z in zip(comp_codes.tolist(),
                            transcript_coords[valid, 2].tolist()):
                ent_to_z3[c].append(z)
            ent_to_z3_arr = {c: np.asarray(zs) for c, zs in ent_to_z3.items()}
            kept3 = set()
            thr_dz = float(min_close_edges_dz)
            thr_n = int(min_close_edges_n)
            for (a, b) in candidate_pairs:
                za = ent_to_z3_arr.get(a)
                zb = ent_to_z3_arr.get(b)
                if za is None or zb is None:
                    continue
                dz = np.abs(za[:, None] - zb[None, :]).ravel()
                if int((dz < thr_dz).sum()) >= thr_n:
                    kept3.add((a, b))
            candidate_pairs = kept3

        edges = list(candidate_pairs)

        # Indices are no longer needed after initial enumeration; release memory.
        del bin_to_comps

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

    # Decomposable-coherence state (only populated when
    # use_decomposable_stitch=True). Per-root running primitives
    # (n_above, n_below, n_finite) updated on union via the 6-segment
    # cross arithmetic. Initialised here from each original's self-prim;
    # combine on union below.
    root_prims: list[tuple[int, int, int]] | None = None
    if use_decomposable_stitch and mode == "count":
        try:
            from . import _cy_prune as _cyp
            # Cython kernel needs float32 dense W
            if isinstance(npmi_mat, np.ndarray) and npmi_mat.dtype == np.float32:
                W_f32 = npmi_mat
            else:
                import scipy.sparse as _sp
                if _sp.issparse(npmi_mat):
                    W_f32 = npmi_mat.toarray().astype(np.float32)
                else:
                    W_f32 = np.ascontiguousarray(npmi_mat, dtype=np.float32)
            root_prims = [
                _cyp.coherence_count_primitives(
                    np.ascontiguousarray(g, dtype=np.int32), W_f32, float(threshold)
                ) if g.size >= 2 else (0, 0, 0)
                for g in gene_id_lists
            ]
        except Exception:
            # Any setup failure → fall back gracefully (no primitive path).
            root_prims = None

    # Reset diagnostic gate-fire counters (visible to caller via
    # tracer.stitching._LAST_GATE_STATS after the call returns).
    _LAST_GATE_STATS.clear()
    _LAST_GATE_STATS.update({
        "K": int(spatial_centroid_k) if spatial_centroid_gate else 0,
        "checks_total": 0,        # _spatial_overlap calls
        "checks_pass": 0,         # _spatial_overlap returned True
        "init_bypasses": 0,       # heap-init pairs that took the bypass
        "merges_via_bypass": 0,   # actual unions that fired through bypass
        "merges_total": 0,        # all unions
    })

    # ----------------------------------------------------------------
    # Per-root spatial state.
    # K=1 (default): bbox check via min/max columns from summary_df.
    # K≥2: count-based check requiring K tx-coords above AND K below
    # the smaller entity's centroid per axis. Requires per-root tx
    # coord arrays (maintained as concatenated ndarrays on union).
    # ----------------------------------------------------------------
    root_centroid: np.ndarray | None = None  # [N, n_dim]
    root_bbox_min: np.ndarray | None = None  # [N, n_dim]  K=1 only
    root_bbox_max: np.ndarray | None = None  # [N, n_dim]  K=1 only
    root_n_tx: np.ndarray | None = None      # [N]
    root_tx_coords: list | None = None        # [N] list of (n_tx, n_dim) arrays — K≥2 only
    if spatial_centroid_gate:
        coord_keys = ["x", "y", "z"] if use_3d else ["x", "y"]
        try:
            root_centroid = summary_df[coord_keys].to_numpy(dtype=np.float64).copy()
            min_keys = [f"{c}_min" for c in coord_keys]
            max_keys = [f"{c}_max" for c in coord_keys]
            root_bbox_min = summary_df[min_keys].to_numpy(dtype=np.float64).copy()
            root_bbox_max = summary_df[max_keys].to_numpy(dtype=np.float64).copy()
            if "n_tx" in summary_df.columns:
                root_n_tx = summary_df["n_tx"].to_numpy(dtype=np.int64).copy()
            else:
                root_n_tx = np.ones(N, dtype=np.int64)
            if spatial_centroid_k >= 2:
                if entity_tx_coords is None:
                    # Need per-entity tx coords for K≥2 → fall back to K=1
                    spatial_centroid_k = 1
                    root_tx_coords = None
                else:
                    root_tx_coords = [
                        np.asarray(entity_tx_coords.get(str(eid), np.zeros((0, len(coord_keys)))),
                                    dtype=np.float64)
                        for eid in summary_df["entity_id"].astype(str)
                    ]
        except KeyError:
            spatial_centroid_gate = False
            root_centroid = root_bbox_min = root_bbox_max = root_n_tx = None
            root_tx_coords = None

    def _spatial_overlap(ra: int, rb: int) -> bool:
        """Default rule: smaller's centroid inside larger's tx cloud.
        K=1 → bbox check. K≥2 → ≥K tx of larger above AND below
        smaller's centroid per axis.

        Flipped rule (spatial_gate_flipped=True): swap roles. Test
        whether the LARGER's centroid lies inside the SMALLER's tx
        cloud. Effective K capped at floor(n_smaller / 3) so a small
        partial uses a lighter K. Detects "cell arranged AROUND
        partial" (legitimate fragment) instead of "partial embedded
        IN cell" (often contamination).
        """
        _LAST_GATE_STATS["checks_total"] += 1
        n_a = int(root_n_tx[ra])
        n_b = int(root_n_tx[rb])
        if n_a <= n_b:
            small_idx, large_idx = ra, rb
            n_small, n_large = n_a, n_b
        else:
            small_idx, large_idx = rb, ra
            n_small, n_large = n_b, n_a

        if spatial_gate_flipped:
            # test point = larger's centroid; reference cloud = smaller's tx
            c = root_centroid[large_idx]
            ref_idx = small_idx
            # Dynamic K cap: small partial → lighter K. Uses ceiling
            # division (n+2)//3 so a 5-tx partial gets K=2, not K=1
            # (bbox-only). Floor gave K=1 for n∈{3,4,5} — too permissive.
            k_eff = min(int(spatial_centroid_k),
                         max(1, (n_small + 2) // 3))
        else:
            c = root_centroid[small_idx]
            ref_idx = large_idx
            # Original "smaller's centroid in larger" rule with dynamic
            # floor based on LARGER entity's size:
            #   K_eff = max(K, ceil(n_larger / 3))
            # For a 105-tx cell merging with a small partial, K_eff = 35
            # — requires 35 cell tx straddling the partial centroid in
            # each axis. Stricter for big cells (the typical contamination
            # host), no-op for small (n<30) cells.
            k_eff = max(int(spatial_centroid_k),
                         (n_large + 2) // 3)

        if k_eff <= 1:
            # Bbox check (cheap)
            bb_min = root_bbox_min[ref_idx]
            bb_max = root_bbox_max[ref_idx]
            ok = bool(np.all(c >= bb_min) and np.all(c <= bb_max))
            if ok:
                _LAST_GATE_STATS["checks_pass"] += 1
            return ok

        # K≥2: per-axis count of tx in reference cloud above/below c.
        ref_coords = root_tx_coords[ref_idx]
        if ref_coords.size == 0 or ref_coords.shape[0] < 2 * k_eff:
            return False
        for d in range(c.shape[0]):
            col = ref_coords[:, d]
            n_above = int(np.sum(col > c[d]))
            if n_above < k_eff:
                return False
            n_below = int(np.sum(col < c[d]))
            if n_below < k_eff:
                return False
        _LAST_GATE_STATS["checks_pass"] += 1
        return True

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
        # Decomposable path: derive (C, purity, conflict) from the
        # root's running primitive sums — no gene-pair iteration.
        if use_decomposable_stitch and root_prims is not None and mode == "count":
            na, nb, nf = root_prims[root_idx]
            if nf == 0:
                triple = (0.0, 0.0, 0.0)
            else:
                purity = na / nf
                conflict = nb / nf
                triple = (purity - conflict, purity, conflict)
            root_C_cache[root_idx] = triple
            return triple
        triple = coherence(
            root_genes[root_idx], npmi_mat,
            mode=mode, threshold=threshold, metric=metric,
        )
        root_C_cache[root_idx] = triple
        return triple

    # Helper: combine two roots' primitives into the union's via the
    # 6-segment decomposition (validated bit-exact in
    # /tmp/validate_decomp_coh.py against direct coherence). Returns
    # (n_above_union, n_below_union, n_finite_union, union_genes_array).
    # Only invoked when use_decomposable_stitch=True and root_prims is
    # populated (mode == 'count' + dense float32 W).
    def _combine_prims(ra, rb):
        ga = root_genes[ra]
        gb = root_genes[rb]
        if ga.size == 0 and gb.size == 0:
            return (0, 0, 0), np.empty(0, dtype=np.int32)
        if ga.size == 0:
            return root_prims[rb], gb
        if gb.size == 0:
            return root_prims[ra], ga
        # 3-segment partition: a_only, b_only, common
        common = np.intersect1d(ga, gb, assume_unique=True)
        a_only = np.setdiff1d(ga, common, assume_unique=True).astype(np.int32)
        b_only = np.setdiff1d(gb, common, assume_unique=True).astype(np.int32)
        common32 = common.astype(np.int32)
        # primitives needed: 3 self (a_only, b_only, common)
        # + 3 cross (a×b, a×c, b×c).  Compose triu(union) from these.
        from . import _cy_prune as _cyp_local
        sa = _cyp_local.coherence_count_primitives(a_only, W_f32, float(threshold)) if a_only.size >= 2 else (0, 0, 0)
        sb = _cyp_local.coherence_count_primitives(b_only, W_f32, float(threshold)) if b_only.size >= 2 else (0, 0, 0)
        sc = _cyp_local.coherence_count_primitives(common32, W_f32, float(threshold)) if common32.size >= 2 else (0, 0, 0)
        cab = _cyp_local.coherence_cross_primitives(a_only, b_only, W_f32, float(threshold))
        cac = _cyp_local.coherence_cross_primitives(a_only, common32, W_f32, float(threshold))
        cbc = _cyp_local.coherence_cross_primitives(b_only, common32, W_f32, float(threshold))
        union_prims = (
            sa[0] + sb[0] + sc[0] + cab[0] + cac[0] + cbc[0],
            sa[1] + sb[1] + sc[1] + cab[1] + cac[1] + cbc[1],
            sa[2] + sb[2] + sc[2] + cab[2] + cac[2] + cbc[2],
        )
        union_genes = np.concatenate([a_only, b_only, common32])
        union_genes.sort()
        return union_prims, union_genes

    # Sentinel ΔC value for spatial-overlap pairs. Pushes them to the
    # top of the heap (popped first, bypass coherence + threshold).
    # 1e9 is far above any realistic ΔC value (∈ [-1, 1] in practice).
    _SPATIAL_OVERLAP_DC = 1e9

    # compute deltaC between current roots
    def compute_deltaC_roots(ra, rb):
        # Spatial bypass (Tier 1): in mode="pre", if the smaller
        # entity's centroid is inside the larger entity's bbox, treat
        # the pair as a GUARANTEED merge — return a high sentinel ΔC.
        # No coherence / gene-PMI evaluation is performed. In
        # mode="post", we never short-circuit here; the spatial test
        # is checked at pop time as a fallback for ΔC-rejected pairs.
        if (spatial_centroid_gate and spatial_gate_mode == "pre"
                and root_centroid is not None
                and _spatial_overlap(ra, rb)):
            return _SPATIAL_OVERLAP_DC

        # Decomposable-primitive fast path: derive C(union) from the
        # roots' running primitive sums + on-the-fly cross primitives,
        # without re-iterating the union's full gene-pair set. Bit-
        # equivalent to coherence(union) for mode='count' (validated).
        if use_decomposable_stitch and root_prims is not None and mode == "count":
            Cu, _, _ = C_of_root(ra)
            Cv, _, _ = C_of_root(rb)
            (na_u, nb_u, nf_u), _ = _combine_prims(ra, rb)
            if nf_u == 0:
                Cunion = 0.0
            else:
                Cunion = (na_u - nb_u) / nf_u
            if not penalize_simplicity:
                return float(Cunion - max(Cu, Cv))
            nu = max(int(root_genes[ra].size), 1)
            nv = max(int(root_genes[rb].size), 1)
            n_union = nu + nv
            C_sep = max(Cu - 1.0 / nu, Cv - 1.0 / nv)
            return float(Cunion - (1.0 / n_union) - C_sep)

        # Eager path (default): compute C(union) directly via coherence.
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

    # ----------------------------------------------------------------
    # Optional heap-init fast-gate: drop candidate pairs whose top-clique
    # cross-PMI block contains a strong-negative entry. For most cell-
    # pair candidates that are biologically incompatible, this avoids
    # the expensive compute_deltaC_roots call entirely. The gate is
    # applied ONLY to the initial edge list (heap-init); boundary
    # expansion + stale-pop reinserts during the merge loop are unchanged.
    # ----------------------------------------------------------------
    gate_keep_mask: np.ndarray | None = None
    if fast_gate_top_k > 0 and len(edges) > 0:
        try:
            from . import _cy_prune as _cyp_gate
            # Build float32 dense W if not already
            if isinstance(npmi_mat, np.ndarray) and npmi_mat.dtype == np.float32:
                _W_f32 = npmi_mat
            else:
                import scipy.sparse as _sp
                if _sp.issparse(npmi_mat):
                    _W_f32 = npmi_mat.toarray().astype(np.float32)
                else:
                    _W_f32 = np.ascontiguousarray(npmi_mat, dtype=np.float32)
            top_cliques = _cyp_gate.top_k_positive_clique_per_entity(
                gene_id_lists, _W_f32, int(fast_gate_top_k), float(threshold),
            )
            edges_arr = np.asarray(edges, dtype=np.int32)
            if edges_arr.ndim == 1:
                edges_arr = edges_arr.reshape(-1, 2)
            gate_keep_mask = _cyp_gate.fast_gate_pairs(
                top_cliques, edges_arr, _W_f32, float(fast_gate_mean_threshold),
            )
        except Exception:
            gate_keep_mask = None  # graceful fallback: no gating

    heap = []
    for ei, (i, j) in enumerate(edges):
        # Tier 1 (positive override): spatial-overlap bypass. If the
        # smaller entity's centroid is inside the larger's bbox, this
        # pair MUST go on the heap (with sentinel ΔC) regardless of
        # the gate result. compute_deltaC_roots returns 1e9 for these.
        is_spatial = (
            spatial_centroid_gate and root_centroid is not None
            and _spatial_overlap(i, j)
        )
        if is_spatial:
            _LAST_GATE_STATS["init_bypasses"] += 1
        # Tier 2 (cheap rejection): fast-gate skips expensive eval ONLY
        # when there's no spatial bypass. In "post" mode, spatial is a
        # fallback at pop time, so still let the fast-gate cull these.
        if (not is_spatial) and gate_keep_mask is not None and not gate_keep_mask[ei]:
            continue
        # Tier 3 (full ΔC eval; sentinel 1e9 only in "pre" mode)
        di = compute_deltaC_roots(i, j)
        if np.isfinite(di) and di >= deltaC_min:
            heapq.heappush(heap, _heap_item(di, i, j))
        elif (is_spatial and spatial_gate_mode == "post"
              and np.isfinite(di)):
            # Post-mode rescue: ΔC says reject, but spatial matches.
            # Push at the real (low/negative) ΔC priority so genuine
            # ΔC merges happen first; this pair gets revisited at pop
            # time and merged via the spatial-override path.
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
        post_override = False
        if not (np.isfinite(dc_now) and dc_now >= deltaC_min):
            # ΔC says reject. In "post" mode, give the spatial gate
            # one chance: if the (current-root) centroid test still
            # matches, force the merge anyway. This is the only way
            # spatial can intervene in post-mode.
            if (spatial_centroid_gate and spatial_gate_mode == "post"
                    and root_centroid is not None
                    and _spatial_overlap(ra, rb)):
                post_override = True
            else:
                continue

        # merge (choose new root)
        rnew = dsu.union(ra, rb)
        rold = rb if rnew == ra else ra
        _LAST_GATE_STATS["merges_total"] += 1
        # Track which gate drove the merge:
        #   pre-mode bypass → dc_now == 1e9 sentinel
        #   post-mode override → ΔC failed but spatial said merge
        if dc_now >= _SPATIAL_OVERLAP_DC * 0.5:
            _LAST_GATE_STATS["merges_via_bypass"] += 1
        elif post_override:
            _LAST_GATE_STATS["merges_via_bypass"] += 1

        # update cluster metadata onto rnew
        has_cell[rnew] = has_cell[rnew] or has_cell[rold]
        cell_ids[rnew] |= cell_ids[rold]
        partial_ids[rnew] |= partial_ids[rold]
        comp_ids[rnew] |= comp_ids[rold]

        # Spatial state update on union (when gate is active).
        if (spatial_centroid_gate and root_centroid is not None
                and root_n_tx is not None):
            n_new = root_n_tx[rnew]
            n_old = root_n_tx[rold]
            n_total = n_new + n_old
            if n_total > 0:
                root_centroid[rnew] = (
                    (root_centroid[rnew] * n_new + root_centroid[rold] * n_old) / n_total
                )
            root_bbox_min[rnew] = np.minimum(root_bbox_min[rnew], root_bbox_min[rold])
            root_bbox_max[rnew] = np.maximum(root_bbox_max[rnew], root_bbox_max[rold])
            root_n_tx[rnew] = n_total
            root_n_tx[rold] = 0
            # K≥2 path: maintain concatenated tx-coord arrays per root.
            if root_tx_coords is not None:
                if root_tx_coords[rnew].size == 0:
                    root_tx_coords[rnew] = root_tx_coords[rold]
                elif root_tx_coords[rold].size > 0:
                    root_tx_coords[rnew] = np.concatenate(
                        [root_tx_coords[rnew], root_tx_coords[rold]], axis=0
                    )
                root_tx_coords[rold] = np.zeros((0, root_centroid.shape[1]), dtype=np.float64)

        # union genes (and primitive sums when in decomposable mode).
        # In decomposable mode we already computed _combine_prims for
        # (ra, rb) inside compute_deltaC_roots; recompute here for the
        # new root's bookkeeping. Yes, this duplicates work — a future
        # optimisation could cache the result. For now, correctness
        # over speed: the merge path is O(merges), not O(rounds).
        if use_decomposable_stitch and root_prims is not None and mode == "count":
            new_prims, new_genes = _combine_prims(ra, rb)
            root_genes[rnew] = new_genes if new_genes.dtype == np.int32 else new_genes.astype(np.int32)
            root_prims[rnew] = new_prims
            root_prims[rold] = (0, 0, 0)
        else:
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

    # ----------------------------------------------------------------
    # Choose stitched label per final root.
    # Priority: cell > partial > component.
    #
    # Multi-partial merger rule (when partial_ids[r] has > 1 element):
    # the merged entity is GIVEN A FRESH LABEL with a SECOND DASH
    # LEVEL, so the result is distinguishable from any of its inputs.
    #
    # Label form:
    #   Phase 1c partial:  "{cell}-{d1}"        (single dash, e.g. "37962-1")
    #   Stitch merger:     "{cell}-{d1}-{d2}"   (two dashes, e.g. "37962-1-1")
    #
    # The depth-2 namespace `(cell, d1)` is per-(winning cell, winning
    # d1). It is initialised by scanning all input partials and
    # recording the max d2 seen in each (cell, d1) namespace; new
    # mergers increment past that max → guaranteed-unique labels.
    #
    # Decision rule for picking the merger's parent (which (cell, d1)
    # namespace owns the result):
    #   1. Higher d1 suffix wins (more aggregated lineage).
    #   2. Higher d2 suffix wins (already-merged > unmerged).
    #   3. Higher tx count wins (dominant biological signal).
    #   4. Lexicographic (final deterministic tiebreak).
    # ----------------------------------------------------------------
    def _parse_partial(label: str) -> tuple[str, int, int] | None:
        """Parse '{cell}-{d1}' or '{cell}-{d1}-{d2}'. Returns (cell,
        d1, d2) or None if not a valid partial label. d2 = 0 for
        single-dash labels."""
        if "-" not in label:
            return None
        parts = label.rsplit("-", 2)
        # parts can be 1, 2, or 3 elements depending on dash count.
        if len(parts) == 2:
            cell, d1_str = parts
            try:
                return cell, int(d1_str), 0
            except ValueError:
                return None
        elif len(parts) == 3:
            cell, d1_str, d2_str = parts
            try:
                return cell, int(d1_str), int(d2_str)
            except ValueError:
                return None
        return None

    # Initialise the depth-2 counters from input labels.
    next_merger_counter: dict[tuple[str, int], int] = {}
    for i in range(N):
        eid = entity_ids[i]
        if etypes[i] != "partial":
            continue
        parsed = _parse_partial(eid)
        if parsed is None:
            continue
        cell, d1, d2 = parsed
        key = (cell, d1)
        cur = next_merger_counter.get(key, 0)
        if d2 > cur:
            next_merger_counter[key] = d2

    def _pick_partial_label(partials: set[str]) -> str:
        if len(partials) == 1:
            return next(iter(partials))
        rows = []
        for p in partials:
            parsed = _parse_partial(p)
            if parsed is None:
                continue
            cell, d1, d2 = parsed
            n_tx = (entity_n_tx or {}).get(p, 0)
            rows.append((d1, d2, n_tx, p, cell))
        if not rows:
            return sorted(partials)[0]
        # Sort: highest d1 → highest d2 → highest tx → lex-smaller label.
        rows.sort(key=lambda r: (-r[0], -r[1], -r[2], r[3]))
        winner_d1, _, _, _, winner_cell = rows[0]
        key = (winner_cell, winner_d1)
        next_merger_counter[key] = next_merger_counter.get(key, 0) + 1
        return f"{winner_cell}-{winner_d1}-{next_merger_counter[key]}"

    root_to_label = {}
    for i in range(N):
        r = dsu.find(i)
        if r in root_to_label:
            continue
        if cell_ids[r]:
            label = sorted(cell_ids[r])[0]          # deterministic
        elif partial_ids[r]:
            label = _pick_partial_label(partial_ids[r])
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
    G_z: float | None = None,
    z_neighbor_depth: int = 0,
    min_candidate_edges: int | str = 0,
    # Optional per-entity-witness count: drop candidate pair (E1, E2)
    # unless EACH entity contributes at least `min_local_tx_per_entity`
    # unique tx in the shared bin neighborhood (xy 8-Moore + ±depth z
    # bins). Catches single-bridging-tx candidates that sneak through
    # the diagonal-Moore reach (~5.66 µm at G=2). Symmetric in (E1, E2)
    # — resistant to mass-dominated cross-product counts.
    # Default 0 = off (current behavior unchanged).
    min_local_tx_per_entity: int = 0,
    max_pair_median_dz: float | None = None,
    min_close_edges_dz: float | None = None,
    min_close_edges_n: int = 0,
    purity_threshold=_LEGACY_STITCH_KWARG_SENTINEL,
    tau=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relu=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relative=_LEGACY_STITCH_KWARG_SENTINEL,
    # Experimental: lazy DSU+heap with decomposable-coherence primitives.
    # Default False (eager path, byte-unchanged). Bit-match validated on
    # 500/1000 µm ROIs (99.98%+ per-tx label parity, ARI identical to 4
    # decimals). See `_stitch_entities_hierarchical_decomposable` for
    # rationale and `TODO.md` for tissue-scale follow-ups.
    use_decomposable_stitch: bool = False,
    # Experimental: top-K positive-clique fast-gate at heap-init.
    # 0 = disabled (default). ≥1 enables — pre-filters candidate pairs
    # using a small per-entity signature signature; rejects pairs with
    # strong-negative top-clique cross-PMI before expensive ΔC eval.
    fast_gate_top_k: int = 0,
    fast_gate_mean_threshold: float = 0.0,
    # Experimental: spatial centroid-in-bbox gate at merge time.
    # When True, smaller entity's centroid must lie inside the larger
    # entity's per-axis tx-coord range (axis-aligned bbox). Default
    # False (no spatial constraint beyond Stitch's existing
    # `dist_threshold` Delaunay-edge filter at candidate-build time).
    spatial_centroid_gate: bool = False,
    # Tightness of the spatial-overlap test. K=1 → bbox check (at
    # least 1 tx of larger entity above AND 1 below smaller's centroid
    # in each axis). K=2 → require 2 above AND 2 below per axis (more
    # interior). K=3 → 3 each. Higher K = stricter.
    spatial_centroid_k: int = 1,
    # Optional per-entity tx-coord arrays. Required for K≥2; with K=1
    # the gate falls back to bbox check using `summary_df`'s min/max
    # columns. dict[entity_id_str -> (n_tx, n_dim) ndarray].
    entity_tx_coords: dict | None = None,
    # Spatial gate mode: "pre" (current default — spatial bypass overrides
    # ΔC and merges first) or "post" (spatial fires only as a fallback
    # for ΔC-rejected pairs).
    spatial_gate_mode: str = "pre",
    # Flipped overlap test: larger's centroid inside smaller's tx cloud.
    # Effective K capped at floor(n_smaller / 3).
    spatial_gate_flipped: bool = False,
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
        if G_z is not None and len(coord_cols) >= 3:
            transcript_coords = df_final[
                [coord_cols[0], coord_cols[1], coord_cols[2]]
            ].to_numpy(dtype=np.float64)
        else:
            transcript_coords = df_final[
                [coord_cols[0], coord_cols[1]]
            ].to_numpy(dtype=np.float64)

    legacy_kwargs = {}
    if purity_threshold is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["purity_threshold"] = purity_threshold
    if tau is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["tau"] = tau
    if use_relu is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["use_relu"] = use_relu
    if use_relative is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["use_relative"] = use_relative

    # Per-entity tx count for the multi-partial merger tiebreak rule
    # in stitch_entities_hierarchical (majority-tx-count when suffix
    # levels tie). One O(N) value_counts pass over the entity column.
    entity_n_tx_dict = (
        df_final[entity_col].astype(str).value_counts().to_dict()
    )

    # When the K≥2 strict spatial gate is requested, compute per-entity
    # tx-coord arrays. Cheap groupby-by-entity on transcript-level data.
    entity_tx_coords_dict: dict | None = None
    if spatial_centroid_gate and spatial_centroid_k >= 2:
        coord_cols_used = list(coord_cols) if use_3d else list(coord_cols[:2])
        entity_tx_coords_dict = {}
        ent_str_col = df_final[entity_col].astype(str)
        for eid, sub in df_final.groupby(ent_str_col, observed=True):
            entity_tx_coords_dict[str(eid)] = sub[coord_cols_used].to_numpy(
                dtype=np.float64
            )

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
        G_z=G_z,
        z_neighbor_depth=z_neighbor_depth,
        transcript_coords=transcript_coords,
        transcript_entity_codes=transcript_entity_codes,
        min_candidate_edges=min_candidate_edges,
        min_local_tx_per_entity=min_local_tx_per_entity,
        max_pair_median_dz=max_pair_median_dz,
        min_close_edges_dz=min_close_edges_dz,
        min_close_edges_n=min_close_edges_n,
        use_decomposable_stitch=use_decomposable_stitch,
        fast_gate_top_k=fast_gate_top_k,
        fast_gate_mean_threshold=fast_gate_mean_threshold,
        entity_n_tx=entity_n_tx_dict,
        spatial_centroid_gate=spatial_centroid_gate,
        spatial_centroid_k=spatial_centroid_k,
        entity_tx_coords=entity_tx_coords_dict,
        spatial_gate_mode=spatial_gate_mode,
        spatial_gate_flipped=spatial_gate_flipped,
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

    # Propagate `_etype` to match the post-stitch labels. `summary`
    # carries one etype per original entity; the merge target's etype
    # is what survives. Build a label→etype map from summary and
    # remap. Without this, tx of merged-in entities keep their
    # original (now-stale) etype, which can bias downstream
    # entity-level classification (the `.first()` aggregation in
    # build_entity_table picks non-deterministic etype within the
    # merged entity).
    if "_etype" in df_out.columns and "etype" in summary.columns:
        # summary has entity_id (original) and etype. Map original →
        # stitched label, then dedupe by stitched label (winning etype
        # is one of the merged-set etypes; pick the first per stitched-
        # label group, which corresponds to the stitch target's etype
        # since stitch targets retain their own ID).
        s_label = pd.Series(summary["etype"].to_numpy(),
                             index=summary["entity_id"].astype(str))
        # Map each original entity to its stitched label
        stitched_label = pd.Series(
            {k: entity_to_stitched.get(k, k) for k in s_label.index},
            index=s_label.index,
        )
        # For each stitched label, prefer the etype of the entity whose
        # ID equals the stitched label itself (the merge target).
        target_etype: dict[str, str] = {}
        for orig, stitched in stitched_label.items():
            if orig == stitched:
                target_etype[stitched] = str(s_label.loc[orig])
        # For entities that didn't survive but whose target wasn't in
        # the summary (shouldn't happen but defensive), fall back to
        # the original etype.
        for orig, stitched in stitched_label.items():
            target_etype.setdefault(stitched, str(s_label.loc[orig]))
        new_labels = df_out[out_col].astype(str)
        new_etype = new_labels.map(target_etype)
        # Where the new label has no entry in the map (e.g.,
        # unassigned), keep the existing _etype value.
        keep_existing = new_etype.isna()
        if (~keep_existing).any():
            df_out.loc[~keep_existing, "_etype"] = new_etype[~keep_existing].astype(str).to_numpy()

    return df_out, entity_to_stitched
