"""Bench data loader.

Re-exports the project-folder discovery + loading helpers from
``tracer.data`` (the package-level utilities), and adds bench-specific
extras: ROI subsetting, VHD-style degradation, and the lung_cancer
legacy default for back-compat with saved bench parquets.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from tracer.data import discover_data_files
from tracer.data import load_full_df as _pkg_load_full_df

# Legacy default: the lung_cancer project folder. Used when callers
# invoke load_*_df() with no project_dir / parquet_path arg, to keep
# pre-genericization bench scripts working bit-equivalent.
DEFAULT_PROJECT_DIR = (
    Path("/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS")
    / "tutorials" / "lung_cancer"
)
DEFAULT_PARQUET = DEFAULT_PROJECT_DIR / "data" / "lung_cancer_df.parquet"

# Legacy lung_cancer-specific 500 µm × 500 µm ROI (kept for binary
# reproducibility of saved parquets that used these exact bounds).
# Center = (1818.7, 2186.8). Half-side = 250 µm.
_LEGACY_LUNG_CANCER_ROI = (1568.7, 2068.7, 1936.8, 2436.8)


def load_full_df(project_dir: Path | str | None = None,
                 parquet_path: Path | str | None = None) -> pd.DataFrame:
    """Bench wrapper: with no args, falls back to the lung_cancer
    legacy default. Otherwise delegates to :func:`tracer.data.load_full_df`.
    """
    if project_dir is None and parquet_path is None:
        return pd.read_parquet(DEFAULT_PARQUET)
    return _pkg_load_full_df(project_dir=project_dir, parquet_path=parquet_path)


def roi_mask(df: pd.DataFrame,
             bbox: tuple[float, float, float, float] | None = None) -> pd.Series:
    """Boolean mask for tx in a (x_min, x_max, y_min, y_max) bbox.
    With ``bbox=None``, uses the legacy lung_cancer 500 µm bbox.
    """
    x_min, x_max, y_min, y_max = bbox if bbox is not None else _LEGACY_LUNG_CANCER_ROI
    return ((df["x"] >= x_min) & (df["x"] <= x_max)
            & (df["y"] >= y_min) & (df["y"] <= y_max))


def load_roi_df(project_dir: Path | str | None = None,
                half_side_um: float | None = None,
                parquet_path: Path | str | None = None,
                roi_center_xy: tuple[float, float] | None = None) -> pd.DataFrame:
    """Load an ROI subset of a project's transcript table.

    Behavior:
      - With no args (legacy call): loads the lung_cancer 500 µm × 500 µm
        legacy bbox. Bit-equivalent to pre-refactor.
      - With ``project_dir`` set and ``half_side_um=None``: ROI center
        defaults to the data's coordinate median; half_side defaults to
        250 µm.
      - With ``half_side_um`` set: builds a 2*half_side × 2*half_side
        bbox around the ROI center. Center is ``roi_center_xy`` if
        given, else (legacy fallback) the lung_cancer fixed bbox center,
        else the coordinate median.
    """
    df = load_full_df(project_dir=project_dir, parquet_path=parquet_path)
    legacy_fallback = project_dir is None and parquet_path is None
    if legacy_fallback and half_side_um is None and roi_center_xy is None:
        return df.loc[roi_mask(df)].reset_index(drop=True)
    if roi_center_xy is None:
        if legacy_fallback:
            x_min, x_max, y_min, y_max = _LEGACY_LUNG_CANCER_ROI
            roi_center_xy = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
        else:
            roi_center_xy = (float(df["x"].median()), float(df["y"].median()))
    cx, cy = roi_center_xy
    half = half_side_um if half_side_um is not None else 250.0
    mask = ((df["x"] >= cx - half) & (df["x"] <= cx + half)
            & (df["y"] >= cy - half) & (df["y"] <= cy + half))
    return df.loc[mask].reset_index(drop=True)


def degrade_to_vhd(
    df: pd.DataFrame,
    *,
    bin_size_um: float = 2.0,
    drop_z: bool = True,
    drop_segmentation: bool = True,
) -> pd.DataFrame:
    """Apply Visium HD-like constraints to a Xenium-style transcript table.

    Bench-time perturbation, NOT a production transformation. Returns a
    copy with:
      - `x`, `y` snapped to bin centers at `bin_size_um` pitch.
      - `z` set to a constant 0 if `drop_z=True`.
      - `cell_id` set to "-1" everywhere if `drop_segmentation=True`.

    Adds a ``bin_id`` column (``"BIN_<bx>_<by>"``) so the bench can use
    it as a PMI ``group_key`` under VHD-no-seg modes (where cell_id
    becomes constant and uninformative).
    """
    out = df.copy()
    bin_x_idx = None
    bin_y_idx = None
    if bin_size_um and bin_size_um > 0:
        bin_x_idx = np.floor(out["x"].to_numpy() / bin_size_um).astype(np.int64)
        bin_y_idx = np.floor(out["y"].to_numpy() / bin_size_um).astype(np.int64)
        out["x"] = (bin_x_idx.astype(np.float64) * bin_size_um + bin_size_um / 2.0)
        out["y"] = (bin_y_idx.astype(np.float64) * bin_size_um + bin_size_um / 2.0)
    if drop_z and "z" in out.columns:
        out["z"] = 0.0
    if bin_x_idx is not None:
        bin_id = np.array(
            [f"BIN_{bx}_{by}" for bx, by in zip(bin_x_idx.tolist(), bin_y_idx.tolist())],
            dtype=object,
        )
    else:
        bin_id = np.full(len(out), "-1", dtype=object)
    out["bin_id"] = pd.Series(bin_id, index=out.index, dtype=object)
    if drop_segmentation and "cell_id" in out.columns:
        if isinstance(out["cell_id"].dtype, pd.CategoricalDtype):
            cats = list(out["cell_id"].cat.categories)
            if "-1" not in cats:
                out["cell_id"] = out["cell_id"].cat.add_categories(["-1"])
        out["cell_id"] = "-1"
    return out


if __name__ == "__main__":
    import sys
    project_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if project_dir is not None:
        parquet, npmi_cache = discover_data_files(project_dir)
        print(f"project_dir: {project_dir}")
        print(f"  parquet:    {parquet}")
        print(f"  npmi_cache: {npmi_cache}")
    roi = load_roi_df(project_dir=project_dir)
    print(f"ROI shape: {roi.shape}")
    print(f"x: [{roi['x'].min():.1f}, {roi['x'].max():.1f}]")
    print(f"y: [{roi['y'].min():.1f}, {roi['y'].max():.1f}]")
    if "z" in roi.columns:
        print(f"z: [{roi['z'].min():.1f}, {roi['z'].max():.1f}]")
    if "cell_id" in roi.columns:
        print(f"cells: {roi['cell_id'].astype(str).nunique()}")
