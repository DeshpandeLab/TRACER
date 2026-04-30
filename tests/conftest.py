"""Shared pytest fixtures and path setup for the TRACER test suite.

Adds the repo root to ``sys.path`` so ``tests.synthetic`` is importable
even when the package is installed in a way that doesn't include the
test fixtures.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Also ensure src/ is importable for editable installs that may not yet
# be built. Normal `pip install -e .` configures this, so this is just a
# defensive fallback.
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def seed() -> int:
    """Default reproducibility seed for tests."""
    return 42


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Path:
    """Create a tmp ``<project>/data/`` containing a single synthetic
    parquet, suitable for ``tracer.data.discover_data_files`` tests.

    Returns
    -------
    Path to the project root (i.e. the parent of ``data/``).
    """
    proj = tmp_path / "syntheticproj"
    data = proj / "data"
    data.mkdir(parents=True)

    df = pd.DataFrame({
        "transcript_id": ["t0", "t1", "t2"],
        "feature_name": ["A", "B", "A"],
        "cell_id": ["c0", "c0", "c1"],
        "x": np.array([0.0, 1.0, 5.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 5.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    })
    df.to_parquet(data / "syntheticproj_df.parquet")
    return proj
