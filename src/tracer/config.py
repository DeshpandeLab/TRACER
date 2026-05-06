"""Pipeline configuration — typed dataclasses + TOML loader.

Phase A of the config migration: this module defines the dataclasses
that codify every tunable knob in the segmentation pipeline, plus a
loader that builds a `PipelineConfig` from layered TOML files.

Design notes
------------
* Code defaults are canonical. `configs/defaults.toml` is a
  human-readable export of those defaults. `tests/test_config.py`
  verifies the two agree, so the TOML stays in lock-step.
* Layered composition: ``defaults`` ← ``platforms/<name>.toml`` ← user
  override file. Each layer patches keys; sections are merged, not
  replaced wholesale.
* `[final_rescue]` accepts an ``inherit = "rescue"`` directive:
  resolved values from `[rescue]` are copied first, then the local
  keys override. One-level inherit only — no transitive chains.
* Frozen dataclasses → configs are hashable, can pin a run.
* `dump_receipt(cfg, path)` writes resolved values as JSON for the
  per-run receipt that ships alongside outputs (reproducibility).

Phase B will switch the runner to consume `PipelineConfig`; this
module is currently standalone — importing it has no effect on the
pipeline.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Per-stage configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase1Config:
    """Phase 1 (a/b/c) — nuclear-anchored greedy prune + admission."""
    pmi_threshold: float = 0.05
    seed_coherence_floor: float = 0.10
    tx_weighted_prune: bool = True
    nuclear_only_admit: bool = True

    def __post_init__(self) -> None:
        if not (-1.0 <= self.pmi_threshold <= 1.0):
            raise ValueError(
                f"phase1.pmi_threshold out of range: {self.pmi_threshold}"
            )
        if not (0.0 <= self.seed_coherence_floor <= 1.0):
            raise ValueError(
                f"phase1.seed_coherence_floor out of range: "
                f"{self.seed_coherence_floor}"
            )


@dataclass(frozen=True)
class SplitPhase1Config:
    """Post-Phase-1 z-gap splitter (no-op when z column absent)."""
    dz_threshold_um: float = 2.0
    min_tx: int = 1
    min_entity_size: int = 2

    def __post_init__(self) -> None:
        if self.dz_threshold_um <= 0:
            raise ValueError(
                f"split_phase1.dz_threshold_um must be > 0; got {self.dz_threshold_um}"
            )
        if self.min_entity_size < 2:
            raise ValueError(
                f"split_phase1.min_entity_size must be >= 2; got {self.min_entity_size}"
            )


@dataclass(frozen=True)
class Phase1QcConfig:
    """Demote Phase-1 entities below this size threshold."""
    min_tx: int = 3

    def __post_init__(self) -> None:
        if self.min_tx < 1:
            raise ValueError(f"phase1_qc.min_tx must be >= 1; got {self.min_tx}")


@dataclass(frozen=True)
class RescueConfig:
    """Spatial-prior rescue veto (used by main Rescue and Final Rescue)."""
    veto_mode: Literal["min", "mean", "hybrid"] = "hybrid"
    min_admit_threshold: float = 0.0      # hybrid: unanimous-pos cutoff
    mean_admit_threshold: float = 0.1     # hybrid/mean: aggregate-pos cutoff
    neg_threshold: float = -0.05          # cluster-guard / min-mode veto
    max_passes: int = 3
    bin_size_um: float = 2.0
    z_bound_um: float | None = None       # None → G * sqrt(2)
    cluster_guard_n: int = 3
    small_entity_guard_n: int = 0

    def __post_init__(self) -> None:
        if self.veto_mode not in ("min", "mean", "hybrid"):
            raise ValueError(
                f"rescue.veto_mode must be 'min'/'mean'/'hybrid'; got {self.veto_mode!r}"
            )
        if self.max_passes < 1:
            raise ValueError(
                f"rescue.max_passes must be >= 1; got {self.max_passes}"
            )
        if self.bin_size_um <= 0:
            raise ValueError(
                f"rescue.bin_size_um must be > 0; got {self.bin_size_um}"
            )


@dataclass(frozen=True)
class GroupConfig:
    """Stage-2 grouping — connected-components on bad-edge-pruned k-NN graph."""
    neighbor_threshold: float = -0.1
    k_neighbors: int = 8
    dist_threshold_um: float = 1.5
    min_comp_size: int = 4

    def __post_init__(self) -> None:
        if self.k_neighbors < 1:
            raise ValueError(f"group.k_neighbors must be >= 1; got {self.k_neighbors}")
        if self.min_comp_size < 1:
            raise ValueError(
                f"group.min_comp_size must be >= 1; got {self.min_comp_size}"
            )


@dataclass(frozen=True)
class StitchConfig:
    """Stage-3 stitcher — partial→partial, partial→cell, cell→cell merges."""
    metric: Literal["pmi", "magnitude", "pmi_sym"] = "pmi"
    threshold: float = 0.05
    dist_threshold_um: float = 5.0
    bin_size_xy_um: float = 2.0
    bin_size_z_um: float | None = None    # None → auto from data
    z_neighbor_depth: int = 1
    min_close_edges_dz: float | None = None   # None → auto
    min_close_edges_n: int = 5
    penalize_simplicity: bool = True
    delta_c_min: float = 0.0
    candidate_source: Literal["grid", "knn"] = "grid"
    neighborhood: Literal["4", "8"] = "8"
    mode: Literal["count", "tx"] = "count"

    def __post_init__(self) -> None:
        if self.metric not in ("pmi", "magnitude", "pmi_sym"):
            raise ValueError(f"stitch.metric invalid: {self.metric!r}")
        if self.candidate_source not in ("grid", "knn"):
            raise ValueError(
                f"stitch.candidate_source invalid: {self.candidate_source!r}"
            )


@dataclass(frozen=True)
class DemoteConfig:
    """Post-Stitch entity-size cutoff."""
    min_entity_size: int = 5

    def __post_init__(self) -> None:
        if self.min_entity_size < 1:
            raise ValueError(
                f"demote.min_entity_size must be >= 1; got {self.min_entity_size}"
            )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline config. `final_rescue` defaults to a copy of
    `rescue` with `small_entity_guard_n = 0`; override by passing an
    explicit `RescueConfig` or via the `[final_rescue] inherit = "rescue"`
    pattern in TOML."""
    phase1: Phase1Config = field(default_factory=Phase1Config)
    split_phase1: SplitPhase1Config = field(default_factory=SplitPhase1Config)
    phase1_qc: Phase1QcConfig = field(default_factory=Phase1QcConfig)
    rescue: RescueConfig = field(default_factory=RescueConfig)
    group: GroupConfig = field(default_factory=GroupConfig)
    stitch: StitchConfig = field(default_factory=StitchConfig)
    demote: DemoteConfig = field(default_factory=DemoteConfig)
    final_rescue: RescueConfig = field(
        default_factory=lambda: RescueConfig(small_entity_guard_n=0)
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIGS_DIR = _PKG_DIR / "configs"

_SECTION_TO_CLS: dict[str, type] = {
    "phase1": Phase1Config,
    "split_phase1": SplitPhase1Config,
    "phase1_qc": Phase1QcConfig,
    "rescue": RescueConfig,
    "group": GroupConfig,
    "stitch": StitchConfig,
    "demote": DemoteConfig,
    "final_rescue": RescueConfig,
}


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge — override wins, sections merge, scalars replace."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_inherit(merged: dict[str, Any]) -> dict[str, Any]:
    """Resolve any `inherit = "<other_section>"` directives.

    One level only: the listed section's resolved values are copied
    in, then the local keys override. The `inherit` key itself is
    stripped from the output. Raises on cycles or bad targets.
    """
    out = dict(merged)
    for section, body in list(merged.items()):
        if not isinstance(body, dict):
            continue
        target = body.get("inherit")
        if target is None:
            continue
        if target == section:
            raise ValueError(f"[{section}] inherits from itself")
        if target not in merged or not isinstance(merged[target], dict):
            raise ValueError(
                f"[{section}] inherits from missing section [{target}]"
            )
        if "inherit" in merged[target]:
            raise ValueError(
                f"[{section}] inherits from [{target}] which itself inherits "
                f"— transitive inherit not supported"
            )
        resolved = dict(merged[target])
        for k, v in body.items():
            if k == "inherit":
                continue
            resolved[k] = v
        out[section] = resolved
    return out


def _to_dataclass(merged: dict[str, Any]) -> PipelineConfig:
    """Map a resolved dict to PipelineConfig, ignoring unknown sections."""
    kwargs: dict[str, Any] = {}
    for section, cls in _SECTION_TO_CLS.items():
        body = merged.get(section, {})
        if not isinstance(body, dict):
            raise ValueError(f"section [{section}] must be a table; got {type(body).__name__}")
        # Filter unknown keys with a clear error rather than silently dropping.
        valid_fields = {f.name for f in fields(cls)}
        unknown = set(body) - valid_fields
        if unknown:
            raise ValueError(
                f"[{section}] contains unknown keys: {sorted(unknown)} "
                f"(valid: {sorted(valid_fields)})"
            )
        kwargs[section] = cls(**body)
    return PipelineConfig(**kwargs)


def load_config(
    path: str | Path | None = None,
    *,
    platform: str | None = None,
) -> PipelineConfig:
    """Load a pipeline config.

    Layering: ``configs/defaults.toml``  ← (optional) ``configs/platforms/<platform>.toml``
    ← (optional) ``path``. Each layer patches keys.

    Parameters
    ----------
    path
        Optional user-override TOML file. Top of the layer stack.
    platform
        Optional platform-preset name (file under ``configs/platforms/``,
        without the ``.toml`` suffix). E.g. ``"xenium_3d"`` or
        ``"vhd_unsegmented"``.

    Returns
    -------
    PipelineConfig
        Frozen, fully-resolved config.
    """
    defaults_path = _DEFAULT_CONFIGS_DIR / "defaults.toml"
    merged: dict[str, Any] = _load_toml(defaults_path) if defaults_path.exists() else {}

    if platform is not None:
        plat_path = _DEFAULT_CONFIGS_DIR / "platforms" / f"{platform}.toml"
        if not plat_path.exists():
            available = sorted(
                p.stem for p in (_DEFAULT_CONFIGS_DIR / "platforms").glob("*.toml")
            ) if (_DEFAULT_CONFIGS_DIR / "platforms").exists() else []
            raise FileNotFoundError(
                f"Unknown platform {platform!r}; available: {available}"
            )
        merged = _deep_merge(merged, _load_toml(plat_path))

    if path is not None:
        merged = _deep_merge(merged, _load_toml(Path(path)))

    merged = _resolve_inherit(merged)
    return _to_dataclass(merged)


# ---------------------------------------------------------------------------
# Run-receipt dumper (JSON — deps-free, machine-readable, easy to diff)
# ---------------------------------------------------------------------------


def to_dict(cfg: PipelineConfig) -> dict[str, Any]:
    """Recursively convert a PipelineConfig to a plain nested dict."""
    return asdict(cfg)


def dump_receipt(cfg: PipelineConfig, path: str | Path) -> None:
    """Write resolved config to JSON. Companion to a pipeline run; lets
    anyone reading the output later replay the exact same parameters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_dict(cfg), f, indent=2, sort_keys=True)


__all__ = [
    "Phase1Config",
    "SplitPhase1Config",
    "Phase1QcConfig",
    "RescueConfig",
    "GroupConfig",
    "StitchConfig",
    "DemoteConfig",
    "PipelineConfig",
    "load_config",
    "to_dict",
    "dump_receipt",
]
