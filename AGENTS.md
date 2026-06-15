# Repository Guidelines

## Branch lineage
This is the `gbm` branch of `DeshpandeLab/TRACER` — the active TRACER fork. It
adds the GBM tutorial (`tutorials/gbm/`) on top of `main`. To track upstream
changes from the fork's main: `git fetch origin && git merge origin/main`.
The container that runs this code is `ghcr.io/deshpandelab/tracer:gbm`, built
by `.github/workflows/build-docker.yml` on push to `gbm`.

## Project Structure & Module Organization
`src/tracer/` is the library. After the post-bootstrap refactor, `core.py` is a
compatibility shim re-exporting from per-phase modules: `pruning.py`
(`prune_transcripts_fast`, `build_dense_pmi_matrix_small_panel`,
`build_sparse_pmi_matrix`), `spatial.py` (`annotate_unassigned_components_fast`,
`enforce_spatial_coherence_fast`, reassignment helpers), `stitching.py`
(`apply_stitching_to_transcripts_memory_efficient`, entity stitching),
`cc_scoring.py` (purity/conflict), `graph.py` (graph builders), `_repro.py`
(reproducibility seeding), and `_cy_*.pyx` (Cython speedups). NPMI bootstrap
lives in `metrics.py` as `compute_pmi_bootstrap` (the legacy `compute_npmi` is
retired). Tutorials live under `tutorials/<dataset>/`; the GBM workflow uses
`tutorials/gbm/`.

## Build, Test, and Development Commands
`python -m pip install -e '.[dev]'` installs TRACER in editable mode with `pytest`, `black`, and `flake8`.

`python -m build --wheel --no-isolation -o dist` builds a wheel and compiles the Cython extensions for reproducible installs.

`python examples/refine_segmentation.py` runs the lightweight demo and writes images to `examples/output/`.

`python tutorials/mouse_ileum/run_mouse_ileum.py --run-smoke-test` runs the deterministic smoke test without a full dataset.

`docker build -t tracer .` mirrors the GitHub Actions container build.

## Coding Style & Naming Conventions
Use 4-space indentation, snake_case for functions and variables, and parameter names consistent with the current API (`cell_id_col`, `gene_col`, `coord_cols`). Format with Black defaults and keep Flake8 clean before opening a PR. Prefer reusable logic in `src/tracer/` and keep notebooks focused on analysis and visualization rather than core implementation.

## Testing Guidelines
The repo includes `pytest` as a dev dependency but does not yet have a committed `tests/` tree. Add new coverage under `tests/test_<module>.py` for library changes and keep tests deterministic by seeding RNG-dependent code. For data-heavy tutorial updates, pair unit tests with a smoke command or a small fixture instead of committing large generated outputs.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects such as `Update Docker image tag in build workflow` and `Add GitHub Actions workflow for Docker build`; follow that pattern and avoid vague messages like `Save changes`. Pull requests should describe the affected module or tutorial, list the commands you ran, note data or runtime implications, and attach screenshots only when plot or notebook output changed.

## Data & Output Hygiene
Large `.parquet`, `.csv.gz`, `.tsv`, most generated plots, and tutorial outputs are gitignored. Keep raw inputs under the matching `tutorials/*/data/` directory, write derived artifacts to `output/`, and only commit figures that are referenced by the README or tutorial documentation.
