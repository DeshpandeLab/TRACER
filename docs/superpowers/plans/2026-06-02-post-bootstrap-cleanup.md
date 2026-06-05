# Post-bootstrap-merge cleanup plan

**Branch:** `feature/post-bootstrap-cleanup` (off `optimization/core-refactor` @ `bd239c4` = `upstream/main`)
**Goal:** Remove functions/modules/notebooks that no longer serve the current SEG and NOSEG production pipelines.

## Method (how to read this plan)

Each candidate was verified by **direct grep** of every external file in `src/`, `tests/`, `tutorials/`, `benchmarks/`. The agent's first pass over-claimed (e.g. flagged `plot.py` and `density_cascade.py` as dead — they're not). The list below is **only what survived direct verification.**

| | Production callers |
|---|---|
| **SEG pipeline** | `tutorials/lung_cancer/segmented_workflow.ipynb` + `tests/_pipeline_runner.py` |
| **NOSEG pipeline** | `tutorials/lung_cancer/noseg_workflow.ipynb` |
| **Other current tutorials** | `lung_cancer.ipynb`, `algorithmic_changes.ipynb`, `kidney/compute_pmi_bootstrap.py`, `breast_cancer/run_breast_cancer.py`, `melanoma/run_melanoma.py`, `mouse_ileum/run_mouse_ileum.py`, `metrics_umap.ipynb` (3 copies) |
| **Live drivers** | `tests/_pipeline_runner.py`, `_pipeline_runner_tiled.py` |

A symbol is removable iff it has **zero callers** in *any* of the above.

---

## Tier 1 — Zero-risk removals (verified 0 external callers)

These delete with full confidence. No deprecation period needed; they already are deprecated.

### T1.1 `_legacy_dense_compute_npmi` — `src/tracer/metrics.py:107–249`
- Docstring header literally says `RETIRED`.
- `grep -rn "_legacy_dense_compute_npmi"` over the whole repo: **3 hits**, all in `src/tracer/{metrics.py,__init__.py}` (definition + comments). Zero test/tutorial/benchmark imports.
- **Action:** delete function body + the two referencing comments in `__init__.py`.

### T1.2 Four deprecated stitching wrappers — `src/tracer/stitching.py`
| symbol | line |
|---|---|
| `coherence_C_from_genes` | 660 |
| `coherence_C_from_genes_relu` | 678 |
| `deltaC_between_clusters` | 712 |
| `deltaC_between_clusters_relu` | 738 |

- Each emits `DeprecationWarning` on call.
- `grep "<name>("` outside `stitching.py`: **0 callers each** across `src/`, `tests/`, `tutorials/`, `benchmarks/`.
- **Action:** delete the 4 wrapper functions. Re-export removal (if any in `__init__.py`).

### T1.3 `src/tracer/spatial_kernel.py` — whole module
- `grep -rn "spatial_kernel"` outside `spatial_kernel.py`: **0 hits**. Not imported by `__init__.py`, not used by any pipeline file.
- **Action:** delete the file. Verify no `from .spatial_kernel import` lurks in any module before deletion.

### T1.4 `tutorials/lung_cancer/lung_cancer_npmi.ipynb`
- Referenced by **0 external files** (no other notebook/script/doc points at it).
- **Action:** delete notebook.

---

## Tier 2 — Low-risk removals (1 docs-only reference to update)

### T2.1 `tutorials/lung_cancer/legacy_workflow.ipynb`
- One external reference: `algorithmic_changes.ipynb` mentions it descriptively (`"the pre-this-branch pipeline"`) in markdown — not a code dependency.
- **Action:** delete the notebook; update the one markdown bullet in `algorithmic_changes.ipynb` to drop the legacy link.

---

## Tier 3 — Conditional removals (depend on retiring a sibling tutorial)

These modules have **exactly one** caller, and that caller is itself a candidate-retired notebook. If you confirm the notebook is retired, the module goes with it.

### T3.1 `src/tracer/tiling.py` (whole module) — pending `breast_cancer_npmi.ipynb` status
- Tiling exports (`metis_partition_cells`, `build_metis_partition_hulls`, `plot_metis_partitions`, `plot_metis_hulls`, `chunk_transcripts`) are imported only by:
  - `src/tracer/__init__.py` (re-export, lines 88–94)
  - **one notebook**: `tutorials/breast_cancer/breast_cancer_npmi.ipynb`
- The current breast_cancer pipeline is `run_breast_cancer.py` (does NOT import tiling). So `breast_cancer_npmi.ipynb` is structurally analogous to the already-retired `lung_cancer_npmi.ipynb`.
- **Decision required:** is `breast_cancer_npmi.ipynb` retired?
  - **If yes** → delete the notebook + tiling.py + its `__init__.py` re-export block.
  - **If no** → keep both.

---

## Not removable (verified active)

- **`src/tracer/plot.py`** — used by `metrics_umap.ipynb` in lung_cancer, breast_cancer, mouse_ileum (current tutorials).
- **`src/tracer/cc_scoring.py`** — used by `tests/refine_segmentation.py` and `3D_simulation.ipynb`.
- **`src/tracer/density_cascade.py`** — imported by `tests/_pipeline_runner.py` (`cascade_as_residual_handler`).
- **`src/tracer/phase1_rescue.py`** — used in the active SEG flow (need to grep before removing; flagged for re-verification only if a future cleanup goes after it).
- **`src/tracer/{spatial,pruning,stitching,graph,metrics,config,core,data,_etype,_kernels,_repro,_utils}.py`** — all active.

---

## Execution order

Do this in **one PR**, one commit per logical group, in this order so each step is independently revertable:

1. **Commit A — `chore(metrics): delete _legacy_dense_compute_npmi`**
   - Delete function body + comments in `__init__.py`.
   - Run `pytest tests/test_pmi_bootstrap.py` (must stay 41 passed).
2. **Commit B — `chore(stitching): drop 4 deprecated coherence/deltaC wrappers`**
   - Delete the 4 wrappers + any `__init__.py` re-exports.
   - Run full `pytest tests/` (whatever subset is green pre-change must stay green).
3. **Commit C — `chore(tracer): remove unused spatial_kernel module`**
   - Delete `src/tracer/spatial_kernel.py`.
4. **Commit D — `chore(tutorials): remove retired lung_cancer_npmi notebook`**
   - Delete `tutorials/lung_cancer/lung_cancer_npmi.ipynb`.
5. **Commit E — `chore(tutorials): remove legacy_workflow notebook + update algorithmic_changes link`**
   - Delete notebook; remove the markdown bullet.
6. **Commit F (conditional)** — only if user confirms `breast_cancer_npmi.ipynb` is retired:
   - `chore(tutorials,tracer): retire breast_cancer_npmi notebook + tiling module`
   - Delete notebook, `tiling.py`, and the `__init__.py` re-export block.

Each commit independently keeps tests green. PR title: `chore: post-bootstrap cleanup (retired functions, modules, notebooks)`.

---

## Estimated impact

- **Code**: ~250 lines (metrics legacy fn) + ~200 lines (4 stitching wrappers) + ~200 lines (spatial_kernel.py) ≈ **~650 lines** of `src/tracer/` removed.
- **Notebooks**: 1–2 files removed.
- **Conditional**: tiling.py (~600 lines) + breast_cancer_npmi.ipynb if T3.1 fires.
- **Public API surface**: 4 stitching exports + 5 tiling exports (conditional) drop from `__init__.py`. Anyone importing them externally gets a clean `ImportError` rather than a silently-changed behavior.

## Open questions before execution

1. **Is `breast_cancer_npmi.ipynb` retired?** (gates T3.1 / Commit F)
2. **Do you want a deprecation period** for the four stitching wrappers, or is "they emit a DeprecationWarning and have zero callers" enough? My recommendation: just delete (no period needed — there's no live caller to warn).
3. **Any external scripts (outside this repo) that might import the removed symbols?** This is the only true blind spot — the grep I ran is repo-internal only.
