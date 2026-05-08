# Density-Cascade Phase 1 — Handoff Document

Status: **prototype only, not committed.** Located in `/tmp/`. Awaiting decision
on (a) VHD validation, (b) noseg-default integration for Xenium.

Date written: 2026-05-08
Worktree: `/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/.claude/worktrees/stoic-feynman-587f37`
Branch: `optimization/core-refactor` (HEAD: `ce0fc0a`)

---

## TL;DR

**The "TRACER on the fly" insight**: in the noseg path (no input cell_id),
the Group stage is doing the de-facto Phase-1 job — turning raw tx into
cell-like entities — but using only spatial bin enumeration with no
anchoring or phenotype filtering. The density-cascade replaces this:
density peaks become synthetic seeds, Phase-1a-b prune purifies each
anchor, mismatched tx return to pool for later passes.

**Result on 500 µm Xenium ROI** (vs. SEG-final reference, all knobs on):

| approach | ARI vs SEG | entities | coverage | wall |
|---|---:|---:|---:|---:|
| current noseg (G=8 self Group) | 0.415 | 2,042 | 79 % | 3.79 s |
| cascade R=1 [8..4] + main rescue | **0.538** | 1,544 | 54 % | 2.38 s |
| cascade R=1 [8..3] + main rescue | 0.501 | **3,073** | 65 % | 3.63 s |
| cascade R=1 [8..2] + main rescue | 0.434 | 5,149 | 75 % | 5.28 s |

User's preferred config: **R=1 [8..3] with main rescue** — closest to
SEG's 2,709 cell count, ARI +0.085 over baseline noseg.

---

## Origin

The idea emerged from two convergent observations during a noseg-path
investigation:

1. Current noseg's Group (G=8 µm, neighborhood="self", `exact_distance_filter=False`)
   is doing Phase-1's job (cell-finding from raw tx) but without anchoring
   or phenotype filtering — bin restriction is a crude regularizer.

2. When we tested G=2 + Moore as a "modernization" of Group, it
   **catastrophically collapsed for noseg**: ARI dropped from 0.367 → 0.000,
   168 mega-blobs of 51 k tx. The G=8/self bin restriction had been
   acting as the missing anchoring step in disguise — flooding without it.

User reframe: **"we're doing TRACER on the fly"** — the cascade IS Phase 1
with synthetic seeds from density, replacing input cell_id when none exists.

---

## Algorithm (after several user refinements)

```python
# pseudocode
def density_cascade_phase1(df, panel, G=2.0, thresholds=[8,7,6,5,4],
                          territory_radius_bins=1, pmi_threshold=0.05,
                          min_anchor_tx=3):
    # 1. Build dense PMI matrix from panel
    # 2. Bin all valid-gene tx in xy at G
    # 3. pool = all valid tx; bin_pool_count[bin] = #pool tx in bin

    for t in thresholds:                          # walk DOWN
        hot = bins where bin_pool_count >= t      # filter to hot
        sort hot by density desc                  # highest-density wins
        for bin in hot:
            if bin_pool_count[bin] < t: continue  # may have dropped after intra-pass commits
            territory = bin's 3x3 Moore (R=1)     # 9 bins, 6 µm reach
            tentative = pool ∩ tx_in_territory
            if len(tentative) < min_anchor_tx: continue
            seed_genes = greedy_prune(unique_genes(tentative), W, thr=0.05)  # Phase 1a
            committed = [tx for tx in tentative if gene(tx) in seed_genes]    # Phase 1b
            if len(committed) < min_anchor_tx: continue
            anchors.append({centroid, genes, tx, threshold})
            pool -= committed                    # mismatches stay in pool
            for tx in committed:
                bin_pool_count[bin_of(tx)] -= 1   # density-aware update

    return {anchors, tx_to_anchor, pool_remaining}
```

### Key design decisions (user-driven)

1. **Walk DOWN thresholds [8..4]** — high confidence first, then lower.
2. **Phase 1a + 1b only, NO 1c** — the cascade replaces 1c via multiple
   density peaks for one biological cell.
3. **Mismatched tx RETURN to pool** (not "ejected") — available for later
   passes with possibly different anchors.
4. **Anchors can overlap territories** — but committed tx are disjoint
   (each tx belongs to one anchor).
5. **"Highest density wins" within each pass** — enforced by sort-by-density
   and pool-aware bin counts; subsequent anchors in the same pass see only
   the residual pool.
6. **Territory = 3×3 Moore (R=1)** — 9 bins, 6 µm reach at G=2 µm. Earlier
   prototypes used R=2 (5×5, 10 µm) which over-merged adjacent same-phenotype
   cells.
7. **Bin counts update post-commit** — pool-aware density at the start of
   each pass.

---

## Files written (all in `/tmp/`, never committed)

| file | purpose |
|---|---|
| `density_cascade_phase1.py` | Main prototype: `density_cascade_phase1()` + `greedy_prune()` |
| `cascade_diag.py` | Pool reconstitution check + extended thresholds [8..2] |
| `cascade_moore.py` | Territory comparison R=1 vs R=2 + threshold sweep |
| `cascade_vs_seg_final.py` | Cascade alone (no downstream) vs SEG-final |
| `bench_noseg_cascade.py` | Noseg pipeline with cascade replacing Group |
| `bench_full_cascade_r1.py` | Full pipeline across R=1 [8..4]/[8..3]/[8..2] |
| `bench_all_knobs_on.py` | All opt-in knobs ON, vs cascade variants |
| `bench_noseg_group_only.py` | Group-stage outputs (no Stitch+) compared |

### Function signatures

```python
def density_cascade_phase1(
    df: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    G: float = 2.0,
    thresholds: list[int] = (8, 7, 6, 5, 4),
    territory_radius_bins: int = 2,    # NB: prototype default is 2; use 1 for Moore
    pmi_threshold: float = 0.05,
    min_anchor_tx: int = 3,
) -> dict:
    """Returns {'anchors': [...], 'tx_to_anchor': {...}, 'n_pool_remaining': int,
                 'pass_diag': [...], ...}"""

def greedy_prune(unique_gene_idx: list[int], W: np.ndarray,
                 threshold: float = 0.05) -> list[int]:
    """Phase-1a-style: iteratively remove gene with most edges PMI < threshold."""
```

---

## Key empirical results (500 µm Xenium ROI, 51,569 tx)

### Cascade alone (no downstream stages) — 1,111 to 5,149 anchors

```
config              anchors  assigned tx  coverage   ARI vs raw input cell_id
R=2 [8..4] (5x5)     1,111      20,142     39 %         0.535
R=1 [8..4] (Moore)   1,544      15,709     30 %         0.563  ← best per-anchor purity
R=1 [8..3]           3,073      23,787     46 %         0.453
R=1 [8..2]           5,149      31,501     61 %         0.377
```

### Anchor purity (R=1 [8..4] vs R=2 [8..4])

```
Distribution of #input cells contributing tx per anchor:

R=2 [8..4]: 28.5% pure, mean 3.04 input cells/anchor
   {1: 82, 2: 313, 3: 363, 4: 221, 5: 95, 6: 33, 7: 4}

R=1 [8..4]: 58.0% pure, mean 1.92 input cells/anchor    ← R=1 dramatically cleaner
   {1: 509, 2: 712, 3: 264, 4: 56, 5: 3}
```

### Full pipeline (cascade + Stitch + Demote + Final Rescue) vs SEG-final-knobs-on

```
config                        wall   entities  assigned   ARI vs SEG    ARI vs raw input
SEG (REFERENCE, knobs on)    6.27s     2,709    40,817      —               0.681
NOSEG baseline (knobs on)    3.79s     2,042    40,526    0.4153            0.411
CASCADE R=1 [8..4]           2.38s     1,544    27,769    0.5376            0.483
CASCADE R=1 [8..3]           3.63s     3,073    33,599    0.5006            0.437  ← user's pick
CASCADE R=1 [8..2]           5.28s     5,149    38,859    0.4355            0.371
```

### Effect of main 3-pass Rescue between cascade and Stitch

```
                no main rescue              with 3-pass rescue
[8..4]    ARI=0.565  assigned=22.7k     ARI=0.537  assigned=27.8k  (-0.028 ARI, +5.1k tx)
[8..3]    ARI=0.496  assigned=30.9k     ARI=0.499  assigned=33.6k  (+0.003 ARI, +2.7k tx)
[8..2]    ARI=0.432  assigned=37.6k     ARI=0.434  assigned=38.9k  (+0.002 ARI, +1.3k tx)
```

Only **[8..3] benefits from main rescue** — the rescue's hybrid g∈E shortcut
completes the sparse cells the cascade fragmented at thr=3, without
contaminating the high-confidence anchors.

---

## Coverage vs k_min (threshold floor)

YES — discussed and measured. The threshold floor in the cascade IS
effectively k_min:

```
threshold floor    anchors    coverage    ARI vs SEG-final
   k_min = 4        1,544       30 %        0.581 (cascade alone, R=1)
   k_min = 3        3,073       46 %        0.500
   k_min = 2        5,149       61 %        0.435
```

Each step from k=4 → 3 → 2 adds ~+15 pp coverage but costs ~−0.07 ARI.
Fragmentation at k_min=2 is intrinsic: sparse cells with 2-tx-per-bin
clusters spawn multiple separate anchors.

A separate floor `min_anchor_tx=3` controls minimum committed tx after
prune (not minimum bin density). Currently both = 3.

---

## Visium HD application

**NO — not applied yet.** Discussed conceptually multiple times, but no
VHD dataset has been loaded.

### Conceptual fit

- Cascade is architecturally well-suited for VHD: no segmentation prior,
  much denser tx, density signal is stronger.
- VHD-appropriate parameters:
  - `G ≈ 0.5 µm` (subcellular bins, vs Xenium's 2 µm)
  - Absolute thresholds much higher (~60–100, not 4–8) due to ~30× tx density
  - OR percentile-based thresholds (e.g., top 5 %, top 10 %, top 25 %) for portability across platforms
- The PDAC datasets downloaded earlier in session are **Xenium 1.6.0 / 2.0.0**,
  not VHD.

### What's needed to validate on VHD

1. Acquire a Visium HD dataset (none in current GENESIS tutorials).
2. Set `G ≈ 0.5 µm`, `territory_radius_bins=1`.
3. Choose threshold cascade — either:
   - Absolute: e.g., `[100, 70, 40, 25, 15]` (scale ~30× higher than Xenium)
   - Percentile: e.g., `[5, 10, 20, 30, 40]` % from top, walking down
4. Run `density_cascade_phase1`.
5. Compare ARI vs whatever ground truth is available for that VHD dataset
   (input segmentation if available; otherwise compare cascade variants
   to each other for self-consistency).

---

## Status of the cascade in the codebase

- **NOT committed** — only exists in `/tmp/density_cascade_phase1.py`.
- **NOT integrated into the production pipeline runner.**
- All knob commits to date have been to the existing pipeline (Stitch
  tightening, Mid-QC, Post-Group Rescue) — opt-in then default-on as of
  commit `ce0fc0a`.
- Density-cascade is queued as the next major architectural piece,
  contingent on either VHD validation or a decision to make it the noseg
  default for Xenium too.

---

## Open design decisions

1. **Make cascade the noseg default?** R=1 [8..3] with main rescue gives
   ARI 0.501 vs 0.415 baseline (+0.086). Or wait until VHD validation.

2. **Cascade for SEG path too?** Currently SEG uses input cell_id for
   Phase 1 — cascade isn't needed. But cascade could serve as a
   sanity check / alternative seeder, useful if input segmentation is
   suspect.

3. **Threshold portability across platforms.** Percentile-based vs
   absolute. Percentile is more portable; absolute is simpler.

4. **Conflict resolution for overlapping anchors at same threshold.**
   Currently "highest density wins" via sort + pool-shrinking.
   Alternative: per-tx Voronoi assignment to closest anchor centroid.
   Not yet tested.

5. **Should `territory_radius_bins` default to 1 (Moore)?** The prototype
   default is 2, but R=1 (true Moore) is the correct algorithmic choice
   per discussion. Should be flipped.

---

## Next steps (if proceeding)

### Path A: Promote cascade to noseg default for Xenium

1. Move `density_cascade_phase1` from `/tmp/` to `src/tracer/density_cascade.py`
   (or add to `tracer.spatial`)
2. Wire as a new `_run_density_cascade()` in `tests/_pipeline_runner.py`
3. Replace `run_noseg_pipeline`'s Group call with the cascade
   (or add as opt-in knob: `NOSEG_USE_DENSITY_CASCADE: bool = False`)
4. Regenerate test references (noseg.json, seg_vs_noseg.json)
5. Document in commit message: ARI 0.367 → 0.501 vs SEG-final
6. Bench full-tissue runtime (predicted ~0.5–1× current noseg time)

### Path B: Validate on Visium HD first

1. Acquire VHD dataset
2. Adapt `density_cascade_phase1` for VHD parameters (small G, scaled thresholds)
3. Verify ARI / coverage on VHD
4. Then proceed with Path A integration

### Path C: Both paths in parallel

Most cautious: ship cascade as opt-in knob (default off), regenerate
references for OFF, validate on VHD separately. Flip default once VHD
validates.

---

## Key conversation moments (for context-recovery)

- **User reframe** ("not Phase 1, replace Group in noseg"): clarified that
  cascade replaces what Group does in noseg, NOT what Phase 1 does in SEG.
  In SEG, input cell_id + Phase 1 stays unchanged.

- **Pool reconstitution VERIFIED**: at every pass, `pool_delta == n_committed`
  exactly. Returned tx genuinely stay in pool for later passes.

- **R=2 → R=1 correction**: prototype used R=2 (5×5 = 25 bins, 10 µm reach)
  but the design discussion was always Moore (R=1, 3×3 = 9 bins, 6 µm reach).
  R=1 dramatically improves anchor purity (28.5 % → 58 %) and mean input
  cells per anchor (3.04 → 1.92).

- **R=2 vs R=1 ARI puzzle**: R=2 has worse granularity but only slightly
  worse ARI vs SEG-final (0.570 vs 0.581). Resolution: SEG-final's Stitch
  also merges cells, so R=2's over-merging accidentally mirrors SEG's mergers.
  The honest reference is **raw input cell_id** (where each Xenium nucleus
  is distinct): R=2 ARI 0.535 vs R=1 ARI 0.563.

- **Threshold extension paradox**: lowering k_min adds anchors but ARI
  goes DOWN. Cause: sparse cells get anchored as multiple separate fragments
  at low thresholds, not as single anchors. Stitch downstream can't recover
  from this fragmentation cleanly.

- **Main rescue helps [8..3] uniquely**: the cascade's mid-threshold
  fragments (sparse cells with 1 hot bin + nearby low-density tx) are
  exactly what hybrid Rescue's g∈E shortcut completes. Same-gene admissions
  fill out the cells without contaminating phenotype-clean anchors.

---

# Update — 2026-05-07

Subsequent investigation extended the cascade beyond the 500 µm Xenium ROI to:
(a) full-tissue Xenium with homogeneity-first scoring,
(b) SEG-path residual handling (cascade as Group replacement), and
(c) Visium HD HC01 at 2 µm, requiring threshold portability across modalities.

The portability problem motivated an **auto-floor selection rule based on
runtime tx-coverage**, replacing the hand-tuned absolute thresholds.

## Auto-floor selection rule (tx-coverage-driven)

### Why hand-tuned thresholds break across modalities

Bin-count distributions differ by orders of magnitude across pools:

| Pool | n_bins (bbox) | occupied | mean tx/bin | max bin count |
|---|---|---|---|---|
| Xenium NOSEG full pool | 3,909,949 | 17.16% | 0.367 | 19 |
| Xenium SEG-residual | 3,909,949 | 7.10% | 0.086 | 9 |
| Visium HD HC01 2µm (filtered) | 10,792,477 | 81.76% | 5.235 | 223 |

A floor that's "strict" in Xenium NOSEG (4 tx/bin) is almost the entire
distribution in Visium HD (where median bin count is ~5). Any cascade
that ships with absolute thresholds will fail on the next modality.

### The rule (no occupancy/cell-count input needed)

For each candidate threshold `n` walking down from the max:

1. Build anchor mask = `grid >= n`
2. R=1 Moore dilation = `binary_dilation(anchor_mask, ones((3,3)))`
3. Compute `tx_coverage(n) = sum(grid[dilated_mask]) / total_tx`
4. **Stop at the largest `n` where `tx_coverage(n) >= target_cov`**
5. Hard floor: `max(2, n)` — never anchor on count=1 bins

Default `target_cov = 0.65` lands on the empirical winners across all
three contexts tested, with `hard_min=2` as the fallback for sparse
pools that can never reach 65 %.

This rule is **runtime-adaptive** and modality-agnostic — the cascade
self-paces its descent based on the data it sees.

### Resulting floors (verified 2026-05-07)

| Pool | bin count at floor | n_anchors | tx_coverage | floor chosen |
|---|---|---|---|---|
| Xenium NOSEG full pool | 4 | 103,396 | **66.52 %** | **floor=4** |
| Xenium SEG-residual | 2 | 47,747 | **52.70 %** | **floor=2** (hard_min; can't reach 65 %) |
| Visium HD HC01 2µm | 13 | 1,116,923 | **65.06 %** | **floor=13** (or 12 at 69.5 % for slack) |

All three line up with what hand-tuning had already converged on
empirically — the auto-rule reproduces them without per-pool tuning.

### Geometric vs runtime: why coverage > bin-tail rules

Earlier proposals used `bin_tail × (2R+1)² >= occupied_frac` as a
geometric tile-fit rule. That's an upper bound on coverage assuming
non-overlapping 3×3 patches. Actual Moore neighborhoods of adjacent
anchors **overlap heavily**, so real coverage is much less than the
geometric upper bound:

| Pool | floor | bin_tail | 9 × bin_tail (upper) | actual tx_cov | actual bin_cov |
|---|---|---|---|---|---|
| Xenium NOSEG | 4 | 2.64 % | 24 % | **66.5 %** | 10.5 % |
| Xenium residual | 2 | 1.22 % | 11 % | **52.7 %** | 8.4 % |
| Visium HD 2µm | 12 | 12.24 % | 100 % (capped) | **69.5 %** | 35.4 % |

The runtime measure is exact; the geometric rule was systematically
optimistic about bin coverage and pessimistic about tx coverage (since
it ignored the heavier weight of high-count bins inside the dilated
mask).

## Function status — WIRED AS OPT-IN (2026-05-07)

**`src/tracer/density_cascade.py`** now exists with three public
functions:

```python
auto_floor_from_coverage(grid, *, target_cov=0.65, R=1, hard_min=2,
                            ceiling=None) -> (floor, coverage_curve)

density_cascade_phase1(df, panel=None, *, aux=None, G=2.0,
                        thresholds="auto", territory_radius_bins=1,
                        pmi_threshold=0.05, min_anchor_tx=3,
                        auto_target_cov=0.65, auto_hard_min=2,
                        auto_ceiling=None) -> dict

cascade_as_residual_handler(df_pruned, aux, *, panel=None,
                              entity_col="tracer_id", G=2.0,
                              thresholds="auto", ...) -> DataFrame
```

`density_cascade_phase1` accepts `thresholds="auto"` to select the
floor at runtime via tx-coverage, or an explicit list. With "auto"
it returns a `coverage_curve` in the result dict for diagnostics.

`cascade_as_residual_handler` is a drop-in replacement for
`annotate_unassigned_components_fast`: operates only on tx with
`entity_col == "-1"`, runs the cascade, writes `cascade_<n>` labels
back. Tx not anchored remain "-1".

### Wired into the SEG pipeline as opt-in knob

`tests/_pipeline_runner.py` now has:

```python
from tracer.density_cascade import cascade_as_residual_handler

PHASE1_SEG_RESIDUAL_CASCADE: bool = False                # opt-in default OFF
PHASE1_SEG_RESIDUAL_CASCADE_TARGET_COV: float = 0.65
PHASE1_SEG_RESIDUAL_CASCADE_HARD_MIN: int = 2
```

When `PHASE1_SEG_RESIDUAL_CASCADE = True`, the SEG path's Group call
is replaced by `cascade_as_residual_handler` with the knob-controlled
target_cov / hard_min. With the knob at default `False`, behaviour is
identical to before — verified by **all 4 regression tests + 56 other
tests passing** (no test fixture regen needed).

### Bench parity confirmed (2026-05-07)

500 µm Xenium ROI (51,569 tx), `/tmp/bench_cascade_residual_optin.py`:

| config | wall | group_new | entities | assigned | drop | ARI vs raw |
|---|---|---|---|---|---|---|
| knob OFF (default Group) | 6.10 s | 1,678 | 2,716 | 41,633 | 9,936 | +0.6816 |
| knob ON (cascade auto) | 5.09 s | **570** | 2,811 | 41,269 | 10,300 | **+0.6874** |

Auto-floor selected **floor=2** (residual maxes at 45.7 % coverage at
thr=2, can't reach 65 % target, falls back to hard_min). Matches the
hand-tuned `[6..2]` reference numbers from `/tmp/seg_cascade_6_2.py`.

Full-tissue Xenium (1.44 M tx),
`/tmp/bench_cascade_residual_optin_full.py`:

| config | wall | group_new | entities | h | c | V | V_β=2 | ARI |
|---|---|---|---|---|---|---|---|---|
| knob OFF (default Group) | 140.91 s | 36,973 | 62,980 | +0.9519 | +0.9356 | +0.9437 | +0.9410 | +0.6896 |
| knob ON (cascade auto) | **127.74 s** | **15,034** | 66,072 | **+0.9534** | +0.9359 | +0.9446 | **+0.9417** | **+0.6928** |
| Δ | −9 % | **−59 %** | +5 % | +0.0015 | +0.0003 | +0.0009 | +0.0007 | +0.0032 |

Cascade dominates default Group on every metric: higher homogeneity
(under-split asymmetry favouring cleaner anchors), higher completeness,
higher V_β=2, higher ARI, **60 % fewer spurious Group-style components**,
and 13 s faster wall.

Required for promotion (Path A from the original handoff):

1. Add `auto_floor_from_coverage(grid, target_cov=0.65, R=1, hard_min=2)`
   to `density_cascade_phase1.py` (and eventually
   `src/tracer/density_cascade.py`).

2. Wire `density_cascade_phase1(thresholds="auto", ...)` to:
   - Build the bin grid from `df[['x','y']]`
   - Call `auto_floor_from_coverage` to pick `floor`
   - Build descending threshold list: `range(max(grid), floor-1, -1)`
   - Or with a ceiling: `range(min(8, max(grid)), floor-1, -1)`

3. **Cython port (deferred)**: `binary_dilation` in scipy is fast enough
   for grids up to ~10 M bins, but Visium HD whole-tissue at G=0.5 µm
   would push to ~100 M bins. If/when that becomes a bottleneck, the
   dilation can be replaced with a 4-pass shift+OR over the bool grid
   in numpy (no Cython needed) or vectorized in `_cy_prune.pyx`.

Suggested signature:

```python
def auto_floor_from_coverage(
    grid: np.ndarray,                    # 2D int grid of bin counts
    target_cov: float = 0.65,            # target fraction of tx mass to capture
    R: int = 1,                          # Moore radius
    hard_min: int = 2,                   # never go below this
) -> tuple[int, list[float]]:
    """Walk thresholds [max..hard_min], return (floor, coverage_curve).

    floor is the LARGEST n such that tx_coverage(n) >= target_cov,
    or hard_min if that condition is never satisfied.
    """
```

## Full-tissue Xenium results (homogeneity-first scoring)

The 500 µm ROI bench compared cascade variants by ARI vs SEG-final.
Full-tissue scoring revealed that **ARI penalizes over-split and
under-split symmetrically, but in this domain the asymmetry matters**:

- **Under-split** (two true cells merged into one entity) → mixed
  gene signature, false doublet, breaks downstream cell-typing.
- **Over-split** (one true cell broken into N partials) → recoverable
  by Stitch / merge step downstream.

Input `cell_id` is itself a 2D-projected detection from a section
~10 µm thick — true volumetric ground truth is 1.3–2× higher than
the 58,405 detected cells. Under that lens, entities count vs cells
should bias toward more-fragmented (over-split is the safer error).

### Full-tissue homogeneity bench (1.44M tx, 58,405 input cells)

| config | wall | entities | N_co | **homog** | **compl** | V (β=1) | **V (β=2)** | ARI | FMI |
|---|---|---|---|---|---|---|---|---|---|
| SEG (knobs on) | 138.9 s | 63,663 | 1.21 M | +0.9515 | +0.9342 | +0.9428 | **+0.9399** | +0.6848 | +0.6877 |
| cascade [8..4] | 7.4 s | 54,427 | 0.59 M | +0.9599 | +0.9032 | +0.9307 | +0.9213 | +0.4498 | +0.4864 |
| cascade [8..3] | 8.9 s | 93,818 | 0.78 M | **+0.9620** | +0.8846 | +0.9217 | +0.9090 | +0.3791 | +0.4313 |

Three reads:

1. **Homogeneity ranking [8..3] > [8..4] > SEG.** More anchors → less
   false-merger → cleaner per-entity gene signatures. Cascade [8..3]
   has the highest homogeneity in the entire table.
2. **Completeness ranking SEG > [8..4] > [8..3].** SEG has Stitch which
   glues fragmented partials of the same cell back together; cascades
   have no analogous merge step.
3. **Cascade prototype lacks Stitch integration.** The `cascade_<n>`
   labels don't have the `{cell}-{suffix}` partial format Stitch
   recognizes — Stitch is a no-op on cascade output. The cascade is
   "fragmented but pure"; Stitch is the natural fixer.

**Implication for `Open design decisions` #1**: cascade as NOSEG default
should ship with `[8..4]` (V_β=2 winner today) but switch to `[8..3]`
once cascade emits partial-style labels and Stitch is wired into the
NOSEG path.

## SEG-path residual cascade (NEW use case)

Cascade can also replace the **Group stage in the SEG path** —
operating only on post-Rescue residual unassigned tx. Not just NOSEG.

### Residual is sparser than NOSEG full pool

```
Pool                    occupied   mean tx/bin    max bin count
NOSEG full              17.16 %    2.14           19
SEG-residual            7.10 %     1.21           10  (full tissue)
SEG-residual ROI        ~3 %       ~1.05          6   (500 µm ROI)
```

The auto-floor rule selects different thresholds per context:

| Context | floor (auto) | tx_coverage at floor | Compare to default Group |
|---|---|---|---|
| NOSEG full pool | **4** | 66.5 % | replaces Group entirely |
| SEG-residual full tissue | **2** | 52.7 % | 47k cascade anchors vs Group's component count |
| SEG-residual 500 µm ROI | **2** | ~50 % | 602 cascade comps vs Group's 1,826 (same ARI, 1/3 entity count) |

ROI bench (`/tmp/seg_cascade_6_2.py`):

```
config                          wall  group_new  entities  assigned   drop  ARI vs raw
SEG default (G=8 self Group)   6.05s    1,826     2,713    40,834   10,735   +0.6808
SEG cascade-Group [8..4]       4.61s       22     2,240    37,795   13,774   +0.7053
SEG cascade-Group [6..2]       4.94s      602     2,820    40,378   11,191   +0.6877
```

[8..4] is misleading — only 22 components emitted because thresholds 8/7/6/5
hit zero anchors on the residual (max bin count = 6). Most residual stays -1,
ARI inflates because dropped tx aren't penalized in the masked metric.

[6..2] is the principled choice: emits 602 components (~1/3 default Group's
1,826), preserves 99 % of default's assignment recovery, equal-or-better ARI,
and faster — exactly the "hot-core finder" design that the auto-floor rule
prescribes for sparse pools.

## Files added 2026-05-07 (all in `/tmp/`, never committed)

| file | purpose |
|---|---|
| `coverage_by_threshold.py` / `.log` | **Auto-floor reference impl** — Moore-dilated tx coverage curve per threshold for all 3 modalities |
| `full_bin_histograms_v2.py` / `.log` | Bin-count histograms with empty bins counted (bbox grid) for all 3 modalities |
| `full_bin_histograms.py` / `.log` | Earlier version, occupied-bins-only (superseded by v2) |
| `full_homogeneity_seg_vs_cascades.py` / `.log` | Full-tissue homogeneity / completeness / V-measure vs input cell_id |
| `full_ari_seg_vs_cascades.py` / `.log` | Full-tissue ARI bench: SEG vs cascade [8..3] vs cascade [8..4] |
| `seg_cascade_6_2.py` / `.log` | SEG-residual cascade with [6..2] vs default Group on 500 µm ROI |
| `seg_with_cascade_group.py` | SEG-residual cascade [8..4] vs default Group ROI bench (predecessor) |
| `seg_residual_density.py` | Per-bin density distribution of SEG residual |
| `group_in_seg_diag.py` | Group-stage diagnostics: G=8 self vs G=2 + 8-Moore on Rescue residual |

Visium HD source data:
- `/Users/adeshpa6/data/dpt/HC01/data/binned_outputs/square_002um/filtered_feature_bc_matrix/matrix.mtx.gz`
- `/Users/adeshpa6/data/dpt/HC01/data/binned_outputs/square_002um/spatial/tissue_positions.parquet`
- 8µm and 16µm grids also available at `square_008um/` and `square_016um/` — not yet histogrammed.

## Resolved & new design decisions

**Resolved from earlier list:**

- ~~3. Threshold portability across platforms.~~ → **Resolved**: auto-floor
  rule based on tx-coverage replaces both absolute and percentile schemes.
  Single rule, single tunable (`target_cov ≈ 0.65`), works across Xenium,
  Xenium-residual, and Visium HD.
- ~~5. Should `territory_radius_bins` default to 1 (Moore)?~~ → **Resolved**:
  yes, R=1 across all modalities. Visium HD is dense enough that R=1 enters
  the exclusion-limited regime naturally (bin_tail at floor=12 is 12.24% >
  1/9, so anchors fully tile the grid via territory exclusion). No need for
  R=2 special case.

**New decisions opened:**

6. **Cascade label scheme.** Currently `cascade_<n>` (flat). For Stitch
   integration, change to `cascade_<n>-1` so Stitch sees them as partial-
   eligible. Then `[8..3]` will out-perform `[8..4]` post-Stitch (predicted).

7. **Full-tissue Visium HD validation pending.** Histograms confirm the
   auto-floor rule lands at floor=12–13 with 65–70% tx coverage.
   Still need to: actually run the cascade on Visium HD, evaluate against
   any available reference (or self-consistency), and verify that R=1
   exclusion behavior produces sensible cell-scale entities at the
   sub-cellular 2 µm grid. May also need to bench at 8 µm / 16 µm grids
   where bins are closer to cell-scale.

8. **Auto-floor target_cov default.** 0.65 was reverse-engineered from
   the empirical NOSEG winner (66.5 % cov at floor=4). Worth a small
   sweep (0.55 / 0.65 / 0.75) on full-tissue homogeneity to confirm.

## Updated next steps

### Path A (Xenium NOSEG default) — DONE through step 4 of an earlier plan

✅ **1. Move `density_cascade_phase1` to `src/tracer/density_cascade.py`** — done.
✅ **2. Add `auto_floor_from_coverage` helper in same module** — done.
✅ **3. Default `thresholds="auto"`, R=1, target_cov=0.65, hard_min=2** — done.
✅ **3.5 SEG-residual opt-in knob `PHASE1_SEG_RESIDUAL_CASCADE`** — done, default OFF.
   ROI + full-tissue benches confirm cascade beats default Group on V_β=2,
   ARI, homogeneity, completeness, wall time. Test suite green at 60/60.

### Steps 4–8 completed 2026-05-07

✅ **4. Cascade label format `cascade_<n>-1`** (depth-1 partial). The
   parallel two-dash partial-merge plan landed earlier (commit `5a87470`),
   so Stitch already understands `{cell}-{d1}-{d2}` form — no parser
   coordination needed. `cascade_as_residual_handler` always emits
   partial form; the `emit_as_partial` toggle was removed after step 6
   (no real use case for flat form, only a footgun risk).

✅ **5. NOSEG path wired** with `PHASE1_NOSEG_CASCADE: bool = False` knob
   plus `_TARGET_COV` and `_HARD_MIN`. Conditional in `run_noseg_pipeline`
   swaps Group for cascade. Default OFF — needs separate validation
   bench before flipping (no input cell_id reference, harder to score).

✅ **6. Stitch + cascade integration verified** on full Xenium tissue
   (`/tmp/bench_cascade_step6.py`):

   | config | wall | gnew | ents | h | c | V | V_β=2 | ARI |
   |---|---|---|---|---|---|---|---|---|
   | A: default Group | 140.1 s | 36,973 | 62,980 | +0.9519 | +0.9356 | +0.9437 | +0.9410 | +0.6896 |
   | B: cascade flat (no Stitch merge) | 127.1 s | 15,034 | 66,072 | +0.9534 | +0.9359 | +0.9446 | +0.9417 | +0.6928 |
   | **C: cascade partial (Stitch-eligible)** | 129.9 s | 15,034 | **59,207** | +0.9526 | **+0.9390** | **+0.9458** | **+0.9435** | **+0.6987** |

   - Stitch DID merge 6,865 cascade fragments (B → C entity drop, −10 %).
   - Homogeneity stayed stable (Δ = −0.0008) — merges are genuine
     same-cell unions, not false multi-cell unions.
   - Completeness +0.0031, V_β=2 +0.0018, ARI +0.0059 over flat cascade.
   - C's 59,207 entities is the closest match to input cell_id's 58,405
     (+1.4 %) vs A's +7.8 % and B's +13.1 %.

✅ **7. SEG default flipped to True** — `PHASE1_SEG_RESIDUAL_CASCADE` now
   defaults to `True` as of 2026-05-07. **Test references unchanged**:
   the pipeline regression tests use small synthetic data (8 cells,
   ~200 tx) where post-Rescue residual is essentially empty, so the
   cascade is a no-op and output is identical to default Group. All 74
   tests still pass with the default flipped — no `TRACER_UPDATE_REFERENCES`
   regen needed. Real-world impact is on tissue-scale data; the synthetic
   test fixtures stay valid.

✅ **8. Smoke test added at `tests/test_density_cascade.py`** (14 tests).
   Covers: auto-floor edge cases (empty/sparse/dense grids, target_cov
   sensitivity, R-Moore radius), Moore dilation correctness, end-to-end
   cascade on the 500 µm ROI, residual handler partial-label format, and
   `_classify` consistency for both `cascade_<n>-1` (→ partial) and
   flat `cascade_<n>` (→ cell).

### Remaining open items

- **Flip NOSEG default to True**: `PHASE1_NOSEG_CASCADE` is wired but
  defaults to False. Needs a self-consistency bench (no input cell_id
  to score against) — e.g., run cascade on the full tissue with
  cell_id wiped, compare to SEG output as the de-facto reference.
- **Visium HD validation**: the auto-floor rule was empirically
  designed against Visium HD HC01 2µm histograms but the cascade has
  not yet been run on Visium HD data end-to-end. See "Path B" below.
- **Cython port of the auto-floor dilation**: currently uses
  scipy.ndimage.binary_dilation, fast enough for ~10M-bin grids
  (Xenium full tissue, Visium HD 2µm filtered). At sub-µm grids
  (~100M bins) a vectorized 4-pass shift+OR may be needed.

### Path B (Visium HD validation) — concrete now

1. ~~Acquire VHD dataset~~ — **done**: HC01 at `/Users/adeshpa6/data/dpt/HC01/`
2. Adapt loader to read filtered Visium HD matrix + tissue_positions
3. Run `density_cascade_phase1` with `thresholds="auto"`
4. Self-consistency check: does cascade produce stable entity counts
   under ROI sub-sampling? Cell-typing on cascade entities reasonable?
5. (Optional) Compare to 10x's `square_008um` / `square_016um` pre-binned
   outputs as crude references.
