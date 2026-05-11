# Phase1-Rerank bench plan

Run after this implementation merges. See spec §8 for the design:
[docs/superpowers/specs/2026-05-11-phase1-rerank-design.md](../docs/superpowers/specs/2026-05-11-phase1-rerank-design.md).

The bench should reuse `benchmarks/bench_reassign_1c_coherence.py` as a
template — same ROI/loader/coherence definitions. Sweep both toggles
(`PHASE1_RERANK_ENABLED`, `PHASE1_REASSIGN_AFTER_1C`) × 3 ROIs (NW, C,
SE). Output: `benchmarks/phase1_rerank_sweep.{json,log,partitions.parquet}`.

Promotion-to-default-on gate: positive cell-count Δ from Case B + no
coherence regression on shared cells + ARI vs off/off baseline ≥ ~0.97.
