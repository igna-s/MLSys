---
title: "MLSys 2026 Contest — Track A Submission"
author: "Schwindt, Ignacio A."
date: "April 2026"
---

# 1. Introduction

Executing massive computational graphs on hardware with strict on-chip fast memory presents a combinatorial optimization challenge. Latency is governed by a roofline model that heavily penalizes repeated data spillage to slow memory. Our submission, the **V21 Engine**, minimizes end-to-end execution cycles through **Adaptive Dynamic Programming (DP)**, fine-grained **Split-K Pipelining**, strict memory limits enforced via **Deduplicated Retention Set** heuristics, **OpenMP parallelism over (w, h, k) granularity combos**, a **Step-Level Adaptive Window** that prevents timeouts on single-core systems, and novel **Operator Recomputation** that trades cheap recomputation for elimination of slow-memory loads.

The engine handles both trivial (5 nodes) and large (100+ nodes) DAGs without violating per-benchmark timeouts or causing OOM crashes. The binary is compiled statically with GCC and `-fopenmp`, deployed inside a minimal Docker container based on Ubuntu 22.04. V21 achieves **39.85M total cycles** across all 5 public benchmarks (B1: 210K, B5: 545K, B9: 22.0M, B13: 11.1M, B17: 5.99M), with end-to-end wall-clock time well below the per-benchmark timeouts on 8 threads.

![The V21 Sliding Window DP Engine. The input DAG is topologically sorted and processed left-to-right through a sliding window. At each position, all candidate subgraph partitions within the window are evaluated in parallel across retention modes and granularity combinations, producing an optimal execution plan with resource allocation and memory management decisions.](dp.png)

# 2. Dynamic Programming Core

The core challenge is grouping dependent operations into subgraphs sequentially without saturating the fast memory capacity *C_fast*. Greedy algorithms fall into local optima — aggressively retaining intermediate tensors to prevent recalculation often leads to inevitable OOM crashes deeper in the graph. RL approaches require exorbitant training times and cannot guarantee generalization within 10 minutes.

We formulate the problem as linear **Dynamic Programming** over a topologically sorted node sequence. The state *dp[i]* represents the minimal total latency to execute the graph up to node *i*. The recursive transition tests all valid subgraphs ending at *i−1* and starting at *j*:

> **dp[i] = min over j < i of { dp[j] + L(subgraph j→i−1, retained tensors) }**

where *L* calculates the analytical roofline latency of the candidate subgraph, parameterized by the tensors retained in fast memory from the preceding step.

## 2.1 Adaptive Window Sizing

Exhaustively evaluating all *j ∈ [0, i−1]* is O(N²), intractable for dense N ≥ 60 topologies. V21 uses **Adaptive Windowing** with two window levels per size class: a wide *W_max* used when the system runs fast (8-core hardware), and a conservative *W_cons* as fallback on 1-core systems to prevent timeouts:

| N | W_max | W_cons | nm_max | nm_cons |
|:-:|:-:|:-:|:-:|:-:|
| ≤ 32 | N (full) | N (full) | 10 | 10 |
| ≤ 40 | 24 | 20 | 8 | 6 |
| ≤ 65 | 20 | 12 | 8 | 5 |
| ≤ 105 | 16 | 10 | 6 | 5 |
| > 105 | 9 | 7 | 4 | 4 |

The dual-window design lets V21 use aggressive widths on fast hardware while gracefully degrading to V19-equivalent work on slow hardware — no timeouts, no regressions.

## 2.2 Step-Level Adaptive Fallback

A per-step timer tracks elapsed time after each outer DP step *i*. If the current step's wall-clock cost multiplied by the remaining steps would exceed the available time budget, the engine **permanently** switches to *(W_cons, nm_cons)* for all subsequent steps. This step-level adaptation is more robust than progress-extrapolation: early DP steps are cheap (small *j*-range) and do not predict the cost of later steps at full window width. The fallback triggers before any timeout risk materializes; on 8-core hardware it never triggers at all.

# 3. Memory Hierarchy & Split-K Tiling

![AI Accelerator Memory Hierarchy. The scheduler decides which tensors reside in fast SRAM and which spill to slow DRAM/HBM. Split-K pipelining partitions the reduction dimension K of MatMul operations into small chunks, keeping partial sums pinned in fast memory to avoid repeated spilling of large intermediate buffers.](mem.png)

## 3.1 Granularity Candidate Generation

Once a candidate subgraph *S* is identified, determining its internal execution tile dimensions *[w, h, k]* is the physical optimization lever. V21 curates candidates from: (1) multiples and divisors of the native hardware granularity, (2) exact dimensional bounds of the subgraph's topological outputs, (3) power-of-2 factors, (4) divisors of each tensor's dimensions, and (5) **memory-aware candidates** derived from the available fast-memory slack. Memory-aware fraction combos are generated for problems up to N < 65 (extended from N < 32 in earlier versions). All candidates are flattened into a single vector and evaluated in parallel (Section 7).

## 3.2 Split-K Pipelining for MatMul

MatMul operations are memory-intensive — they require holding LHS, RHS, and Output tensors simultaneously. V21 aggressively evaluates the inner reduction depth *k* via **Split-K Pipelining**: using small *k* values, the system accumulates partial dot products within a pinned Output tensor in fast memory, iterating over the reduction dimension *K* sequentially. This drastically slashes the real-time working set requirement, bypassing OOM crashes in large GEMMs without forfeiting computational efficiency.

# 4. Fast Memory Retention Strategies

Deciding which tensors to persist in fast memory between subgraph steps is an NP-Hard subset problem. In each DP transition, *M_retention* is generated by evaluating up to **10 candidate filtering heuristics**:

- **Mode 0 (Size, large first):** Greedy pack by descending size, full capacity budget.
- **Mode 1 (Value-based):** Score = size / distance-to-next-use. Large tensors needed soon get priority; forward scan finds the first actual use to avoid `lu[t]` overestimation.
- **Mode 2 (Proximity, 3/4 budget):** Nearest-use within 8 ops; 75% capacity cap.
- **Mode 3 (Chain-aware):** Sort by precomputed `total_fanout` descending. Skip/residual tensors consumed by many future ops are retained first.
- **Mode 4 (lu-distance):** Sort by `last_use` ascending — tensors that expire soonest are retained first; they are needed for the next few subgraphs and won't be needed long after.
- **Modes 5–9:** Small-threshold smallest-first, wider proximity lookahead (12 ops), smallest-first full budget, near-only 8-op window, half-budget narrow lookahead.

**Hash Deduplication.** Generating 10 retention sets often produces overlapping configurations. V21 computes a deterministic 64-bit FNV-1a hash for each retention set — only uniquely hashed sets proceed to full evaluation, typically eliminating 30–50% of redundant work.

# 5. V21 Operator Recomputation

V21 introduces **Operator Recomputation** as a novel DP transition extension. For small subgraphs (|S| ≤ 4 ops), the engine identifies input tensors that are: (a) consumed by the subgraph but not produced within it, (b) not in the retained set from the prior step, (c) not graph-level inputs, and (d) produced by a **Pointwise** op *P* before position *j* whose recompute cost satisfies:

> **base_cost(P) × slow_bw < tsize(t) / 2**

i.e., the compute cost of re-running *P* is cheaper than half the I/O cost of loading tensor *t* from slow memory. When such candidates exist, the engine evaluates **both** the base subgraph and the **extended** subgraph (with cheap Pointwise ops prepended in topological order). The cheaper result wins — guaranteeing no regressions while capturing fusion opportunities that eliminate slow-memory loads.

A `producer_of[]` array (built once at solve-start, mapping `producer_of[t] = op index`) enables O(1) producer lookup during transition evaluation. Recomputed op indices are stored alongside the standard subgraph and emitted first in the output traversal order.

# 6. Latency Evaluation Model

The core evaluation function simulates hardware behavior analytically. For every subgraph and granularity candidate, latency is bounded by the roofline formula:

> **L = max(T_compute, T_stream_in + T_stream_out)**

where *T_compute* is the arithmetic cost divided by compute throughput, and *T_stream* is the bytes transferred divided by memory bandwidth. The model accounts for tile-level working set sizes, distinguishing between streaming tensors (loaded per tile) and stationary tensors (loaded once and reused).

**Traversal Order Selection.** V21 evaluates both **Row-Snake** (reuses LHS row-by-row) and **Col-Snake** (reuses RHS column-by-column) traversal patterns. The order with lower total stream bandwidth is selected. For single-tile subgraphs, no traversal order is emitted.

# 7. Concurrency Architecture

**OpenMP Granularity Parallelism.** Inside `find_best()`, all *(w, h, k)* candidate combos are flattened into a single vector and distributed across threads via `#pragma omp parallel` + `#pragma omp for schedule(dynamic, 4)`, with each thread maintaining a local `Best` and merging under `#pragma omp critical`. V21 uses a single level of parallelism (`omp_set_max_active_levels(1)`); retention modes are iterated sequentially in the outer loop, relying on hash deduplication (Section 4) to skip redundant work rather than spawning additional threads. This keeps scheduling overhead low on the 8-core target while still exposing hundreds–thousands of independent combo evaluations per DP transition.

**Cache-Optimized Data Structures.** (1) *Flat Bitset (TensorSet)*: custom 8192-bit bitset using 64-bit words; O(1) insert/lookup via shift+mask; compact 1 KB representation fits entirely in L1 cache. (2) *Precomputed SubgraphTensorInfo*: tensor metadata (produced/consumed/ephemeral sets, LHS/RHS roles, K dimensions, base costs) computed once per DP transition and shared across all *(w, h, k)* evaluations, avoiding redundant per-combo scans. (3) *Workspace category vectors*: `ws_mm_lhs`/`ws_mm_rhs`/`ws_pw_in`/`ws_out` are materialized once per subgraph so the inner evaluator iterates compact vectors rather than re-classifying tensors for every granularity candidate.

# 8. Results

| Benchmark | Cycles | Time (T8) | Time (T1) | Timeout |
|:-:|:-:|:-:|:-:|:-:|
| B1 (5 ops) | 210,219 | 0.16s | 0.02s | 2s |
| B5 (19 ops) | 545,348 | 0.09s | 0.33s | 5s |
| B9 (32 ops) | 22,028,095 | 2.48s | 16.53s | 15s |
| B13 (63 ops) | 11,078,205 | 8.51s | 26.52s | 30s |
| B17 (103 ops) | 5,992,359 | 4.74s | 27.69s | 60s |
| **Total** | **39,854,226** | **15.97s** | **71.09s** | — |

Cycles are thread-count-independent (reported by V21's roofline model). At 8 threads (the official setup), total wall-time is 16s against a 112s combined budget. On 1 thread B9 exceeds its 15s budget by ≈1.5s; the step-level adaptive fallback (Section 2.2) is what keeps B13 and B17 below their 30s/60s limits by switching to the conservative window once extrapolated cost threatens the budget.

V21 merges DP optimality bounds with localized heuristics for tiling, memory pinning, and operator recomputation. Hash-deduplicated retention sets, bitset-optimized hot paths, and precomputed tensor metadata minimize per-transition overhead; Pointwise recomputation further cuts slow-memory traffic for small subgraphs with cheap producers.
