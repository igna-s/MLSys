---
title: "MLSys 2026 Contest — Track A Submission"
author: "Schwindt, Ignacio A."
date: "April 2026"
---

# 1. Introduction

Executing massive computational graphs on hardware with strict on-chip fast memory presents a combinatorial optimization challenge. Latency is governed by a roofline model that heavily penalizes repeated data spillage to slow memory. Our submission, the **V15 Engine**, minimizes end-to-end execution cycles through **Adaptive Dynamic Programming (DP)**, fine-grained **Split-K Pipelining**, strict memory limits enforced via **Deduplicated Retention Set** heuristics, and **Two-Level OpenMP Parallelism** that fully utilizes all available CPU cores.

The engine handles both trivial (5 nodes) and massive (105+ nodes) DAGs without violating the 10-minute timeout or causing OOM crashes. The binary is compiled statically with GCC and `-fopenmp`, and deployed inside a minimal Docker container based on Ubuntu 22.04. V15 achieves **4.12M total cycles** across all 5 public benchmarks in **8.75 seconds** wall-clock on 8 cores — a **17.9× speedup** over the previous V14 engine while producing bit-identical scheduling results.

![The V15 Sliding Window DP Engine. The input DAG is topologically sorted and processed left-to-right through a sliding window. At each position, all candidate subgraph partitions within the window are evaluated in parallel, producing an optimal execution plan with resource allocation and memory management decisions.](dp.png)

# 2. Dynamic Programming Core

The core challenge is grouping dependent operations into subgraphs sequentially without saturating the fast memory capacity *C_fast*. Greedy algorithms fall into local optima — aggressively retaining intermediate tensors to prevent recalculation often leads to inevitable OOM crashes deeper in the graph. RL approaches require exorbitant training times and cannot guarantee generalization within 10 minutes.

We formulate the problem as linear **Dynamic Programming** over a topologically sorted node sequence. The state *dp[i]* represents the minimal total latency to execute the graph up to node *i*. The recursive transition tests all valid subgraphs ending at *i−1* and starting at *j*:

> **dp[i] = min over j < i of { dp[j] + L(subgraph j→i−1, retained tensors) }**

where *L* calculates the analytical roofline latency of the candidate subgraph, parameterized by the tensors retained in fast memory from the preceding step.

## 2.1 Adaptive Window Sizing

Exhaustively evaluating all *j ∈ [0, i−1]* is O(N²), which combined with combinatorial retention exploration is intractable for dense N ≥ 60 topologies. V15 uses **Adaptive Windowing** — restricting backward search depth to a window *W* that varies by problem size. Crucially, V15 uses **significantly wider windows** than V14, enabled by the concurrency optimizations described in Section 5:

| Problem Size N | Window W | Retention Modes R |
|:-:|:-:|:-:|
| ≤ 32 | N (full search) | 10 |
| ≤ 40 | 16 | 10 |
| ≤ 65 | **32** (was 12) | **10** (was 6) |
| ≤ 105 | **20** (was 10) | **10** (was 5) |
| > 105 | 8 | 4 |

These wider windows allow the DP to discover better subgraph partitions that narrower searches miss entirely. Benchmark B13 improved from 424K to 378K cycles (−11%) solely from finding a superior partition with W=20 instead of W=10.

# 3. Memory Hierarchy & Split-K Tiling

![AI Accelerator Memory Hierarchy. The scheduler must decide which tensors reside in fast SRAM and which spill to slow DRAM/HBM. Split-K pipelining partitions the reduction dimension K of MatMul operations into small chunks, keeping partial sums pinned in fast memory to avoid repeated spilling of large intermediate buffers.](mem.png)

## 3.1 Granularity Candidate Generation

Once a candidate subgraph *S* is identified, determining its internal execution tile dimensions *[w, h, k]* is the physical optimization lever. Sweeping all possible values is intractable, so V15 mathematically curates candidates from: (1) multiples and divisors of the native hardware granularity, (2) exact dimensional bounds of the subgraph's topological outputs, (3) power-of-2 factors, and (4) divisors of each tensor's dimensions. All candidates for a given subgraph are flattened into a single vector and evaluated in parallel (see Section 5.1).

## 3.2 Split-K Pipelining for MatMul

MatMul operations are memory-intensive — they require holding LHS, RHS, and Output tensors simultaneously. V15 aggressively evaluates the inner reduction depth *k*. Instead of native materialization (large *k*), the scheduler simulates **Split-K Pipelining**: using small *k* values, the system accumulates partial dot products within a pinned Output tensor in fast memory, iterating over the reduction dimension *K* sequentially. This drastically slashes the real-time working set requirement, bypassing OOM crashes in large GEMMs without forfeiting computational efficiency.

# 4. Fast Memory Retention Strategies

Deciding which tensors to persist in fast memory between subgraph steps is an NP-Hard subset problem — tensors vary wildly in size and future utility. In each DP transition, *M_retention* is generated by simulating up to **10 concurrent filtering heuristics**:

- **Mode 1 (Baseline):** Evict everything — safe but strictly bandwidth-bound.
- **Modes 0, 3, 4, 5 (Size-Based Packing):** Sort candidate tensors by physical size and greedily pack *C_fast* using full capacity or fractional thresholds (e.g., 75% limit) to leave headroom.
- **Modes 2, 6, 9 (Next-Use Proximity):** Calculate the shortest forward distance to each tensor's next consumer. Tensors needed within 6 operations take priority over distant ones.
- **Modes 7, 8 (Hybrid):** Combine size and proximity, balancing immediate reuse with long-term memory pressure.

**Hash Deduplication.** Generating 10 retention sets often produces overlapping configurations. Passing duplicates into the heavy evaluation function *L* is a severe bottleneck. V15 computes a deterministic 64-bit FNV-1a hash for each retention set — only uniquely hashed sets proceed to full evaluation, typically eliminating 30–50% of redundant work.

# 5. V15 Concurrency Architecture

V15 fundamentally rearchitects the parallelism model, fixing a critical nested OpenMP bug from V14 and introducing cache-optimized data structures that enable wider DP windows within the same time budget.

## 5.1 Two-Level OpenMP Parallelism

V14 had a bug where `omp_set_max_active_levels(1)` prevented the inner granularity search from actually running in parallel — the Level 2 threads were serialized. V15 fixes this with `omp_set_max_active_levels(2)` and a clean nested architecture:

**Level 1 — Retention Mode Parallelism.** Each DP transition distributes up to *R* retention modes across threads via `#pragma omp parallel for`. Every thread independently builds its retention set, evaluates it through `find_best()`, and stores results in a thread-local struct. After all modes complete, a sequential reduction picks the global minimum-latency configuration.

**Level 2 — Granularity Search Parallelism.** Inside each `find_best()` call, all *(w, h, k)* granularity combinations (including transposed axes and multiple snake/raster traversal orders) are flattened into a single vector and distributed across inner threads with `#pragma omp for schedule(dynamic,4)`. Dynamic scheduling handles the variable cost of evaluating different granularities. Each thread maintains a local best-result struct and merges via `#pragma omp critical`.

This true nested parallelism yields a **17.9× wall-clock speedup** (54.58s → 8.75s on 8 cores) while producing **bit-identical cycle counts** — parallelism accelerates the *search*, not the *evaluation model*.

## 5.2 Cache-Optimized Data Structures

Three key optimizations eliminate hot-path overhead:

1. **Flat Bitset (TensorSet).** Replaces `std::unordered_set<size_t>` with a custom 8192-bit flat bitset using 64-bit words. Insert and lookup are O(1) with a single shift+mask, and the compact representation (1 KB) fits entirely in L1 cache. The innermost evaluation loop calls `count()` millions of times per benchmark — this change alone accounts for a significant fraction of the speedup.

2. **Precomputed SubgraphTensorInfo.** For each DP transition, V15 computes tensor metadata (produced/consumed/ephemeral sets, LHS/RHS roles, actual K dimensions, base costs) **once** and shares the struct across all *(w, h, k)* evaluations. V14 redundantly recomputed this for every granularity candidate.

3. **OpType Enum.** Replaces runtime string comparisons (`"MatMul"` vs `"Pointwise"`) with an integer enum parsed once at load time, eliminating thousands of `strcmp` calls per DP state.

# 6. Latency Evaluation Model

The core evaluation function simulates hardware behavior analytically. For every subgraph and granularity candidate, latency is bounded by the roofline formula:

> **L = max(T_compute, T_stream_in + T_stream_out)**

where *T_compute* is the arithmetic cost divided by compute throughput, and *T_stream* is the bytes transferred divided by memory bandwidth. The model accounts for tile-level working set sizes, distinguishing between streaming tensors (loaded per tile) and stationary tensors (loaded once and reused).

**Traversal Order Selection.** When the execution divides the output into a grid of tiles, the traversal order modulates memory reuse. V15 evaluates both **Raster** (row-major) and **Snake** (alternating sweep) patterns. Snake traversals naturally reuse tensors at the boundary between adjacent tiles already mapped in fast memory, and V15 selects snake ordering whenever it mathematically reduces the total stream bandwidth.

# 7. Results

| Benchmark | Cycles | Wall Time | Timeout |
|:-:|:-:|:-:|:-:|
| B1 (5 ops) | 87,840 | 0.005s | 2s |
| B5 (16 ops) | 122,336 | 0.028s | 5s |
| B9 (33 ops) | 2,768,241 | 0.995s | 15s |
| B13 (66 ops) | 377,815 | 7.099s | 30s |
| B17 (105 ops) | 767,764 | 0.618s | 60s |
| **Total** | **4,123,996** | **8.75s** | — |

The V15 engine merges the provable optimality bounds of Dynamic Programming with hyper-localized heuristics for tiling and memory pinning. The combination of properly nested OpenMP parallelism, bitset-optimized hot paths, precomputed tensor metadata, and hash-deduplicated retention sets delivers a 17.9× wall-clock speedup over V14 while maintaining deterministic, bit-identical latency schedules across all benchmark sizes.
