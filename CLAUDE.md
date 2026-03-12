# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

MLSys 2026 Contest entry (Track A: Systems Engineering). The goal is to schedule a directed acyclic graph (DAG) of tensor operations on hardware with limited fast memory, minimizing total execution latency.

The binary reads a JSON problem file and writes a JSON solution file.

## Build Commands

```bash
cd solution

# Quick build (Linux/macOS with GCC or Clang + OpenMP)
c++ -O3 -std=c++20 -fopenmp main.cpp scheduler.cpp -I. -o mlsys

# Python build harness (auto-downloads nlohmann/json, builds, runs all benchmarks)
python run_test.py

# Docker build (production - static binary on Ubuntu 22.04)
docker build -t mlsys-builder .
```

The `nlohmann/json.hpp` single-header library is auto-downloaded by `run_test.py` if not present.

## Running / Testing

```bash
# Run against a single benchmark
./mlsys <input.json> <output.json>

# Run all public benchmarks and print scores
cd solution && python run_test.py

# WSL benchmark runner (Windows → Linux cross-testing)
cd solution && python wsl_bench.py
```

Public benchmarks are in `/benchmarks/` (mlsys-2026-{1,5,9,13,17}.json). Each has its own timeout (2s–60s).

## Architecture

### Files

- [solution/main.cpp](solution/main.cpp) — JSON I/O, problem parsing, calls `Solve()`, writes output
- [solution/scheduler.h](solution/scheduler.h) — `Problem` / `Solution` / `Subgraph` data structures, `Solve()` declaration
- [solution/scheduler.cpp](solution/scheduler.cpp) — Core engine (~22KB): DP solver, roofline latency model, retention heuristics, granularity search

### Algorithm (V14 Engine)

**Dynamic Programming** over topological op order:
- `dp[i]` = min total latency to schedule ops `[0, i)`
- `par[i]` = backpointer for solution reconstruction
- **Adaptive window**: how far back each DP state searches depends on problem size (full search for N≤32, W=8 for N>105)

**For each DP transition** (candidate subgraph), it evaluates:

1. **Two-level OpenMP parallelism:**
   - Level 1: 10 retention modes in parallel (how to decide which tensors stay in fast memory)
   - Level 2: all (w, h, k) granularity candidates in parallel

2. **Retention modes** (0–9): Different heuristics for packing tensors into fast memory (greedy size, proximity to last use, evict-all baseline, etc.). FNV-1a hashing deduplicates identical retention sets.

3. **Granularity candidates**: Generated from native hardware granularity, tensor dimensions, power-of-2 factors, divisors, and multiples of the native tile.

4. **Latency model** (`eval_inner`): Roofline model with split-K pipelining for MatMul ops. Also evaluates raster vs. snake traversal orders.

### Problem / Solution JSON Structure

**Input:**
```json
{
  "widths": [], "heights": [],          // tensor dimensions
  "inputs": [], "outputs": [],          // op→tensor mappings
  "base_costs": [], "op_types": [],     // per-op compute cost and type ("Pointwise"/"MatMul")
  "fast_memory_capacity": N,
  "slow_memory_bandwidth": B,
  "native_granularity": [w, h]
}
```

**Output:**
```json
{
  "subgraphs": [],           // op indices per step
  "granularities": [],       // [w, h, k] per subgraph
  "tensors_to_retain": [],   // tensors kept in fast memory after each step
  "traversal_orders": [],    // null (raster) or snake pattern
  "subgraph_latencies": []   // computed latency per step
}
```

### Scoring

Lower total latency is better. Current best public scores: ~4.17M cycles total across all 5 benchmarks.
