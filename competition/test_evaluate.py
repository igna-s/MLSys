#!/usr/bin/env python3
"""Tests for evaluate.py latency calculator using PROBLEM.md examples."""
import sys, math
from evaluate import validate_solution, compute_latency

def check(name, problem, solution, expected_latency, should_be_valid=True):
    valid, errors, warnings, reported = validate_solution(problem, solution)
    computed = compute_latency(problem, solution)
    computed_total = sum(computed)
    ok = True
    if valid != should_be_valid:
        print(f"  FAIL {name}: valid={valid}, expected={should_be_valid}")
        for e in errors: print(f"    ! {e}")
        ok = False
    if expected_latency is not None and abs(computed_total - expected_latency) > 0.1:
        print(f"  FAIL {name}: computed={computed_total:.1f}, expected={expected_latency:.1f}")
        print(f"    per-sg: {[f'{x:.1f}' for x in computed]}")
        ok = False
    if ok:
        print(f"  OK   {name}: latency={computed_total:.1f} (expected={expected_latency})")
    return ok

# ─── Example 1: Baseline ───
P1 = {
    "widths": [128,128,128], "heights": [128,128,128],
    "inputs": [[0],[1]], "outputs": [[1],[2]],
    "base_costs": [1000,100], "op_types": ["Pointwise","Pointwise"],
    "fast_memory_capacity": 35000, "slow_memory_bandwidth": 10,
    "native_granularity": [128,128]
}

# ─── Example 2: Fast Memory Capacity ───
P2 = {
    "widths": [256,256,256], "heights": [256,256,256],
    "inputs": [[0],[1]], "outputs": [[1],[2]],
    "base_costs": [1000,100], "op_types": ["Pointwise","Pointwise"],
    "fast_memory_capacity": 25000, "slow_memory_bandwidth": 10,
    "native_granularity": [128,128]
}

# ─── Example 3: Diamond (skip connection) ───
P3 = {
    "widths": [128,128,128,128], "heights": [128,128,128,128],
    "inputs": [[0],[1],[1,2]], "outputs": [[1],[2],[3]],
    "base_costs": [1500,1500,1500], "op_types": ["Pointwise","Pointwise","Pointwise"],
    "fast_memory_capacity": 50000, "slow_memory_bandwidth": 10,
    "native_granularity": [128,128]
}

# ─── Example 4: MatMul Revisit ───
P4 = {
    "widths": [128,128,128], "heights": [128,128,128],
    "inputs": [[0,1]], "outputs": [[2]],
    "base_costs": [1500], "op_types": ["MatMul"],
    "fast_memory_capacity": 25000, "slow_memory_bandwidth": 10,
    "native_granularity": [128,128]
}

# ─── Example 5: Chained MatMul Split-K ───
P5 = {
    "widths": [128,128,128,128,128], "heights": [128,128,128,128,128],
    "inputs": [[0,1],[3,2]], "outputs": [[3],[4]],
    "base_costs": [2000,2000], "op_types": ["MatMul","MatMul"],
    "fast_memory_capacity": 45000, "slow_memory_bandwidth": 10,
    "native_granularity": [128,128]
}

def run_tests():
    results = []
    print("=== Example 1: Baseline ===")
    # Strategy A: Always Spill
    results.append(check("Ex1-A", P1,
        {"subgraphs":[[0],[1]], "granularities":[[128,128,1],[128,128,1]],
         "tensors_to_retain":[[],[]], "traversal_orders":[None,None],
         "subgraph_latencies":[3276.8, 3276.8]}, 6553.6))
    # Strategy B: Mega-group 128x128
    results.append(check("Ex1-B", P1,
        {"subgraphs":[[0,1]], "granularities":[[128,128,1]],
         "tensors_to_retain":[[]], "traversal_orders":[None],
         "subgraph_latencies":[3276.8]}, 3276.8))
    # Strategy C: Mega-group 64x64
    results.append(check("Ex1-C", P1,
        {"subgraphs":[[0,1]], "granularities":[[64,64,1]],
         "tensors_to_retain":[[]], "traversal_orders":[None],
         "subgraph_latencies":[4400.0]}, 4400.0))

    print("\n=== Example 2: Fast Memory Capacity ===")
    # Strategy A: Always Spill — NOTE: OOM invalid (ws=32768 > fast_mem=25000)
    results.append(check("Ex2-A", P2,
        {"subgraphs":[[0],[1]], "granularities":[[128,128,1],[128,128,1]],
         "tensors_to_retain":[[],[]], "traversal_orders":[None,None],
         "subgraph_latencies":[13107.2, 13107.2]}, 26214.4, should_be_valid=False))
    # Strategy B: Mega-group 128x128
    results.append(check("Ex2-B", P2,
        {"subgraphs":[[0,1]], "granularities":[[128,128,1]],
         "tensors_to_retain":[[]], "traversal_orders":[None],
         "subgraph_latencies":[13107.2]}, 13107.2))

    print("\n=== Example 3: Spilling vs Recomputation ===")
    # Strategy A: Spilling
    results.append(check("Ex3-A", P3,
        {"subgraphs":[[0],[1],[2]], "granularities":[[128,128,1],[128,128,1],[128,128,1]],
         "tensors_to_retain":[[],[],[]], "traversal_orders":[None,None,None],
         "subgraph_latencies":[3276.8, 3276.8, 4915.2]}, 11468.8))
    # Strategy B: Recomputation
    results.append(check("Ex3-B", P3,
        {"subgraphs":[[0,1],[0,2]], "granularities":[[128,128,1],[128,128,1]],
         "tensors_to_retain":[[2],[]], "traversal_orders":[None,None],
         "subgraph_latencies":[3000, 3276.8]}, 6276.8))
    # Strategy C: Selective Residency
    results.append(check("Ex3-C", P3,
        {"subgraphs":[[0],[1,2]], "granularities":[[128,128,1],[128,128,1]],
         "tensors_to_retain":[[1],[]], "traversal_orders":[None,None],
         "subgraph_latencies":[1638.4, 3000]}, 4638.4))

    print("\n=== Example 4: MatMul Revisit ===")
    # Strategy A: Naive tiling (raster, no reuse)
    results.append(check("Ex4-A", P4,
        {"subgraphs":[[0]], "granularities":[[64,64,128]],
         "tensors_to_retain":[[]], "traversal_orders":[None],
         "subgraph_latencies":[8192]}, 8192.0))
    # Strategy B: Optimized traversal (zig-zag, reuse)
    results.append(check("Ex4-B", P4,
        {"subgraphs":[[0]], "granularities":[[64,64,128]],
         "tensors_to_retain":[[]], "traversal_orders":[[0,1,3,2]],
         "subgraph_latencies":[6548]}, 6548.0))

    print("\n=== Example 5: Split-K Pipelining ===")
    # Strategy A: full k=128, should OOM (MatMul accumulator forces ws=49152 > 45000)
    results.append(check("Ex5-A (OOM)", P5,
        {"subgraphs":[[0,1]], "granularities":[[128,128,128]],
         "tensors_to_retain":[[]], "traversal_orders":[None],
         "subgraph_latencies":[5000]}, None, should_be_valid=False))
    # Strategy B: Split-K k=32, valid (ws fits)
    results.append(check("Ex5-B", P5,
        {"subgraphs":[[0,1]], "granularities":[[128,128,32]],
         "tensors_to_retain":[[]], "traversal_orders":[None],
         "subgraph_latencies":[6915.2]}, 6915.2))

    print("\n=== Edge Cases: Retained Tensor Validation ===")
    # Retaining a tensor not produced/loaded by the subgraph → invalid
    results.append(check("Retain-invalid", P3,
        {"subgraphs":[[0],[1],[2]], "granularities":[[128,128,1],[128,128,1],[128,128,1]],
         "tensors_to_retain":[[3],[],[]], "traversal_orders":[None,None,None],
         "subgraph_latencies":[3276.8, 3276.8, 4915.2]}, None, should_be_valid=False))
    # Retaining an output tensor of the subgraph → valid
    results.append(check("Retain-valid-output", P3,
        {"subgraphs":[[0],[1,2]], "granularities":[[128,128,1],[128,128,1]],
         "tensors_to_retain":[[1],[]], "traversal_orders":[None,None],
         "subgraph_latencies":[1638.4, 3000]}, 4638.4))

    print(f"\n{'='*50}")
    passed = sum(results)
    print(f"Results: {passed}/{len(results)} passed")
    return passed == len(results)

if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
