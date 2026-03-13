#!/usr/bin/env python3
"""
MLSys 2026 Competition Evaluator
Runs V10, V14, V15 binaries under competition conditions (8 cores, Ubuntu 22.04)
and validates outputs per competition rules + reports latency scores.

Checks performed:
  1. Valid JSON output with all required fields
  2. All ops in the DAG are covered by at least one subgraph
  3. Topological ordering within subgraphs is correct
  4. Granularity values are positive integers
  5. Tensor indices in tensors_to_retain are valid
  6. Working set fits in fast_memory_capacity (OOM check)
  7. Retained tensors fit in fast memory between subgraphs
  8. Graph outputs end up in slow memory (not retained at end)
  9. Timeout enforcement per competition table
"""

import json
import math
import os
import subprocess
import sys
import time

# Competition timeouts per benchmark group
TIMEOUTS = {
    "mlsys-2026-1": 2, "mlsys-2026-2": 2, "mlsys-2026-3": 2, "mlsys-2026-4": 2,
    "mlsys-2026-5": 5, "mlsys-2026-6": 5, "mlsys-2026-7": 5, "mlsys-2026-8": 5,
    "mlsys-2026-9": 15, "mlsys-2026-10": 15, "mlsys-2026-11": 15, "mlsys-2026-12": 15,
    "mlsys-2026-13": 30, "mlsys-2026-14": 30, "mlsys-2026-15": 30, "mlsys-2026-16": 30,
    "mlsys-2026-17": 60, "mlsys-2026-18": 60, "mlsys-2026-19": 60, "mlsys-2026-20": 60,
}

VERSIONS = ["v10", "v14", "v15", "v16"]
CORE_CONFIGS = [1, 8]  # Run each version with 1 core and 8 cores
BIN_DIR = "/app/bin"
BENCH_DIR = "/app/benchmarks"
OUT_DIR = "/app/output"


def get_timeout(benchmark_name):
    base = benchmark_name.replace(".json", "")
    return TIMEOUTS.get(base, 60)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def validate_solution(problem, solution):
    """
    Validate solution against all competition constraints.
    Returns (valid, errors, warnings, reported_total_cycles).
    """
    errors = []
    warnings = []

    widths = problem["widths"]
    heights = problem["heights"]
    inputs_list = problem["inputs"]
    outputs_list = problem["outputs"]
    base_costs = problem["base_costs"]
    op_types = problem["op_types"]
    fast_mem = problem["fast_memory_capacity"]
    slow_bw = problem["slow_memory_bandwidth"]
    native_w, native_h = problem["native_granularity"]

    num_ops = len(base_costs)
    num_tensors = len(widths)

    # --- Check required fields ---
    required = ["subgraphs", "granularities", "tensors_to_retain", "traversal_orders", "subgraph_latencies"]
    for field in required:
        if field not in solution:
            errors.append(f"Missing required field: {field}")
    if errors:
        return False, errors, warnings, 0

    subgraphs = solution["subgraphs"]
    granularities = solution["granularities"]
    tensors_to_retain = solution["tensors_to_retain"]
    traversal_orders = solution["traversal_orders"]
    subgraph_latencies = solution["subgraph_latencies"]
    n_sg = len(subgraphs)

    # --- Check parallel list lengths ---
    for field_name, lst in [("granularities", granularities), ("tensors_to_retain", tensors_to_retain),
                            ("traversal_orders", traversal_orders), ("subgraph_latencies", subgraph_latencies)]:
        if len(lst) != n_sg:
            errors.append(f"{field_name} length ({len(lst)}) != subgraphs length ({n_sg})")
    if errors:
        return False, errors, warnings, 0

    # --- Check all ops covered ---
    covered_ops = set()
    for sg_ops in subgraphs:
        for op_id in sg_ops:
            if op_id < 0 or op_id >= num_ops:
                errors.append(f"Invalid op index: {op_id}")
            covered_ops.add(op_id)
    missing_ops = set(range(num_ops)) - covered_ops
    if missing_ops:
        errors.append(f"Ops not covered: {sorted(missing_ops)}")

    # --- Build tensor producer/consumer maps ---
    tensor_producer = {}  # tensor_id -> op_id that produces it
    for op_id in range(num_ops):
        for t_id in outputs_list[op_id]:
            tensor_producer[t_id] = op_id

    # Graph inputs: tensors not produced by any op
    # Graph outputs: tensors not consumed by any op
    all_produced = set()
    all_consumed = set()
    for op_id in range(num_ops):
        for t in outputs_list[op_id]:
            all_produced.add(t)
        for t in inputs_list[op_id]:
            all_consumed.add(t)
    graph_inputs = set(range(num_tensors)) - all_produced
    graph_outputs = set(range(num_tensors)) - all_consumed

    # --- Check granularity validity ---
    for i, gran in enumerate(granularities):
        if len(gran) != 3:
            errors.append(f"Subgraph {i}: granularity must have 3 elements, got {len(gran)}")
            continue
        w, h, k = gran
        if w <= 0 or h <= 0 or k <= 0:
            errors.append(f"Subgraph {i}: granularity values must be positive: [{w},{h},{k}]")

    # --- Check tensor indices in tensors_to_retain ---
    for i, retain in enumerate(tensors_to_retain):
        for t_id in retain:
            if t_id < 0 or t_id >= num_tensors:
                errors.append(f"Subgraph {i}: invalid tensor index in tensors_to_retain: {t_id}")

    # --- Check topological ordering within subgraphs ---
    for sg_idx, sg_ops in enumerate(subgraphs):
        op_set = set(sg_ops)
        op_pos = {op_id: pos for pos, op_id in enumerate(sg_ops)}
        for op_id in sg_ops:
            for t_id in inputs_list[op_id]:
                if t_id in tensor_producer and tensor_producer[t_id] in op_set:
                    prod_op = tensor_producer[t_id]
                    if op_pos[prod_op] >= op_pos[op_id]:
                        errors.append(
                            f"Subgraph {sg_idx}: op {op_id} uses tensor {t_id} produced by op {prod_op}, "
                            f"but op {prod_op} appears at position {op_pos[prod_op]} >= {op_pos[op_id]}"
                        )

    # --- OOM / Working Set Check ---
    # For each subgraph, verify the working set fits in fast_memory_capacity.
    # Working set = sum of input slices + output slices for a single execution tile.
    for sg_idx, sg_ops in enumerate(subgraphs):
        if sg_idx >= len(granularities) or len(granularities[sg_idx]) != 3:
            continue
        w, h, k = granularities[sg_idx]
        op_set = set(sg_ops)

        # Identify ephemeral tensors (produced AND consumed within subgraph, not graph output)
        ephemeral = set()
        for op_id in sg_ops:
            for t_id in outputs_list[op_id]:
                if t_id in graph_outputs:
                    continue
                # Check if all consumers are within this subgraph
                all_consumers_in_sg = True
                for op2 in range(num_ops):
                    if op2 not in op_set and t_id in inputs_list[op2]:
                        all_consumers_in_sg = False
                        break
                if all_consumers_in_sg:
                    # Check if at least one consumer exists in subgraph
                    has_consumer_in_sg = any(
                        t_id in inputs_list[op2] for op2 in sg_ops if op2 != op_id
                    )
                    if has_consumer_in_sg:
                        ephemeral.add(t_id)

        # Calculate working set for each op in the subgraph
        # The worst-case working set is the max across all ops
        max_working_set = 0
        for op_id in sg_ops:
            ws = 0
            if op_types[op_id] == "MatMul":
                # LHS input slice: k x h
                # RHS input slice: w x k
                # Output slice: w x h
                for t_id in inputs_list[op_id]:
                    if t_id in ephemeral:
                        continue  # ephemeral = 0 capacity
                    # First input = LHS (height=h, width=k), Second = RHS (height=k, width=w)
                lhs_t = inputs_list[op_id][0]
                rhs_t = inputs_list[op_id][1]
                if lhs_t not in ephemeral:
                    ws += k * h  # LHS slice
                if rhs_t not in ephemeral:
                    ws += w * k  # RHS slice
                for t_id in outputs_list[op_id]:
                    if t_id not in ephemeral:
                        ws += w * h  # output slice
            else:
                # Pointwise: all inputs and outputs are w x h
                for t_id in inputs_list[op_id]:
                    if t_id not in ephemeral:
                        ws += w * h
                for t_id in outputs_list[op_id]:
                    if t_id not in ephemeral:
                        ws += w * h

            max_working_set = max(max_working_set, ws)

        # Also add retained tensors from previous subgraph that are still resident
        # (retained tensors occupy space alongside the working set)
        # The retained tensors have their full size, not sliced
        retain_set = set(tensors_to_retain[sg_idx]) if sg_idx < len(tensors_to_retain) else set()

        # Tensors retained FROM PREVIOUS subgraph that are used as inputs here
        # They occupy their slice size in the working set (already counted above)
        # But tensors retained that are NOT used in this subgraph still occupy full space
        # Actually per the problem: retained tensors stay in fast memory but working set
        # is about the execution tile. The retained tensors should not exceed capacity either.

        # Check: retained tensors after this subgraph fit
        retained_size = sum(widths[t] * heights[t] for t in retain_set)
        if retained_size > fast_mem:
            errors.append(
                f"Subgraph {sg_idx}: retained tensors after subgraph ({retained_size}) "
                f"exceed fast memory ({fast_mem})"
            )

        if max_working_set > fast_mem:
            errors.append(
                f"Subgraph {sg_idx}: working set ({max_working_set}) exceeds "
                f"fast memory ({fast_mem}) — OOM! [w={w},h={h},k={k}]"
            )

    # --- Check graph outputs are NOT retained at the end ---
    # All graph outputs must end up in slow memory
    if n_sg > 0:
        final_retain = set(tensors_to_retain[-1])
        retained_graph_outputs = final_retain & graph_outputs
        if retained_graph_outputs:
            warnings.append(
                f"Graph outputs {sorted(retained_graph_outputs)} are retained after last subgraph "
                f"(should be evicted to slow memory)"
            )

    # --- Check subgraph_latencies are positive ---
    for i, lat in enumerate(subgraph_latencies):
        if lat <= 0:
            errors.append(f"Subgraph {i}: latency must be positive, got {lat}")

    reported_total = sum(subgraph_latencies)
    valid = len(errors) == 0
    return valid, errors, warnings, reported_total


def run_benchmark(binary, benchmark_path, output_path, timeout_sec, num_cores=8):
    """Run binary on benchmark with timeout. Returns (success, elapsed, error_msg)."""
    try:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(num_cores)
        start = time.time()
        result = subprocess.run(
            [binary, benchmark_path, output_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec + 1,  # +1s grace for I/O
            env=env,
        )
        elapsed = time.time() - start
        if result.returncode != 0:
            return False, elapsed, f"Exit code {result.returncode}: {result.stderr.strip()[:200]}"
        return True, elapsed, None
    except subprocess.TimeoutExpired:
        return False, timeout_sec, "TIMEOUT"
    except Exception as e:
        return False, 0, str(e)


def print_header():
    print("=" * 120)
    print(f"{'MLSys 2026 Competition Evaluator':^120}")
    print(f"{'Ubuntu 22.04 | Competition Timeouts | 1-core & 8-core':^120}")
    print("=" * 120)


def run_config(version, num_cores, benchmarks, results_key, all_results):
    """Run all benchmarks for a given version and core count."""
    binary = os.path.join(BIN_DIR, f"mlsys_{version}")
    if not os.path.exists(binary):
        print(f"\n  [SKIP] {results_key}: binary not found at {binary}")
        return

    print(f"\n{'─' * 120}")
    print(f"  {results_key.upper()} — {binary} — OMP_NUM_THREADS={num_cores}")
    print(f"{'─' * 120}")
    print(f"  {'Benchmark':<20} {'Status':<9} {'Time(s)':<9} {'TL(s)':<7} {'In TL?':<7} "
          f"{'Cycles':>14} {'#SG':>5} {'Valid':>6} {'OOM':>5} {'Notes'}")
    print(f"  {'─'*20} {'─'*9} {'─'*9} {'─'*7} {'─'*7} {'─'*14} {'─'*5} {'─'*6} {'─'*5} {'─'*20}")

    total_cycles = 0.0
    n_valid = 0

    for bench_file in benchmarks:
        bench_name = bench_file.replace(".json", "")
        bench_path = os.path.join(BENCH_DIR, bench_file)
        out_path = os.path.join(OUT_DIR, f"{results_key}_{bench_file}")
        timeout = get_timeout(bench_name)

        success, elapsed, err = run_benchmark(binary, bench_path, out_path, timeout, num_cores)

        if not success:
            status = "TIMEOUT" if err == "TIMEOUT" else "FAIL"
            within_tl = "NO" if err == "TIMEOUT" else "—"
            notes = err[:30] if err != "TIMEOUT" else ""
            print(f"  {bench_name:<20} {status:<9} {elapsed:<9.2f} {timeout:<7} {within_tl:<7} "
                  f"{'—':>14} {'—':>5} {'—':>6} {'—':>5} {notes}")
            all_results[results_key][bench_name] = {
                "status": status, "elapsed": elapsed, "timeout": timeout,
                "within_tl": False, "cycles": None, "error": err, "valid": False
            }
            continue

        try:
            problem = load_json(bench_path)
            solution = load_json(out_path)
            valid, errs, warns, reported_total = validate_solution(problem, solution)

            n_subgraphs = len(solution.get("subgraphs", []))
            within_tl = elapsed <= timeout

            has_oom = any("OOM" in e for e in errs)
            status_str = "OK" if valid else "INVALID"
            valid_str = "YES" if valid else "NO"
            within_str = "YES" if within_tl else "NO"
            oom_str = "YES" if has_oom else "no"

            notes_parts = []
            if errs:
                non_oom = [e for e in errs if "OOM" not in e]
                if non_oom:
                    notes_parts.append(non_oom[0][:40])
            if warns:
                notes_parts.append(warns[0][:30])
            notes = "; ".join(notes_parts)

            if valid:
                total_cycles += reported_total
                n_valid += 1

            print(f"  {bench_name:<20} {status_str:<9} {elapsed:<9.2f} {timeout:<7} {within_str:<7} "
                  f"{reported_total:>14.1f} {n_subgraphs:>5} {valid_str:>6} {oom_str:>5} {notes}")

            if not valid or not within_tl:
                for e in errs[:3]:
                    print(f"    ! {e}")

            all_results[results_key][bench_name] = {
                "status": "OK" if valid else "INVALID",
                "elapsed": elapsed, "timeout": timeout,
                "within_tl": within_tl,
                "cycles": reported_total if valid else None,
                "n_subgraphs": n_subgraphs, "valid": valid,
                "errors": errs, "warnings": warns,
            }

        except Exception as e:
            print(f"  {bench_name:<20} {'ERROR':<9} {elapsed:<9.2f} {timeout:<7} {'—':<7} "
                  f"{'—':>14} {'—':>5} {'—':>6} {'—':>5} {str(e)[:40]}")
            all_results[results_key][bench_name] = {
                "status": "ERROR", "elapsed": elapsed, "timeout": timeout, "error": str(e), "valid": False
            }

    # Summary row
    all_within = all(r.get("within_tl", False) for r in all_results[results_key].values())
    all_valid = all(r.get("valid", False) for r in all_results[results_key].values())
    pass_str = "PASS" if (all_within and all_valid) else "FAIL"
    print(f"  {'─'*20} {'─'*9} {'─'*9} {'─'*7} {'─'*7} {'─'*14} {'─'*5} {'─'*6} {'─'*5} {'─'*20}")
    print(f"  {'TOTAL':<20} {pass_str:<9} {'':9} {'':7} "
          f"{'ALL:'+('Y' if all_within else 'N'):<7} "
          f"{total_cycles:>14.1f} {'':>5} "
          f"{'ALL:'+('Y' if all_valid else 'N'):>6} {'':>5} "
          f"{n_valid}/{len(benchmarks)} passed")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    benchmarks = sorted(
        [f for f in os.listdir(BENCH_DIR) if f.endswith(".json")]
    )

    if not benchmarks:
        print("ERROR: No benchmarks found!")
        sys.exit(1)

    print_header()

    # Build config keys: v10_1c, v10_8c, v14_1c, v14_8c, v15_1c, v15_8c
    config_keys = []
    for v in VERSIONS:
        for nc in CORE_CONFIGS:
            config_keys.append((v, nc, f"{v}_{nc}c"))

    all_results = {key: {} for _, _, key in config_keys}

    # Run all configurations
    for version, num_cores, results_key in config_keys:
        run_config(version, num_cores, benchmarks, results_key, all_results)

    # ═══════════════════════════════════════════════════
    # Final comparison table
    # ═══════════════════════════════════════════════════
    print(f"\n{'═' * 130}")
    print(f"{'COMPARISON SUMMARY — CYCLES':^130}")
    print(f"{'═' * 130}")

    col_w = 16
    labels = [key for _, _, key in config_keys]

    print(f"  {'Benchmark':<20}", end="")
    for lbl in labels:
        print(f" {lbl.upper():>{col_w}}", end="")
    print()
    print(f"  {'─'*20}", end="")
    for _ in labels:
        print(f" {'─'*col_w}", end="")
    print()

    for bench_file in benchmarks:
        bench_name = bench_file.replace(".json", "")
        print(f"  {bench_name:<20}", end="")
        for lbl in labels:
            r = all_results[lbl].get(bench_name, {})
            cycles = r.get("cycles")
            if cycles is not None:
                within = r.get("within_tl", False)
                valid = r.get("valid", False)
                if within and valid:
                    print(f" {cycles:>{col_w},.1f}", end="")
                elif not within:
                    print(f" {cycles:>10,.1f} *TL  ", end="")
                else:
                    print(f" {cycles:>10,.1f} *INV ", end="")
            else:
                status = r.get("status", "N/A")
                print(f" {status:>{col_w}}", end="")
        print()

    # Totals row
    print(f"  {'─'*20}", end="")
    for _ in labels:
        print(f" {'─'*col_w}", end="")
    print()

    print(f"  {'TOTAL CYCLES':<20}", end="")
    for lbl in labels:
        total = sum(r.get("cycles", 0) or 0 for r in all_results[lbl].values())
        all_ok = all(
            r.get("valid", False) and r.get("within_tl", False)
            for r in all_results[lbl].values()
        )
        tag = "OK" if all_ok else "FAIL"
        print(f" {total:>11,.1f} {tag:>4}", end="")
    print()

    # Time row
    print(f"  {'TOTAL TIME (s)':<20}", end="")
    for lbl in labels:
        total_time = sum(r.get("elapsed", 0) for r in all_results[lbl].values())
        print(f" {total_time:>{col_w},.2f}", end="")
    print()

    # Competition compliance
    print(f"\n  {'COMPETITION PASS?':<20}", end="")
    for lbl in labels:
        all_ok = all(
            r.get("valid", False) and r.get("within_tl", False)
            for r in all_results[lbl].values()
        )
        tag = "PASS" if all_ok else "FAIL"
        print(f" {tag:>{col_w}}", end="")
    print()

    print(f"\n  Legend: *TL = exceeded time limit | *INV = invalid solution (OOM/constraint violation)")
    print(f"  Competition: 8-core Linux workstation, Ubuntu 22.04, 32GB RAM")
    print(f"  Timeouts: b1-4=2s, b5-8=5s, b9-12=15s, b13-16=30s, b17-20=60s")
    print(f"  Note: V10/V14 are single-threaded (no OpenMP), V15 uses OpenMP parallelism")
    print()


if __name__ == "__main__":
    main()
