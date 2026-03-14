#!/usr/bin/env python3
"""
MLSys 2026 Competition Evaluator
Runs V10, V14, V15 binaries under competition conditions (8 cores, Ubuntu 22.04)
and validates outputs per competition rules + reports latency scores.

Checks performed:
  1.  Valid JSON output with all required fields
  2.  All ops in the DAG are covered by at least one subgraph
  3.  No duplicate ops within the same subgraph
  4.  Topological ordering within subgraphs is correct
  5.  Inter-subgraph data flow: each op's inputs produced in an earlier or same subgraph
  6.  Granularity values are positive integers
  7.  Tensor indices in tensors_to_retain are valid
  8.  Joint working set fits in fast_memory_capacity (OOM check)
       - MatMul LHS: full K loaded once per tile (widths[lhs]*h), not a k-strip
       - MatMul RHS: w*k strip per k-step
       - Ephemeral MatMul outputs: zero capacity (pass-through)
       - All non-ephemeral outputs: w*h each
       - Previously retained tensors coexist, not double-counted
  9.  Retained tensors fit in fast memory between subgraphs
  10. Graph outputs end up in slow memory (not retained at end)
  11. Timeout enforcement per competition table
  12. Reported latencies verified against roofline model (warnings only)
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

VERSIONS = ["v10", "v16", "v17"]
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
    # Per PROBLEM.md: only output tensors or loaded inputs of the subgraph
    # (or previously retained tensors) may be retained.
    for i, retain in enumerate(tensors_to_retain):
        sg_ops_set = set(subgraphs[i]) if i < len(subgraphs) else set()
        # Tensors accessible in fast memory during this subgraph:
        #  - outputs of ops in this subgraph
        #  - inputs loaded by ops in this subgraph (non-ephemeral, non-graph-output)
        #  - previously retained tensors
        accessible = set()
        for op_id in sg_ops_set:
            for t_id in outputs_list[op_id]:
                accessible.add(t_id)
            for t_id in inputs_list[op_id]:
                accessible.add(t_id)
        if i > 0:
            accessible |= set(tensors_to_retain[i - 1])
        for t_id in retain:
            if t_id < 0 or t_id >= num_tensors:
                errors.append(f"Subgraph {i}: invalid tensor index in tensors_to_retain: {t_id}")
            elif t_id not in accessible:
                errors.append(
                    f"Subgraph {i}: tensor {t_id} in tensors_to_retain was not "
                    f"produced, loaded, or previously retained by this subgraph"
                )

    # --- Check for duplicate ops within the same subgraph ---
    for sg_idx, sg_ops in enumerate(subgraphs):
        seen = set()
        dups = []
        for op_id in sg_ops:
            if op_id in seen:
                dups.append(op_id)
            seen.add(op_id)
        if dups:
            errors.append(f"Subgraph {sg_idx}: contains duplicate ops: {sorted(set(dups))}")

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

    # --- Check inter-subgraph data flow consistency ---
    # For each op C in subgraph i, every input tensor must be either:
    #   (a) a graph input (not produced by any op), or
    #   (b) produced by an op scheduled in some earlier subgraph j < i, or
    #   (c) produced by an op earlier in the same subgraph (covered by topo check above).
    op_first_sg = {}  # op_id -> earliest subgraph index where it is scheduled
    for sg_idx, sg_ops in enumerate(subgraphs):
        for op_id in sg_ops:
            if op_id not in op_first_sg:
                op_first_sg[op_id] = sg_idx

    for sg_idx, sg_ops in enumerate(subgraphs):
        op_set = set(sg_ops)
        for op_id in sg_ops:
            for t_id in inputs_list[op_id]:
                if t_id not in tensor_producer:
                    continue  # graph input — always available from slow memory
                prod_op = tensor_producer[t_id]
                if prod_op in op_set:
                    continue  # same subgraph — topological order check handles this
                earliest = op_first_sg.get(prod_op)
                if earliest is None or earliest >= sg_idx:
                    errors.append(
                        f"Subgraph {sg_idx}: op {op_id} needs tensor {t_id} "
                        f"produced by op {prod_op}, but op {prod_op} is not "
                        f"scheduled before subgraph {sg_idx}"
                    )

    # --- Validate traversal_orders ---
    for sg_idx, (sg_ops, trav, gran) in enumerate(zip(subgraphs, traversal_orders, granularities)):
        if trav is None:
            continue
        if len(gran) != 3:
            continue
        w, h, k = gran
        if w <= 0 or h <= 0:
            continue
        # Determine number of spatial tiles from the tensors touched by this subgraph
        # Use the first output tensor of the first op to get the output dimensions
        max_W, max_H = 0, 0
        for op_id in sg_ops:
            for t_id in outputs_list[op_id]:
                max_W = max(max_W, widths[t_id])
                max_H = max(max_H, heights[t_id])
        if max_W == 0 or max_H == 0:
            continue
        n_tiles_w = math.ceil(max_W / w)
        n_tiles_h = math.ceil(max_H / h)
        n_tiles = n_tiles_w * n_tiles_h
        if len(trav) != n_tiles:
            errors.append(
                f"Subgraph {sg_idx}: traversal_order length ({len(trav)}) != "
                f"expected number of tiles ({n_tiles} = {n_tiles_w}x{n_tiles_h})"
            )
        elif sorted(trav) != list(range(n_tiles)):
            errors.append(
                f"Subgraph {sg_idx}: traversal_order is not a valid permutation of [0..{n_tiles-1}]"
            )

    # --- OOM / Working Set Check ---
    # Compute the JOINT working set across all ops in the subgraph tile.
    # Peak footprint occurs on the first k-step:
    #   - MatMul LHS: full K dimension loaded ONCE per spatial tile (widths[lhs]*h),
    #     then stays resident for all subsequent k-steps (bandwidth 0 on mid-steps).
    #     This matches PROBLEM.md Example 5 where Tensor0 (128x128) is kept resident.
    #   - MatMul RHS: w*k strip loaded every k-step.
    #   - MatMul output (non-ephemeral): w*h accumulator always in fast memory.
    #   - MatMul output (ephemeral): passes directly to next op — zero capacity.
    #     (Example 5: Tensor3 ephemeral intermediate is NOT in the 40960 working set.)
    #   - Pointwise inputs/outputs (non-ephemeral): w*h each.
    #   - Previously retained tensors already occupy fast memory and are NOT
    #     double-counted in the joint working set.
    for sg_idx, sg_ops in enumerate(subgraphs):
        if sg_idx >= len(granularities) or len(granularities[sg_idx]) != 3:
            continue
        w, h, k = granularities[sg_idx]

        # Identify ephemeral tensors (produced AND consumed within subgraph, not graph output)
        # External consumers don't prevent ephemeral status — they're served
        # by tensors_to_retain (if retained) or recomputation (if discarded).
        ephemeral = set()
        for op_id in sg_ops:
            for t_id in outputs_list[op_id]:
                if t_id in graph_outputs:
                    continue
                has_consumer_in_sg = any(
                    t_id in inputs_list[op2] for op2 in sg_ops if op2 != op_id
                )
                if has_consumer_in_sg:
                    ephemeral.add(t_id)

        prev_retain_set = set(tensors_to_retain[sg_idx - 1]) if sg_idx > 0 else set()
        prev_retained_size = sum(widths[t] * heights[t] for t in prev_retain_set)

        # Collect unique boundary tensors across all ops for the joint working set.
        matmul_lhs = {}  # t_id -> True: non-ephemeral, non-retained MatMul LHS
        matmul_rhs = {}  # t_id -> True: non-ephemeral, non-retained MatMul RHS
        pw_inputs = {}   # t_id -> True: non-ephemeral, non-retained Pointwise inputs
        out_tiles = {}   # t_id -> True: non-ephemeral outputs (accumulator or pointwise)

        for op_id in sg_ops:
            if op_types[op_id] == "MatMul":
                lhs_t = inputs_list[op_id][0]
                rhs_t = inputs_list[op_id][1]
                if lhs_t not in ephemeral and lhs_t not in prev_retain_set:
                    matmul_lhs[lhs_t] = True
                if rhs_t not in ephemeral and rhs_t not in prev_retain_set:
                    matmul_rhs[rhs_t] = True
                for t_id in outputs_list[op_id]:
                    if t_id not in ephemeral:
                        out_tiles[t_id] = True  # non-ephemeral accumulator only
            else:  # Pointwise
                for t_id in inputs_list[op_id]:
                    if t_id not in ephemeral and t_id not in prev_retain_set:
                        pw_inputs[t_id] = True
                for t_id in outputs_list[op_id]:
                    if t_id not in ephemeral:
                        out_tiles[t_id] = True

        # Joint working set (peak, during first k-step of a spatial tile)
        joint_ws = 0
        for t_id in matmul_lhs:
            joint_ws += widths[t_id] * h   # full K dimension, loaded once per tile
        for _t in matmul_rhs:
            joint_ws += w * k               # RHS strip per k-step
        for _t in pw_inputs:
            joint_ws += w * h               # pointwise input, loaded once per tile
        for _t in out_tiles:
            joint_ws += w * h               # output slice / accumulator in fast memory

        total_ws = joint_ws + prev_retained_size

        # Check: retained tensors after this subgraph alone fit in fast memory
        retain_set = set(tensors_to_retain[sg_idx]) if sg_idx < len(tensors_to_retain) else set()
        retained_size = sum(widths[t] * heights[t] for t in retain_set)
        if retained_size > fast_mem:
            errors.append(
                f"Subgraph {sg_idx}: retained tensors after subgraph ({retained_size}) "
                f"exceed fast memory ({fast_mem})"
            )

        # Check: joint working set + previously retained tensors must fit
        if total_ws > fast_mem:
            errors.append(
                f"Subgraph {sg_idx}: joint working set ({joint_ws}) + "
                f"prev retained ({prev_retained_size}) = {total_ws} exceeds "
                f"fast memory ({fast_mem}) — OOM! [w={w},h={h},k={k}]"
            )

    # --- Check graph outputs are NOT retained at the end ---
    # All graph outputs must end up in slow memory (this is a hard constraint)
    if n_sg > 0:
        final_retain = set(tensors_to_retain[-1])
        retained_graph_outputs = final_retain & graph_outputs
        if retained_graph_outputs:
            errors.append(
                f"Graph outputs {sorted(retained_graph_outputs)} are retained after last subgraph "
                f"(must be evicted to slow memory)"
            )

    # --- Check subgraph_latencies are positive ---
    for i, lat in enumerate(subgraph_latencies):
        if lat <= 0:
            errors.append(f"Subgraph {i}: latency must be positive, got {lat}")

    # --- Verify subgraph_latencies against independently computed roofline model ---
    try:
        computed_lats = compute_latency(problem, solution)
        for i, (reported, computed) in enumerate(zip(subgraph_latencies, computed_lats)):
            if computed > 0:
                rel_err = abs(reported - computed) / computed
                if rel_err > 0.01:  # 1% tolerance for floating-point differences
                    warnings.append(
                        f"Subgraph {i}: reported latency {reported:.2f} differs from "
                        f"roofline model {computed:.2f} (rel error {rel_err:.2%})"
                    )
    except Exception as e:
        warnings.append(f"Could not verify latencies against roofline model: {e}")

    reported_total = sum(subgraph_latencies)
    valid = len(errors) == 0
    return valid, errors, warnings, reported_total


def compute_latency(problem, solution):
    """
    Independently compute latencies for all subgraphs using the roofline model.
    Returns list of computed latencies per subgraph.

    Model:
      - Per step: latency = max(compute, mem_in + mem_out)
      - traversal_orders=null → no intra-subgraph spatial data reuse
      - traversal_orders=[perm] → MatMul LHS reused when row unchanged, RHS when col unchanged
      - split-K: LHS loaded once per spatial tile, RHS per k-step, output accumulator stays
    """
    widths = problem["widths"]
    heights = problem["heights"]
    inputs_list = problem["inputs"]
    outputs_list = problem["outputs"]
    base_costs = problem["base_costs"]
    op_types = problem["op_types"]
    slow_bw = problem["slow_memory_bandwidth"]
    native_w, native_h = problem["native_granularity"]
    num_ops = len(base_costs)

    subgraphs = solution["subgraphs"]
    granularities = solution["granularities"]
    tensors_to_retain = solution["tensors_to_retain"]
    traversal_orders = solution["traversal_orders"]
    n_sg = len(subgraphs)

    # Precompute: tensor_producer, graph_inputs, graph_outputs
    tensor_producer = {}
    all_produced = set()
    all_consumed = set()
    for op_id in range(num_ops):
        for t in outputs_list[op_id]:
            tensor_producer[t] = op_id
            all_produced.add(t)
        for t in inputs_list[op_id]:
            all_consumed.add(t)
    graph_outputs = set(range(len(widths))) - all_consumed

    computed_latencies = []

    for sg_idx in range(n_sg):
        sg_ops = subgraphs[sg_idx]
        w, h, k = granularities[sg_idx]
        trav = traversal_orders[sg_idx]
        op_set = set(sg_ops)
        prev_retained = set(tensors_to_retain[sg_idx - 1]) if sg_idx > 0 else set()
        curr_retained = set(tensors_to_retain[sg_idx])

        # Identify ephemeral tensors: produced AND consumed within subgraph.
        # External consumers don't prevent ephemeral status — they're served
        # by tensors_to_retain (if retained) or recomputation (if discarded).
        ephemeral = set()
        for op_id in sg_ops:
            for t_id in outputs_list[op_id]:
                if t_id in graph_outputs:
                    continue
                has_consumer_in_sg = any(
                    t_id in inputs_list[op2] for op2 in sg_ops if op2 != op_id
                )
                if has_consumer_in_sg:
                    ephemeral.add(t_id)

        # Determine spatial dimensions from output tensors
        max_W, max_H = 0, 0
        for op_id in sg_ops:
            for t_id in outputs_list[op_id]:
                max_W = max(max_W, widths[t_id])
                max_H = max(max_H, heights[t_id])

        n_tiles_w = math.ceil(max_W / w) if max_W > 0 else 1
        n_tiles_h = math.ceil(max_H / h) if max_H > 0 else 1
        n_tiles = n_tiles_w * n_tiles_h

        # Determine K (reduction dimension) from MatMul LHS inputs
        K = 0
        for op_id in sg_ops:
            if op_types[op_id] == "MatMul":
                lhs_t = inputs_list[op_id][0]
                K = max(K, widths[lhs_t])
        n_k = max(1, math.ceil(K / k)) if K > 0 else 1

        # Compute cost per step (with native granularity padding)
        pad = math.ceil(w / native_w) * math.ceil(h / native_h)
        matmul_compute_per_step = sum(
            base_costs[op_id] * pad / n_k
            for op_id in sg_ops if op_types[op_id] == "MatMul"
        )
        pointwise_compute = sum(
            base_costs[op_id] * pad
            for op_id in sg_ops if op_types[op_id] != "MatMul"
        )

        # Classify unique boundary input tensors
        # Each entry: (role, first_k_load, mid_k_load)
        #   mm_lhs: load widths[t]*h once per tile, 0 on subsequent k-steps
        #   mm_rhs: load w*k every k-step
        #   pw_input: load w*h once per tile, 0 on subsequent k-steps
        boundary_inputs = {}  # t_id -> (role, first_k_load, mid_k_load)
        for op_id in sg_ops:
            for i, t_id in enumerate(inputs_list[op_id]):
                if t_id in ephemeral or t_id in prev_retained or t_id in boundary_inputs:
                    continue
                if op_types[op_id] == "MatMul":
                    if i == 0:  # LHS
                        boundary_inputs[t_id] = ("mm_lhs", widths[t_id] * h, 0)
                    else:  # RHS
                        boundary_inputs[t_id] = ("mm_rhs", w * k, w * k)
                else:
                    boundary_inputs[t_id] = ("pw_input", w * h, 0)

        # Classify boundary output tensors (non-ephemeral, non-retained)
        boundary_output_size = 0
        for op_id in sg_ops:
            for t_id in outputs_list[op_id]:
                if t_id not in ephemeral and t_id not in curr_retained:
                    boundary_output_size += w * h

        # Build traversal sequence
        tile_seq = trav if trav is not None else list(range(n_tiles))
        has_reuse = trav is not None

        # Execute tiles and accumulate latency
        total_latency = 0.0
        prev_row, prev_col = -1, -1

        for tile_pos, tile_idx in enumerate(tile_seq):
            row = tile_idx // n_tiles_w
            col = tile_idx % n_tiles_w

            for kstep in range(n_k):
                # Compute
                compute = matmul_compute_per_step
                if kstep == n_k - 1:
                    compute += pointwise_compute

                # Memory in
                mem_in = 0.0
                for t_id, (role, first_load, mid_load) in boundary_inputs.items():
                    if kstep == 0:
                        # First k-step: check spatial reuse
                        if role == "mm_lhs":
                            if has_reuse and tile_pos > 0 and row == prev_row:
                                pass  # Reuse LHS (same row)
                            else:
                                mem_in += first_load
                        elif role == "mm_rhs":
                            if has_reuse and tile_pos > 0 and col == prev_col:
                                pass  # Reuse RHS (same col)
                            else:
                                mem_in += first_load
                        else:  # pw_input
                            mem_in += first_load
                    else:
                        mem_in += mid_load  # Only RHS loads on mid k-steps

                # Memory out (only on last k-step)
                mem_out = 0.0
                if kstep == n_k - 1:
                    mem_out = boundary_output_size

                mem_time = (mem_in + mem_out) / slow_bw
                step_latency = max(compute, mem_time)
                total_latency += step_latency

            prev_row, prev_col = row, col

        computed_latencies.append(total_latency)

    return computed_latencies


def run_benchmark(binary, benchmark_path, output_path, timeout_sec, num_cores=8):
    """Run binary on benchmark with timeout. Returns (success, elapsed, error_msg)."""
    try:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(num_cores)
        # Pin to first N cores with taskset for faithful competition simulation
        # (competition runs on an 8-core machine; without affinity, a >8-core
        # host lets the process spread across more cores than allowed).
        import shutil
        cmd = [binary, benchmark_path, output_path]
        if shutil.which("taskset"):
            # Build CPU mask: cores 0..num_cores-1
            mask = (1 << num_cores) - 1
            cmd = ["taskset", f"0x{mask:x}"] + cmd
        start = time.time()
        result = subprocess.run(
            cmd,
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
