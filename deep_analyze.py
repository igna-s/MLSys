"""Analyze benchmark structures and find optimization opportunities."""
import json, sys

def analyze(path):
    d = json.load(open(path))
    widths = d['widths']
    heights = d['heights']
    inputs = d['inputs']
    outputs = d['outputs']
    costs = d['base_costs']
    types = d['op_types']
    cap = d['fast_memory_capacity']
    bw = d['slow_memory_bandwidth']
    ng = d['native_granularity']
    
    N = len(inputs)
    T = len(widths)
    
    print(f"=== {path} ===")
    print(f"Ops: {N}, Tensors: {T}")
    print(f"FastMem: {cap}, BW: {bw}, Native: {ng}")
    
    # Graph inputs and outputs
    produced = set()
    consumed = set()
    for i in range(N):
        for t in inputs[i]: consumed.add(t)
        for t in outputs[i]: produced.add(t)
    
    graph_in = consumed - produced
    graph_out = produced - consumed
    print(f"Graph inputs: {sorted(graph_in)} ({len(graph_in)})")
    print(f"Graph outputs: {sorted(graph_out)} ({len(graph_out)})")
    
    # Total input/output bytes
    in_bytes = sum(widths[t]*heights[t] for t in graph_in)
    out_bytes = sum(widths[t]*heights[t] for t in graph_out)
    print(f"Input bytes: {in_bytes:,}, Output bytes: {out_bytes:,}")
    print(f"Min memory time (load+store): {(in_bytes+out_bytes)/bw:,.1f}")
    print(f"Total compute: {sum(costs):,}")
    print(f"Theoretical min: {max(sum(costs), (in_bytes+out_bytes)/bw):,.1f}")
    
    # Per-op info
    print(f"\nOps:")
    for i in range(min(N, 20)):
        ins = inputs[i]
        outs = outputs[i]
        in_sizes = [f"T{t}({widths[t]}x{heights[t]})" for t in ins]
        out_sizes = [f"T{t}({widths[t]}x{heights[t]})" for t in outs]
        print(f"  Op{i}: {types[i]} cost={costs[i]} in={in_sizes} out={out_sizes}")
    if N > 20:
        print(f"  ... ({N-20} more ops)")

for b in ['benchmarks/mlsys-2026-9.json', 'benchmarks/mlsys-2026-13.json', 'benchmarks/mlsys-2026-1.json']:
    analyze(b)
    print()
