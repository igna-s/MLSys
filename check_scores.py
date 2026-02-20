import json, glob, os

total = 0
for f in sorted(glob.glob('solution/build/output_*.json')):
    with open(f) as fh:
        data = json.load(fh)
    s = sum(data['subgraph_latencies'])
    total += s
    n = len(data['subgraph_latencies'])
    print(f"{os.path.basename(f):40s} {s:>15,.2f} cycles  ({n} subgraphs)")
print("-" * 70)
print(f"{'TOTAL':40s} {total:>15,.2f} cycles")
