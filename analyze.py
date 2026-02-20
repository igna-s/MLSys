import json, glob, os

total = 0
for f in sorted(glob.glob('solution/build/output_*.json')):
    with open(f) as fh:
        data = json.load(fh)
    s = sum(data['subgraph_latencies'])
    total += s
    n = len(data['subgraph_latencies'])
    name = os.path.basename(f)
    print(f"{name}: {s:,.0f} cycles ({n} subgraphs)")
    
    # Show top 3 most expensive subgraphs
    lats = data['subgraph_latencies']
    grans = data['granularities']
    for idx in sorted(range(len(lats)), key=lambda i: -lats[i])[:3]:
        print(f"  SG{idx}: {lats[idx]:,.0f} gran={grans[idx]}")

print("=" * 60)
print(f"TOTAL: {total:,.0f} cycles")
