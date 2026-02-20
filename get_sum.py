import json
import glob

total = 0
for f in sorted(glob.glob('solution/build/output_*.json')):
    with open(f) as file:
        data = json.load(file)
    s = sum(data['subgraph_latencies'])
    total += s
    print(f"{f}: {s:,.2f} cycles")
print("-" * 40)
print(f"TOTAL: {total:,.2f} cycles")
