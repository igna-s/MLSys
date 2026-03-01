import json, glob, os

build_dir = os.path.join(r'd:\GitHub\Golge algo\MLSys\solution', 'build')
files = sorted(glob.glob(os.path.join(build_dir, 'output_*.json')))
results = {}
total = 0.0
for f in files:
    with open(f) as fp:
        data = json.load(fp)
    lats = data.get('subgraph_latencies', [])
    s = sum(lats)
    total += s
    name = os.path.basename(f).replace('output_', '')
    results[name] = {"score": s, "subgraphs": len(lats)}

results["TOTAL"] = {"score": total, "under_10m": total < 10_000_000}
with open(os.path.join(build_dir, 'results.json'), 'w') as fp:
    json.dump(results, fp, indent=2)
