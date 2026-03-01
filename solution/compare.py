import json, glob, os

base = r'd:\GitHub\Golge algo\MLSys\solution\build'

def get_scores(subdir):
    d = os.path.join(base, subdir)
    files = sorted(glob.glob(os.path.join(d, 'output_*.json')))
    scores = {}
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        lats = data.get('subgraph_latencies', [])
        name = os.path.basename(f).replace('output_', '').replace('.json', '')
        scores[name] = {'score': sum(lats), 'subgraphs': len(lats)}
    return scores

v8 = get_scores('v8_outputs')
v10 = get_scores('v10_outputs')

result = {}
for k in sorted(set(list(v8.keys()) + list(v10.keys()))):
    s8 = v8.get(k, {}).get('score', 0)
    s10 = v10.get(k, {}).get('score', 0)
    sg8 = v8.get(k, {}).get('subgraphs', 0)
    sg10 = v10.get(k, {}).get('subgraphs', 0)
    pct = ((s10 - s8) / s8 * 100) if s8 > 0 else 0
    result[k] = {'v8': s8, 'v10': s10, 'change_pct': round(pct, 1), 'sg_v8': sg8, 'sg_v10': sg10}

t8 = sum(v8[k]['score'] for k in v8)
t10 = sum(v10[k]['score'] for k in v10)
result['TOTAL'] = {'v8': t8, 'v10': t10, 'change_pct': round((t10 - t8) / t8 * 100, 1)}

with open(os.path.join(base, 'comparison.json'), 'w') as fp:
    json.dump(result, fp, indent=2)
