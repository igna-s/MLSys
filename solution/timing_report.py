import json, glob, os, subprocess, time

solution_dir = r'd:\GitHub\Golge algo\MLSys\solution'
build_dir = os.path.join(solution_dir, 'build')
benchmarks_dir = r'd:\GitHub\Golge algo\MLSys\benchmarks'
binary = os.path.join(build_dir, 'mlsys.exe')

timeouts = {
    'mlsys-2026-1': 2, 'mlsys-2026-5': 5, 'mlsys-2026-9': 15,
    'mlsys-2026-13': 30, 'mlsys-2026-17': 60
}

files = sorted(glob.glob(os.path.join(build_dir, 'output_*.json')))
results = {}
for f in files:
    with open(f) as fp:
        data = json.load(fp)
    lats = data.get('subgraph_latencies', [])
    name = os.path.basename(f).replace('output_', '').replace('.json', '')
    
    bench_file = os.path.join(benchmarks_dir, name + '.json')
    out_file = os.path.join(build_dir, 'temp_timing.json')
    t0 = time.time()
    subprocess.run([binary, bench_file, out_file], capture_output=True, timeout=120)
    elapsed = time.time() - t0
    
    tmax = timeouts.get(name, 60)
    results[name] = {
        'score': sum(lats), 'subgraphs': len(lats),
        'time_s': round(elapsed, 2), 'timeout_s': tmax,
        'within_timeout': elapsed < tmax
    }

results['TOTAL'] = {'score': sum(r['score'] for r in results.values() if 'score' in r)}
with open(os.path.join(build_dir, 'timing_report.json'), 'w') as fp:
    json.dump(results, fp, indent=2)
