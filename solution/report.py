import json, glob, os, subprocess, time

solution_dir = r'd:\GitHub\Golge algo\MLSys\solution'
build_dir = os.path.join(solution_dir, 'build')
benchmarks_dir = r'd:\GitHub\Golge algo\MLSys\benchmarks'
binary = os.path.join(build_dir, 'mlsys.exe')

# Timeouts per README
timeouts = {
    'mlsys-2026-1': 2, 'mlsys-2026-5': 5, 'mlsys-2026-9': 15,
    'mlsys-2026-13': 30, 'mlsys-2026-17': 60
}

files = sorted(glob.glob(os.path.join(build_dir, 'output_*.json')))
total = 0.0
print(f'{"Benchmark":20s} {"Score":>15s} {"Subgraphs":>10s} {"Time(s)":>10s} {"Timeout(s)":>10s} {"OK?":>5s}')
print('-' * 75)

for f in files:
    with open(f) as fp:
        data = json.load(fp)
    lats = data.get('subgraph_latencies', [])
    s = sum(lats)
    total += s
    name = os.path.basename(f).replace('output_', '').replace('.json', '')
    
    # Re-run to get timing
    bench_file = os.path.join(benchmarks_dir, name + '.json')
    out_file = os.path.join(build_dir, 'temp_timing.json')
    t0 = time.time()
    subprocess.run([binary, bench_file, out_file], capture_output=True, timeout=120)
    elapsed = time.time() - t0
    
    tmax = timeouts.get(name, 60)
    ok = 'YES' if elapsed < tmax else 'NO'
    print(f'{name:20s} {s:>15,.2f} {len(lats):>10d} {elapsed:>10.2f} {tmax:>10d} {ok:>5s}')

print('-' * 75)
print(f'{"TOTAL":20s} {total:>15,.2f}')
