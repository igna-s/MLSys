import time, subprocess, glob, os

print("Timing final V8/V9 benchmarks (execution time only):")
print("-" * 50)
print(f"{'Benchmark':<20} | {'Time (s)':<10} | {'Timeout (s)':<10}")
print("-" * 50)

timeouts = {
    'mlsys-2026-1': 2,
    'mlsys-2026-5': 5,
    'mlsys-2026-9': 15,
    'mlsys-2026-13': 30,
    'mlsys-2026-17': 60
}

for f in sorted(glob.glob('../benchmarks/*.json')):
    name = os.path.basename(f).replace('.json', '')
    exe = "build/mlsys.exe"
    if os.name == 'nt':
        exe = "build\\mlsys.exe"
        f = os.path.normpath(f)
    
    t0 = time.time()
    res = subprocess.run([exe, f, "build/temp.json"], capture_output=True)
    dt = time.time() - t0
    
    timeout = timeouts.get(name, "N/A")
    status = "✅ PASS" if dt < timeout else "❌ FAIL"
    
    print(f"{name:<20} | {dt:<10.3f} | {timeout:<10} {status}")

print("-" * 50)
