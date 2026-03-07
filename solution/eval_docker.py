import os
import glob
import time
import subprocess
import json

benchmarks = glob.glob(r"D:\GitHub\Golge algo\benchmarks\*.json")
benchmarks.sort()

print("--- MLSys V14 Docker Execution Report ---")
total_cycles = 0
total_time = 0

for b in benchmarks:
    name = os.path.basename(b)
    out_file = fr"D:\GitHub\Golge algo\solution\out_{name}"
    
    start = time.time()
    subprocess.run(
        ["docker", "run", "--rm", "-v", "D:\\GitHub\\Golge algo:/work", "mlsys-builder", f"/work/benchmarks/{name}", f"/work/solution/out_{name}"],
        capture_output=True
    )
    t = time.time() - start
    
    try:
        with open(out_file, 'r') as f:
            data = json.load(f)
            cycles = sum(sg.get('subgraph_latency', 0) for sg in data.get('subgraphs', []))
    except Exception as e:
        cycles = -1
        print(f"Error reading {out_file}: {e}")
        
    print(f"{name:<20} | Time: {t:05.2f}s | Cycles: {cycles:,}")
    total_time += t
    if cycles > 0:
        total_cycles += cycles

print("-" * 45)
print(f"Total Time:   {total_time:.2f}s")
print(f"Total Cycles: {total_cycles:,}")
