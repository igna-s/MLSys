#!/usr/bin/env python3
"""Run v21 binary in WSL against all benchmarks with OMP_NUM_THREADS=1 and 8."""
import os, subprocess, json, time, sys, shutil

BENCH_DIR = "/mnt/d/GitHub/Golge algo/benchmarks"
BINARY    = "/tmp/v21build/mlsys_v21"
TMP       = "/tmp/v21_bench"

BENCHMARKS = [
    ("mlsys-2026-1.json",    2),
    ("mlsys-2026-5.json",    5),
    ("mlsys-2026-9.json",   15),
    ("mlsys-2026-13.json",  30),
    ("mlsys-2026-17.json",  60),
]

def setup():
    if os.path.exists(TMP): shutil.rmtree(TMP)
    os.makedirs(TMP)
    for name, _ in BENCHMARKS:
        shutil.copy2(os.path.join(BENCH_DIR, name), os.path.join(TMP, name))

def run_one(name, threads, to):
    inp = os.path.join(TMP, name)
    out = os.path.join(TMP, f"out_t{threads}_{name}")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    t0 = time.perf_counter()
    timed_out = False
    try:
        r = subprocess.run([BINARY, inp, out], capture_output=True, text=True, timeout=to+2, env=env)
        ok = r.returncode == 0
    except subprocess.TimeoutExpired:
        timed_out = True; ok = False
    el = time.perf_counter()-t0
    cycles = None
    if ok and os.path.exists(out):
        try:
            data = json.load(open(out))
            cycles = sum(data.get("subgraph_latencies", []))
        except Exception as e:
            print("parse err", e)
    return el, cycles, timed_out

def main():
    setup()
    results = {}
    for threads in [8, 1]:
        print(f"\n=== OMP_NUM_THREADS={threads} ===")
        print(f"{'Benchmark':<22} {'Time':>8} {'Limit':>6} {'Cycles':>18}")
        rows=[]
        tot_t=0.0; tot_c=0.0
        for name, to in BENCHMARKS:
            el, cy, tmo = run_one(name, threads, to)
            tot_t += el
            if cy is not None: tot_c += cy
            cs = f"{cy:,.0f}" if cy is not None else ("TIMEOUT" if tmo else "ERR")
            print(f"{name:<22} {el:>7.2f}s {to:>5}s {cs:>18}")
            rows.append((name, el, cy, tmo))
        print(f"{'TOTAL':<22} {tot_t:>7.2f}s        {tot_c:>18,.0f}")
        results[threads]=rows

    # JSON summary for the PDF update
    out = {}
    for t in (1,8):
        out[str(t)] = [
            {"bench": n, "time_s": el, "cycles": cy, "timeout": tmo}
            for (n,el,cy,tmo) in results[t]
        ]
    with open("/tmp/v21_bench/summary.json","w") as f:
        json.dump(out, f, indent=2)
    print("\nSummary written to /tmp/v21_bench/summary.json")

if __name__ == "__main__":
    main()
