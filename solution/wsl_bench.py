#!/usr/bin/env python3
"""
WSL benchmark runner for mlsys (Track A).

Run this INSIDE WSL:
  python3 /mnt/d/GitHub/Golge\ algo/solution/wsl_bench.py

Usa el binario estático de submission/mlsys (ya compilado con -fopenmp).
Testea OMP_NUM_THREADS=8 y OMP_NUM_THREADS=1 y compara contra los timeouts
oficiales del README.

Copia el binario y los benchmarks a /tmp para evitar overhead del filesystem
NTFS y obtener tiempos representativos del entorno de evaluación (Linux nativo).
"""

import os
import subprocess
import json
import time
import sys
import shutil

# ---------------------------------------------------------------------------
# Paths (WSL-relative)
# ---------------------------------------------------------------------------
SOLUTION_DIR   = "/mnt/d/GitHub/Golge algo/solution"
BENCHMARKS_DIR = "/mnt/d/GitHub/Golge algo/benchmarks"
BUILD_DIR      = f"{SOLUTION_DIR}/build"

# Binario estático de la submission (compilado con Docker + -fopenmp)
BINARY_SRC = f"{SOLUTION_DIR}/submission/mlsys"

# Copiar a /tmp para timing en filesystem Linux nativo (sin overhead NTFS)
TMP_DIR = "/tmp/mlsys_bench"
BINARY  = f"{TMP_DIR}/mlsys"

# ---------------------------------------------------------------------------
# Official timeouts from README.md (Track A)
# ---------------------------------------------------------------------------
BENCHMARKS = [
    ("mlsys-2026-1.json",    2),
    ("mlsys-2026-5.json",    5),
    ("mlsys-2026-9.json",   15),
    ("mlsys-2026-13.json",  30),
    ("mlsys-2026-17.json",  60),
]

# ---------------------------------------------------------------------------
# Setup: copy binary + benchmarks to /tmp
# ---------------------------------------------------------------------------
def setup():
    if not os.path.exists(BINARY_SRC):
        print(f"ERROR: Binary not found: {BINARY_SRC}")
        sys.exit(1)

    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)
    os.makedirs(BUILD_DIR, exist_ok=True)

    shutil.copy2(BINARY_SRC, BINARY)
    os.chmod(BINARY, 0o755)

    for name, _ in BENCHMARKS:
        src = os.path.join(BENCHMARKS_DIR, name)
        dst = os.path.join(TMP_DIR, name)
        shutil.copy2(src, dst)

    r = subprocess.run(["file", BINARY], capture_output=True, text=True)
    print(f"Binary : {BINARY}")
    print(f"Type   : {r.stdout.strip().split(':', 1)[-1].strip()}")
    print(f"Note   : benchmarks + output written to {TMP_DIR} (Linux native FS)")
    print()

# ---------------------------------------------------------------------------
# Run one benchmark, return (wall_time, cycles | None, timed_out)
# ---------------------------------------------------------------------------
def run_one(name: str, threads: int, timeout_s: int):
    bench_path = os.path.join(TMP_DIR, name)
    out_path   = os.path.join(TMP_DIR, f"out_t{threads}_{name}")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    t0 = time.perf_counter()
    timed_out = False
    try:
        r = subprocess.run(
            [BINARY, bench_path, out_path],
            capture_output=True, text=True,
            timeout=timeout_s + 2,  # 2s grace period
            env=env
        )
        ok = (r.returncode == 0)
    except subprocess.TimeoutExpired:
        timed_out = True
        ok = False
    elapsed = time.perf_counter() - t0

    if timed_out:
        return elapsed, None, True

    cycles = None
    if ok and os.path.exists(out_path):
        try:
            with open(out_path) as f:
                data = json.load(f)
            cycles = sum(data.get("subgraph_latencies", []))
            # copy result back to Windows build dir for inspection
            shutil.copy2(out_path, os.path.join(BUILD_DIR, f"wsl_t{threads}_{name}"))
        except Exception as e:
            print(f"  [warn] parse error {out_path}: {e}")

    return elapsed, cycles, False

# ---------------------------------------------------------------------------
# Run and print table for one thread config
# ---------------------------------------------------------------------------
def run_config(threads: int):
    rows = []
    print(f"{'='*72}")
    print(f"  OMP_NUM_THREADS = {threads}")
    print(f"{'='*72}")
    print(f"  {'Benchmark':<22}  {'Time':>8}  {'Limit':>7}  {'% used':>8}  {'Cycles':>22}")
    print(f"  {'-'*69}")

    total_time   = 0.0
    total_cycles = 0.0

    for name, timeout_s in BENCHMARKS:
        elapsed, cycles, timed_out = run_one(name, threads, timeout_s)
        total_time += elapsed

        if timed_out:
            pct_str    = "TIMEOUT"
            cycles_str = "---"
        elif cycles is None:
            pct_str    = "ERROR"
            cycles_str = "---"
        else:
            pct_str    = f"{elapsed / timeout_s * 100:6.1f}%"
            cycles_str = f"{cycles:,.2f}"
            total_cycles += cycles

        print(f"  {name:<22}  {elapsed:>7.2f}s  {timeout_s:>5}s  {pct_str:>8}  {cycles_str:>22}")
        rows.append((name, elapsed, cycles, timed_out))

    print(f"  {'-'*69}")
    print(f"  {'TOTAL':<22}  {total_time:>7.2f}s  {'':>6}  {'':>8}  {total_cycles:>22,.2f}")
    print()
    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup()

    results = {}
    for threads in [8, 1]:
        results[threads] = run_config(threads)

    # -----------------------------------------------------------------------
    # Side-by-side comparison
    # -----------------------------------------------------------------------
    print(f"{'='*72}")
    print(f"  Comparison: 8 threads vs 1 thread")
    print(f"{'='*72}")
    print(f"  {'Benchmark':<22}  {'T8':>8}  {'T1':>8}  {'Speedup':>8}  {'Cycles(T8)':>22}  {'Cycles(T1)':>22}  {'Match?':>7}")
    print(f"  {'-'*100}")

    total_t8_time   = 0.0
    total_t1_time   = 0.0
    total_t8_cycles = 0.0
    total_t1_cycles = 0.0

    for i, (name, _) in enumerate(BENCHMARKS):
        _, t8_time, t8_cycles, t8_to = results[8][i]
        _, t1_time, t1_cycles, t1_to = results[1][i]

        speedup_str = f"{t1_time/t8_time:.2f}x" if t8_time > 0 else "---"
        t8_str = f"{t8_cycles:,.2f}" if t8_cycles is not None else ("TIMEOUT" if t8_to else "ERROR")
        t1_str = f"{t1_cycles:,.2f}" if t1_cycles is not None else ("TIMEOUT" if t1_to else "ERROR")

        if t8_cycles is not None and t1_cycles is not None:
            match = "OK" if abs(t8_cycles - t1_cycles) < 1e-4 else "DIFFER"
        else:
            match = "---"

        print(f"  {name:<22}  {t8_time:>7.2f}s  {t1_time:>7.2f}s  {speedup_str:>8}  {t8_str:>22}  {t1_str:>22}  {match:>7}")

        total_t8_time += t8_time
        total_t1_time += t1_time
        if t8_cycles: total_t8_cycles += t8_cycles
        if t1_cycles: total_t1_cycles += t1_cycles

    speedup_total = total_t1_time / total_t8_time if total_t8_time > 0 else 0
    print(f"  {'-'*100}")
    print(f"  {'TOTAL':<22}  {total_t8_time:>7.2f}s  {total_t1_time:>7.2f}s  {speedup_total:>7.2f}x  {total_t8_cycles:>22,.2f}  {total_t1_cycles:>22,.2f}")
    print()

if __name__ == "__main__":
    main()
