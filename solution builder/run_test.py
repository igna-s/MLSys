import os
import subprocess
import glob
import time
import json
import shutil
import urllib.request
import sys

def run_tests():
    # 1. Download json.hpp if it doesn't exist
    solution_dir = os.path.dirname(os.path.abspath(__file__))
    nlohmann_dir = os.path.join(solution_dir, "nlohmann")
    json_hpp = os.path.join(nlohmann_dir, "json.hpp")
    if not os.path.exists(json_hpp):
        print("Downloading nlohmann/json.hpp...")
        os.makedirs(nlohmann_dir, exist_ok=True)
        urllib.request.urlretrieve("https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp", json_hpp)

    build_dir = os.path.join(solution_dir, "build")
    print(f"Building solution in {build_dir}...")
    os.makedirs(build_dir, exist_ok=True)
    
    binary_path = os.path.join(build_dir, "mlsys")
    if os.name == 'nt':
        binary_path += ".exe"

    src_files = ["main.cpp", "scheduler.cpp"]

    # 2. Try compiling
    # Use zig c++ bundled from python dependencies since Windows doesn't generally provide a compiler out of the box
    print(f"Using zig c++ via python module ({sys.executable})...")
    cmd = [sys.executable, "-m", "ziglang", "c++", "-O3", "-std=c++20"] + src_files + ["-I.", "-o", binary_path]
    try:
        subprocess.run(cmd, cwd=solution_dir, check=True)
        print("Build successful.")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return

    # 3. Find benchmarks
    project_root = os.path.dirname(solution_dir)
    benchmarks_dir = os.path.join(project_root, "benchmarks")
    benchmark_files = glob.glob(os.path.join(benchmarks_dir, "*.json"))
    
    if not benchmark_files:
        print(f"No benchmarks found in {benchmarks_dir}")
        return

    print(f"\nFound {len(benchmark_files)} benchmarks. Running tests...\n")
    
    for benchmark in sorted(benchmark_files):
        basename = os.path.basename(benchmark)
        output_file = os.path.join(build_dir, f"output_{basename}")
        
        print(f"Running benchmark: {basename}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [binary_path, benchmark, output_file],
                capture_output=True,
                text=True,
                check=True,
                timeout=120
            )
            elapsed = time.time() - start_time
            print(f"  [SUCCESS] Completed in {elapsed:.2f} seconds")
            
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    try:
                        data = json.load(f)
                        print(f"  [VALID] Output is valid JSON. Number of subgraphs: {len(data.get('subgraphs', []))}")
                    except json.JSONDecodeError:
                        print(f"  [ERROR] Output file generated but is not valid JSON!")
            else:
                print(f"  [ERROR] Output file not generated!")
                
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] Benchmark exceeded 120 seconds!")
        except subprocess.CalledProcessError as e:
            print(f"  [FAILED] Execution error: {e}")
            if e.stderr:
                print(f"  stderr: {e.stderr.strip()}")

if __name__ == "__main__":
    run_tests()