import json
import glob
import os

def main():
    solution_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(solution_dir, "build")
    
    benchmark_files = glob.glob(os.path.join(build_dir, "output_*.json"))
    
    if not benchmark_files:
        print("No output files found to evaluate.")
        return
        
    print(f"Evaluating {len(benchmark_files)} benchmarks...\n")
    print(f"{'Benchmark':<25} | {'Total Latency (Score)':<25}")
    print("-" * 55)
    
    total_score = 0.0
    scores = {}
    
    for bf in sorted(benchmark_files):
        with open(bf, "r") as f:
            data = json.load(f)
            
        latencies = data.get("subgraph_latencies", [])
        score = sum(latencies)
        
        name = os.path.basename(bf).replace("output_", "")
        scores[name] = score
        total_score += score
        
        print(f"{name:<25} | {score:,.2f}")
        
    print("-" * 55)
    print(f"{'TOTAL LATENCY:':<25} | {total_score:,.2f}")
    
    print("\nEstimation of Points (Reference: min_team_cost / our_cost)")
    print("Assuming the median cost across teams is around our score (1.0 per benchmark):")
    print(f"Expected Minimum Points Baseline: {len(benchmark_files) * 0.8:.2f} - {len(benchmark_files) * 1.2:.2f} points (out of {len(benchmark_files)})")

if __name__ == "__main__":
    main()
