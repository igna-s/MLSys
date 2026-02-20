#ifndef SCHEDULER_H_
#define SCHEDULER_H_

#include <vector>
#include <string>
#include <cstdint>
#include <optional>

namespace mlsys_solver {

using Width = int64_t;
using Height = int64_t;
using Depth = int64_t;
using BaseCost = int64_t;
using FastMemCap = int64_t;
using Bandwidth = int64_t;

struct Tensor {
    Width width;
    Height height;
    
    int64_t size() const { return width * height; }
};

struct Op {
    std::string op_type;
    std::vector<size_t> inputs;
    std::vector<size_t> outputs;
    BaseCost base_cost;
};

struct Granularity {
    Width width;
    Height height;
    Depth depth;
};

struct Problem {
    std::vector<Tensor> tensors;
    std::vector<Op> ops;
    FastMemCap fast_memory_capacity;
    Bandwidth slow_memory_bandwidth;
    Granularity native_granularity;
};

struct Subgraph {
    std::vector<size_t> ops;
    std::vector<size_t> tensors_to_retain;
    Granularity granularity;
    std::vector<int64_t> traversal_order; // empty means null/raster
    double subgraph_latency;
};

struct Solution {
    std::vector<Subgraph> subgraphs;
};

// Main solver entry point
Solution Solve(const Problem& problem);

} // namespace mlsys_solver

#endif // SCHEDULER_H_
