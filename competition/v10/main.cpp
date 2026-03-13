#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "scheduler.h"
#include <nlohmann/json.hpp> // Needs nlohmann/json

using json = nlohmann::json;
using namespace mlsys_solver;

Problem ReadProblem(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(1);
    }
    
    json j;
    f >> j;

    Problem prob;
    
    // Parse tensors
    auto widths = j["widths"];
    auto heights = j["heights"];
    for (size_t i = 0; i < widths.size(); ++i) {
        prob.tensors.push_back({widths[i], heights[i]});
    }

    // Parse ops
    auto inputs = j["inputs"];
    auto outputs = j["outputs"];
    auto op_types = j["op_types"];
    auto base_costs = j["base_costs"];
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        Op op;
        op.op_type = op_types[i];
        op.base_cost = base_costs[i];
        
        for (auto in : inputs[i]) op.inputs.push_back(in);
        for (auto out : outputs[i]) op.outputs.push_back(out);
        
        prob.ops.push_back(op);
    }

    prob.fast_memory_capacity = j["fast_memory_capacity"];
    prob.slow_memory_bandwidth = j["slow_memory_bandwidth"];
    
    auto native_granularity = j["native_granularity"];
    prob.native_granularity = {native_granularity[0], native_granularity[1], 1}; // Depth is per-subgraph, not native

    return prob;
}

void WriteSolution(const Solution& sol, const std::string& filename) {
    json j;
    j["subgraphs"] = json::array();
    j["granularities"] = json::array();
    j["tensors_to_retain"] = json::array();
    j["traversal_orders"] = json::array();
    j["subgraph_latencies"] = json::array();

    for (const auto& sg : sol.subgraphs) {
        j["subgraphs"].push_back(sg.ops);
        j["granularities"].push_back({sg.granularity.width, sg.granularity.height, sg.granularity.depth});
        j["tensors_to_retain"].push_back(sg.tensors_to_retain);
        
        if (sg.traversal_order.empty()) {
            j["traversal_orders"].push_back(nullptr);
        } else {
            j["traversal_orders"].push_back(sg.traversal_order);
        }
        
        j["subgraph_latencies"].push_back(sg.subgraph_latency);
    }

    std::ofstream o(filename);
    o << j.dump(2) << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_input.json> <path_to_output.json>\n";
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    try {
        Problem problem = ReadProblem(input_file);
        Solution solution = Solve(problem);
        WriteSolution(solution, output_file);
        
        std::cout << "Successfully generated schedule -> " << output_file << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
