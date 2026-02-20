#include "scheduler.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <vector>
#include <cstdint>

namespace mlsys_solver {

static inline int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static std::vector<int64_t> make_snake(int64_t cols, int64_t rows) {
    std::vector<int64_t> ord;
    ord.reserve(cols * rows);
    for (int64_t r = 0; r < rows; ++r) {
        if (r & 1) for (int64_t c = cols-1; c >= 0; --c) ord.push_back(r*cols+c);
        else       for (int64_t c = 0; c < cols; ++c)    ord.push_back(r*cols+c);
    }
    return ord;
}

// ---------------------------------------------------------------------------
// Evaluate subgraph latency.  Returns {total_latency, valid}.
//
// retained_outputs: tensors from this subgraph that will be RETAINED (not evicted).
//   These save store bandwidth.
// resident: tensors already in fast memory from the previous subgraph.
//   These save load bandwidth.
// ---------------------------------------------------------------------------
struct EvalResult { double latency; bool valid; };

static EvalResult eval_latency(
    const Problem& prob,
    const std::vector<size_t>& ops,
    int64_t gw, int64_t gh, int64_t gk,
    const std::set<size_t>& resident,
    const std::set<size_t>& retained_outputs,
    const std::vector<int>& last_use,
    int sg_last_op
) {
    int64_t nw = prob.native_granularity.width;
    int64_t nh = prob.native_granularity.height;
    double bw = std::max(1.0, (double)prob.slow_memory_bandwidth);

    std::set<size_t> produced, consumed, all_t;
    bool has_mm = false;
    int64_t K_dim = 0;

    for (size_t oi : ops) {
        const auto& op = prob.ops[oi];
        if (op.op_type == "MatMul") {
            has_mm = true;
            K_dim = std::max(K_dim, prob.tensors[op.inputs[0]].width);
        }
        for (auto t : op.inputs)  { consumed.insert(t); all_t.insert(t); }
        for (auto t : op.outputs) { produced.insert(t); all_t.insert(t); }
    }

    // Ephemeral: produced AND consumed locally AND not needed later
    std::set<size_t> ephemeral;
    for (size_t t : produced)
        if (consumed.count(t) && last_use[t] <= sg_last_op)
            ephemeral.insert(t);

    // Output dims and grid
    int64_t outW = 0, outH = 0;
    for (size_t oi : ops)
        for (auto t : prob.ops[oi].outputs) {
            auto& T = prob.tensors[t];
            if (T.width * T.height > outW * outH) { outW = T.width; outH = T.height; }
        }
    if (outW == 0) return {0, false};

    int64_t cols = ceildiv(outW, gw);
    int64_t rows = ceildiv(outH, gh);
    int64_t tiles = cols * rows;
    int64_t ksteps = (has_mm && K_dim > 0 && gk > 0) ? ceildiv(K_dim, gk) : 1;

    // Working set check
    int64_t ws = 0;
    for (size_t t : all_t) {
        if (ephemeral.count(t)) continue;
        bool lhs = false, rhs = false;
        for (size_t oi : ops) {
            auto& op = prob.ops[oi];
            if (op.op_type == "MatMul" && op.inputs.size() >= 2) {
                if (op.inputs[0] == t) lhs = true;
                if (op.inputs[1] == t) rhs = true;
            }
        }
        if      (lhs) ws += gh * gk;
        else if (rhs) ws += gw * gk;
        else          ws += gw * gh;
    }
    if (ws > prob.fast_memory_capacity) return {0, false};

    // Compute costs
    double pw_cost = 0, mm_cost_per_kstep = 0;
    for (size_t oi : ops) {
        auto& op = prob.ops[oi];
        if (op.op_type == "MatMul") mm_cost_per_kstep += (double)op.base_cost / ksteps;
        else                        pw_cost += op.base_cost;
    }
    double pad = (double)(std::max(gw, nw) * std::max(gh, nh)) / (double)(gw * gh);
    pw_cost *= pad;
    mm_cost_per_kstep *= pad;

    // Per-tensor IO costs
    double lhs_load = 0, rhs_load = 0, pw_load = 0, store_cost = 0;

    for (size_t t : all_t) {
        if (ephemeral.count(t)) continue;
        bool is_p = produced.count(t), is_c = consumed.count(t);
        bool is_lhs = false, is_rhs = false;
        for (size_t oi : ops) {
            auto& op = prob.ops[oi];
            if (op.op_type == "MatMul" && op.inputs.size() >= 2) {
                if (op.inputs[0] == t) is_lhs = true;
                if (op.inputs[1] == t) is_rhs = true;
            }
        }
        int64_t sl = is_lhs ? gh*gk : (is_rhs ? gw*gk : gw*gh);
        double cost = (double)sl / bw;

        if (is_c && !is_p && !resident.count(t)) {
            if (is_lhs)       lhs_load += cost;
            else if (is_rhs)  rhs_load += cost;
            else              pw_load += cost;
        }
        // Only store if NOT retained
        if (is_p && !retained_outputs.count(t)) {
            store_cost += cost;
        }
    }

    // Analytical latency with snake traversal
    double total_lat = 0;

    for (int64_t kk = 0; kk < ksteps; ++kk) {
        double comp = mm_cost_per_kstep;
        if (kk == 0) comp += pw_cost;

        double pw_mem = (kk == 0) ? pw_load : 0;
        double st_mem = (kk == ksteps - 1) ? store_cost : 0;

        if (has_mm && tiles > 1 && (lhs_load > 0 || rhs_load > 0)) {
            double m_00 = lhs_load + rhs_load + pw_mem + st_mem;
            total_lat += std::max(comp, m_00);

            if (cols > 1) {
                double m_0c = rhs_load + pw_mem + st_mem;
                total_lat += (cols - 1) * std::max(comp, m_0c);
            }

            for (int64_t r = 1; r < rows; ++r) {
                double m_r0 = lhs_load + pw_mem + st_mem;
                total_lat += std::max(comp, m_r0);
                if (cols > 1) {
                    double m_rc = rhs_load + pw_mem + st_mem;
                    total_lat += (cols - 1) * std::max(comp, m_rc);
                }
            }
        } else {
            double m = lhs_load + rhs_load + pw_mem + st_mem;
            total_lat += tiles * std::max(comp, m);
        }
    }

    return {total_lat, true};
}

// ---------------------------------------------------------------------------
// Find best granularity for a subgraph.
// ---------------------------------------------------------------------------
struct BestResult {
    double latency; int64_t w, h, k;
    std::vector<int64_t> traversal; bool valid;
};

static BestResult find_best(
    const Problem& prob, const std::vector<size_t>& ops,
    const std::set<size_t>& resident,
    const std::set<size_t>& retained_outputs,
    const std::vector<int>& last_use
) {
    int64_t nw = prob.native_granularity.width;
    int64_t nh = prob.native_granularity.height;
    bool has_mm = false;
    int64_t K_dim = 0;
    int64_t outW = 0, outH = 0;
    int sg_last = (int)ops.back();

    for (size_t oi : ops) {
        auto& op = prob.ops[oi];
        if (op.op_type == "MatMul") {
            has_mm = true;
            K_dim = std::max(K_dim, prob.tensors[op.inputs[0]].width);
        }
        for (auto t : op.outputs) {
            auto& T = prob.tensors[t];
            if (T.width * T.height > outW * outH) { outW = T.width; outH = T.height; }
        }
    }

    // Candidate spatial sizes
    std::vector<int64_t> ws, hs;
    for (int64_t v = nw; v >= 16; v /= 2) ws.push_back(v);
    for (int64_t v = nh; v >= 16; v /= 2) hs.push_back(v);

    auto add_p2 = [](std::vector<int64_t>& v, int64_t val) {
        if (val >= 16 && (val & (val-1)) == 0)
            if (std::find(v.begin(), v.end(), val) == v.end()) v.push_back(val);
    };
    add_p2(ws, outW);
    add_p2(hs, outH);

    BestResult best;
    best.valid = false;
    best.latency = 1e30;

    auto try_cand = [&](int64_t sw, int64_t sh, int64_t sk) {
        auto ev = eval_latency(prob, ops, sw, sh, sk, resident, retained_outputs, last_use, sg_last);
        if (ev.valid && ev.latency < best.latency) {
            best.latency = ev.latency; best.w = sw; best.h = sh; best.k = sk;
            best.valid = true;
        }
    };

    if (has_mm && K_dim > 0) {
        for (int64_t sw : ws)
            for (int64_t sh : hs)
                for (int64_t kk = K_dim; kk >= 16; kk /= 2)
                    try_cand(sw, sh, kk);
    } else {
        for (int64_t sw : ws)
            for (int64_t sh : hs)
                try_cand(sw, sh, 1);
    }

    if (best.valid) {
        int64_t cs = ceildiv(outW, best.w);
        int64_t rs = ceildiv(outH, best.h);
        best.traversal = (cs * rs > 1) ? make_snake(cs, rs) : std::vector<int64_t>{};
    }
    return best;
}

// ---------------------------------------------------------------------------
// Compute which tensors to retain from a subgraph, and find the best
// granularity with those retention decisions baked into the latency.
// ---------------------------------------------------------------------------
struct PartitionResult {
    BestResult best;
    std::vector<size_t> retain;
};

static PartitionResult evaluate_partition(
    const Problem& prob,
    const std::vector<size_t>& ops,
    const std::set<size_t>& resident,
    const std::vector<int>& last_use,
    int next_op_idx  // = i (the op index after this subgraph)
) {
    int sg_last = (int)ops.back();

    // First pass: find tensors eligible for retention
    std::set<size_t> sg_t;
    for (size_t oi : ops) {
        for (auto t : prob.ops[oi].outputs) sg_t.insert(t);
        for (auto t : prob.ops[oi].inputs)  sg_t.insert(t);
    }

    // Identify produced (non-ephemeral) tensors
    std::set<size_t> produced;
    for (size_t oi : ops)
        for (auto t : prob.ops[oi].outputs) produced.insert(t);

    struct RetainCandidate {
        size_t idx;
        int64_t size;
    };
    std::vector<RetainCandidate> candidates;

    for (size_t t : sg_t) {
        if (last_use[t] > sg_last) {  // Needed by later ops
            int64_t sz = prob.tensors[t].width * prob.tensors[t].height;
            candidates.push_back({t, sz});
        }
    }

    // Sort by size descending (retain largest first to save most bandwidth)
    std::sort(candidates.begin(), candidates.end(),
        [](const RetainCandidate& a, const RetainCandidate& b) {
            return a.size > b.size;
        });

    // Greedily assign retention budget
    std::vector<size_t> retain_list;
    std::set<size_t> retain_set;
    int64_t budget = prob.fast_memory_capacity;
    for (auto& c : candidates) {
        if (budget >= c.size) {
            retain_list.push_back(c.idx);
            retain_set.insert(c.idx);
            budget -= c.size;
        }
    }

    // Now find best granularity with retention info
    auto best = find_best(prob, ops, resident, retain_set, last_use);

    return {best, retain_list};
}

// ---------------------------------------------------------------------------
// O(N²) DP solver
// ---------------------------------------------------------------------------
Solution Solve(const Problem& problem) {
    int N = problem.ops.size();

    std::vector<int> first_use(problem.tensors.size(), -1);
    std::vector<int> last_use(problem.tensors.size(), -1);
    for (int s = 0; s < N; ++s) {
        auto& op = problem.ops[s];
        for (auto t : op.inputs)  { if(first_use[t]==-1)first_use[t]=s; last_use[t]=s; }
        for (auto t : op.outputs) { if(first_use[t]==-1)first_use[t]=s; last_use[t]=s; }
    }

    std::vector<double> dp(N+1, 1e30);
    std::vector<int> par(N+1, -1);

    struct Info {
        int64_t w,h,k;
        std::vector<int64_t> trav;
        double lat;
        std::vector<size_t> retain;
    };
    std::vector<Info> info(N+1);
    dp[0] = 0;

    int WINDOW = 8;

    for (int i = 1; i <= N; ++i) {
        for (int j = std::max(0, i - WINDOW); j < i; ++j) {
            if (dp[j] >= 1e29) continue;

            std::vector<size_t> op_list;
            for (int k = j; k < i; ++k) op_list.push_back(k);

            std::set<size_t> res;
            if (j > 0 && par[j] >= 0)
                for (auto t : info[j].retain) res.insert(t);

            auto pr = evaluate_partition(problem, op_list, res, last_use, i);
            if (!pr.best.valid) continue;

            double total = dp[j] + pr.best.latency;
            if (total < dp[i]) {
                dp[i] = total;
                par[i] = j;
                info[i] = {pr.best.w, pr.best.h, pr.best.k,
                           pr.best.traversal, pr.best.latency, pr.retain};
            }
        }
    }

    // Reconstruct
    std::vector<Subgraph> sgs;
    int cur = N;
    while (cur > 0) {
        int p = par[cur];
        Subgraph sg;
        for (int k = p; k < cur; ++k) sg.ops.push_back(k);
        sg.granularity = {info[cur].w, info[cur].h, info[cur].k};
        sg.traversal_order = info[cur].trav;
        sg.subgraph_latency = info[cur].lat;
        sg.tensors_to_retain = info[cur].retain;
        sgs.push_back(sg);
        cur = p;
    }
    std::reverse(sgs.begin(), sgs.end());

    Solution sol;
    sol.subgraphs = sgs;
    return sol;
}

} // namespace mlsys_solver
