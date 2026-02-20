#include "scheduler.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <vector>
#include <cstdint>
#include <map>

namespace mlsys_solver {

static inline int64_t ceildiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

static std::vector<int64_t> make_snake(int64_t cols, int64_t rows) {
    std::vector<int64_t> ord;
    ord.reserve(cols * rows);
    for (int64_t r = 0; r < rows; ++r) {
        if (r & 1) for (int64_t c = cols-1; c >= 0; --c) ord.push_back(r*cols+c);
        else       for (int64_t c = 0; c < cols; ++c)    ord.push_back(r*cols+c);
    }
    return ord;
}

// Column-major snake: iterate columns first, reuse RHS strips across rows
static std::vector<int64_t> make_col_snake(int64_t cols, int64_t rows) {
    std::vector<int64_t> ord;
    ord.reserve(cols * rows);
    for (int64_t c = 0; c < cols; ++c) {
        if (c & 1) for (int64_t r = rows-1; r >= 0; --r) ord.push_back(r*cols+c);
        else       for (int64_t r = 0; r < rows; ++r)    ord.push_back(r*cols+c);
    }
    return ord;
}

// ---------------------------------------------------------------------------
// Core latency evaluator.
//
// KEY INSIGHT from PROBLEM.md Example 5: In split-K mode, "stationary" tensors
// (like the LHS that doesn't change across k-steps for a given spatial tile,
// or the output accumulator) are loaded ONCE and reused across all k-steps.
// Only "streaming" inputs (whose k-slice changes each step) are loaded per step.
//
// However, a tensor is only "stationary" if its FULL extent (across all k-slices)
// fits in the working set. Otherwise it must be streamed per k-step.
// ---------------------------------------------------------------------------
struct EvalResult { double latency; bool valid; int snake_mode; }; // snake_mode: 0=row, 1=col

static EvalResult eval_latency_inner(
    const Problem& prob,
    const std::vector<size_t>& ops,
    int64_t gw, int64_t gh, int64_t gk,
    const std::set<size_t>& resident,
    const std::set<size_t>& retained_outputs,
    const std::vector<int>& last_use,
    int sg_last_op,
    int snake_mode  // 0=row-snake, 1=col-snake
) {
    int64_t nw = prob.native_granularity.width;
    int64_t nh = prob.native_granularity.height;
    double bw = std::max(1.0, (double)prob.slow_memory_bandwidth);

    std::set<size_t> produced, consumed, all_t;
    bool has_mm = false;
    int64_t K_dim = 0;
    for (size_t oi : ops) {
        auto& op = prob.ops[oi];
        if (op.op_type == "MatMul") {
            has_mm = true;
            K_dim = std::max(K_dim, prob.tensors[op.inputs[0]].width);
        }
        for (auto t : op.inputs)  { consumed.insert(t); all_t.insert(t); }
        for (auto t : op.outputs) { produced.insert(t); all_t.insert(t); }
    }

    std::set<size_t> ephemeral;
    for (size_t t : produced)
        if (consumed.count(t) && last_use[t] <= sg_last_op)
            ephemeral.insert(t);

    int64_t outW = 0, outH = 0;
    for (size_t oi : ops)
        for (auto t : prob.ops[oi].outputs) {
            auto& T = prob.tensors[t];
            if (T.width * T.height > outW * outH) { outW = T.width; outH = T.height; }
        }
    if (outW == 0) return {0, false, 0};

    int64_t cols = ceildiv(outW, gw);
    int64_t rows = ceildiv(outH, gh);
    int64_t tiles = cols * rows;
    int64_t ksteps = (has_mm && K_dim > 0 && gk > 0) ? ceildiv(K_dim, gk) : 1;

    // --- Classify each boundary tensor ---
    struct TInfo {
        size_t idx;
        bool is_lhs, is_rhs, is_pw_in, is_output;
        int64_t full_size;    // full tensor size
        int64_t slice_k;      // per-k-step slice (h*gk or gw*gk)
        int64_t slice_spatial; // per-spatial-tile slice (gw*gh)
        bool stationary;      // can be kept fully resident across k-steps?
        bool needs_load;
        bool needs_store;
    };
    std::vector<TInfo> btensors;

    // First pass: determine roles and sizes
    int64_t ws_check = 0;
    for (size_t t : all_t) {
        if (ephemeral.count(t)) continue;
        bool is_p = produced.count(t), is_c = consumed.count(t);
        bool lhs = false, rhs = false;
        for (size_t oi : ops) {
            auto& op = prob.ops[oi];
            if (op.op_type == "MatMul" && op.inputs.size() >= 2) {
                if (op.inputs[0] == t) lhs = true;
                if (op.inputs[1] == t) rhs = true;
            }
        }

        TInfo ti;
        ti.idx = t;
        ti.is_lhs = lhs; ti.is_rhs = rhs;
        ti.is_pw_in = is_c && !lhs && !rhs;
        ti.is_output = is_p;
        ti.full_size = prob.tensors[t].width * prob.tensors[t].height;

        if (lhs)      { ti.slice_k = gh * gk; ti.slice_spatial = gh * K_dim; }
        else if (rhs)  { ti.slice_k = gw * gk; ti.slice_spatial = K_dim * gw; }
        else           { ti.slice_k = gw * gh; ti.slice_spatial = gw * gh; }

        // Stationary: loaded once per spatial tile, reused across k-steps.
        // Per PROBLEM.md Ex5: LHS (Tensor0) fully loaded and reused.
        // Working set for stationary = full spatial extent (gh*K or K*gw)
        // Working set for streaming = per-k slice
        ti.stationary = false; // decided below
        ti.needs_load = is_c && !is_p && !resident.count(t);
        ti.needs_store = is_p && !retained_outputs.count(t);

        btensors.push_back(ti);
    }

    // Try stationary mode for MatMul inputs if it helps and fits
    // Stationary means: WS uses full_spatial extent but load only once per spatial tile
    // Streaming means: WS uses k-slice but load every k-step
    // Only relevant when ksteps > 1
    if (ksteps > 1) {
        // Compute WS with all streaming first
        int64_t ws_streaming = 0;
        for (auto& ti : btensors) {
            if (ti.is_lhs || ti.is_rhs) ws_streaming += ti.slice_k;
            else ws_streaming += ti.slice_spatial;
        }

        // Try making LHS stationary
        int64_t ws_lhs_stat = 0;
        for (auto& ti : btensors) {
            if (ti.is_lhs) ws_lhs_stat += ti.slice_spatial; // full spatial
            else if (ti.is_rhs) ws_lhs_stat += ti.slice_k;  // streaming
            else ws_lhs_stat += ti.slice_spatial;
        }

        if (ws_lhs_stat <= prob.fast_memory_capacity) {
            for (auto& ti : btensors) if (ti.is_lhs) ti.stationary = true;
            ws_check = ws_lhs_stat;
        } else if (ws_streaming <= prob.fast_memory_capacity) {
            ws_check = ws_streaming;
        } else {
            return {0, false, 0}; // OOM
        }
    } else {
        // Single k-step: all spatial
        for (auto& ti : btensors) {
            ws_check += (ti.is_lhs || ti.is_rhs) ? ti.slice_k : ti.slice_spatial;
        }
    }

    if (ws_check > prob.fast_memory_capacity) return {0, false, 0};

    // --- Compute cost ---
    double pw_cost = 0, mm_cost_per_kstep = 0;
    for (size_t oi : ops) {
        auto& op = prob.ops[oi];
        if (op.op_type == "MatMul") mm_cost_per_kstep += (double)op.base_cost / ksteps;
        else pw_cost += op.base_cost;
    }
    double pad = (double)(std::max(gw, nw) * std::max(gh, nh)) / (double)(gw * gh);
    pw_cost *= pad;
    mm_cost_per_kstep *= pad;

    // --- Per-tensor memory costs ---
    // Stationary tensors: loaded once per spatial tile (on first k-step)
    //   - With snake: reused across tiles in same row (LHS) or column (RHS)
    // Streaming tensors: loaded every k-step
    //   - With snake: reused across tiles in same row (LHS) or column (RHS)

    double lhs_stat_load = 0, lhs_stream_load = 0;
    double rhs_stat_load = 0, rhs_stream_load = 0;
    double pw_load_cost = 0, store_cost = 0;

    for (auto& ti : btensors) {
        if (!ti.needs_load && !ti.needs_store) continue;
        if (ti.needs_load) {
            if (ti.is_lhs) {
                if (ti.stationary)
                    lhs_stat_load += (double)ti.slice_spatial / bw; // full row, once per tile
                else
                    lhs_stream_load += (double)ti.slice_k / bw;    // k-slice per step
            } else if (ti.is_rhs) {
                if (ti.stationary)
                    rhs_stat_load += (double)ti.slice_spatial / bw;
                else
                    rhs_stream_load += (double)ti.slice_k / bw;
            } else {
                pw_load_cost += (double)ti.slice_spatial / bw;
            }
        }
        if (ti.needs_store) {
            store_cost += (double)(gw * gh) / bw;
        }
    }

    // --- Analytical latency ---
    // For row-snake: LHS reused across tiles in same row
    // For col-snake: RHS reused across tiles in same column
    double total_lat = 0;

    // Determine reuse direction
    // Row-snake: LHS (row strips) reused across cols. RHS changes per col.
    // Col-snake: RHS (col strips) reused across rows. LHS changes per row.
    bool row_snake = (snake_mode == 0);

    for (int64_t kk = 0; kk < ksteps; ++kk) {
        double comp = mm_cost_per_kstep;
        if (kk == 0) comp += pw_cost;

        // Stationary loads only on kk==0
        double lhs_s = (kk == 0) ? lhs_stat_load : 0;
        double rhs_s = (kk == 0) ? rhs_stat_load : 0;
        double pw_m  = (kk == 0) ? pw_load_cost : 0;
        double st_m  = (kk == ksteps - 1) ? store_cost : 0;

        // Streaming loads every k-step
        double lhs_str = lhs_stream_load;
        double rhs_str = rhs_stream_load;

        if (has_mm && tiles > 1) {
            if (row_snake) {
                // Row 0: first tile loads all; rest reuse LHS
                double m_first = lhs_s + lhs_str + rhs_s + rhs_str + pw_m + st_m;
                total_lat += std::max(comp, m_first);
                if (cols > 1) {
                    double m_same = rhs_s + rhs_str + pw_m + st_m;
                    total_lat += (cols - 1) * std::max(comp, m_same);
                }
                for (int64_t r = 1; r < rows; ++r) {
                    double m_new = lhs_s + lhs_str + pw_m + st_m; // RHS reused from prev row end
                    total_lat += std::max(comp, m_new);
                    if (cols > 1) {
                        double m_mid = rhs_s + rhs_str + pw_m + st_m;
                        total_lat += (cols - 1) * std::max(comp, m_mid);
                    }
                }
            } else {
                // Col-snake: first tile loads all; rest reuse RHS
                double m_first = lhs_s + lhs_str + rhs_s + rhs_str + pw_m + st_m;
                total_lat += std::max(comp, m_first);
                if (rows > 1) {
                    double m_same = lhs_s + lhs_str + pw_m + st_m;
                    total_lat += (rows - 1) * std::max(comp, m_same);
                }
                for (int64_t c = 1; c < cols; ++c) {
                    double m_new = rhs_s + rhs_str + pw_m + st_m; // LHS reused from prev col end
                    total_lat += std::max(comp, m_new);
                    if (rows > 1) {
                        double m_mid = lhs_s + lhs_str + pw_m + st_m;
                        total_lat += (rows - 1) * std::max(comp, m_mid);
                    }
                }
            }
        } else {
            double m = lhs_s + lhs_str + rhs_s + rhs_str + pw_m + st_m;
            total_lat += tiles * std::max(comp, m);
        }
    }

    return {total_lat, true, snake_mode};
}

// Try both snake directions
static EvalResult eval_latency(
    const Problem& prob, const std::vector<size_t>& ops,
    int64_t gw, int64_t gh, int64_t gk,
    const std::set<size_t>& resident, const std::set<size_t>& retained,
    const std::vector<int>& last_use, int sg_last
) {
    auto r0 = eval_latency_inner(prob, ops, gw, gh, gk, resident, retained, last_use, sg_last, 0);
    auto r1 = eval_latency_inner(prob, ops, gw, gh, gk, resident, retained, last_use, sg_last, 1);
    if (r0.valid && r1.valid) return r0.latency <= r1.latency ? r0 : r1;
    if (r0.valid) return r0;
    return r1;
}

struct BestResult {
    double latency; int64_t w, h, k;
    std::vector<int64_t> traversal; bool valid;
};

static BestResult find_best(
    const Problem& prob, const std::vector<size_t>& ops,
    const std::set<size_t>& resident, const std::set<size_t>& retained,
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

    std::vector<int64_t> ws, hs;
    for (int64_t v = nw; v >= 16; v /= 2) ws.push_back(v);
    for (int64_t v = nh; v >= 16; v /= 2) hs.push_back(v);
    auto add_p2 = [](std::vector<int64_t>& v, int64_t val) {
        if (val >= 16 && (val & (val-1)) == 0)
            if (std::find(v.begin(), v.end(), val) == v.end()) v.push_back(val);
    };
    add_p2(ws, outW); add_p2(hs, outH);

    BestResult best; best.valid = false; best.latency = 1e30;

    auto try_cand = [&](int64_t sw, int64_t sh, int64_t sk) {
        auto ev = eval_latency(prob, ops, sw, sh, sk, resident, retained, last_use, sg_last);
        if (ev.valid && ev.latency < best.latency) {
            best.latency = ev.latency; best.w = sw; best.h = sh; best.k = sk;
            best.valid = true;
            int64_t cs = ceildiv(outW, sw), rs = ceildiv(outH, sh);
            if (cs * rs > 1) {
                best.traversal = (ev.snake_mode == 0) ? make_snake(cs, rs) : make_col_snake(cs, rs);
            } else {
                best.traversal = {};
            }
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
    return best;
}

// ---------------------------------------------------------------------------
// DP solver
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
    struct Info { int64_t w,h,k; std::vector<int64_t> trav; double lat; std::vector<size_t> retain; };
    std::vector<Info> info(N+1);
    dp[0] = 0;

    int WINDOW = 10;

    for (int i = 1; i <= N; ++i) {
        for (int j = std::max(0, i - WINDOW); j < i; ++j) {
            if (dp[j] >= 1e29) continue;
            std::vector<size_t> op_list;
            for (int k = j; k < i; ++k) op_list.push_back(k);

            std::set<size_t> res;
            if (j > 0 && par[j] >= 0)
                for (auto t : info[j].retain) res.insert(t);

            // Compute retention candidates
            int sg_last = i - 1;
            std::set<size_t> sg_t;
            for (size_t oi : op_list) {
                for (auto t : problem.ops[oi].outputs) sg_t.insert(t);
                for (auto t : problem.ops[oi].inputs)  sg_t.insert(t);
            }
            struct RC { size_t idx; int64_t sz; };
            std::vector<RC> rcands;
            for (size_t t : sg_t) {
                if (last_use[t] > sg_last) {
                    rcands.push_back({t, problem.tensors[t].width * problem.tensors[t].height});
                }
            }
            std::sort(rcands.begin(), rcands.end(), [](const RC& a, const RC& b) { return a.sz > b.sz; });
            std::vector<size_t> retain_list;
            std::set<size_t> retain_set;
            int64_t budget = problem.fast_memory_capacity;
            for (auto& c : rcands) {
                if (budget >= c.sz) { retain_list.push_back(c.idx); retain_set.insert(c.idx); budget -= c.sz; }
            }

            auto br = find_best(problem, op_list, res, retain_set, last_use);
            if (!br.valid) continue;

            double total = dp[j] + br.latency;
            if (total < dp[i]) {
                dp[i] = total; par[i] = j;
                info[i] = {br.w, br.h, br.k, br.traversal, br.latency, retain_list};
            }
        }
    }

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
    Solution sol; sol.subgraphs = sgs;
    return sol;
}

} // namespace mlsys_solver
