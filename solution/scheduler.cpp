#include "scheduler.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <vector>
#include <cstdint>

namespace mlsys_solver {

static inline int64_t ceildiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

static std::vector<int64_t> make_snake(int64_t cols, int64_t rows) {
    std::vector<int64_t> o; o.reserve(cols * rows);
    for (int64_t r = 0; r < rows; ++r)
        if (r & 1) for (int64_t c = cols-1; c >= 0; --c) o.push_back(r*cols+c);
        else       for (int64_t c = 0; c < cols; ++c) o.push_back(r*cols+c);
    return o;
}

static std::vector<int64_t> make_col_snake(int64_t cols, int64_t rows) {
    std::vector<int64_t> o; o.reserve(cols * rows);
    for (int64_t c = 0; c < cols; ++c)
        if (c & 1) for (int64_t r = rows-1; r >= 0; --r) o.push_back(r*cols+c);
        else       for (int64_t r = 0; r < rows; ++r) o.push_back(r*cols+c);
    return o;
}

struct EvalResult { double latency; bool valid; int snake_mode; };

static EvalResult eval_inner(
    const Problem& prob, const std::vector<size_t>& ops,
    int64_t gw, int64_t gh, int64_t gk,
    const std::set<size_t>& resident, const std::set<size_t>& retained,
    const std::vector<int>& last_use, int sg_last, int snake_mode
) {
    int64_t nw = prob.native_granularity.width;
    int64_t nh = prob.native_granularity.height;
    double bw = std::max(1.0, (double)prob.slow_memory_bandwidth);

    std::set<size_t> produced, consumed, all_t;
    bool has_mm = false; int64_t K_dim = 0;
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
        if (consumed.count(t) && last_use[t] <= sg_last) ephemeral.insert(t);

    int64_t outW = 0, outH = 0;
    for (size_t oi : ops)
        for (auto t : prob.ops[oi].outputs) {
            auto& T = prob.tensors[t];
            if (T.width * T.height > outW * outH) { outW = T.width; outH = T.height; }
        }
    if (!outW) return {0, false, 0};

    int64_t cols = ceildiv(outW, gw), rows = ceildiv(outH, gh);
    int64_t tiles = cols * rows;
    int64_t ksteps = (has_mm && K_dim > 0 && gk > 0) ? ceildiv(K_dim, gk) : 1;

    // Classify tensors and compute working set
    struct TI { bool lhs, rhs; int64_t slice; bool need_ld, need_st; bool stationary; };
    std::vector<TI> bt;
    int64_t ws_streaming = 0, ws_lhs_stat = 0;
    bool can_stat = (ksteps > 1);

    for (size_t t : all_t) {
        if (ephemeral.count(t)) continue;
        bool is_p = produced.count(t), is_c = consumed.count(t);
        bool l = false, r = false;
        for (size_t oi : ops) {
            auto& op = prob.ops[oi];
            if (op.op_type == "MatMul" && op.inputs.size() >= 2) {
                if (op.inputs[0] == t) l = true;
                if (op.inputs[1] == t) r = true;
            }
        }
        int64_t sk = l ? gh*gk : (r ? gw*gk : gw*gh);
        int64_t sf = l ? gh*K_dim : (r ? K_dim*gw : gw*gh);
        ws_streaming += sk;
        ws_lhs_stat += (l ? sf : sk);

        TI ti;
        ti.lhs = l; ti.rhs = r;
        ti.need_ld = is_c && !is_p && !resident.count(t);
        ti.need_st = is_p && !retained.count(t);
        ti.stationary = false;
        ti.slice = sk; // will be updated if stationary
        bt.push_back(ti);
    }

    // Choose best residency mode
    bool use_stat = can_stat && ws_lhs_stat <= prob.fast_memory_capacity;
    if (use_stat) {
        int idx = 0;
        for (size_t t : all_t) {
            if (ephemeral.count(t)) continue;
            if (bt[idx].lhs) {
                bt[idx].stationary = true;
                bt[idx].slice = gh * K_dim; // full spatial extent
            }
            idx++;
        }
    } else if (ws_streaming > prob.fast_memory_capacity) {
        return {0, false, 0};
    }

    // Compute costs
    double pw_cost = 0, mm_per_kstep = 0;
    for (size_t oi : ops) {
        auto& op = prob.ops[oi];
        if (op.op_type == "MatMul") mm_per_kstep += (double)op.base_cost / ksteps;
        else pw_cost += op.base_cost;
    }
    double pad = (double)(std::max(gw, nw) * std::max(gh, nh)) / (double)(gw * gh);
    pw_cost *= pad; mm_per_kstep *= pad;

    // Per-tensor memory costs
    double lhs_ld = 0, rhs_ld = 0, pw_ld = 0, store_c = 0;
    for (auto& ti : bt) {
        double c = (double)ti.slice / bw;
        if (ti.need_ld) {
            if (ti.lhs)      lhs_ld += c;
            else if (ti.rhs) rhs_ld += c;
            else             pw_ld += c;
        }
        if (ti.need_st) store_c += (double)(gw * gh) / bw;
    }

    // Analytical latency — snake reuse for LHS (row) and RHS (boundary)
    double total = 0;
    bool rs = (snake_mode == 0);

    for (int64_t kk = 0; kk < ksteps; ++kk) {
        double comp = mm_per_kstep;
        if (kk == 0) comp += pw_cost;

        // Stationary tensors: loaded once on kk=0 (first tile of row/col)
        // Non-stationary: loaded every kstep
        double l_cost = use_stat ? ((kk == 0) ? lhs_ld : 0) : lhs_ld;
        double r_cost = rhs_ld;
        double pw = (kk == 0) ? pw_ld : 0;
        double st = (kk == ksteps - 1) ? store_c : 0;

        if (has_mm && tiles > 1) {
            if (rs) {
                // Row-snake: LHS reused across cols
                double m0 = l_cost + r_cost + pw + st;
                total += std::max(comp, m0);
                if (cols > 1) {
                    double mc = r_cost + pw + st;
                    total += (cols - 1) * std::max(comp, mc);
                }
                for (int64_t r = 1; r < rows; ++r) {
                    double mr = l_cost + pw + st; // RHS reused
                    total += std::max(comp, mr);
                    if (cols > 1) {
                        double mrc = r_cost + pw + st;
                        total += (cols - 1) * std::max(comp, mrc);
                    }
                }
            } else {
                // Col-snake: RHS reused across rows
                double m0 = l_cost + r_cost + pw + st;
                total += std::max(comp, m0);
                if (rows > 1) {
                    double mr = l_cost + pw + st;
                    total += (rows - 1) * std::max(comp, mr);
                }
                for (int64_t c = 1; c < cols; ++c) {
                    double mc = r_cost + pw + st; // LHS reused
                    total += std::max(comp, mc);
                    if (rows > 1) {
                        double mr2 = l_cost + pw + st;
                        total += (rows - 1) * std::max(comp, mr2);
                    }
                }
            }
        } else {
            double m = l_cost + r_cost + pw + st;
            total += tiles * std::max(comp, m);
        }
    }
    return {total, true, snake_mode};
}

static EvalResult eval_latency(
    const Problem& p, const std::vector<size_t>& ops,
    int64_t gw, int64_t gh, int64_t gk,
    const std::set<size_t>& res, const std::set<size_t>& ret,
    const std::vector<int>& lu, int sl
) {
    auto r0 = eval_inner(p, ops, gw, gh, gk, res, ret, lu, sl, 0);
    auto r1 = eval_inner(p, ops, gw, gh, gk, res, ret, lu, sl, 1);
    if (r0.valid && r1.valid) return r0.latency <= r1.latency ? r0 : r1;
    return r0.valid ? r0 : r1;
}

struct Best { double lat; int64_t w, h, k; std::vector<int64_t> trav; bool valid; };

static Best find_best(
    const Problem& p, const std::vector<size_t>& ops,
    const std::set<size_t>& res, const std::set<size_t>& ret,
    const std::vector<int>& lu
) {
    int64_t nw = p.native_granularity.width, nh = p.native_granularity.height;
    bool has_mm = false; int64_t K_dim = 0, outW = 0, outH = 0;
    int sl = (int)ops.back();
    for (size_t oi : ops) {
        auto& op = p.ops[oi];
        if (op.op_type == "MatMul") { has_mm = true; K_dim = std::max(K_dim, p.tensors[op.inputs[0]].width); }
        for (auto t : op.outputs) {
            auto& T = p.tensors[t]; if (T.width*T.height > outW*outH) { outW = T.width; outH = T.height; }
        }
    }

    // Wide granularity search: powers of 2 from 16 up to max tensor dim
    auto gen = [](int64_t native, int64_t maxd) {
        std::set<int64_t> s;
        for (int64_t v = native; v >= 16; v /= 2) s.insert(v);
        for (int64_t v = native * 2; v <= maxd && v <= 4096; v *= 2) s.insert(v);
        return std::vector<int64_t>(s.begin(), s.end());
    };
    auto ws = gen(nw, outW); auto hs = gen(nh, outH);

    Best best; best.valid = false; best.lat = 1e30;
    auto try_c = [&](int64_t sw, int64_t sh, int64_t sk) {
        auto ev = eval_latency(p, ops, sw, sh, sk, res, ret, lu, sl);
        if (ev.valid && ev.latency < best.lat) {
            best.lat = ev.latency; best.w = sw; best.h = sh; best.k = sk; best.valid = true;
            int64_t cs = ceildiv(outW, sw), rs = ceildiv(outH, sh);
            best.trav = (cs*rs > 1) ? ((ev.snake_mode == 0) ? make_snake(cs, rs) : make_col_snake(cs, rs)) : std::vector<int64_t>{};
        }
    };

    if (has_mm && K_dim > 0) {
        for (int64_t sw : ws) for (int64_t sh : hs)
            for (int64_t kk = K_dim; kk >= 16; kk /= 2) try_c(sw, sh, kk);
    } else {
        for (int64_t sw : ws) for (int64_t sh : hs) try_c(sw, sh, 1);
    }
    return best;
}

Solution Solve(const Problem& problem) {
    int N = problem.ops.size();
    std::vector<int> fu(problem.tensors.size(), -1), lu(problem.tensors.size(), -1);
    for (int s = 0; s < N; ++s) {
        auto& op = problem.ops[s];
        for (auto t : op.inputs)  { if (fu[t]==-1) fu[t]=s; lu[t]=s; }
        for (auto t : op.outputs) { if (fu[t]==-1) fu[t]=s; lu[t]=s; }
    }

    std::vector<double> dp(N+1, 1e30);
    std::vector<int> par(N+1, -1);
    struct Info { int64_t w, h, k; std::vector<int64_t> tr; double lat; std::vector<size_t> ret; };
    std::vector<Info> info(N+1);
    dp[0] = 0;
    int W = std::min(12, N);

    for (int i = 1; i <= N; ++i) {
        for (int j = std::max(0, i-W); j < i; ++j) {
            if (dp[j] >= 1e29) continue;
            std::vector<size_t> ol; for (int k=j; k<i; ++k) ol.push_back(k);
            std::set<size_t> res;
            if (j > 0 && par[j] >= 0) for (auto t : info[j].ret) res.insert(t);

            int sg_last = i - 1;
            std::set<size_t> sg_t;
            for (size_t oi : ol) {
                for (auto t : problem.ops[oi].outputs) sg_t.insert(t);
                for (auto t : problem.ops[oi].inputs) sg_t.insert(t);
            }
            struct RC { size_t i; int64_t s; };
            std::vector<RC> rc;
            for (size_t t : sg_t) if (lu[t] > sg_last)
                rc.push_back({t, problem.tensors[t].width * problem.tensors[t].height});
            std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.s > b.s; });
            std::vector<size_t> rl; std::set<size_t> rs;
            int64_t bud = problem.fast_memory_capacity;
            for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }

            auto br = find_best(problem, ol, res, rs, lu);
            if (!br.valid) continue;
            double tot = dp[j] + br.lat;
            if (tot < dp[i]) {
                dp[i] = tot; par[i] = j;
                info[i] = {br.w, br.h, br.k, br.trav, br.lat, rl};
            }
        }
    }

    std::vector<Subgraph> sgs;
    int cur = N;
    while (cur > 0) {
        int p = par[cur]; Subgraph sg;
        for (int k = p; k < cur; ++k) sg.ops.push_back(k);
        sg.granularity = {info[cur].w, info[cur].h, info[cur].k};
        sg.traversal_order = info[cur].tr;
        sg.subgraph_latency = info[cur].lat;
        sg.tensors_to_retain = info[cur].ret;
        sgs.push_back(sg); cur = p;
    }
    std::reverse(sgs.begin(), sgs.end());
    Solution sol; sol.subgraphs = sgs; return sol;
}

} // namespace mlsys_solver
