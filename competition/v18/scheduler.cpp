#include "scheduler.h"
#include <algorithm>
#include <cmath>
#include <climits>
#include <numeric>
#include <set>
#include <vector>
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <chrono>
#include <tuple>
#include <bitset>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace mlsys_solver {

static inline int64_t ceildiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

static std::vector<int64_t> make_snake(int64_t cols, int64_t rows) {
    std::vector<int64_t> o; o.reserve(cols*rows);
    for (int64_t r=0;r<rows;++r)
        if(r&1)for(int64_t c=cols-1;c>=0;--c)o.push_back(r*cols+c);
        else for(int64_t c=0;c<cols;++c)o.push_back(r*cols+c);
    return o;
}
static std::vector<int64_t> make_col_snake(int64_t cols, int64_t rows) {
    std::vector<int64_t> o; o.reserve(cols*rows);
    for (int64_t c=0;c<cols;++c)
        if(c&1)for(int64_t r=rows-1;r>=0;--r)o.push_back(r*cols+c);
        else for(int64_t r=0;r<rows;++r)o.push_back(r*cols+c);
    return o;
}

// Compact bitset for tensor membership
static constexpr size_t MAX_TENSORS = 8192;
struct TensorSet {
    std::vector<uint64_t> bits;
    TensorSet() : bits((MAX_TENSORS + 63) / 64, 0) {}
    void insert(size_t t) { bits[t >> 6] |= (1ULL << (t & 63)); }
    bool count(size_t t) const { return (bits[t >> 6] >> (t & 63)) & 1; }
    void clear() { std::fill(bits.begin(), bits.end(), 0); }
    bool operator==(const TensorSet& o) const { return bits == o.bits; }
};

struct GraphInfo {
    TensorSet graph_outputs;
    TensorSet graph_inputs;
};

static GraphInfo build_graph_info(const Problem& p) {
    GraphInfo gi;
    TensorSet consumed, produced;
    for (auto& op : p.ops) {
        for (auto t : op.inputs) consumed.insert(t);
        for (auto t : op.outputs) produced.insert(t);
    }
    for (size_t i = 0; i < p.tensors.size(); ++i) {
        if (!consumed.count(i)) gi.graph_outputs.insert(i);
        if (!produced.count(i)) gi.graph_inputs.insert(i);
    }
    return gi;
}

// Precomputed data about a subgraph's tensors
struct SubgraphTensorInfo {
    TensorSet produced, consumed, all_t, eph;
    bool has_mm;
    int64_t K_dim;
    int64_t outW, outH;  // V18: independent max per dim, ALL outputs

    // Per-tensor info for non-ephemeral tensors
    struct TInfo {
        size_t tensor_id;
        bool lhs, rhs;
        bool pw_use;
        int64_t actual_K;
        bool is_consumed_not_produced;
        bool is_produced;
    };
    std::vector<TInfo> tinfos;

    // Precomputed base costs
    double pw_base;
    double mm_base;

    // V18: Evaluator-matching workspace categories (precomputed from ops)
    // These mirror the evaluator's matmul_lhs, matmul_rhs, pw_inputs, out_tiles dicts
    std::vector<size_t> ws_mm_lhs;   // non-eph MatMul LHS tensor IDs
    std::vector<size_t> ws_mm_rhs;   // non-eph MatMul RHS tensor IDs
    std::vector<size_t> ws_pw_in;    // non-eph Pointwise input tensor IDs
    std::vector<size_t> ws_out;      // ALL non-eph output tensor IDs
};

static SubgraphTensorInfo precompute_sg_tensors(
    const Problem& p, const std::vector<size_t>& ops,
    const std::vector<int>& last_use, int sg_last,
    const GraphInfo& gi)
{
    SubgraphTensorInfo si;
    si.has_mm = false;
    si.K_dim = 0;
    si.outW = 0;
    si.outH = 0;
    si.pw_base = 0;
    si.mm_base = 0;

    for (size_t oi : ops) {
        auto& op = p.ops[oi];
        if (op.op_type_enum == OpType::MatMul) {
            si.has_mm = true;
            si.K_dim = std::max(si.K_dim, p.tensors[op.inputs[0]].width);
            si.mm_base += (double)op.base_cost;
        } else {
            si.pw_base += (double)op.base_cost;
        }
        for (auto t : op.inputs) { si.consumed.insert(t); si.all_t.insert(t); }
        for (auto t : op.outputs) { si.produced.insert(t); si.all_t.insert(t); }
    }

    // Collect unique tensor IDs
    std::vector<size_t> all_t_vec;
    for (size_t oi : ops) {
        for (auto t : p.ops[oi].inputs) all_t_vec.push_back(t);
        for (auto t : p.ops[oi].outputs) all_t_vec.push_back(t);
    }
    std::sort(all_t_vec.begin(), all_t_vec.end());
    all_t_vec.erase(std::unique(all_t_vec.begin(), all_t_vec.end()), all_t_vec.end());

    // Compute ephemeral tensors
    for (size_t t : all_t_vec) {
        if (si.produced.count(t) && si.consumed.count(t) &&
            last_use[t] <= sg_last && !gi.graph_outputs.count(t))
            si.eph.insert(t);
    }

    // V18: Compute output dimensions from ALL outputs (including ephemeral),
    // using independent max per dimension (matching evaluator)
    for (size_t oi : ops) {
        for (auto t : p.ops[oi].outputs) {
            auto& T = p.tensors[t];
            si.outW = std::max(si.outW, T.width);
            si.outH = std::max(si.outH, T.height);
        }
    }

    // Build per-tensor info for non-ephemeral tensors (used for latency computation)
    for (size_t t : all_t_vec) {
        if (si.eph.count(t)) continue;
        bool ip = si.produced.count(t), ic = si.consumed.count(t);
        bool l = false, r = false, pw = false;
        int64_t actual_K = si.K_dim;
        for (size_t oi : ops) {
            auto& op = p.ops[oi];
            if (op.op_type_enum == OpType::MatMul && op.inputs.size() >= 2) {
                if (op.inputs[0] == t) { l = true; actual_K = p.tensors[t].width; }
                if (op.inputs[1] == t) { r = true; actual_K = p.tensors[t].height; }
            }
            if (op.op_type_enum == OpType::Pointwise) {
                for (auto ti : op.inputs) if (ti == t) pw = true;
                for (auto to : op.outputs) if (to == t) pw = true;
            }
        }
        si.tinfos.push_back({t, l, r, pw, actual_K, ic && !ip, ip});
    }

    // V18: Build evaluator-matching workspace categories
    // These exactly mirror the evaluator's logic for OOM checking
    std::set<size_t> _mm_lhs, _mm_rhs, _pw_in, _out;
    for (size_t oi : ops) {
        auto& op = p.ops[oi];
        if (op.op_type_enum == OpType::MatMul) {
            if (op.inputs.size() >= 2) {
                if (!si.eph.count(op.inputs[0])) _mm_lhs.insert(op.inputs[0]);
                if (!si.eph.count(op.inputs[1])) _mm_rhs.insert(op.inputs[1]);
            }
            for (auto t : op.outputs) if (!si.eph.count(t)) _out.insert(t);
        } else {
            for (auto t : op.inputs) if (!si.eph.count(t)) _pw_in.insert(t);
            for (auto t : op.outputs) if (!si.eph.count(t)) _out.insert(t);
        }
    }
    si.ws_mm_lhs.assign(_mm_lhs.begin(), _mm_lhs.end());
    si.ws_mm_rhs.assign(_mm_rhs.begin(), _mm_rhs.end());
    si.ws_pw_in.assign(_pw_in.begin(), _pw_in.end());
    si.ws_out.assign(_out.begin(), _out.end());

    return si;
}

struct EvalResult { double latency; bool valid; int snake; };

// V18: Rewritten to match evaluator's OOM and latency models exactly
static EvalResult eval_inner_precomp(
    const Problem& p,
    const SubgraphTensorInfo& si,
    int64_t gw, int64_t gh, int64_t gk,
    const TensorSet& resident, const TensorSet& retained,
    int smode,
    int64_t resident_full_size
) {
    int64_t nw = p.native_granularity.width, nh = p.native_granularity.height;
    double bw = std::max(1.0, (double)p.slow_memory_bandwidth);

    if (!si.outW) return {0, false, 0};

    int64_t cols = ceildiv(si.outW, gw), rows = ceildiv(si.outH, gh), tiles = cols * rows;
    int64_t ksteps = (si.has_mm && si.K_dim > 0 && gk > 0) ? ceildiv(si.K_dim, gk) : 1;

    // V18: OOM check matching evaluator exactly
    // joint_ws = sum of workspace for each category (matching evaluator's logic)
    int64_t joint_ws = 0;
    for (auto t : si.ws_mm_lhs)
        if (!resident.count(t)) joint_ws += p.tensors[t].width * gh;  // LHS: full K * tile height
    for (auto t : si.ws_mm_rhs)
        if (!resident.count(t)) joint_ws += gw * gk;                  // RHS: tile width * k-strip
    for (auto t : si.ws_pw_in)
        if (!resident.count(t)) joint_ws += gw * gh;                  // PW input: tile size
    for (auto t : si.ws_out)
        joint_ws += gw * gh;                                          // All non-eph outputs: tile size

    int64_t total_ws = joint_ws + resident_full_size;
    if (total_ws > p.fast_memory_capacity) return {0, false, 0};

    // V18: Also check retained tensors fit after this subgraph
    // (This is checked in the DP loop too, but early rejection helps)

    // Latency computation
    // V18: LHS always loaded as full K (once per spatial tile, not per k-step)
    // RHS loaded as k-strip every k-step
    // PW inputs loaded once per spatial tile
    double lhs_ld = 0, rhs_ld = 0, pw_ld = 0, st_c = 0;
    for (auto& ti : si.tinfos) {
        double load_cost;
        if (ti.lhs && !ti.pw_use) {
            load_cost = (double)(gh * ti.actual_K) / bw;  // LHS: full K * h
        } else if (ti.rhs && !ti.pw_use) {
            load_cost = (double)(gw * gk) / bw;           // RHS: w * k-strip
        } else {
            load_cost = (double)(gw * gh) / bw;           // PW / other: w * h
        }

        // Multi-role: if MatMul input AND also used as Pointwise
        // The evaluator would count it in both categories; we use the larger cost
        if (ti.pw_use && ti.lhs) {
            load_cost = std::max(load_cost, (double)(gw * gh) / bw);
        }
        if (ti.pw_use && ti.rhs) {
            load_cost = std::max(load_cost, (double)(gw * gh) / bw);
        }

        bool needs_load = ti.is_consumed_not_produced && !resident.count(ti.tensor_id);
        // V18: needs_store doesn't check resident — match evaluator
        bool needs_store = ti.is_produced && !retained.count(ti.tensor_id);

        if (needs_load) {
            if (ti.pw_use && (ti.lhs || ti.rhs)) pw_ld += load_cost;
            else if (ti.lhs) lhs_ld += load_cost;
            else if (ti.rhs) rhs_ld += load_cost;
            else pw_ld += load_cost;
        }
        if (needs_store) st_c += (double)(gw * gh) / bw;
    }

    // V18: Padding formula matching evaluator: ceil(gw/nw) * ceil(gh/nh)
    double pad = (double)(ceildiv(gw, nw) * ceildiv(gh, nh));

    double mm = si.mm_base / ksteps * pad;
    double pw = si.pw_base * pad;

    double total = 0;
    bool rs = (smode == 0);
    for (int64_t kk = 0; kk < ksteps; ++kk) {
        // V18: PW compute on last k-step (matching evaluator), not first
        double comp = mm;
        if (kk == ksteps - 1) comp += pw;

        // V18: LHS always loaded once per tile (k=0 only), never on mid k-steps
        double lc = (kk == 0) ? lhs_ld : 0;
        double rc = rhs_ld;
        double pwm = (kk == 0) ? pw_ld : 0;
        double stm = (kk == ksteps - 1) ? st_c : 0;

        if (si.has_mm && tiles > 1) {
            if (rs) {
                // Row-snake: LHS reused when same row, RHS reused at row transitions
                total += std::max(comp, lc + rc + pwm + stm);
                if (cols > 1) total += (cols - 1) * std::max(comp, rc + pwm + stm);
                for (int64_t r = 1; r < rows; ++r) {
                    total += std::max(comp, lc + pwm + stm);
                    if (cols > 1) total += (cols - 1) * std::max(comp, rc + pwm + stm);
                }
            } else {
                // Col-snake: RHS reused when same col, LHS reused at col transitions
                total += std::max(comp, lc + rc + pwm + stm);
                if (rows > 1) total += (rows - 1) * std::max(comp, lc + pwm + stm);
                for (int64_t c = 1; c < cols; ++c) {
                    total += std::max(comp, rc + pwm + stm);
                    if (rows > 1) total += (rows - 1) * std::max(comp, lc + pwm + stm);
                }
            }
        } else {
            total += tiles * std::max(comp, lc + rc + pwm + stm);
        }
    }
    return {total, true, smode};
}

static EvalResult eval_precomp(const Problem& p,
    const SubgraphTensorInfo& si,
    int64_t gw, int64_t gh, int64_t gk,
    const TensorSet& res, const TensorSet& ret,
    int64_t resident_full_size)
{
    auto r0 = eval_inner_precomp(p, si, gw, gh, gk, res, ret, 0, resident_full_size);
    auto r1 = eval_inner_precomp(p, si, gw, gh, gk, res, ret, 1, resident_full_size);
    if (r0.valid && r1.valid) return r0.latency <= r1.latency ? r0 : r1;
    return r0.valid ? r0 : r1;
}

struct Best { double lat; int64_t w, h, k; std::vector<int64_t> tr; bool valid; };

static Best find_best(const Problem& p, const std::vector<size_t>& ops,
    const TensorSet& res, const TensorSet& ret,
    const std::vector<int>& lu, const GraphInfo& gi)
{
    int64_t nw = p.native_granularity.width, nh = p.native_granularity.height;

    int sl = (int)ops.back();
    SubgraphTensorInfo si = precompute_sg_tensors(p, ops, lu, sl, gi);

    if (!si.outW) return {1e30, 0, 0, 0, {}, false};

    int64_t oW = si.outW, oH = si.outH;

    std::set<int64_t> all_dims;
    for (size_t oi : ops) {
        auto& op = p.ops[oi];
        for (auto t : op.inputs) { all_dims.insert(p.tensors[t].width); all_dims.insert(p.tensors[t].height); }
        for (auto t : op.outputs) { all_dims.insert(p.tensors[t].width); all_dims.insert(p.tensors[t].height); }
    }

    auto gen = [&](int64_t nat, int64_t mx) -> std::vector<int64_t> {
        std::set<int64_t> s;
        for (int64_t v = 16; v <= mx && v <= 8192; v *= 2) s.insert(v);
        for (int64_t v = nat; v >= 16; v /= 2) s.insert(v);
        for (int64_t m : {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16}) {
            int64_t v = m * nat;
            if (v >= 16 && v <= mx && v <= 8192) s.insert(v);
        }
        if (mx >= 16) s.insert(mx);
        if (mx / 2 >= 16) s.insert(mx / 2);
        if (mx / 3 >= 16) s.insert(mx / 3);
        if (mx / 4 >= 16) s.insert(mx / 4);
        if (mx / 6 >= 16) s.insert(mx / 6);
        if (mx / 8 >= 16) s.insert(mx / 8);
        for (int64_t d = nat; d <= mx; d += nat) {
            if (mx % d == 0) s.insert(d);
        }
        if (mx <= 8192) {
            for (int64_t d = 16; d * d <= mx; ++d) {
                if (mx % d == 0) {
                    s.insert(d);
                    int64_t q = mx / d;
                    if (q >= 16 && q <= 8192) s.insert(q);
                }
            }
        }
        for (int64_t dim : all_dims) {
            if (dim >= 16 && dim <= mx) {
                s.insert(dim);
                for (int64_t d = nat; d <= dim && d <= mx; d += nat) {
                    if (dim % d == 0 && mx % d == 0) s.insert(d);
                }
            }
            if (dim <= 8192 && dim >= 16) {
                for (int64_t d = 16; d * d <= dim; ++d) {
                    if (dim % d == 0) {
                        if (d <= mx && mx % d == 0) s.insert(d);
                        int64_t q = dim / d;
                        if (q >= 16 && q <= mx && mx % q == 0) s.insert(q);
                    }
                }
            }
        }
        return std::vector<int64_t>(s.begin(), s.end());
    };
    auto ws = gen(nw, oW), hs = gen(nh, oH);

    // V18: Add memory-aware h candidates for MatMul subgraphs
    if (si.has_mm && si.K_dim > 0 && !si.ws_mm_lhs.empty()) {
        int64_t lhs_k_per_h = 0;
        for (auto t : si.ws_mm_lhs)
            lhs_k_per_h += p.tensors[t].width;
        int64_t out_per_h = (int64_t)si.ws_out.size() * nw;
        int64_t pw_per_h = (int64_t)si.ws_pw_in.size() * nw;
        int64_t min_rhs = (int64_t)si.ws_mm_rhs.size() * nw;
        int64_t avail = p.fast_memory_capacity - min_rhs;
        int64_t cost_per_h = lhs_k_per_h + out_per_h + pw_per_h;
        if (avail > 0 && cost_per_h > 0) {
            int64_t max_h = std::min(avail / cost_per_h, oH);
            if (max_h >= 16 && max_h < nh) {
                if (max_h <= oH) hs.push_back(max_h);
                int64_t h75 = max_h * 3 / 4;
                if (h75 >= 16 && h75 <= oH) hs.push_back(h75);
                int64_t h16 = (max_h / 16) * 16;
                if (h16 >= 16 && h16 <= oH && h16 != max_h) hs.push_back(h16);
                std::sort(hs.begin(), hs.end());
                hs.erase(std::unique(hs.begin(), hs.end()), hs.end());
            }
        }
    }

    std::vector<int64_t> ks;
    if (si.has_mm && si.K_dim > 0) {
        std::set<int64_t> kset;
        int64_t K = si.K_dim;
        for (int64_t v = 16; v <= K; v *= 2) kset.insert(v);
        kset.insert(K);
        for (int64_t base : {nw, nh}) {
            for (int64_t m = 1; m * base <= K; ++m) {
                int64_t v = m * base;
                if (v >= 16 && K % v == 0) kset.insert(v);
            }
        }
        for (int64_t d = 16; d * d <= K; ++d) {
            if (K % d == 0) {
                kset.insert(d);
                if (K / d >= 16) kset.insert(K / d);
            }
        }
        for (size_t oi : ops) {
            auto& op = p.ops[oi];
            if (op.op_type_enum == OpType::MatMul && op.inputs.size() >= 2) {
                int64_t thisK = p.tensors[op.inputs[0]].width;
                for (int64_t v = 16; v <= thisK; v *= 2) kset.insert(v);
                kset.insert(thisK);
                for (int64_t d = 16; d * d <= thisK; ++d) {
                    if (thisK % d == 0) { kset.insert(d); if (thisK / d >= 16) kset.insert(thisK / d); }
                }
            }
        }
        // k=8 as safety net for tight OOM
        if (8 <= K) kset.insert(8);
        ks.assign(kset.begin(), kset.end());
    } else {
        ks = {1};
    }

    // Flatten all (w, h, k) combos
    std::vector<std::tuple<int64_t, int64_t, int64_t>> combos;
    combos.reserve(ws.size() * hs.size() * ks.size() * 2);
    for (int64_t sw : ws) for (int64_t sh : hs)
        for (int64_t kk : ks) combos.emplace_back(sw, sh, kk);
    // Transposed combos
    for (int64_t sw : hs) for (int64_t sh : ws)
        for (int64_t kk : ks) {
            if (std::find(ws.begin(), ws.end(), sw) != ws.end() &&
                std::find(hs.begin(), hs.end(), sh) != hs.end()) continue;
            combos.emplace_back(sw, sh, kk);
        }

    // Precompute total size of resident tensors
    int64_t resident_full_size = 0;
    for (size_t t = 0; t < p.tensors.size(); ++t) {
        if (res.count(t)) resident_full_size += p.tensors[t].size();
    }

    Best best; best.valid = false; best.lat = 1e30;
    int64_t combo_n = (int64_t)combos.size();

    #pragma omp parallel
    {
        Best local_best; local_best.valid = false; local_best.lat = 1e30;

        #pragma omp for schedule(dynamic, 4) nowait
        for (int64_t ci = 0; ci < combo_n; ++ci) {
            auto [sw, sh, sk] = combos[ci];
            auto ev = eval_precomp(p, si, sw, sh, sk, res, ret, resident_full_size);
            if (ev.valid && ev.latency < local_best.lat) {
                local_best.lat = ev.latency; local_best.w = sw; local_best.h = sh; local_best.k = sk; local_best.valid = true;
                int64_t cs = ceildiv(oW, sw), rs_ = ceildiv(oH, sh);
                local_best.tr = (cs * rs_ > 1) ? ((ev.snake == 0) ? make_snake(cs, rs_) : make_col_snake(cs, rs_)) : std::vector<int64_t>{};
            }
        }

        #pragma omp critical
        {
            if (local_best.valid && local_best.lat < best.lat) best = local_best;
        }
    }

    return best;
}

// Hash a retention set for deduplication
static uint64_t hash_tensor_set(const TensorSet& ts) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (auto w : ts.bits) {
        h ^= w;
        h *= 0x100000001b3ULL;
    }
    return h;
}

Solution Solve(const Problem& problem) {
    int N = problem.ops.size();
    GraphInfo gi = build_graph_info(problem);

    std::vector<int> fu(problem.tensors.size(), -1), lu(problem.tensors.size(), -1);
    for (int s = 0; s < N; ++s) {
        auto& op = problem.ops[s];
        for (auto t : op.inputs) { if (fu[t] == -1) fu[t] = s; lu[t] = s; }
        for (auto t : op.outputs) { if (fu[t] == -1) fu[t] = s; lu[t] = s; }
    }

#ifdef _OPENMP
    omp_set_max_active_levels(1);
#endif

    int W;
    int num_rmodes;
    if (N <= 32) {
        W = N;
        num_rmodes = 10;
    } else if (N <= 40) {
        W = std::min(20, N);
        num_rmodes = 6;
    } else if (N <= 65) {
        W = std::min(10, N);
        num_rmodes = 5;
    } else if (N <= 105) {
        W = std::min(8, N);
        num_rmodes = 4;
    } else {
        W = std::min(6, N);
        num_rmodes = 4;
    }

    std::vector<double> dp(N + 1, 1e30);
    std::vector<int> par(N + 1, -1);
    struct Info { int64_t w, h, k; std::vector<int64_t> tr; double lat; std::vector<size_t> ret; };
    std::vector<Info> info(N + 1);
    dp[0] = 0;

    for (int i = 1; i <= N; ++i) {
        for (int j = std::max(0, i - W); j < i; ++j) {
            if (dp[j] >= 1e29) continue;
            std::vector<size_t> ol; for (int k = j; k < i; ++k) ol.push_back(k);
            TensorSet res;
            if (j > 0 && par[j] >= 0) for (auto t : info[j].ret) res.insert(t);

            int sg_last_common = i - 1;

            // Pre-build common tensor set for all rmodes
            std::vector<size_t> sg_tensors_vec;
            {
                TensorSet seen;
                for (size_t oi : ol) {
                    for (auto t : problem.ops[oi].outputs) {
                        if (!seen.count(t)) { seen.insert(t); sg_tensors_vec.push_back(t); }
                    }
                    for (auto t : problem.ops[oi].inputs) {
                        if (!seen.count(t)) { seen.insert(t); sg_tensors_vec.push_back(t); }
                    }
                }
            }

            struct RModeResult {
                double tot;
                int64_t w, h, k;
                std::vector<int64_t> tr;
                double lat;
                std::vector<size_t> ret;
                bool valid;
            };
            std::vector<RModeResult> rmode_results(num_rmodes);
            for (int rm = 0; rm < num_rmodes; ++rm) rmode_results[rm].valid = false;

            std::vector<TensorSet> ret_sets(num_rmodes);
            std::vector<std::vector<size_t>> ret_lists(num_rmodes);

            for (int rmode = 0; rmode < num_rmodes; ++rmode) {
                std::vector<size_t>& rl = ret_lists[rmode];
                TensorSet& rs = ret_sets[rmode];

                if (rmode == 0) {
                    struct RC { size_t i; int64_t s; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common)
                        rc.push_back({t, problem.tensors[t].width * problem.tensors[t].height});
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.s > b.s; });
                    int64_t bud = problem.fast_memory_capacity;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 1) {
                    // Retain nothing
                }
                else if (rmode == 2) {
                    struct RC { size_t i; int64_t s; int nu; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        int next_use = -1;
                        for (int k2 = i; k2 < std::min(N, i + 8); ++k2) {
                            for (auto t2 : problem.ops[k2].inputs) if (t2 == t) { next_use = k2; break; }
                            if (next_use >= 0) break;
                        }
                        if (next_use >= 0)
                            rc.push_back({t, problem.tensors[t].width * problem.tensors[t].height, next_use});
                    }
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.nu < b.nu; });
                    int64_t bud = problem.fast_memory_capacity * 3 / 4;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 3) {
                    size_t best_t = 0; int64_t best_s = 0; bool found = false;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        int64_t s = problem.tensors[t].width * problem.tensors[t].height;
                        if (s > best_s) { best_s = s; best_t = t; found = true; }
                    }
                    if (found && best_s <= problem.fast_memory_capacity) {
                        rl.push_back(best_t); rs.insert(best_t);
                    }
                }
                else if (rmode == 4) {
                    size_t best_t = 0; int64_t best_s = INT64_MAX; bool found = false;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        int64_t s = problem.tensors[t].width * problem.tensors[t].height;
                        if (s < best_s && s > 0) { best_s = s; best_t = t; found = true; }
                    }
                    if (found && best_s <= problem.fast_memory_capacity) {
                        rl.push_back(best_t); rs.insert(best_t);
                    }
                }
                else if (rmode == 5) {
                    struct RC { size_t i; int64_t s; };
                    std::vector<RC> rc;
                    int64_t thresh = problem.fast_memory_capacity / 4;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        int64_t s = problem.tensors[t].width * problem.tensors[t].height;
                        if (s <= thresh) rc.push_back({t, s});
                    }
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.s < b.s; });
                    int64_t bud = problem.fast_memory_capacity * 3 / 4;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 6) {
                    struct RC { size_t i; int64_t s; int nu; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        int next_use = -1;
                        for (int k2 = i; k2 < std::min(N, i + 12); ++k2) {
                            for (auto t2 : problem.ops[k2].inputs) if (t2 == t) { next_use = k2; break; }
                            if (next_use >= 0) break;
                        }
                        if (next_use >= 0)
                            rc.push_back({t, problem.tensors[t].width * problem.tensors[t].height, next_use});
                    }
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.nu < b.nu; });
                    int64_t bud = problem.fast_memory_capacity;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 7) {
                    struct RC { size_t i; int64_t s; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common)
                        rc.push_back({t, problem.tensors[t].width * problem.tensors[t].height});
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.s < b.s; });
                    int64_t bud = problem.fast_memory_capacity;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 8) {
                    struct RC { size_t i; int64_t s; int nu; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        bool near = false;
                        for (int k2 = i; k2 < std::min(N, i + 8); ++k2) {
                            for (auto t2 : problem.ops[k2].inputs) if (t2 == t) { near = true; break; }
                            if (near) break;
                        }
                        if (near) {
                            int next_use = -1;
                            for (int k2 = i; k2 < std::min(N, i + 8); ++k2) {
                                for (auto t2 : problem.ops[k2].inputs) if (t2 == t) { next_use = k2; break; }
                                if (next_use >= 0) break;
                            }
                            rc.push_back({t, problem.tensors[t].width * problem.tensors[t].height, next_use});
                        }
                    }
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.nu < b.nu; });
                    int64_t bud = problem.fast_memory_capacity;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 9) {
                    struct RC { size_t i; int64_t s; int nu; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        int next_use = -1;
                        for (int k2 = i; k2 < std::min(N, i + 6); ++k2) {
                            for (auto t2 : problem.ops[k2].inputs) if (t2 == t) { next_use = k2; break; }
                            if (next_use >= 0) break;
                        }
                        if (next_use >= 0)
                            rc.push_back({t, problem.tensors[t].width * problem.tensors[t].height, next_use});
                    }
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.nu < b.nu; });
                    int64_t bud = problem.fast_memory_capacity / 2;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
            }

            // Dedup retention sets
            std::vector<uint64_t> ret_hashes(num_rmodes);
            std::vector<int> dedup_map(num_rmodes, -1);
            for (int rm = 0; rm < num_rmodes; ++rm) {
                ret_hashes[rm] = hash_tensor_set(ret_sets[rm]);
                dedup_map[rm] = rm;
                for (int prev = 0; prev < rm; ++prev) {
                    if (ret_hashes[prev] == ret_hashes[rm] && ret_sets[prev] == ret_sets[rm]) {
                        dedup_map[rm] = prev;
                        break;
                    }
                }
            }

            // Evaluate only unique retention sets
            for (int rmode = 0; rmode < num_rmodes; ++rmode) {
                if (dedup_map[rmode] != rmode) {
                    int src = dedup_map[rmode];
                    if (rmode_results[src].valid) {
                        rmode_results[rmode] = rmode_results[src];
                        rmode_results[rmode].ret = ret_lists[rmode];
                    }
                    continue;
                }

                auto br = find_best(problem, ol, res, ret_sets[rmode], lu, gi);
                if (!br.valid) continue;
                double tot = dp[j] + br.lat;
                rmode_results[rmode] = {tot, br.w, br.h, br.k, br.tr, br.lat, ret_lists[rmode], true};
            }

            // Sequential reduction with retained-size safety check
            for (int rmode = 0; rmode < num_rmodes; ++rmode) {
                if (!rmode_results[rmode].valid) continue;
                auto& r = rmode_results[rmode];
                int64_t ret_size = 0;
                for (auto t : r.ret) ret_size += problem.tensors[t].width * problem.tensors[t].height;
                if (ret_size > problem.fast_memory_capacity) continue;
                if (r.tot < dp[i]) {
                    dp[i] = r.tot; par[i] = j;
                    info[i] = {r.w, r.h, r.k, r.tr, r.lat, r.ret};
                }
            }
        }
    }

    std::vector<Subgraph> sgs; int cur = N;
    while (cur > 0) {
        int p_ = par[cur]; Subgraph sg;
        for (int k = p_; k < cur; ++k) sg.ops.push_back(k);
        sg.granularity = {info[cur].w, info[cur].h, info[cur].k};
        sg.traversal_order = info[cur].tr; sg.subgraph_latency = info[cur].lat;
        sg.tensors_to_retain = info[cur].ret; sgs.push_back(sg); cur = p_;
    }
    std::reverse(sgs.begin(), sgs.end());
    Solution sol; sol.subgraphs = sgs; return sol;
}

} // namespace mlsys_solver
