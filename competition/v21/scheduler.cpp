// V20 improvements over V19:
// 1. Time-budget adaptive DP window: W_max (larger) used when fast, W_cons (v19 fallback) when slow
// 2. New retention modes 1 (value-based: size/distance), 3 (chain-aware: fan-out), 4 (lu-distance)
// 3. More retention modes for medium-N: 8 modes for N≤65 (was 5), 8 for N≤40 (was 6)
// 4. Memory-aware combo generation extended from N<32 to N<65
// 5. Precomputed total_fanout for chain-aware mode
// 6. Larger W_max for each size class (22 for N≤65, 16 for N≤105)

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
    int64_t outW, outH;

    struct TInfo {
        size_t tensor_id;
        bool lhs, rhs;
        bool pw_use;
        int64_t actual_K;
        bool is_consumed_not_produced;
        bool is_produced;
        bool skip_store;  // produced AND consumed in subgraph → evaluator treats as ephemeral for store
    };
    std::vector<TInfo> tinfos;

    double pw_base;
    double mm_base;

    // Evaluator-matching workspace categories
    std::vector<size_t> ws_mm_lhs;
    std::vector<size_t> ws_mm_rhs;
    std::vector<size_t> ws_pw_in;
    std::vector<size_t> ws_out;
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

    // Compute output dimensions from ALL outputs (including ephemeral)
    for (size_t oi : ops) {
        for (auto t : p.ops[oi].outputs) {
            auto& T = p.tensors[t];
            si.outW = std::max(si.outW, T.width);
            si.outH = std::max(si.outH, T.height);
        }
    }

    // Build per-tensor info for non-ephemeral tensors
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
        bool skip_st = ip && ic && !gi.graph_outputs.count(t);
        si.tinfos.push_back({t, l, r, pw, actual_K, ic && !ip, ip, skip_st});
    }

    // Build evaluator-matching workspace categories
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

    // OOM check matching evaluator
    int64_t joint_ws = 0;
    for (auto t : si.ws_mm_lhs)
        if (!resident.count(t)) joint_ws += p.tensors[t].width * gh;
    for (auto t : si.ws_mm_rhs)
        if (!resident.count(t)) joint_ws += gw * gk;
    for (auto t : si.ws_pw_in)
        if (!resident.count(t)) joint_ws += gw * gh;
    for (auto t : si.ws_out)
        joint_ws += gw * gh;

    int64_t total_ws = joint_ws + resident_full_size;
    if (total_ws > p.fast_memory_capacity) return {0, false, 0};

    // Latency computation
    double lhs_ld = 0, rhs_ld = 0, pw_ld = 0, st_c = 0;
    for (auto& ti : si.tinfos) {
        double load_cost;
        if (ti.lhs && !ti.pw_use) {
            load_cost = (double)(gh * ti.actual_K) / bw;
        } else if (ti.rhs && !ti.pw_use) {
            load_cost = (double)(gw * gk) / bw;
        } else {
            load_cost = (double)(gw * gh) / bw;
        }

        if (ti.pw_use && ti.lhs) {
            load_cost = std::max(load_cost, (double)(gw * gh) / bw);
        }
        if (ti.pw_use && ti.rhs) {
            load_cost = std::max(load_cost, (double)(gw * gh) / bw);
        }

        bool needs_load = ti.is_consumed_not_produced && !resident.count(ti.tensor_id);
        bool needs_store = ti.is_produced && !retained.count(ti.tensor_id) && !ti.skip_store;

        if (needs_load) {
            if (ti.pw_use && (ti.lhs || ti.rhs)) pw_ld += load_cost;
            else if (ti.lhs) lhs_ld += load_cost;
            else if (ti.rhs) rhs_ld += load_cost;
            else pw_ld += load_cost;
        }
        if (needs_store) st_c += (double)(gw * gh) / bw;
    }

    double pad = (double)(ceildiv(gw, nw) * ceildiv(gh, nh));

    double mm = si.mm_base / ksteps * pad;
    double pw = si.pw_base * pad;

    double total = 0;
    bool rs = (smode == 0);
    for (int64_t kk = 0; kk < ksteps; ++kk) {
        double comp = mm;
        if (kk == ksteps - 1) comp += pw;

        double lc = (kk == 0) ? lhs_ld : 0;
        double rc = rhs_ld;
        double pwm = (kk == 0) ? pw_ld : 0;
        double stm = (kk == ksteps - 1) ? st_c : 0;

        if (si.has_mm && tiles > 1) {
            if (rs) {
                // Row-snake: LHS reused same row, RHS reused at row transitions
                total += std::max(comp, lc + rc + pwm + stm);
                if (cols > 1) total += (cols - 1) * std::max(comp, rc + pwm + stm);
                for (int64_t r = 1; r < rows; ++r) {
                    total += std::max(comp, lc + pwm + stm);
                    if (cols > 1) total += (cols - 1) * std::max(comp, rc + pwm + stm);
                }
            } else {
                // Col-snake: RHS reused same col, LHS reused at col transitions
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

    // Compute resident_full_size early (needed for memory-aware combos)
    int64_t resident_full_size = 0;
    for (size_t t = 0; t < p.tensors.size(); ++t) {
        if (res.count(t)) resident_full_size += p.tensors[t].size();
    }

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

    // Memory-aware h candidates (nw proxy — for backward compat)
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
        // k=4 and k=8 as safety nets for tight OOM
        if (4 <= K) kset.insert(4);
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

    // V20: Memory-aware combos extended to N<65 (was N<32)
    if (si.has_mm && si.K_dim > 0 && !si.ws_mm_lhs.empty()) {
        std::set<int64_t> target_ws;
        if (oW >= 16) target_ws.insert(oW);
        if (oH >= 16) target_ws.insert(oH);
        if (nw >= 16) target_ws.insert(nw);
        if (nh >= 16) target_ws.insert(nh);
        // Fraction combos for smaller problems — extended from N<32 to N<65
        if ((int)p.ops.size() < 65) {
            for (int64_t d : {(int64_t)2, (int64_t)3, (int64_t)4, (int64_t)6, (int64_t)8}) {
                if (oW / d >= 16) target_ws.insert(oW / d);
                if (oH / d >= 16) target_ws.insert(oH / d);
            }
        }

        for (int64_t sw : target_ws) {
            int64_t out_cost_per_h = (int64_t)si.ws_out.size() * sw;
            int64_t pw_cost_per_h = (int64_t)si.ws_pw_in.size() * sw;

            int64_t adj_lhs_K = 0;
            for (auto t : si.ws_mm_lhs)
                if (!res.count(t)) adj_lhs_K += p.tensors[t].width;
            int64_t adj_cost_per_h = adj_lhs_K + out_cost_per_h + pw_cost_per_h;
            if (adj_cost_per_h <= 0) continue;

            for (int64_t sk : ks) {
                int64_t rhs_fixed = 0;
                for (auto t : si.ws_mm_rhs)
                    if (!res.count(t)) rhs_fixed += sw * sk;

                int64_t avail = p.fast_memory_capacity - rhs_fixed - resident_full_size;
                if (avail <= 0) continue;

                int64_t max_h = std::min(avail / adj_cost_per_h, oH);
                if (max_h < 1) continue;

                // Add max_h and nearby values
                for (int64_t dh = 0; dh <= 2; ++dh) {
                    int64_t h_try = max_h - dh;
                    if (h_try >= 1) combos.emplace_back(sw, h_try, sk);
                }
                // Native-aligned
                int64_t aligned_h = (max_h / nh) * nh;
                if (aligned_h >= 1 && aligned_h != max_h)
                    combos.emplace_back(sw, aligned_h, sk);
                // 75% of max_h
                int64_t h75 = max_h * 3 / 4;
                if (h75 >= 1) combos.emplace_back(sw, h75, sk);
            }
        }
    }

    // Deduplicate combos
    std::sort(combos.begin(), combos.end());
    combos.erase(std::unique(combos.begin(), combos.end()), combos.end());

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

    // V20: Precompute total fanout for chain-aware retention mode
    std::vector<int> total_fanout(problem.tensors.size(), 0);
    for (int s = 0; s < N; ++s) {
        auto& op = problem.ops[s];
        TensorSet counted_op;
        for (auto t : op.inputs) {
            if (!counted_op.count(t)) { total_fanout[t]++; counted_op.insert(t); }
        }
    }

    // V21: producer_of[t] = op index that produces tensor t (-1 if graph input)
    std::vector<int> producer_of(problem.tensors.size(), -1);
    for (int s = 0; s < N; ++s)
        for (auto t : problem.ops[s].outputs)
            producer_of[t] = s;

#ifdef _OPENMP
    omp_set_max_active_levels(1);
#endif

    // V20: Timer for adaptive window selection
    auto t0 = std::chrono::steady_clock::now();
    auto elapsed_ms = [&]() -> int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
    };

    // V20: Adaptive window parameters
    // W_max / nm_max: wide window + more modes used when running fast (8-core)
    // W_cons / nm_cons: conservative window + fewer modes (= v19 settings) as fallback for 1-core
    // Step-level adaptive: after each DP step, check if remaining steps at current pace would
    // exceed the time budget. If so, permanently switch to conservative settings.
    // This is more robust than progress-extrapolation because early steps are cheap (small j-range)
    // and don't reflect the true per-step cost of later steps with full window width.
    int W_max, W_cons, nm_max, nm_cons;
    int64_t budget_ms;
    if (N <= 32) {
        W_max = N; W_cons = N; budget_ms = 0; nm_max = 10; nm_cons = 10;
    } else if (N <= 40) {
        W_max = std::min(24, N); W_cons = 20; budget_ms = 4000; nm_max = 8; nm_cons = 6;
    } else if (N <= 65) {
        // Key improvement: W_max=20 (was 12 in v19), nm_max=8 (was 5).
        // On 8-core (fast): uses W_max + nm_max throughout → ~11% gain on b13.
        // On 1-core (slow): step-level timer detects slowness and falls back to
        //   W_cons=12, nm_cons=5 (identical work budget to v19) → no timeout.
        W_max = std::min(20, N); W_cons = 12; budget_ms = 22000; nm_max = 8; nm_cons = 5;
    } else if (N <= 105) {
        W_max = std::min(16, N); W_cons = 10; budget_ms = 50000; nm_max = 6; nm_cons = 5;
    } else {
        W_max = std::min(9, N); W_cons = 7; budget_ms = 50000; nm_max = 4; nm_cons = 4;
    }

    std::vector<double> dp(N + 1, 1e30);
    std::vector<int> par(N + 1, -1);
    // V21: recomputed = ops prepended from before j (recomputed within this subgraph)
    struct Info { int64_t w, h, k; std::vector<int64_t> tr; double lat; std::vector<size_t> ret; std::vector<size_t> recomputed; };
    std::vector<Info> info(N + 1);
    dp[0] = 0;

    // V20: Step-level adaptive state
    int W_cur = (N <= 32) ? N : W_max;
    int num_rmodes = (N <= 32) ? 10 : nm_max;
    bool gone_conservative = (N <= 32);  // small problems never need fallback

    for (int i = 1; i <= N; ++i) {
        int64_t step_start_ms = elapsed_ms();

        for (int j = std::max(0, i - W_cur); j < i; ++j) {
            if (dp[j] >= 1e29) continue;
            std::vector<size_t> base_ol; for (int k = j; k < i; ++k) base_ol.push_back(k);
            TensorSet res;
            if (j > 0 && par[j] >= 0) for (auto t : info[j].ret) res.insert(t);

            int sg_last_common = i - 1;

            // V21: Find candidate recomputed ops (guard: small subgraphs only to prevent runaway fusions).
            // base_ol and extended_ol are evaluated separately; cheaper result wins → no regressions.
            std::vector<size_t> recomputed_ops_cand;
            if ((int)base_ol.size() <= 4) {
                TensorSet produced_in_base;
                for (size_t oi : base_ol) for (auto t : problem.ops[oi].outputs) produced_in_base.insert(t);
                TensorSet already_added;
                for (size_t oi : base_ol) {
                    for (auto t : problem.ops[oi].inputs) {
                        if (produced_in_base.count(t)) continue;
                        if (res.count(t)) continue;
                        if (gi.graph_inputs.count(t)) continue;
                        int P = producer_of[t];
                        if (P < 0 || P >= j) continue;
                        if (already_added.count((size_t)P)) continue;
                        auto& pop = problem.ops[(size_t)P];
                        if (pop.op_type_enum != OpType::Pointwise) continue;
                        if (lu[t] > sg_last_common) continue;
                        if (gi.graph_outputs.count(t)) continue;
                        int64_t Tsize = problem.tensors[t].size();
                        if (pop.base_cost * problem.slow_memory_bandwidth < Tsize / 2) {
                            recomputed_ops_cand.push_back((size_t)P);
                            already_added.insert((size_t)P);
                        }
                    }
                }
                if (!recomputed_ops_cand.empty()) {
                    std::sort(recomputed_ops_cand.begin(), recomputed_ops_cand.end());
                    recomputed_ops_cand.erase(std::unique(recomputed_ops_cand.begin(), recomputed_ops_cand.end()), recomputed_ops_cand.end());
                }
            }

            struct EvalBest { bool valid; double tot; int64_t w, h, k; std::vector<int64_t> tr; double lat; std::vector<size_t> ret; };
            auto eval_ol_fn = [&](const std::vector<size_t>& ol) -> EvalBest {

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
                    // Greedy by size (largest first), full budget
                    struct RC { size_t i; int64_t s; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common)
                        rc.push_back({t, problem.tensors[t].size()});
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.s > b.s; });
                    int64_t bud = problem.fast_memory_capacity;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 1) {
                    // V20: Value-based retention: score = size / distance_to_first_next_use
                    // Large tensors needed soon get highest priority.
                    // Uses lu[t] as a fast upper bound, then scans forward for first actual use.
                    struct RC { size_t i; int64_t s; double score; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        int64_t s = problem.tensors[t].size();
                        // Find first use of t after the current subgraph end (i-1)
                        int next_use = lu[t];
                        for (int k2 = i; k2 <= lu[t] && k2 < N; ++k2) {
                            bool found = false;
                            for (auto t2 : problem.ops[k2].inputs) {
                                if (t2 == t) { next_use = k2; found = true; break; }
                            }
                            if (found) break;
                        }
                        int dist = std::max(1, next_use - sg_last_common);
                        rc.push_back({t, s, (double)s / dist});
                    }
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.score > b.score; });
                    int64_t bud = problem.fast_memory_capacity;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 2) {
                    // Nearest use, 3/4 budget
                    struct RC { size_t i; int64_t s; int nu; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        int next_use = -1;
                        for (int k2 = i; k2 < std::min(N, i + 8); ++k2) {
                            for (auto t2 : problem.ops[k2].inputs) if (t2 == t) { next_use = k2; break; }
                            if (next_use >= 0) break;
                        }
                        if (next_use >= 0)
                            rc.push_back({t, problem.tensors[t].size(), next_use});
                    }
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.nu < b.nu; });
                    int64_t bud = problem.fast_memory_capacity * 3 / 4;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 3) {
                    // V20: Chain-aware retention: prefer tensors used by multiple future ops (skip connections).
                    // Uses precomputed total_fanout — tensors with high fan-out are likely skip/residual tensors.
                    struct RC { size_t i; int64_t s; int fanout; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        rc.push_back({t, problem.tensors[t].size(), total_fanout[t]});
                    }
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) {
                        if (a.fanout != b.fanout) return a.fanout > b.fanout;
                        return a.s > b.s;
                    });
                    int64_t bud = problem.fast_memory_capacity;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 4) {
                    // V20: lu-distance retention: sort by last_use ascending (soonest-last-used first).
                    // Retains tensors that expire soonest — they need to be in fast memory for the
                    // next few subgraphs and won't be needed long after.
                    struct RC { size_t i; int64_t s; int last_use; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        rc.push_back({t, problem.tensors[t].size(), lu[t]});
                    }
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) {
                        return a.last_use < b.last_use;
                    });
                    int64_t bud = problem.fast_memory_capacity;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 5) {
                    // Small threshold (1/4 capacity), smallest first
                    struct RC { size_t i; int64_t s; };
                    std::vector<RC> rc;
                    int64_t thresh = problem.fast_memory_capacity / 4;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        int64_t s = problem.tensors[t].size();
                        if (s <= thresh) rc.push_back({t, s});
                    }
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.s < b.s; });
                    int64_t bud = problem.fast_memory_capacity * 3 / 4;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 6) {
                    // Nearest use, full budget, wider lookahead
                    struct RC { size_t i; int64_t s; int nu; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        int next_use = -1;
                        for (int k2 = i; k2 < std::min(N, i + 12); ++k2) {
                            for (auto t2 : problem.ops[k2].inputs) if (t2 == t) { next_use = k2; break; }
                            if (next_use >= 0) break;
                        }
                        if (next_use >= 0)
                            rc.push_back({t, problem.tensors[t].size(), next_use});
                    }
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.nu < b.nu; });
                    int64_t bud = problem.fast_memory_capacity;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 7) {
                    // Greedy smallest first, full budget
                    struct RC { size_t i; int64_t s; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common)
                        rc.push_back({t, problem.tensors[t].size()});
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.s < b.s; });
                    int64_t bud = problem.fast_memory_capacity;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 8) {
                    // Near tensors only (next 8 ops), nearest first, full budget
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
                            rc.push_back({t, problem.tensors[t].size(), next_use});
                        }
                    }
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.nu < b.nu; });
                    int64_t bud = problem.fast_memory_capacity;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 9) {
                    // Nearest use, half budget, narrow lookahead
                    struct RC { size_t i; int64_t s; int nu; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        int next_use = -1;
                        for (int k2 = i; k2 < std::min(N, i + 6); ++k2) {
                            for (auto t2 : problem.ops[k2].inputs) if (t2 == t) { next_use = k2; break; }
                            if (next_use >= 0) break;
                        }
                        if (next_use >= 0)
                            rc.push_back({t, problem.tensors[t].size(), next_use});
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

            // Return the best result across all rmodes
            EvalBest best{false, 1e30, 0, 0, 0, {}, 0.0, {}};
            for (int rmode = 0; rmode < num_rmodes; ++rmode) {
                if (!rmode_results[rmode].valid) continue;
                auto& r = rmode_results[rmode];
                int64_t ret_size = 0;
                for (auto t : r.ret) ret_size += problem.tensors[t].size();
                if (ret_size > problem.fast_memory_capacity) continue;
                if (r.tot < best.tot) {
                    best = {true, r.tot, r.w, r.h, r.k, r.tr, r.lat, r.ret};
                }
            }
            return best;
            }; // end eval_ol_fn

            // Evaluate base_ol; also extended_ol if recompute candidates exist — keep the cheaper result
            EvalBest base_res = eval_ol_fn(base_ol);
            EvalBest ext_res{false, 1e30, 0, 0, 0, {}, 0.0, {}};
            if (!recomputed_ops_cand.empty()) {
                std::vector<size_t> ext_ol = recomputed_ops_cand;
                ext_ol.insert(ext_ol.end(), base_ol.begin(), base_ol.end());
                ext_res = eval_ol_fn(ext_ol);
            }

            std::vector<size_t> best_recomputed;
            EvalBest* best_res_ptr = nullptr;
            if (base_res.valid && (!ext_res.valid || base_res.tot <= ext_res.tot)) {
                best_res_ptr = &base_res;
            } else if (ext_res.valid) {
                best_res_ptr = &ext_res;
                best_recomputed = recomputed_ops_cand;
            }
            if (best_res_ptr && best_res_ptr->tot < dp[i]) {
                dp[i] = best_res_ptr->tot; par[i] = j;
                info[i] = {best_res_ptr->w, best_res_ptr->h, best_res_ptr->k, best_res_ptr->tr, best_res_ptr->lat, best_res_ptr->ret, best_recomputed};
            }
        }

        // V20: Step-level adaptive — after each outer i step, check if remaining steps
        // at the current pace would exceed the time budget. If so, permanently switch
        // to conservative settings (W_cons + nm_cons = same as v19).
        // This correctly handles the case where early steps are cheap (small j-range)
        // but late steps are expensive (full W window) — progress extrapolation misses this.
        if (!gone_conservative) {
            int64_t step_ms = elapsed_ms() - step_start_ms;
            int64_t remaining_steps = (int64_t)(N - i);
            int64_t elapsed_so_far = elapsed_ms();
            int64_t budget_left = budget_ms - elapsed_so_far;
            // If this step's cost * remaining steps would blow the budget, fall back
            if (remaining_steps > 0 && step_ms * remaining_steps > budget_left) {
                W_cur = W_cons;
                num_rmodes = nm_cons;
                gone_conservative = true;
            }
        }
    }

    std::vector<Subgraph> sgs; int cur = N;
    while (cur > 0) {
        int p_ = par[cur]; Subgraph sg;
        // V21: prepend recomputed ops (indices < p_) in topological order, then original ops [p_..cur-1]
        for (size_t P : info[cur].recomputed) sg.ops.push_back(P);
        for (int k = p_; k < cur; ++k) sg.ops.push_back(k);
        sg.granularity = {info[cur].w, info[cur].h, info[cur].k};
        sg.traversal_order = info[cur].tr; sg.subgraph_latency = info[cur].lat;
        sg.tensors_to_retain = info[cur].ret; sgs.push_back(sg); cur = p_;
    }
    std::reverse(sgs.begin(), sgs.end());
    Solution sol; sol.subgraphs = sgs; return sol;
}

} // namespace mlsys_solver
