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

// Compact bitset for tensor membership — O(1) lookup, cache-friendly
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
    };
    std::vector<TInfo> tinfos;

    double pw_base;
    double mm_base;
};

static SubgraphTensorInfo precompute_sg_tensors(
    const Problem& p, const std::vector<size_t>& ops,
    const std::vector<int>& last_use, int sg_last,
    const GraphInfo& gi)
{
    SubgraphTensorInfo si;
    si.has_mm = false; si.K_dim = 0; si.outW = 0; si.outH = 0;
    si.pw_base = 0; si.mm_base = 0;

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

    std::vector<size_t> all_t_vec;
    for (size_t oi : ops) {
        for (auto t : p.ops[oi].inputs) all_t_vec.push_back(t);
        for (auto t : p.ops[oi].outputs) all_t_vec.push_back(t);
    }
    std::sort(all_t_vec.begin(), all_t_vec.end());
    all_t_vec.erase(std::unique(all_t_vec.begin(), all_t_vec.end()), all_t_vec.end());

    for (size_t t : all_t_vec) {
        if (si.produced.count(t) && si.consumed.count(t) &&
            last_use[t] <= sg_last && !gi.graph_outputs.count(t))
            si.eph.insert(t);
    }

    for (size_t oi : ops) {
        for (auto t : p.ops[oi].outputs) {
            if (si.eph.count(t)) continue;
            auto& T = p.tensors[t];
            if (T.width * T.height > si.outW * si.outH) {
                si.outW = T.width; si.outH = T.height;
            }
        }
    }
    if (!si.outW) {
        for (size_t oi : ops) {
            for (auto t : p.ops[oi].outputs) {
                auto& T = p.tensors[t];
                if (T.width * T.height > si.outW * si.outH) {
                    si.outW = T.width; si.outH = T.height;
                }
            }
        }
    }

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

    // Pass 1: workspace sizes for feasibility
    int64_t ws_stream = 0, ws_stat = 0;
    for (auto& ti : si.tinfos) {
        int64_t sk = ti.lhs ? gh * gk : (ti.rhs ? gw * gk : gw * gh);
        int64_t sf = ti.lhs ? gh * ti.actual_K : (ti.rhs ? ti.actual_K * gw : gw * gh);
        if (ti.pw_use && (ti.lhs || ti.rhs)) {
            sk = std::max(sk, gw * gh);
            sf = std::max(sf, gw * gh);
        }
        if (!resident.count(ti.tensor_id)) {
            ws_stream += sk;
            ws_stat += sf;
        }
    }

    // OOM check: account for resident tensors
    int64_t avail = p.fast_memory_capacity - resident_full_size;
    if (avail <= 0) return {0, false, 0};
    bool use_stat = ksteps > 1 && ws_stat <= avail;
    if (!use_stat && ws_stream > avail) return {0, false, 0};
    if (use_stat && ws_stat > avail) return {0, false, 0};

    // Pass 2: memory transfer costs
    double lhs_ld = 0, rhs_ld = 0, pw_ld = 0, st_c = 0;
    for (auto& ti : si.tinfos) {
        int64_t sk = ti.lhs ? gh * gk : (ti.rhs ? gw * gk : gw * gh);
        int64_t sf = ti.lhs ? gh * ti.actual_K : (ti.rhs ? ti.actual_K * gw : gw * gh);
        if (ti.pw_use && (ti.lhs || ti.rhs)) {
            sk = std::max(sk, gw * gh);
            sf = std::max(sf, gw * gh);
        }

        bool needs_load = ti.is_consumed_not_produced && !resident.count(ti.tensor_id);
        bool needs_store = ti.is_produced && !retained.count(ti.tensor_id) && !resident.count(ti.tensor_id);

        double c;
        if (use_stat && ti.lhs && !ti.pw_use) {
            c = (double)sf / bw;
        } else {
            c = (double)sk / bw;
        }
        if (needs_load) {
            if (ti.pw_use && (ti.lhs || ti.rhs)) pw_ld += c;
            else if (ti.lhs) lhs_ld += c;
            else if (ti.rhs) rhs_ld += c;
            else pw_ld += c;
        }
        if (needs_store) st_c += (double)(gw * gh) / bw;
    }

    // Compute costs
    double mm = si.mm_base / ksteps;
    double pw = si.pw_base;
    double pad = (double)(std::max(gw, nw) * std::max(gh, nh)) / (double)(gw * gh);
    pw *= pad; mm *= pad;

    double total = 0;
    bool rs = (smode == 0);
    for (int64_t kk = 0; kk < ksteps; ++kk) {
        double comp = mm; if (kk == 0) comp += pw;
        double lc = use_stat ? ((kk == 0) ? lhs_ld : 0) : lhs_ld;
        double rc = rhs_ld, pwm = (kk == 0) ? pw_ld : 0, stm = (kk == ksteps - 1) ? st_c : 0;

        if (si.has_mm && tiles > 1) {
            if (rs) {
                total += std::max(comp, lc + rc + pwm + stm);
                if (cols > 1) total += (cols - 1) * std::max(comp, rc + pwm + stm);
                for (int64_t r = 1; r < rows; ++r) {
                    total += std::max(comp, lc + pwm + stm);
                    if (cols > 1) total += (cols - 1) * std::max(comp, rc + pwm + stm);
                }
            } else {
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

    // V17: Extended granularity generation
    // min_g adapts: small subgraphs use finer granularity, large ones stay at 16
    int64_t min_g = (ops.size() <= 10) ? 8 : 16;
    auto gen = [&](int64_t nat, int64_t mx) -> std::vector<int64_t> {
        std::set<int64_t> s;
        // Powers of 2
        for (int64_t v = min_g; v <= mx && v <= 8192; v *= 2) s.insert(v);
        for (int64_t v = nat; v >= min_g; v /= 2) s.insert(v);
        // Multiples of native
        for (int64_t m : {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32}) {
            int64_t v = m * nat;
            if (v >= min_g && v <= mx && v <= 8192) s.insert(v);
        }
        // Fractions of output dim
        if (mx >= min_g) s.insert(mx);
        for (int d : {2, 3, 4, 5, 6, 8, 10, 12, 16}) {
            int64_t v = mx / d;
            if (v >= min_g) s.insert(v);
        }
        // Divisors of mx that are multiples of native
        for (int64_t d = nat; d <= mx; d += nat) {
            if (mx % d == 0) s.insert(d);
        }
        // Full factorization of mx
        if (mx <= 8192) {
            for (int64_t d = min_g; d * d <= mx; ++d) {
                if (mx % d == 0) {
                    s.insert(d);
                    int64_t q = mx / d;
                    if (q >= min_g && q <= 8192) s.insert(q);
                }
            }
        }
        // Cross-dimensional divisors
        for (int64_t dim : all_dims) {
            if (dim >= min_g && dim <= mx) {
                s.insert(dim);
                for (int64_t d = nat; d <= dim && d <= mx; d += nat) {
                    if (dim % d == 0 && mx % d == 0) s.insert(d);
                }
            }
            if (dim <= 8192 && dim >= min_g) {
                for (int64_t d = min_g; d * d <= dim; ++d) {
                    if (dim % d == 0) {
                        if (d <= mx && mx % d == 0) s.insert(d);
                        int64_t q = dim / d;
                        if (q >= min_g && q <= mx && mx % q == 0) s.insert(q);
                    }
                }
            }
        }
        return std::vector<int64_t>(s.begin(), s.end());
    };
    auto ws = gen(nw, oW), hs = gen(nh, oH);

    std::vector<int64_t> ks;
    if (si.has_mm && si.K_dim > 0) {
        std::set<int64_t> kset;
        int64_t K = si.K_dim;
        for (int64_t v = 8; v <= K; v *= 2) kset.insert(v);
        kset.insert(K);
        for (int64_t base : {nw, nh}) {
            for (int64_t m = 1; m * base <= K; ++m) {
                int64_t v = m * base;
                if (v >= 8 && K % v == 0) kset.insert(v);
            }
        }
        for (int64_t d = 8; d * d <= K; ++d) {
            if (K % d == 0) {
                kset.insert(d);
                if (K / d >= 8) kset.insert(K / d);
            }
        }
        for (size_t oi : ops) {
            auto& op = p.ops[oi];
            if (op.op_type_enum == OpType::MatMul && op.inputs.size() >= 2) {
                int64_t thisK = p.tensors[op.inputs[0]].width;
                for (int64_t v = 8; v <= thisK; v *= 2) kset.insert(v);
                kset.insert(thisK);
                for (int64_t d = 8; d * d <= thisK; ++d) {
                    if (thisK % d == 0) { kset.insert(d); if (thisK / d >= 8) kset.insert(thisK / d); }
                }
            }
        }
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

    // Precompute resident tensor full size
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

// FNV-1a hash for deduplication
static uint64_t hash_tensor_set(const TensorSet& ts) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (auto w : ts.bits) { h ^= w; h *= 0x100000001b3ULL; }
    return h;
}

// V17: Single DP pass with given window size
struct DPInfo { int64_t w, h, k; std::vector<int64_t> tr; double lat; std::vector<size_t> ret; };

static double solve_with_window(const Problem& problem, int W, int num_rmodes,
    const GraphInfo& gi, const std::vector<int>& lu, int N,
    std::vector<double>& dp, std::vector<int>& par, std::vector<DPInfo>& info)
{
    dp.assign(N + 1, 1e30);
    par.assign(N + 1, -1);
    info.resize(N + 1);
    dp[0] = 0;

    for (int i = 1; i <= N; ++i) {
        for (int j = std::max(0, i - W); j < i; ++j) {
            if (dp[j] >= 1e29) continue;
            std::vector<size_t> ol; for (int k = j; k < i; ++k) ol.push_back(k);
            TensorSet res;
            if (j > 0 && par[j] >= 0) for (auto t : info[j].ret) res.insert(t);

            int sg_last_common = i - 1;

            // Build tensor set for retention modes
            std::vector<size_t> sg_tensors_vec;
            {
                TensorSet seen;
                for (size_t oi : ol) {
                    for (auto t : problem.ops[oi].outputs)
                        if (!seen.count(t)) { seen.insert(t); sg_tensors_vec.push_back(t); }
                    for (auto t : problem.ops[oi].inputs)
                        if (!seen.count(t)) { seen.insert(t); sg_tensors_vec.push_back(t); }
                }
            }

            struct RModeResult {
                double tot; int64_t w, h, k;
                std::vector<int64_t> tr; double lat;
                std::vector<size_t> ret; bool valid;
            };
            std::vector<RModeResult> rmode_results(num_rmodes);
            for (int rm = 0; rm < num_rmodes; ++rm) rmode_results[rm].valid = false;

            std::vector<TensorSet> ret_sets(num_rmodes);
            std::vector<std::vector<size_t>> ret_lists(num_rmodes);

            for (int rmode = 0; rmode < num_rmodes; ++rmode) {
                std::vector<size_t>& rl = ret_lists[rmode];
                TensorSet& rs = ret_sets[rmode];

                if (rmode == 0) {
                    // Retain all used-later, largest first
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
                    // Near-future (8 ops), 3/4 capacity
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
                    // Retain single largest
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
                    // Retain single smallest
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
                    // Small tensors only, 3/4 budget
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
                    // Near-future (12 ops), full capacity
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
                    // All future-use, smallest first
                    struct RC { size_t i; int64_t s; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common)
                        rc.push_back({t, problem.tensors[t].width * problem.tensors[t].height});
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.s < b.s; });
                    int64_t bud = problem.fast_memory_capacity;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 8) {
                    // Near-future (8 ops) with capacity check
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
                    // Near-future (6 ops), half capacity
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
                else if (rmode == 10) {
                    // V17: Retain all future-use, largest first, half capacity
                    // (V10-style conservative retention)
                    struct RC { size_t i; int64_t s; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common)
                        rc.push_back({t, problem.tensors[t].width * problem.tensors[t].height});
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.s > b.s; });
                    int64_t bud = problem.fast_memory_capacity / 2;
                    for (auto& c : rc) if (bud >= c.s) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; }
                }
                else if (rmode == 11) {
                    // V17: Near-future (4 ops), full capacity (V10-like proximity)
                    struct RC { size_t i; int64_t s; int nu; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        int next_use = -1;
                        for (int k2 = i; k2 < std::min(N, i + 4); ++k2) {
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
                else if (rmode == 12) {
                    // V17: Retain two largest reusable tensors
                    struct RC { size_t i; int64_t s; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common)
                        rc.push_back({t, problem.tensors[t].width * problem.tensors[t].height});
                    std::sort(rc.begin(), rc.end(), [](const RC& a, const RC& b) { return a.s > b.s; });
                    int64_t bud = problem.fast_memory_capacity;
                    int cnt = 0;
                    for (auto& c : rc) if (bud >= c.s && cnt < 2) { rl.push_back(c.i); rs.insert(c.i); bud -= c.s; ++cnt; }
                }
                else if (rmode == 13) {
                    // V17: Near-future (16 ops), 3/4 capacity — very long lookahead
                    struct RC { size_t i; int64_t s; int nu; };
                    std::vector<RC> rc;
                    for (size_t t : sg_tensors_vec) if (lu[t] > sg_last_common) {
                        int next_use = -1;
                        for (int k2 = i; k2 < std::min(N, i + 16); ++k2) {
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

    return dp[N];
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

    // V17: Multi-pass DP — try multiple window sizes, keep the best result
    // This captures V10's advantage on b17 (narrow window) while still
    // getting V16's advantage on b13/b9 (wide window)
    int num_rmodes = 14;

    // Generate window sizes to try
    std::vector<int> windows;
    if (N <= 32) {
        windows = {N};
    } else if (N <= 40) {
        windows = {N, std::min(20, N), std::min(12, N)};
    } else if (N <= 65) {
        windows = {std::min(32, N), std::min(20, N), std::min(12, N), std::min(10, N)};
    } else if (N <= 105) {
        windows = {std::min(20, N), std::min(14, N), std::min(10, N), std::min(8, N)};
    } else {
        windows = {std::min(16, N), std::min(12, N), std::min(10, N), std::min(8, N)};
    }

    // Deduplicate window sizes
    std::sort(windows.begin(), windows.end());
    windows.erase(std::unique(windows.begin(), windows.end()), windows.end());

    double best_total = 1e30;
    std::vector<double> best_dp;
    std::vector<int> best_par;
    std::vector<DPInfo> best_info;

    for (int W : windows) {
        std::vector<double> dp;
        std::vector<int> par;
        std::vector<DPInfo> info;

        double total = solve_with_window(problem, W, num_rmodes, gi, lu, N, dp, par, info);

        if (total < best_total) {
            best_total = total;
            best_dp = std::move(dp);
            best_par = std::move(par);
            best_info = std::move(info);
        }
    }

    // Reconstruct solution from best DP result
    std::vector<Subgraph> sgs; int cur = N;
    while (cur > 0) {
        int p_ = best_par[cur]; Subgraph sg;
        for (int k = p_; k < cur; ++k) sg.ops.push_back(k);
        sg.granularity = {best_info[cur].w, best_info[cur].h, best_info[cur].k};
        sg.traversal_order = best_info[cur].tr;
        sg.subgraph_latency = best_info[cur].lat;
        sg.tensors_to_retain = best_info[cur].ret;
        sgs.push_back(sg); cur = p_;
    }
    std::reverse(sgs.begin(), sgs.end());
    Solution sol; sol.subgraphs = sgs; return sol;
}

} // namespace mlsys_solver
