#include "scheduler.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <vector>
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <map>

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

struct GraphInfo {
    std::unordered_set<size_t> graph_outputs;
    std::unordered_set<size_t> graph_inputs;
};

static GraphInfo build_graph_info(const Problem& p) {
    GraphInfo gi;
    std::unordered_set<size_t> consumed, produced;
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

struct EvalResult { double latency; bool valid; int snake; };

static EvalResult eval_inner(
    const Problem& p, const std::vector<size_t>& ops,
    int64_t gw, int64_t gh, int64_t gk,
    const std::set<size_t>& resident, const std::set<size_t>& retained,
    const std::vector<int>& last_use, int sg_last,
    const GraphInfo& gi, int smode
) {
    int64_t nw=p.native_granularity.width, nh=p.native_granularity.height;
    double bw=std::max(1.0,(double)p.slow_memory_bandwidth);

    std::set<size_t> produced,consumed,all_t;
    bool has_mm=false; int64_t K_dim=0;
    for(size_t oi:ops){
        auto&op=p.ops[oi];
        if(op.op_type=="MatMul"){
            has_mm=true;
            K_dim=std::max(K_dim,p.tensors[op.inputs[0]].width);
        }
        for(auto t:op.inputs){consumed.insert(t);all_t.insert(t);}
        for(auto t:op.outputs){produced.insert(t);all_t.insert(t);}
    }

    // Ephemeral: produced AND consumed in subgraph, last_use within subgraph, NOT a graph output
    std::set<size_t> eph;
    for(size_t t:produced) {
        if(consumed.count(t) && last_use[t]<=sg_last && !gi.graph_outputs.count(t))
            eph.insert(t);
    }

    int64_t outW=0,outH=0;
    for(size_t oi:ops)for(auto t:p.ops[oi].outputs){
        if(eph.count(t)) continue;
        auto&T=p.tensors[t];if(T.width*T.height>outW*outH){outW=T.width;outH=T.height;}
    }
    if(!outW){
        for(size_t oi:ops)for(auto t:p.ops[oi].outputs){
            auto&T=p.tensors[t];if(T.width*T.height>outW*outH){outW=T.width;outH=T.height;}
        }
    }
    if(!outW) return{0,false,0};

    int64_t cols=ceildiv(outW,gw),rows=ceildiv(outH,gh),tiles=cols*rows;
    int64_t ksteps=(has_mm&&K_dim>0&&gk>0)?ceildiv(K_dim,gk):1;

    // Working set calculation with PER-TENSOR K dimensions
    int64_t ws_stream=0, ws_stat=0;
    struct TI{bool lhs,rhs; int64_t sk,sf; bool ld,st; int64_t actual_K;};
    std::vector<TI> bt;
    for(size_t t:all_t){
        if(eph.count(t))continue;
        bool ip=produced.count(t),ic=consumed.count(t);
        bool l=false,r=false;
        int64_t actual_K = K_dim;
        for(size_t oi:ops){auto&op=p.ops[oi];
            if(op.op_type=="MatMul"&&op.inputs.size()>=2){
                if(op.inputs[0]==t){l=true; actual_K=p.tensors[t].width;}
                if(op.inputs[1]==t){r=true; actual_K=p.tensors[t].height;}
            }
        }

        int64_t sk=l?gh*gk:(r?gw*gk:gw*gh);
        int64_t sf=l?gh*actual_K:(r?actual_K*gw:gw*gh);

        // Resident tensors are already in fast memory — don't add their WS
        if(!resident.count(t)) {
            ws_stream+=sk;
            ws_stat+=sf;
        }

        bool needs_load = ic && !ip && !resident.count(t);
        // Retained tensors stay in fast memory for next subgraph — no store needed
        bool needs_store = ip && !retained.count(t) && !resident.count(t);

        bt.push_back({l,r,sk,sf,needs_load,needs_store,actual_K});
    }

    bool use_stat=ksteps>1&&ws_stat<=p.fast_memory_capacity;
    if(!use_stat&&ws_stream>p.fast_memory_capacity)return{0,false,0};
    if(use_stat&&ws_stat>p.fast_memory_capacity)return{0,false,0};

    // Compute cost with padding
    double pw=0,mm=0;
    for(size_t oi:ops){auto&op=p.ops[oi];
        if(op.op_type=="MatMul")mm+=(double)op.base_cost/ksteps;
        else pw+=op.base_cost;}
    double pad=(double)(std::max(gw,nw)*std::max(gh,nh))/(double)(gw*gh);
    pw*=pad;mm*=pad;

    // Memory costs per tile
    double lhs_ld=0,rhs_ld=0,pw_ld=0,st_c=0;
    for(auto&ti:bt){
        double c;
        if(use_stat && ti.lhs) {
            // In stationary mode, LHS is loaded once (full actual_K dimension)
            c = (double)ti.sf / bw;
        } else {
            c = (double)ti.sk / bw;
        }
        if(ti.ld){if(ti.lhs)lhs_ld+=c;else if(ti.rhs)rhs_ld+=c;else pw_ld+=c;}
        if(ti.st)st_c+=(double)(gw*gh)/bw;
    }

    // Analytical latency with snake traversal modeling
    double total=0;
    bool rs=(smode==0); // row-snake vs col-snake
    for(int64_t kk=0;kk<ksteps;++kk){
        double comp=mm;if(kk==0)comp+=pw;
        double lc=use_stat?((kk==0)?lhs_ld:0):lhs_ld;
        double rc=rhs_ld, pwm=(kk==0)?pw_ld:0, stm=(kk==ksteps-1)?st_c:0;

        if(has_mm&&tiles>1){
            if(rs){
                // Row-snake: reuse LHS across columns, reload RHS at row change
                total+=std::max(comp,lc+rc+pwm+stm);
                if(cols>1)total+=(cols-1)*std::max(comp,rc+pwm+stm);
                for(int64_t r=1;r<rows;++r){
                    total+=std::max(comp,lc+pwm+stm);
                    if(cols>1)total+=(cols-1)*std::max(comp,rc+pwm+stm);
                }
            }else{
                // Col-snake: reuse RHS across rows, reload LHS at col change
                total+=std::max(comp,lc+rc+pwm+stm);
                if(rows>1)total+=(rows-1)*std::max(comp,lc+pwm+stm);
                for(int64_t c=1;c<cols;++c){
                    total+=std::max(comp,rc+pwm+stm);
                    if(rows>1)total+=(rows-1)*std::max(comp,lc+pwm+stm);
                }
            }
        }else{
            total+=tiles*std::max(comp,lc+rc+pwm+stm);
        }
    }
    return{total,true,smode};
}

static EvalResult eval(const Problem&p,const std::vector<size_t>&ops,
    int64_t gw,int64_t gh,int64_t gk,const std::set<size_t>&res,
    const std::set<size_t>&ret,const std::vector<int>&lu,int sl,
    const GraphInfo& gi){
    auto r0=eval_inner(p,ops,gw,gh,gk,res,ret,lu,sl,gi,0);
    auto r1=eval_inner(p,ops,gw,gh,gk,res,ret,lu,sl,gi,1);
    if(r0.valid&&r1.valid)return r0.latency<=r1.latency?r0:r1;
    return r0.valid?r0:r1;
}

struct Best{double lat;int64_t w,h,k;std::vector<int64_t>tr;bool valid;};

static Best find_best(const Problem&p,const std::vector<size_t>&ops,
    const std::set<size_t>&res,const std::set<size_t>&ret,
    const std::vector<int>&lu,const GraphInfo& gi){
    int64_t nw=p.native_granularity.width,nh=p.native_granularity.height;
    bool has_mm=false;int64_t K=0,oW=0,oH=0;int sl=(int)ops.back();
    for(size_t oi:ops){auto&op=p.ops[oi];
        if(op.op_type=="MatMul"){has_mm=true;K=std::max(K,p.tensors[op.inputs[0]].width);}
        for(auto t:op.outputs){auto&T=p.tensors[t];if(T.width*T.height>oW*oH){oW=T.width;oH=T.height;}}
    }

    // Collect ALL tensor dimensions in the subgraph for smarter candidate generation
    std::set<int64_t> all_dims;
    for(size_t oi:ops){auto&op=p.ops[oi];
        for(auto t:op.inputs){all_dims.insert(p.tensors[t].width);all_dims.insert(p.tensors[t].height);}
        for(auto t:op.outputs){all_dims.insert(p.tensors[t].width);all_dims.insert(p.tensors[t].height);}
    }

    // Generate dense granularity candidates
    auto gen=[&](int64_t nat, int64_t mx) -> std::vector<int64_t> {
        std::set<int64_t> s;

        // Powers of 2 from 16 up to max
        for(int64_t v=16;v<=mx&&v<=8192;v*=2) s.insert(v);

        // Powers of 2 from nat down to 16
        for(int64_t v=nat;v>=16;v/=2) s.insert(v);

        // Key multiples of nat up to max
        for(int64_t m : {1,2,3,4,6,8,16}) {
            int64_t v=m*nat;
            if(v>=16&&v<=mx&&v<=8192) s.insert(v);
        }

        // The output dimension itself and fractions
        if(mx>=16) s.insert(mx);
        if(mx/2>=16) s.insert(mx/2);
        if(mx/4>=16) s.insert(mx/4);
        if(mx/8>=16) s.insert(mx/8);

        // Divisors of mx that are multiples of nat
        for(int64_t d=nat;d<=mx;d+=nat){
            if(mx%d==0) s.insert(d);
        }

        // Divisors of mx >= 16 (thorough search for reasonable sizes)
        if(mx <= 8192) {
            for(int64_t d=16;d*d<=mx;++d){
                if(mx%d==0){
                    s.insert(d);
                    int64_t q=mx/d;
                    if(q>=16&&q<=8192) s.insert(q);
                }
            }
        }

        // Also add divisors of OTHER tensor dimensions in the subgraph
        // This helps when the output dims don't capture all useful tile sizes
        for(int64_t dim : all_dims) {
            if(dim >= 16 && dim <= mx) {
                s.insert(dim);
                // And divisors of this dim that also divide mx
                for(int64_t d=nat;d<=dim&&d<=mx;d+=nat) {
                    if(dim%d==0 && mx%d==0) s.insert(d);
                }
            }
        }

        return std::vector<int64_t>(s.begin(),s.end());
    };
    auto ws=gen(nw,oW),hs=gen(nh,oH);

    // K candidates: denser set
    std::vector<int64_t> ks;
    if(has_mm&&K>0){
        std::set<int64_t> kset;
        // Powers of 2 from 16 up to K
        for(int64_t v=16;v<=K;v*=2) kset.insert(v);
        // K itself
        kset.insert(K);
        // Key multiples of native dims that divide K
        for(int64_t base : {nw, nh}) {
            for(int64_t m=1;m*base<=K;++m) {
                int64_t v=m*base;
                if(v>=16&&K%v==0) kset.insert(v);
            }
        }
        // Divisors of K >= 16
        for(int64_t d=16;d*d<=K;++d){
            if(K%d==0){
                kset.insert(d);
                if(K/d>=16) kset.insert(K/d);
            }
        }
        // Also add divisors of ALL K dimensions (not just the max)
        for(size_t oi:ops){auto&op=p.ops[oi];
            if(op.op_type=="MatMul"&&op.inputs.size()>=2){
                int64_t thisK = p.tensors[op.inputs[0]].width;
                for(int64_t v=16;v<=thisK;v*=2) kset.insert(v);
                kset.insert(thisK);
                for(int64_t d=16;d*d<=thisK;++d){
                    if(thisK%d==0){kset.insert(d);if(thisK/d>=16) kset.insert(thisK/d);}
                }
            }
        }
        ks.assign(kset.begin(),kset.end());
    }else{
        ks={1};
    }

    Best best;best.valid=false;best.lat=1e30;
    auto tr=[&](int64_t sw,int64_t sh,int64_t sk){
        auto ev=eval(p,ops,sw,sh,sk,res,ret,lu,sl,gi);
        if(ev.valid&&ev.latency<best.lat){
            best.lat=ev.latency;best.w=sw;best.h=sh;best.k=sk;best.valid=true;
            int64_t cs=ceildiv(oW,sw),rs_=ceildiv(oH,sh);
            best.tr=(cs*rs_>1)?((ev.snake==0)?make_snake(cs,rs_):make_col_snake(cs,rs_)):std::vector<int64_t>{};
        }
    };

    for(int64_t sw:ws)for(int64_t sh:hs)
        for(int64_t kk:ks)tr(sw,sh,kk);
    // Asymmetric swap
    for(int64_t sw:hs)for(int64_t sh:ws)
        for(int64_t kk:ks){
            if(std::find(ws.begin(),ws.end(),sw)!=ws.end()&&
               std::find(hs.begin(),hs.end(),sh)!=hs.end())continue;
            tr(sw,sh,kk);
        }

    return best;
}

Solution Solve(const Problem&problem){
    int N=problem.ops.size();
    GraphInfo gi = build_graph_info(problem);

    // Build first-use / last-use maps
    std::vector<int>fu(problem.tensors.size(),-1),lu(problem.tensors.size(),-1);
    for(int s=0;s<N;++s){auto&op=problem.ops[s];
        for(auto t:op.inputs){if(fu[t]==-1)fu[t]=s;lu[t]=s;}
        for(auto t:op.outputs){if(fu[t]==-1)fu[t]=s;lu[t]=s;}}

    std::vector<double>dp(N+1,1e30);
    std::vector<int>par(N+1,-1);
    struct Info{int64_t w,h,k;std::vector<int64_t>tr;double lat;std::vector<size_t>ret;};
    std::vector<Info>info(N+1);
    dp[0]=0;

    // Wider adaptive DP window — must cover full transformer blocks (4 ops)
    int W;
    if(N<=20) W=std::min(20,N);
    else if(N<=40) W=std::min(16,N);
    else if(N<=80) W=std::min(12,N);
    else W=std::min(10,N);

    // Number of retention modes
    int num_rmodes = (N<=50) ? 9 : 3;

    for(int i=1;i<=N;++i){
        for(int j=std::max(0,i-W);j<i;++j){
            if(dp[j]>=1e29)continue;
            std::vector<size_t>ol;for(int k=j;k<i;++k)ol.push_back(k);
            std::set<size_t>res;
            if(j>0&&par[j]>=0)for(auto t:info[j].ret)res.insert(t);

            // Multiple retention strategies
            for(int rmode=0;rmode<num_rmodes;++rmode){
                std::vector<size_t>rl;std::set<size_t>rs;
                int sg_last=i-1;

                if(rmode==0){
                    // Retain all used-later tensors, largest first
                    std::set<size_t>sg_t;
                    for(size_t oi:ol){for(auto t:problem.ops[oi].outputs)sg_t.insert(t);
                                      for(auto t:problem.ops[oi].inputs)sg_t.insert(t);}
                    struct RC{size_t i;int64_t s;};
                    std::vector<RC>rc;
                    for(size_t t:sg_t)if(lu[t]>sg_last)
                        rc.push_back({t,problem.tensors[t].width*problem.tensors[t].height});
                    std::sort(rc.begin(),rc.end(),[](const RC&a,const RC&b){return a.s>b.s;});
                    int64_t bud=problem.fast_memory_capacity;
                    for(auto&c:rc)if(bud>=c.s){rl.push_back(c.i);rs.insert(c.i);bud-=c.s;}
                }
                else if(rmode==1){
                    // Retain nothing
                }
                else if(rmode==2){
                    // Retain only tensors used in the NEXT few ops (forward-looking)
                    std::set<size_t>sg_t;
                    for(size_t oi:ol){for(auto t:problem.ops[oi].outputs)sg_t.insert(t);
                                      for(auto t:problem.ops[oi].inputs)sg_t.insert(t);}
                    struct RC{size_t i;int64_t s;int nu;};
                    std::vector<RC>rc;
                    for(size_t t:sg_t)if(lu[t]>sg_last){
                        int next_use=-1;
                        for(int k2=i;k2<std::min(N,i+8);++k2){
                            for(auto t2:problem.ops[k2].inputs)if(t2==t){next_use=k2;break;}
                            if(next_use>=0)break;
                        }
                        if(next_use>=0)
                            rc.push_back({t,problem.tensors[t].width*problem.tensors[t].height,next_use});
                    }
                    std::sort(rc.begin(),rc.end(),[](const RC&a,const RC&b){return a.nu<b.nu;});
                    int64_t bud=problem.fast_memory_capacity*3/4;
                    for(auto&c:rc)if(bud>=c.s){rl.push_back(c.i);rs.insert(c.i);bud-=c.s;}
                }
                else if(rmode==3){
                    // Retain the single largest reusable tensor
                    std::set<size_t>sg_t;
                    for(size_t oi:ol){for(auto t:problem.ops[oi].outputs)sg_t.insert(t);
                                      for(auto t:problem.ops[oi].inputs)sg_t.insert(t);}
                    size_t best_t=0;int64_t best_s=0;bool found=false;
                    for(size_t t:sg_t)if(lu[t]>sg_last){
                        int64_t s=problem.tensors[t].width*problem.tensors[t].height;
                        if(s>best_s){best_s=s;best_t=t;found=true;}
                    }
                    if(found&&best_s<=problem.fast_memory_capacity){
                        rl.push_back(best_t);rs.insert(best_t);
                    }
                }
                else if(rmode==4){
                    // Retain the smallest reusable tensor
                    std::set<size_t>sg_t;
                    for(size_t oi:ol){for(auto t:problem.ops[oi].outputs)sg_t.insert(t);
                                      for(auto t:problem.ops[oi].inputs)sg_t.insert(t);}
                    size_t best_t=0;int64_t best_s=INT64_MAX;bool found=false;
                    for(size_t t:sg_t)if(lu[t]>sg_last){
                        int64_t s=problem.tensors[t].width*problem.tensors[t].height;
                        if(s<best_s&&s>0){best_s=s;best_t=t;found=true;}
                    }
                    if(found&&best_s<=problem.fast_memory_capacity){
                        rl.push_back(best_t);rs.insert(best_t);
                    }
                }
                else if(rmode==5){
                    // Retain ALL small reusable tensors (threshold: 25% of fast memory)
                    std::set<size_t>sg_t;
                    for(size_t oi:ol){for(auto t:problem.ops[oi].outputs)sg_t.insert(t);
                                      for(auto t:problem.ops[oi].inputs)sg_t.insert(t);}
                    struct RC{size_t i;int64_t s;};
                    std::vector<RC>rc;
                    int64_t thresh=problem.fast_memory_capacity/4;
                    for(size_t t:sg_t)if(lu[t]>sg_last){
                        int64_t s=problem.tensors[t].width*problem.tensors[t].height;
                        if(s<=thresh) rc.push_back({t,s});
                    }
                    std::sort(rc.begin(),rc.end(),[](const RC&a,const RC&b){return a.s<b.s;});
                    int64_t bud=problem.fast_memory_capacity*3/4;
                    for(auto&c:rc)if(bud>=c.s){rl.push_back(c.i);rs.insert(c.i);bud-=c.s;}
                }
                else if(rmode==6){
                    // Retain all reusable tensors with generous budget
                    std::set<size_t>sg_t;
                    for(size_t oi:ol){for(auto t:problem.ops[oi].outputs)sg_t.insert(t);
                                      for(auto t:problem.ops[oi].inputs)sg_t.insert(t);}
                    struct RC{size_t i;int64_t s;int nu;};
                    std::vector<RC>rc;
                    for(size_t t:sg_t)if(lu[t]>sg_last){
                        int next_use=-1;
                        for(int k2=i;k2<std::min(N,i+12);++k2){
                            for(auto t2:problem.ops[k2].inputs)if(t2==t){next_use=k2;break;}
                            if(next_use>=0)break;
                        }
                        if(next_use>=0)
                            rc.push_back({t,problem.tensors[t].width*problem.tensors[t].height,next_use});
                    }
                    // Sort by soonest use first
                    std::sort(rc.begin(),rc.end(),[](const RC&a,const RC&b){return a.nu<b.nu;});
                    int64_t bud=problem.fast_memory_capacity;
                    for(auto&c:rc)if(bud>=c.s){rl.push_back(c.i);rs.insert(c.i);bud-=c.s;}
                }
                else if(rmode==7){
                    // BUDGET-FREE: Retain ALL used-later tensors unconditionally.
                    // The WS check in eval_inner handles validity, since retention
                    // only needs per-tile strip memory, not the full tensor.
                    std::set<size_t>sg_t;
                    for(size_t oi:ol){for(auto t:problem.ops[oi].outputs)sg_t.insert(t);
                                      for(auto t:problem.ops[oi].inputs)sg_t.insert(t);}
                    for(size_t t:sg_t)if(lu[t]>sg_last){
                        rl.push_back(t);rs.insert(t);
                    }
                }
                else if(rmode==8){
                    // BUDGET-FREE: Retain only tensors used in the next few ops
                    std::set<size_t>sg_t;
                    for(size_t oi:ol){for(auto t:problem.ops[oi].outputs)sg_t.insert(t);
                                      for(auto t:problem.ops[oi].inputs)sg_t.insert(t);}
                    for(size_t t:sg_t)if(lu[t]>sg_last){
                        bool near=false;
                        for(int k2=i;k2<std::min(N,i+8);++k2){
                            for(auto t2:problem.ops[k2].inputs)if(t2==t){near=true;break;}
                            if(near)break;
                        }
                        if(near){rl.push_back(t);rs.insert(t);}
                    }
                }

                auto br=find_best(problem,ol,res,rs,lu,gi);
                if(!br.valid)continue;
                double tot=dp[j]+br.lat;
                if(tot<dp[i]){
                    dp[i]=tot;par[i]=j;
                    info[i]={br.w,br.h,br.k,br.tr,br.lat,rl};
                }
            }
        }
    }

    std::vector<Subgraph>sgs;int cur=N;
    while(cur>0){int p_=par[cur];Subgraph sg;
        for(int k=p_;k<cur;++k)sg.ops.push_back(k);
        sg.granularity={info[cur].w,info[cur].h,info[cur].k};
        sg.traversal_order=info[cur].tr;sg.subgraph_latency=info[cur].lat;
        sg.tensors_to_retain=info[cur].ret;sgs.push_back(sg);cur=p_;}
    std::reverse(sgs.begin(),sgs.end());
    Solution sol;sol.subgraphs=sgs;return sol;
}

} // namespace mlsys_solver
