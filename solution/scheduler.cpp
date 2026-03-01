#include "scheduler.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <vector>
#include <cstdint>
#include <unordered_set>

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
};

static GraphInfo build_graph_info(const Problem& p) {
    GraphInfo gi;
    std::unordered_set<size_t> consumed;
    for (auto& op : p.ops)
        for (auto t : op.inputs) consumed.insert(t);
    for (size_t i = 0; i < p.tensors.size(); ++i)
        if (!consumed.count(i)) gi.graph_outputs.insert(i);
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

    // Working set
    int64_t ws_stream=0, ws_stat=0;
    struct TI{bool lhs,rhs; int64_t sk,sf; bool ld,st;};
    std::vector<TI> bt;
    for(size_t t:all_t){
        if(eph.count(t))continue;
        bool ip=produced.count(t),ic=consumed.count(t);
        bool l=false,r=false;
        for(size_t oi:ops){auto&op=p.ops[oi];
            if(op.op_type=="MatMul"&&op.inputs.size()>=2){
                if(op.inputs[0]==t)l=true;if(op.inputs[1]==t)r=true;}}
        int64_t sk=l?gh*gk:(r?gw*gk:gw*gh);
        int64_t sf=l?gh*K_dim:(r?K_dim*gw:gw*gh);
        ws_stream+=sk; ws_stat+=(l?sf:sk);
        bool needs_load = ic && !ip && !resident.count(t);
        bool needs_store = ip && !retained.count(t);
        bt.push_back({l,r,sk,sf,needs_load,needs_store});
    }

    bool use_stat=ksteps>1&&ws_stat<=p.fast_memory_capacity;
    if(!use_stat&&ws_stream>p.fast_memory_capacity)return{0,false,0};
    if(use_stat&&ws_stat>p.fast_memory_capacity)return{0,false,0};

    // Compute
    double pw=0,mm=0;
    for(size_t oi:ops){auto&op=p.ops[oi];
        if(op.op_type=="MatMul")mm+=(double)op.base_cost/ksteps;
        else pw+=op.base_cost;}
    double pad=(double)(std::max(gw,nw)*std::max(gh,nh))/(double)(gw*gh);
    pw*=pad;mm*=pad;

    // Memory costs
    double lhs_ld=0,rhs_ld=0,pw_ld=0,st_c=0;
    for(auto&ti:bt){
        double c=(double)(use_stat&&ti.lhs?ti.sf:ti.sk)/bw;
        if(ti.ld){if(ti.lhs)lhs_ld+=c;else if(ti.rhs)rhs_ld+=c;else pw_ld+=c;}
        if(ti.st)st_c+=(double)(gw*gh)/bw;
    }

    // Analytical latency
    double total=0;
    bool rs=(smode==0);
    for(int64_t kk=0;kk<ksteps;++kk){
        double comp=mm;if(kk==0)comp+=pw;
        double lc=use_stat?((kk==0)?lhs_ld:0):lhs_ld;
        double rc=rhs_ld, pwm=(kk==0)?pw_ld:0, stm=(kk==ksteps-1)?st_c:0;

        if(has_mm&&tiles>1){
            if(rs){
                total+=std::max(comp,lc+rc+pwm+stm);
                if(cols>1)total+=(cols-1)*std::max(comp,rc+pwm+stm);
                for(int64_t r=1;r<rows;++r){
                    total+=std::max(comp,lc+pwm+stm);
                    if(cols>1)total+=(cols-1)*std::max(comp,rc+pwm+stm);
                }
            }else{
                total+=std::max(comp,lc+rc+pwm+stm);
                if(rows>1)total+=(rows-1)*std::max(comp,lc+pwm+stm);
                for(int64_t c=1;c<cols;++c){
                    total+=std::max(comp,rc+pwm+stm);
                    if(rows>1)total+=(rows-1)*std::max(comp,lc+pwm+stm);
                }
            }
        }else{total+=tiles*std::max(comp,lc+rc+pwm+stm);}
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

    // Collect all tensor dims in the subgraph for smart candidate generation
    std::set<int64_t> tensor_dims;
    for(size_t oi:ops){auto&op=p.ops[oi];
        for(auto t:op.inputs){tensor_dims.insert(p.tensors[t].width);tensor_dims.insert(p.tensors[t].height);}
        for(auto t:op.outputs){tensor_dims.insert(p.tensors[t].width);tensor_dims.insert(p.tensors[t].height);}
    }

    // Generate granularity candidates: focused set
    auto gen=[&](int64_t nat, int64_t mx) -> std::vector<int64_t> {
        std::set<int64_t> s;
        // Powers of 2 from 16 up to mx
        for(int64_t v=16;v<=mx&&v<=8192;v*=2) s.insert(v);
        // Nat and its multiples (up to mx)
        for(int64_t m=1;m*nat<=mx&&m*nat<=8192;++m) s.insert(m*nat);
        // Divisors of output that are multiples of nat or >=16
        if(mx>0){
            // Only add divisors that are multiples of min(nat,16)
            int64_t step = std::min(nat, (int64_t)16);
            for(int64_t d=step;d<=mx;d+=step){
                if(mx%d==0) s.insert(d);
            }
        }
        // Also add divisors of each tensor dim that align well  
        for(int64_t td:tensor_dims){
            if(td<=0||td>8192) continue;
            for(int64_t d=nat;d<=td;d+=nat) {
                if(td%d==0 && d<=mx) s.insert(d);
            }
        }
        return std::vector<int64_t>(s.begin(),s.end());
    };
    auto ws=gen(nw,oW),hs=gen(nh,oH);

    // K candidates: focused set
    std::vector<int64_t> ks;
    if(has_mm&&K>0){
        std::set<int64_t> kset;
        for(int64_t v=K;v>=16;v/=2)kset.insert(v);
        // Also add multiples of native dims that divide K
        for(int64_t m=1;m*nw<=K;++m)if(K%(m*nw)==0)kset.insert(m*nw);
        for(int64_t m=1;m*nh<=K;++m)if(K%(m*nh)==0)kset.insert(m*nh);
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

    std::vector<int>fu(problem.tensors.size(),-1),lu(problem.tensors.size(),-1);
    for(int s=0;s<N;++s){auto&op=problem.ops[s];
        for(auto t:op.inputs){if(fu[t]==-1)fu[t]=s;lu[t]=s;}
        for(auto t:op.outputs){if(fu[t]==-1)fu[t]=s;lu[t]=s;}}

    std::vector<double>dp(N+1,1e30);
    std::vector<int>par(N+1,-1);
    struct Info{int64_t w,h,k;std::vector<int64_t>tr;double lat;std::vector<size_t>ret;};
    std::vector<Info>info(N+1);
    dp[0]=0;

    // Adaptive window: wider for reasonable sizes, cap to avoid explosion
    int W = std::min(16, N);

    for(int i=1;i<=N;++i){
        for(int j=std::max(0,i-W);j<i;++j){
            if(dp[j]>=1e29)continue;
            std::vector<size_t>ol;for(int k=j;k<i;++k)ol.push_back(k);
            std::set<size_t>res;
            if(j>0&&par[j]>=0)for(auto t:info[j].ret)res.insert(t);

            // 4 retention strategies
            for(int rmode=0;rmode<4;++rmode){
                std::vector<size_t>rl;std::set<size_t>rs;
                int sg_last=i-1;

                if(rmode==0){
                    // Retain all used-later, largest first
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
                    // Retain only tensors used in the NEXT few ops
                    std::set<size_t>sg_t;
                    for(size_t oi:ol){for(auto t:problem.ops[oi].outputs)sg_t.insert(t);
                                      for(auto t:problem.ops[oi].inputs)sg_t.insert(t);}
                    struct RC{size_t i;int64_t s;int nu;};
                    std::vector<RC>rc;
                    for(size_t t:sg_t)if(lu[t]>sg_last){
                        int next_use=-1;
                        for(int k2=i;k2<std::min(N,i+4);++k2){
                            for(auto t2:problem.ops[k2].inputs)if(t2==t){next_use=k2;break;}
                            if(next_use>=0)break;
                        }
                        if(next_use>=0)
                            rc.push_back({t,problem.tensors[t].width*problem.tensors[t].height,next_use});
                    }
                    std::sort(rc.begin(),rc.end(),[](const RC&a,const RC&b){return a.nu<b.nu;});
                    int64_t bud=problem.fast_memory_capacity;
                    for(auto&c:rc)if(bud>=c.s){rl.push_back(c.i);rs.insert(c.i);bud-=c.s;}
                }
                else if(rmode==3){
                    // Retain only the single largest reusable tensor
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
