#!/usr/bin/env python3
"""Check deferred scratch source invariants and compile actual graph-control methods.

Generation-time checks are static. Executed C++ uses fake graph/stream boundaries;
no kernel, GPU runtime, model, filesystem or driver is invoked by the fixture.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
static_checks = 0


def check(ok, why):
    global static_checks
    static_checks += 1
    if not ok:
        raise AssertionError(why)


def function(text, signature):
    check(text.count(signature) == 1, "unique production method: " + signature)
    start = text.index(signature)
    brace = text.index("{", start)
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                    lambda m: " " * len(m[0]), text)
    depth = 0
    for end in range(brace, len(text)):
        depth += (masked[end] == "{") - (masked[end] == "}")
        if depth == 0:
            return text[start:end + 1]
    raise AssertionError("unclosed method: " + signature)


source = (ROOT / "src/core/verify.cpp").read_text()
header = (ROOT / "include/strata/core/verify.hpp").read_text()
generate = (ROOT / "src/program/generate.cpp").read_text()
capture = function(source, "bool Verifier::capture_deferred(")
launch = function(source, "bool Verifier::batch_launch_rows(")
commit = function(source, "bool Verifier::batch_commit_rows(")
evict = function(source, "bool Verifier::evict_batch_graph(")
stage = function(source, "bool Verifier::stage_batch(")
init = function(source, "bool Verifier::init_deferred_commit(")
check("nG * R * C, nG * R * C, nG * R * HV, nG * R * HV, nQ * R * ID, R * HS" in init,
      "saved GDN/QSA/PLE sizes scale with total handoff rows")
for dst, src, stride in [("dc_qkv_", "qkv_L_", "C"), ("dc_h_", "h_L_", "C"),
                         ("dc_gate_", "gate_L_", "HV"), ("dc_beta_", "beta_L_", "HV")]:
    check(f"copy({dst} + (size_t) (gdn_index * R + hbase) * {stride}, {src} + (size_t) gdn_index * MT * {stride}, S * {stride});" in capture,
          "per-group saved GDN scratch: " + dst)
check("copy(dc_idx_ + (size_t) (qsa_index * R + hbase) * ID, idx_raw_L_ + (size_t) qsa_index * MT * ID, S * ID);" in capture,
      "per-group saved QSA scratch")
check("copy(dc_hist_ + (size_t) hbase * HS, hist_snap_, S * HS);" in capture, "per-group saved PLE history")
for fragment in ["gdn_index * R + hbase + first", "qsa_index * R + hbase + u",
                 "dc_hist_ + (size_t) (hbase + first) * HS", "(gdn_index - sx.gdn_ord0)",
                 "commitd_ + (size_t) rows[first] * CB + 2 + (u - first)"]:
    check(fragment in capture, "deferred replay reads saved/owner-relative rows: " + fragment)
check(launch.index("pending_bm_.emplace(window, key)") < launch.index("cudaGraphLaunch(exec_bm_[key]"),
      "pin captured doorbell layout before partial launch can fail")
check("const auto key = pending->second;" in commit and "bkey(" not in commit, "commit uses captured doorbell identity")
check("n < 1 || n > t - first" in commit and "j < n ? (int32_t) pos[first + j] : -1" in commit,
      "only accepted prefix positions reach commit staging")
check(commit.index("cudaGraphLaunch(it->second") < commit.index("sx.ple_prev[0]") < commit.index("pending_bm_.erase(pending)"),
      "host history/unpin follow successful queued commit")
check("first + keep[rows[first]]" in commit, "PLE host history advances accepted prefixes only")
check(evict.index("cudaStreamSynchronize(cs_)") < evict.index("cudaGraphExecDestroy"), "quiescent eviction")
check("{&exec_bm_, &commit_bm_, &save_bm_, &dcommit_bm_}" in evict, "evict all graph variants together")
check("if (!trim_batch_graphs(key, err)) return false;" in stage and
      "(!staging_deferred_ && !capture_commit_batch(rows, S, hbase, err))" in stage,
      "deferred staging omits immediate commit without changing ordinary capture")
destructor = function(source, "Verifier::~Verifier()")
check(destructor.index("cudaStreamSynchronize(cs_)") < destructor.index("for (auto* m : {&save_bm_, &dcommit_bm_})"),
      "destructor quiesces before destroying deferred variants")
check("if (dc_arena_) cudaFree(dc_arena_);" in destructor and "if (h_commitd_) cudaFreeHost(h_commitd_);" in destructor,
      "deferred arenas released")
check("k.push_back(ar_off_ ? 1 : 0);" in header, "doorbell variant remains part of graph identity")
for fragment in ["for (Verifier* y : v) if (y != x) (void) y->release_gpu_waits(5000);",
                 "for (Verifier* x : v) (void) x->release_gpu_waits(5000);",
                 "*(volatile unsigned*) h_mk_stall_ = 0;"]:
    check(fragment in source, "preserved upstream TP failure/stall fix")
# Pipelined-MTP lifecycle is source checked, not executed by this mock verifier.
for fragment in ["const int pmtp_rows_per_group = GS * pmtp_rows_per_slot;",
                 "G.hbase = pmtp_rows_per_group * pick;", "if (!sl.active) continue;   // an idle slot carries no row",
                 "if (pmtp && !pmtp_verdict(G, vk, outb)) return false;",
                 "while (blk.keep < blk.n && !sl.stop", "!is_eos(outb[f + blk.keep - 1])",
                 "stage_verifier(k).batch_commit_rows(G.rows, G.S, G.hbase, keep.data(), G.tok, G.pos, err)",
                 "if (vk.nonfinite_in(t))", "sl.cached = false;", "sl.max_new - sl.produced - 1",
                 "o.max_context - sl.p - 2", "return pmtp_settle();"]:
    check(fragment in generate, "pipelined-MTP source invariant: " + fragment)
verdict = generate[generate.index("auto pmtp_verdict ="):generate.index("auto pmtp_settle =")]
check(verdict.index("cudaMemcpyAsync(slot_mtp_rows") < verdict.index("cudaEventRecord(pm_copied") <
      verdict.index("cudaStreamWaitEvent(slot_mtp") < verdict.index("->chain_launch("),
      "residual copy on producer stream, copy event, drafter wait, then chain launch")
quiesce = generate[generate.index("PmtpQuiesce pm_quiesce{"):generate.index("// the chain of slot b has landed")]
check(quiesce.index("release_gpu_waits") < quiesce.index("cudaStreamSynchronize") <
      quiesce.index("draft->idle") < quiesce.index("cudaEventDestroy"), "quiescent error-exit event teardown")
settle = generate[generate.index("auto pmtp_settle ="):generate.index("// one turn: serve every running stage")]
check(settle.index("pmtp_poll_draft(b, true)") < settle.index("cudaStreamSynchronize"),
      "settle drafts and every stage's commits before slot reuse")

methods = "\n\n".join(function(source, s) for s in [
    "bool Verifier::batch_launch_rows(", "bool Verifier::batch_commit_rows(",
    "bool Verifier::batch_graph_pending(", "std::set<std::vector<int>> Verifier::batch_graph_keys(",
    "bool Verifier::trim_batch_graphs(", "bool Verifier::evict_batch_graph(", "bool Verifier::instantiate_evicting("])

preamble = r'''
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
static int checks=0, syncs=0, fail_launch=0, fail_sync=0, oom=0;
#define CHECK(x) do { ++checks; if(!(x)) { std::fprintf(stderr,"deferred check at %d: %s\n",__LINE__,#x);std::abort(); } } while(0)
struct Graph { int id; bool alive=true; };
using cudaGraphExec_t=Graph*; using cudaGraph_t=Graph*; using cudaError_t=int;
constexpr int cudaSuccess=0, cudaErrorMemoryAllocation=2;
std::vector<std::unique_ptr<Graph>> graphs;
std::vector<Graph*> launches, destroyed;
Graph* graph() { graphs.push_back(std::make_unique<Graph>(Graph{int(graphs.size())}));return graphs.back().get(); }
int cudaGraphLaunch(Graph* g,int) {
    CHECK(g && g->alive);
    if(fail_launch>0 && --fail_launch==0)return 1;
    launches.push_back(g);return 0;
}
int cudaStreamQuery(int) { return 0; }
int cudaStreamSynchronize(int) { if(fail_sync)return 1; ++syncs;return 0; }
int cudaGraphExecDestroy(Graph* g) { CHECK(syncs>0 && g->alive);g->alive=false;destroyed.push_back(g);return 0; }
int cudaGraphInstantiate(Graph** ex,Graph* g,int) { if(oom>0) {--oom;return cudaErrorMemoryAllocation;} *ex=g;return 0; }
int cudaGetLastError(){return 0;}
const char* cudaGetErrorString(int){return "mock failure";}
namespace strata::core {
struct OnDevice { explicit OnDevice(int) {} };
struct SessionState { int32_t ple_prev[2]={-2,-1}; };
class Verifier {
public:
    int device_=0,cs_=1,dc_rows_=24,max_t_=8,lb_=0,le_=2,b_k_=0,b_steps_=0;
    bool staging_deferred_=false,ar_off_=false,stage_fails=false,capture_fails=false;
    std::atomic<bool> b_running_{false};
    std::function<bool(int,std::string&)> pre_launch_;
    std::vector<SessionState*> slots_;
    int32_t staging[40]={}; int32_t* h_commitd_=staging;
    std::map<std::vector<int>,Graph*> exec_bm_,commit_bm_,save_bm_,dcommit_bm_;
    std::map<std::vector<int>,std::vector<int>> pending_bm_;
    std::map<std::vector<int>,uint64_t> bm_used_;
    uint64_t bm_tick_=0;size_t batch_graph_limit_=0;
    static std::vector<int> batch_key(const int* rows,int S,int hbase) {
        std::vector<int> out{hbase,S};out.insert(out.end(),rows,rows+S);return out;
    }
    std::vector<int> bkey(const int* rows,int S,int hbase) const {
        auto k=batch_key(rows,S,hbase);k.push_back(ar_off_?1:0);return k;
    }
    // Expensive capture/kernels are not executed: scratch-address formulas are checked statically above.
    bool stage_batch(const int* rows,int S,int h,const int32_t*,const int64_t*,std::string& err) {
        CHECK(staging_deferred_);
        if(stage_fails)return false;
        auto key=bkey(rows,S,h);
        if(!trim_batch_graphs(key,err))return false;
        if(!exec_bm_.count(key))exec_bm_[key]=graph();
        bm_used_[key]=++bm_tick_;return true;
    }
    bool capture_deferred(const int* rows,int S,int h,std::string&) {
        if(capture_fails)return false;
        auto key=bkey(rows,S,h);
        if(!save_bm_.count(key))save_bm_[key]=graph();
        if(!dcommit_bm_.count(key))dcommit_bm_[key]=graph();
        return true;
    }
    void ring_now() {} bool ple_stage()const{return true;}
    bool batch_launch_rows(const int*,int,int,const int32_t*,const int64_t*,std::string&);
    bool batch_commit_rows(const int*,int,int,const int*,const int32_t*,const int64_t*,std::string&);
    bool batch_graph_pending(const std::vector<int>&)const;
    std::set<std::vector<int>> batch_graph_keys()const;
    bool trim_batch_graphs(const std::vector<int>&,std::string&);
    bool evict_batch_graph(const std::vector<int>&,bool&,std::string&);
    bool instantiate_evicting(Graph*&,Graph*,const std::vector<int>&,const char*,std::string&);
};
'''

main = r'''
}
using strata::core::Verifier;
using strata::core::SessionState;
void identity_and_prefixes() {
    Verifier v;SessionState a,b;v.slots_={&a,&b};std::string err;
    int rows[]={0,0,0,1,1},other[]={1},keep[]={2,1};
    int32_t tokens[]={10,11,12,20,21};int64_t pos[]={100,101,102,200,201};
    auto key=v.bkey(rows,5,0);
    CHECK(v.batch_launch_rows(rows,5,0,tokens,pos,err));
    CHECK(v.pending_bm_.at(v.batch_key(rows,5,0))==key && v.b_running_);
    CHECK(launches[launches.size()-2]==v.exec_bm_.at(key) && launches.back()==v.save_bm_.at(key));
    const auto saved=v.dcommit_bm_.at(key);
    CHECK(!v.batch_launch_rows(rows,5,0,tokens,pos,err)); // busy
    v.b_running_=false;
    CHECK(!v.batch_launch_rows(rows,5,0,tokens,pos,err)); // verdict still pending
    v.ar_off_=true; // a later group refreshes doorbell selection
    CHECK(v.batch_launch_rows(other,1,12,tokens,pos,err));v.b_running_=false;
    const auto wrong=v.bkey(rows,5,0);v.dcommit_bm_[wrong]=graph();
    CHECK(wrong!=key && v.pending_bm_.size()==2);
    keep[0]=0;CHECK(!v.batch_commit_rows(rows,5,0,keep,tokens,pos,err));
    keep[0]=4;CHECK(!v.batch_commit_rows(rows,5,0,keep,tokens,pos,err));
    keep[0]=2;
    fail_launch=1;CHECK(!v.batch_commit_rows(rows,5,0,keep,tokens,pos,err));
    CHECK(v.pending_bm_.size()==2 && a.ple_prev[1]==-1 && b.ple_prev[1]==-1);
    CHECK(v.batch_commit_rows(rows,5,0,keep,tokens,pos,err));
    CHECK(launches.back()==saved && launches.back()!=v.dcommit_bm_.at(wrong));
    CHECK(v.pending_bm_.size()==1 && !v.batch_graph_pending(key));
    CHECK(v.staging[0]==2 && v.staging[1]==1 && v.staging[2]==100 && v.staging[3]==101);
    for(int i=4;i<10;++i)CHECK(v.staging[i]==-1);
    CHECK(v.staging[10]==1 && v.staging[11]==0 && v.staging[12]==200);
    for(int i=13;i<20;++i)CHECK(v.staging[i]==-1);
    CHECK(a.ple_prev[0]==10 && a.ple_prev[1]==11 && b.ple_prev[0]==-1 && b.ple_prev[1]==20);
    CHECK(!v.batch_commit_rows(rows,5,0,keep,tokens,pos,err)); // no duplicate verdict
}
void failure_pins() {
    Verifier v;SessionState a;v.slots_={&a};std::string err;
    int rows[]={0};int32_t tok[]={7};int64_t pos[]={42};
    v.dc_rows_=0;CHECK(!v.batch_launch_rows(rows,1,0,tok,pos,err));v.dc_rows_=24;
    CHECK(!v.batch_launch_rows(rows,1,-1,tok,pos,err));
    CHECK(!v.batch_launch_rows(rows,1,24,tok,pos,err));
    v.stage_fails=true;CHECK(!v.batch_launch_rows(rows,1,0,tok,pos,err));
    CHECK(!v.staging_deferred_ && v.pending_bm_.empty());v.stage_fails=false;
    v.capture_fails=true;CHECK(!v.batch_launch_rows(rows,1,0,tok,pos,err));
    CHECK(v.pending_bm_.empty());v.capture_fails=false;
    fail_launch=2; // window launch succeeds, save launch fails: partial work remains pinned
    CHECK(!v.batch_launch_rows(rows,1,0,tok,pos,err));
    auto key=v.bkey(rows,1,0);CHECK(v.batch_graph_pending(key));
    bool evicted=true;CHECK(v.evict_batch_graph({99},evicted,err) && !evicted);
    fail_launch=0;
}
void lifetime_and_union() {
    Verifier v;std::string err;
    std::vector<int> a{1},b{2},c{3},keep{9};
    v.exec_bm_[a]=graph();v.commit_bm_[a]=graph();v.save_bm_[a]=graph();v.dcommit_bm_[a]=graph();
    v.save_bm_[b]=graph();v.dcommit_bm_[c]=graph();
    v.bm_used_[a]=1;v.bm_used_[b]=2;v.bm_used_[c]=3;
    CHECK(v.batch_graph_keys().size()==3);v.pending_bm_[{100}]=a;v.batch_graph_limit_=1;
    CHECK(v.trim_batch_graphs(keep,err));
    CHECK(v.batch_graph_keys()==std::set<std::vector<int>>{a}); // soft limit cannot evict pending verdict
    const auto before=destroyed.size();v.pending_bm_.clear();
    bool evicted=true;fail_sync=1;
    CHECK(!v.evict_batch_graph(keep,evicted,err) && !evicted);
    CHECK(destroyed.size()==before && v.batch_graph_keys().count(a)==1);
    fail_sync=0;const int prior=syncs;
    CHECK(v.evict_batch_graph(keep,evicted,err) && evicted && syncs==prior+1);
    CHECK(destroyed.size()==before+4 && v.batch_graph_keys().empty() && v.bm_used_.empty());
    v.exec_bm_[keep]=graph();CHECK(v.evict_batch_graph(keep,evicted,err) && !evicted);
    // Allocation retry evicts eligible layouts, never a pending layout or the current capture.
    v.save_bm_[b]=graph();v.bm_used_[b]=1;Graph* ex=nullptr;Graph* raw=graph();oom=1;
    CHECK(v.instantiate_evicting(ex,raw,keep,"mock instantiate: ",err) && ex==raw);
    CHECK(!v.batch_graph_keys().count(b) && v.batch_graph_keys().count(keep));
    v.pending_bm_[{101}]=keep;oom=1;ex=nullptr;
    CHECK(!v.instantiate_evicting(ex,raw,{11},"mock instantiate: ",err) && !ex);
}
int main() {
    identity_and_prefixes();failure_pins();lifetime_and_union();
    std::printf("deferred_graph_test: %d CPU mock checks passed (3 scenarios); scratch/kernel invariants are static only\n",checks);
}
'''

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: deferred_graph_test.py OUTPUT.cpp")
    Path(sys.argv[1]).write_text(preamble + methods + main)
    print(f"deferred_graph_test: {static_checks} static source checks passed; generated CPU mock fixture")
