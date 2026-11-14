#!/usr/bin/env python3
"""Generate a CPU fixture from production DP methods, plus explicit static checks.

CUDA-shaped boundaries below are in-memory mocks. This is not a GPU/runtime test.
Only the named source files are read; the output C++ belongs in the build tree.
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
    # Mask comments and literals without changing offsets, so braces in diagnostics do not matter.
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                    lambda m: " " * len(m[0]), text)
    depth = 0
    for end in range(brace, len(text)):
        depth += (masked[end] == "{") - (masked[end] == "}")
        if depth == 0:
            return text[start:end + 1]
    raise AssertionError("unclosed method: " + signature)


prefill = (ROOT / "src/prefill/prefill.cpp").read_text()
layer = (ROOT / "src/core/layer.cpp").read_text()
session = (ROOT / "src/core/session.cpp").read_text()
mtp = (ROOT / "src/core/mtp.cpp").read_text()
generate = (ROOT / "src/program/generate.cpp").read_text()
header = (ROOT / "include/strata/prefill/prefill.hpp").read_text()
run = function(prefill, "bool Prefill::run_dp(const std::vector<Prefill*>& paths")
copy = function(prefill, "bool Prefill::dp_copy_layer(")
wait = function(prefill, "bool Prefill::dp_wait(")
publish = function(prefill, "bool Prefill::dp_publish(")
check("using DpChunk = dp::Chunk;" in header, "public planner type")
check("const int peer = (dp_index_ + (int) L.dev.size() - 1) % (int) L.dev.size();" in wait,
      "circular predecessor, not DP2-only peer")
check("L.abort || L.pub[peer][(size_t) l] >= k - 1" in wait, "abortable predecessor wait")
check("dp::handoff_begin(dp_plan_, k, (int) L.dev.size())" in publish and
      "p0 - dp_plan_[(size_t) k].c0 + begin" in publish, "cumulative position-relative handoff")
check(publish.index("dp_copy_layer") < publish.index("cudaEventRecord") < publish.index("L.pub["),
      "copy/event before mailbox publication")
check("(gi - b.gdn_ord0)" in copy and "(gi - a.gdn_ord0)" in copy, "owner-relative GDN source/destination")
check("p0 / blk" in copy and "(p0 + T) / blk + 2" in copy, "cumulative QSA range and tail margin")
check(run.index("decline: alignment") < run.index("struct Saved"), "alignment decline before overrides")
check(run.index("decline: images") < run.index("struct Saved"), "unsupported reads decline before overrides")
check(run.index("saved.reserve") < run.index("} restore{") < run.index("p->dp_active_ = true"),
      "allocations and restore guard precede overrides")
for field in ["stage_lb_", "stage_le_", "next_", "hand_in_", "hand_ev_", "hand_pieces_",
              "host_res", "on_stage_chunk", "dp_active_"]:
    check(field in run[run.index("} restore{"):run.index("for (auto* p : paths) {\n        p->stage_lb_")],
          "restored per-prompt field: " + field)
check("{p->impl_->cs, p->impl_->copy, p->impl_->kv_copy}" in run, "drain all three stream kinds")
check("L.abort = true" in run and "L.cv.notify_all()" in run and
      run.index("for (auto& f : futures) f.get();") < run.index("completed path drain"), "abort/notify/join/drain")
check("auto& last = *paths.back();" in run and "last.dp_copy_layer(*p, l, pos0, n, err)" in run,
      "terminal full-range state returned to each decode owner")
check("return run_dp(*paths[0], *paths[1]" in run, "DP2 compatibility adapter")
check("const bool kv_prefetch = kv_prefetch_on && (!dp || dp_link_->dev.size() == 2);" in prefill,
      "N-path excludes two-path-only prefetch")
init = function(layer, "uint64_t qsa_state_init(")
for guard in ["share_host->kv_mode != p.mode", "share_host->max_cells != max_cells",
              "qsa_kv_format(*share_host) != qsa_kv_format(st)", "share_host->kv_host_bytes != bytes",
              "!share_host->kv_host_base"]:
    check(guard in init, "shared-host compatibility: " + guard)
check(init.index("h = (uint8_t*) share_host->kv_host_base") < init.index("cudaHostGetDevicePointer((void**) &d, h, 0)"),
      "receiver remaps CPU base, not sender device alias")
check("if (!share_host) g_kv_host_bytes += bytes;" in init and "st.kv_host_base = h;" in init and
      "st.kv_host_bytes = bytes;" in init, "host identity and owner-only accounting")
check("share_host_states ? &share_host_states[s.qsa_ord0 + j] : nullptr" in session,
      "borrow host states by actual session ordinal")
release = function(session, "void session_release(")
check(release.count("s.qsa_states[s.qsa_ord0 + j]") == 3, "RoPE release uses owned slice throughout")
check("if (owns_weights_ && dense4_) cudaFree(dense4_);" in mtp and
      "dense4_ = shared->dense4_;\n        q4_off_ = shared->q4_off_;" in mtp,
      "shared Q4 MTP storage/offset pairing and owner-only free")
check(generate.index("dp_arenas.emplace_back(buf") < generate.index("buf, *state, 0, -1, rope, hosts.data()"),
      "register partial-allocation cleanup before foreign session init")
check("if (e.rfind(\"prefill DP: decline:\", 0) != 0) return false;" in generate,
      "post-start errors never re-read mutated state")
# Pin-only TP safeguards must survive the port; these are static, not TP execution coverage.
for fragment in ["struct OwnRowsScope", "bool Prefill::tp_ownrows_env()", "pool_retry();",
                 "pm.own_rows ? pm.R_a", "prefill TP (read): "]:
    check(fragment in prefill, "preserved upstream: " + fragment)
for fragment in ["tpp_pictures", "!lend_no_tp", "if (!own_rows_back(e)) return false;", "tp_clean_cells",
                 "tp_to(true, n - 1, err)", "late_recover", "srcp->note_async_read(b, lt.stream)", "late_sd_release"]:
    check(fragment in generate, "preserved upstream: " + fragment)

methods = "\n\n".join(function(prefill, s) for s in [
    "core::SessionState& dp_gdn_session(", "float* dp_gdn_of(", "core::QsaState& dp_qsa_of(",
    "core::PleRun& dp_ple_of(", "bool Prefill::dp_wait(", "bool Prefill::dp_publish(",
    "bool Prefill::dp_copy_layer(", "bool Prefill::run_dp(const std::vector<Prefill*>& paths"])

preamble = r'''
#include "strata/prefill/dp_plan.hpp"
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
static std::atomic<int> checks{0}, drains{0}, resets{0}, copies{0}, live{0};
#define CHECK(x) do { ++checks; if (!(x)) { std::fprintf(stderr,"DP check failed at %d: %s\n",__LINE__,#x); std::abort(); } } while(0)
using cudaStream_t = int;
using cudaEvent_t = int;
constexpr int cudaSuccess = 0;
int cudaStreamSynchronize(cudaStream_t) { ++drains; return 0; }
int cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, int) { return 0; }
int cudaEventRecord(cudaEvent_t, cudaStream_t) { return 0; }
int cudaGetLastError() { return 0; }
const char* cudaGetErrorString(int) { return "mock failure"; }
int cudaMemcpyPeerAsync(void* dst, int, const void* src, int, size_t n, cudaStream_t) {
    ++copies; std::memcpy(dst,src,n); return 0;
}
namespace strata::aux_cpus { void pin_current_thread() {} }
namespace strata::kernels {
constexpr int NG_HIST = 2, NG_HC_DIM = 2;
struct QsaShapes { int64_t idx_block = 256; };
void kv_stream_reset(int*, int) { ++resets; }
}
namespace strata::core {
struct ModelGeometry {
    int64_t n_layers=8, n_expert=2, ssm_state_size=1, ssm_v_heads=1,
            ssm_conv_channels=1, ssm_d_conv=2, idx_key_dim=1;
};
struct OnDevice { explicit OnDevice(int) {} };
bool is_qsa_layer(const ModelGeometry&, int64_t l) { return l%2 == 1; }
kernels::QsaShapes qsa_shapes(const ModelGeometry&) { return {}; }
struct QsaState {
    float *idx_pooled=nullptr, *idx_tail=nullptr, *idx_dead=nullptr;
    int32_t* idx_block_pos=nullptr;
    int64_t idx_pooled_rows=2048;
    int* map=nullptr;
};
struct PleRun { float* hist=nullptr; bool ready() const { return hist != nullptr; } };
struct SessionState {
    int64_t gdn_ord0=0, gdn_alloc=4, qsa_ord0=0, qsa_alloc=4, max_cells=262144;
    float* gdn_state=nullptr;
    QsaState* qsa_states=nullptr;
    PleRun ple;
    int32_t ple_prev[2]={-2,-1};
};
}
namespace strata::prefill {
namespace core = strata::core;
class Prefill {
public:
    struct DpLink {
        std::mutex mu; std::condition_variable cv; bool abort=false;
        std::vector<int> dev;
        std::vector<std::vector<int64_t>> pub;
        std::vector<std::vector<int>> ev;
    };
    struct Impl { int device=0, cs=1, copy=2, kv_copy=3; int64_t T=32768;
                  core::ModelGeometry* g=nullptr; core::SessionState* ss=nullptr; const int32_t* host_res=nullptr; };
    Impl* impl_=nullptr;
    std::shared_ptr<DpLink> dp_link_;
    core::SessionState* dp_foreign_=nullptr;
    Prefill *dp_peer_=nullptr, *next_=nullptr;
    int dp_index_=0;
    int64_t dp_own_lb_=0, dp_own_le_=0, stage_lb_=0, stage_le_=0, dp_last_p0_=-1;
    const float* hand_in_=nullptr;
    const void* hand_ev_=nullptr;
    int hand_pieces_=0;
    bool dp_active_=false;
    void* embd_rows=nullptr;
    std::vector<int> tap_layers_;
    bool stop_at_chunk=false;
    std::vector<dp::Chunk> dp_plan_;
    std::vector<int32_t> dp_res_;
    std::function<bool(int64_t,std::string&)> on_stage_chunk;
    int fail_mode=0;
    bool run_impl(const int64_t*, int64_t, int64_t, std::string&);
    bool dp_wait(int64_t,int64_t,std::string&);
    bool dp_publish(int64_t,int64_t,int64_t,int64_t,std::string&);
    bool dp_copy_layer(Prefill&,int64_t,int64_t,int64_t,std::string&);
    static bool run_dp(const std::vector<Prefill*>&, const int64_t*,int64_t,int64_t,int64_t,int64_t,std::string&);
    // DP2 production remains separate; this sentinel checks overload dispatch only.
    static bool run_dp(Prefill&, Prefill&,const int64_t*,int64_t,int64_t,int64_t,int64_t,std::string& e) {
        e="DP2 adapter"; return true;
    }
};
'''

main = r'''
bool Prefill::run_impl(const int64_t*, int64_t, int64_t pos0, std::string& err) {
    struct Live { Live(){++live;} ~Live(){--live;} } active;
    CHECK(dp_active_ && !next_ && !hand_in_ && !hand_ev_ && hand_pieces_==0 && !on_stage_chunk);
    CHECK(stage_lb_==0 && stage_le_==impl_->g->n_layers);
    for (int l=0;l<8;++l) for(int e=0;e<2;++e)
        CHECK(impl_->host_res[l*2+e] == (l>=dp_own_lb_ && l<dp_own_le_ ? l*2+e : -1));
    if (fail_mode==1) { err="cancelled"; return false; }
    if (fail_mode==2) throw std::runtime_error("injected path exception");
    for (const auto& c:dp_plan_) if(c.who==dp_index_) {
        for (int l=0;l<8;++l) {
            if(c.k && !dp_wait(l,c.k,err)) return false;
            if (l%2==0) {
                auto* state=dp_gdn_of(*impl_->ss,dp_foreign_,l/2,2);
                CHECK(state[0]==100+l+pos0+c.c0);
                state[0]+=c.T;
            } else {
                auto& q=dp_qsa_of(*impl_->ss,dp_foreign_,l/2);
                for(int b=0;b<(pos0+c.c0)/256;++b) CHECK(q.idx_pooled[b]==b+1);
                for(int b=(pos0+c.c0)/256;b<(pos0+c.c0+c.T+255)/256;++b) q.idx_pooled[b]=float(b+1);
            }
            if(l==1) dp_ple_of(*impl_->ss,dp_foreign_).hist[0]=float(pos0+c.c0+c.T);
            if(!dp_publish(l,c.k,pos0+c.c0,c.T,err)) return false;
        }
    }
    return true;
}
}
using strata::prefill::Prefill;
using strata::core::SessionState;
struct Storage {
    std::vector<float> gdn=std::vector<float>(8,-999), hist=std::vector<float>(4,-999);
    std::vector<std::vector<float>> pools=std::vector<std::vector<float>>(4,std::vector<float>(2048));
    std::vector<std::vector<float>> tails=std::vector<std::vector<float>>(4,std::vector<float>(255));
    float dead[4]={}; int32_t block[4]={};
    strata::core::QsaState q[4]; SessionState s;
    Storage() {
        s.gdn_state=gdn.data(); s.qsa_states=q; s.ple.hist=hist.data();
        for(int i=0;i<4;++i) { q[i].idx_pooled=pools[i].data();q[i].idx_tail=tails[i].data();q[i].idx_dead=&dead[i];q[i].idx_block_pos=&block[i]; }
    }
};
void scenario(int N, int failure, int64_t pos0, int64_t n, int64_t cap) {
    strata::core::ModelGeometry g;
    g.n_layers=N*2;
    // The eight-layer mock model gives each of four owners one GDN and one QSA layer.
    CHECK(N==4);
    std::vector<std::unique_ptr<Storage>> own,foreign;
    std::vector<Prefill> p(N); std::vector<Prefill::Impl> impl(N); std::vector<Prefill*> paths;
    auto L=std::make_shared<Prefill::DpLink>();
    L->dev={0,1,2,3};L->pub.assign(N,std::vector<int64_t>(8,-1));L->ev.assign(N,std::vector<int>(8,1));
    int32_t residency[16]; for(int i=0;i<16;++i)residency[i]=i;
    float hand=1;
    for(int i=0;i<N;++i) {
        own.push_back(std::make_unique<Storage>());foreign.push_back(std::make_unique<Storage>());
        auto& o=*own.back(); o.s.gdn_ord0=i;o.s.gdn_alloc=1;o.s.qsa_ord0=i;o.s.qsa_alloc=1;
        o.gdn[0]=float(100+2*i+pos0);
        for(int b=0;b<pos0/256;++b)o.pools[i][b]=float(b+1);
        if(i!=0)o.s.ple.hist=nullptr; // PLE is owned by the first slice only.
        impl[i].device=i;impl[i].g=&g;impl[i].ss=&o.s;impl[i].host_res=residency;impl[i].T=cap;
        p[i].impl_=&impl[i];p[i].dp_foreign_=&foreign.back()->s;p[i].dp_link_=L;p[i].dp_index_=i;
        p[i].dp_own_lb_=p[i].stage_lb_=2*i;p[i].dp_own_le_=p[i].stage_le_=2*i+2;
        p[i].next_=&p[(i+1)%N];p[i].dp_peer_=&p[(i+1)%N];
        p[i].hand_in_=&hand;p[i].hand_ev_=&hand;p[i].hand_pieces_=7;
        p[i].on_stage_chunk=[](int64_t,std::string&){return true;};
        paths.push_back(&p[i]);
    }
    p[0].fail_mode=failure;
    std::vector<int64_t> tok((size_t)n,42); std::string err;
    const int d0=drains, r0=resets;
    CHECK(strata::prefill::dp::takes(n,cap,N,4224));
    bool ok=Prefill::run_dp(paths,tok.data(),n,pos0,cap,4224,err);
    if(ok!=(failure==0))std::fprintf(stderr,"DP scenario failure=%d n=%lld: %s\n",failure,(long long)n,err.c_str());
    CHECK(ok==(failure==0));CHECK(live==0);CHECK(drains>d0);CHECK(resets-r0==20);
    if(failure)CHECK(L->abort && !err.empty());
    for(int i=0;i<N;++i) {
        CHECK(!p[i].dp_active_ && p[i].next_==&p[(i+1)%N] && impl[i].host_res==residency);
        CHECK(p[i].stage_lb_==2*i && p[i].stage_le_==2*i+2 && p[i].on_stage_chunk);
        CHECK(p[i].hand_in_==&hand && p[i].hand_ev_==&hand && p[i].hand_pieces_==7);
        if(ok) {
            CHECK(own[i]->gdn[0]==100+2*i+pos0+n);
            for(int b=0;b<(pos0+n)/256;++b) CHECK(own[i]->pools[i][b]==b+1);
            CHECK(own[i]->s.ple_prev[0]==42 && own[i]->s.ple_prev[1]==42);
        }
    }
    if(ok) CHECK(own[0]->hist[0]==pos0+n);
    // Unsupported input is rejected without setting the per-prompt overrides.
    p[0].embd_rows=&hand; const int before=drains;
    CHECK(!Prefill::run_dp(paths,tok.data(),n,pos0,cap,4224,err));
    CHECK(err.find("decline:")!=std::string::npos && drains==before && !p[0].dp_active_);
    p[0].embd_rows=nullptr;
    CHECK(!Prefill::run_dp(paths,tok.data(),n,pos0+1,cap,4224,err));
    CHECK(err.find("decline:")!=std::string::npos && drains==before);
    std::vector<Prefill*> two{&p[0],&p[1]};
    CHECK(Prefill::run_dp(two,tok.data(),n,pos0,cap,4224,err) && err=="DP2 adapter");
}
int main() {
    scenario(4,0,0,72627,32768);
    scenario(4,0,512,151123,8192); // multiple ring traversals and a resumed prefix; final chunk >= 4224
    scenario(4,1,0,72627,32768); // cancellation aborts and joins predecessor waiters
    scenario(4,2,0,72627,32768); // thrown path also aborts, joins and restores
    CHECK(live==0 && copies>0);
    std::printf("dp_port_test: %d CPU mock checks passed (4 scenarios); static checks are generation-time only\n",checks.load());
}
'''

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: dp_port_test.py OUTPUT.cpp")
    Path(sys.argv[1]).write_text(preamble + methods + main)
    print(f"dp_port_test: {static_checks} static source checks passed; generated CPU mock fixture")
