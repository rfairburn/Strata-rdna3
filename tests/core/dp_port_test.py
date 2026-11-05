"""Emit a CPU-only fake-runtime fixture using the production DP methods verbatim.

This tests orchestration/ownership, not GPU kernels or the native ABI. Run the
generated C++ fixture separately from any GPU-linked test binary.
"""
import pathlib
import sys


def block(text, anchor):
    start = text.index(anchor)
    first = text.index("{", start)
    depth = 1
    end = first + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


root = pathlib.Path(__file__).resolve().parents[2]
pf = (root / "src/prefill/prefill.cpp").read_text()
layer = (root / "src/core/layer.cpp").read_text()
mtp = (root / "src/core/mtp.cpp").read_text()
gen = (root / "src/program/generate.cpp").read_text()
session = (root / "src/core/session.cpp").read_text()

# Do not silently test a handwritten replacement of the production methods.
methods = "\n\n".join(block(pf, anchor) for anchor in (
    "void Prefill::set_dp(",
    "std::shared_ptr<Prefill::DpLink> Prefill::make_dp_link(int n_layers, const std::vector<int>&",
    "core::SessionState& dp_gdn_session(",
    "core::QsaState& dp_qsa_of(",
    "core::PleRun& dp_ple_of(",
    "bool Prefill::dp_wait(",
    "bool Prefill::dp_copy_layer(",
    "bool Prefill::dp_publish(",
    "bool Prefill::run_dp(const std::vector<Prefill*>&",
))
link = block(pf, "struct Prefill::DpLink") + ";"
host = block(layer, "    if (p.mode != 0) {")
shared = block(mtp, "    if (shared != nullptr) {")
free_q4 = next(line.strip() for line in mtp.splitlines() if "cudaFree(dense4_)" in line)
cleanup = block(gen, "dp_arenas.emplace_back(buf, [dev, state]")
cleanup = cleanup[cleanup.index("{") + 1:-1]
release = block(session, "void session_release(")

prefix = r'''
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
#include <string>
#include <thread>
#include <utility>
using cudaStream_t = int*;
struct Event { int dev; };
using cudaEvent_t = Event*;
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaEventDisableTiming = 0, cudaErrorPeerAccessAlreadyEnabled = 2;
int cudaDeviceCanAccessPeer(int* can, int, int) { *can = 1; return 0; }
int cudaDeviceEnablePeerAccess(int, int) { return 0; }
int cudaGetLastError() { return 0; }
const char* cudaGetErrorString(int) { return "fake"; }
thread_local int device = -1;
std::atomic<int> copies{0}, syncs{0};
int fail_copy_at = -1, rope_releases = 0;
int fail_event = -1, event_count = 0, live_events = 0, frees = 0;
int cudaEventCreateWithFlags(cudaEvent_t* e, int) {
    if (event_count++ == fail_event) return 1;
    *e = new Event{device}; ++live_events; return 0;
}
int cudaEventDestroy(cudaEvent_t e) { delete e; --live_events; return 0; }
int cudaEventRecord(cudaEvent_t e, cudaStream_t s) {
    if (e->dev != device || *s != device) std::abort(); return 0;
}
int cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t, int) {
    if (*s != device) std::abort(); return 0;
}
int cudaStreamSynchronize(cudaStream_t s) {
    if (s && *s != device) std::abort(); ++syncs; return 0;
}
int cudaDeviceSynchronize() { return 0; }
int cudaMemcpyPeerAsync(void* dst, int, const void* src, int from, size_t bytes, cudaStream_t s) {
    if (from != device || *s != device) std::abort();
    if (copies++ == fail_copy_at) return 1;
    std::memcpy(dst, src, bytes); return 0;
}
int cudaFree(void*) { ++frees; return 0; }
int cudaFreeHost(void*) { ++frees; return 0; }
namespace strata::kernels {
constexpr int NG_HIST = 1, NG_HC_DIM = 1, KV_Q8_GROUP = 64;
struct KvHostPools {
    int8_t *k_q = nullptr, *v_q = nullptr;
    uint16_t *k_scale = nullptr, *v_scale = nullptr, *k_pool = nullptr, *v_pool = nullptr;
    uint8_t *k_q4 = nullptr, *v_q4 = nullptr;
};
struct Shapes { int idx_block = 256, n_head_kv = 1, page_size = 4, head_dim = 64; };
uint64_t kv_block_bytes(Shapes, int) { return 512; }
uint64_t kv_q4_bytes_per_head(int) { return 36; }
void kv_stream_reset(int* map, cudaStream_t) { ++*map; }
void rope_table_release(float*) { ++rope_releases; }
}
namespace strata::aux_cpus { void pin_current_thread() {} }
namespace strata::core {
struct OnDevice {
    int old;
    explicit OnDevice(int d) : old(device) { device = d; }
    ~OnDevice() { device = old; }
};
struct ModelGeometry {
    int64_t n_layers = 8, n_expert = 2, idx_key_dim = 1;
    int64_t ssm_state_size = 1, ssm_v_heads = 1, ssm_conv_channels = 0, ssm_d_conv = 1;
};
bool is_qsa_layer(const ModelGeometry&, int64_t l) { return l % 2 == 1; }
kernels::Shapes qsa_shapes(const ModelGeometry&) { return {}; }
struct QsaState {
    float *idx_pooled = nullptr, *idx_tail = nullptr, *idx_dead = nullptr;
    int32_t* idx_block_pos = nullptr;
    int64_t idx_pooled_rows = 512, max_cells = 100000;
    int map_data = 0;
    int* map = &map_data;
    int kv_mode = 1;
    bool kv_hybrid = false, kv_q4 = false, kv_int8 = true, owns_rope = false;
    float* cos_tab = nullptr;
    void* kv_host_base = nullptr;
    uint64_t kv_host_bytes = 0;
    int32_t *host_step = nullptr, *host_pos = nullptr;
    kernels::KvHostPools host;
};
int qsa_kv_format(const QsaState&) { return 1; }
struct PleRun { float* hist = nullptr; bool ready() const { return hist != nullptr; } };
struct SessionState {
    int64_t max_cells = 100000, gdn_ord0 = 0, gdn_alloc = 4, qsa_ord0 = 0, qsa_alloc = 4;
    float* gdn_state = nullptr;
    QsaState* qsa_states = nullptr;
    PleRun ple;
    int32_t ple_prev[2] = {-1, -1};
};
}
namespace strata::prefill {
bool dp_share_env() { return false; }
struct Kva { bool on = false, capture = false; };
Kva kc;
const Kva& kva_config() { return kc; }
class Prefill {
public:
    struct Impl {
        int device;
        int64_t T = 32768;
        core::ModelGeometry* g;
        core::SessionState* ss;
        int stream;
        cudaStream_t cs, copy, kv_copy;
        const int32_t* host_res = nullptr;
    };
    struct DpLink;
    std::unique_ptr<Impl> impl_;
    Prefill *dp_peer_ = nullptr, *next_ = nullptr;
    core::SessionState* dp_foreign_ = nullptr;
    int64_t dp_own_lb_ = 0, dp_own_le_ = 0, stage_lb_ = 0, stage_le_ = 0;
    int64_t dp_last_p0_ = -1;
    int dp_index_ = 0, hand_pieces_ = 0;
    const float *hand_in_ = nullptr, *embd_rows = nullptr;
    const void* hand_ev_ = nullptr;
    std::vector<int> tap_layers_;
    std::shared_ptr<DpLink> dp_link_;
    std::vector<dp::Chunk> dp_plan_;
    std::vector<int32_t> dp_res_;
    bool dp_active_ = false;
    int fail_layer = -1;
    std::function<bool()> stop_at_chunk;
    std::function<bool(int64_t, std::string&)> on_stage_chunk;
    void set_dp(Prefill*, core::SessionState*, int64_t, int64_t, std::shared_ptr<DpLink>, int);
    static std::shared_ptr<DpLink> make_dp_link(int, const std::vector<int>&, std::string&);
    bool dp_wait(int64_t, int64_t, std::string&);
    bool dp_publish(int64_t, int64_t, int64_t, int64_t, std::string&);
    bool dp_copy_layer(Prefill&, int64_t, int64_t, int64_t, std::string&);
    static bool run_dp(Prefill&, Prefill&, const int64_t*, int64_t, int64_t, int64_t, int64_t, std::string&) { return false; }
    static bool run_dp(const std::vector<Prefill*>&, const int64_t*, int64_t, int64_t, int64_t, int64_t, std::string&);
    bool run_impl(const int64_t*, int64_t, int64_t, std::string&);
};
'''

suffix = r'''
bool Prefill::run_impl(const int64_t*, int64_t, int64_t pos0, std::string& err) {
    const core::OnDevice on(impl_->device);
    auto& g = *impl_->g;
    for (const auto& c : dp_plan_) if (c.who == dp_index_) {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (l == fail_layer) { err = "injected failure"; return false; }
            if (c.k && !dp_wait(l, c.k, err)) return false;
            if (core::is_qsa_layer(g, l)) {
                auto& s = dp_qsa_of(*impl_->ss, dp_foreign_, qi++);
                for (int64_t r = 0; r < (pos0 + c.c0) / 256; ++r)
                    if (s.idx_pooled[r] != 1) { err = "missing accumulated indexer row"; return false; }
                for (int64_t r = (pos0 + c.c0) / 256; r < (pos0 + c.c0 + c.T) / 256; ++r) s.idx_pooled[r] = 1;
                *s.idx_block_pos = (int32_t) (pos0 + c.c0 + c.T);
            } else {
                auto& s = dp_gdn_session(*impl_->ss, dp_foreign_, gi);
                float& state = s.gdn_state[gi - s.gdn_ord0];
                if (state != pos0 + c.c0) { err = "wrong GDN handoff"; return false; }
                state += (float) c.T; ++gi;
            }
            if (l == 1) {
                auto& pr = dp_ple_of(*impl_->ss, dp_foreign_);
                if (*pr.hist != pos0 + c.c0) { err = "wrong PLE handoff"; return false; }
                *pr.hist += (float) c.T;
            }
            if (!dp_publish(l, c.k, pos0 + c.c0, c.T, err)) return false;
        }
    }
    return true;
}
} // namespace strata::prefill
using namespace strata;
void check(bool v) { if (!v) { std::fprintf(stderr, "dp_port_test failed\n"); std::abort(); } }
struct State {
    core::SessionState s;
    std::vector<float> gdn = std::vector<float>(4, 256), pooled[4], tail[4], dead = std::vector<float>(4);
    core::QsaState qsa[4]; int32_t positions[4] = {};
    float hist = 256;
    State() {
        s.gdn_state = gdn.data(); s.qsa_states = qsa;
        for (int q = 0; q < 4; ++q) {
            pooled[q].resize(512); pooled[q][0] = 1;
            tail[q].resize(kernels::Shapes{}.idx_block - 1);
            qsa[q].idx_pooled = pooled[q].data(); qsa[q].idx_tail = tail[q].data();
            qsa[q].idx_dead = &dead[q]; qsa[q].idx_block_pos = &positions[q];
        }
    }
};
void scheduler_test(int npaths, int fail) {
    prefill::Prefill p[4]; State owners[4], foreign[4];
    core::ModelGeometry g; std::string err;
    const std::vector<int> devices = npaths == 4 ? std::vector<int>{0, 1, 3, 2} : std::vector<int>{0, 2};
    auto link = prefill::Prefill::make_dp_link(8, devices, err); check(!!link);
    std::vector<prefill::Prefill*> paths;
    for (int i = 0; i < npaths; ++i) {
        const int64_t lo = i * 8 / npaths, hi = (i + 1) * 8 / npaths;
        auto& s = owners[i].s;
        s.gdn_ord0 = lo / 2; s.gdn_alloc = (hi - lo) / 2;
        s.qsa_ord0 = lo / 2; s.qsa_alloc = (hi - lo) / 2;
        if (i == 0) s.ple.hist = &owners[i].hist;
        else foreign[i].s.ple.hist = &foreign[i].hist;
        p[i].impl_ = std::make_unique<prefill::Prefill::Impl>();
        auto& m = *p[i].impl_;
        m.device = devices[i]; m.stream = devices[i]; m.g = &g; m.ss = &s;
        m.cs = m.copy = m.kv_copy = &m.stream;
        p[i].set_dp(&p[(i + 1) % npaths], &foreign[i].s, lo, hi, link, i);
        p[i].stage_lb_ = lo; p[i].stage_le_ = hi;
        p[i].next_ = i + 1 < npaths ? &p[i + 1] : nullptr;
        paths.push_back(&p[i]);
    }
    if (fail == 1) p[0].fail_layer = 0;
    if (fail == 2) fail_copy_at = copies + 2;
    std::vector<int64_t> tokens(72627, 7);
    const bool ok = prefill::Prefill::run_dp(paths, tokens.data(), tokens.size(), 256, 32768, 4224, err);
    if (!ok && !fail) std::fprintf(stderr, "%s\n", err.c_str());
    check(ok == (fail == 0));
    fail_copy_at = -1;
    for (int i = 0; i < npaths; ++i) {
        check(!p[i].dp_active_ && p[i].stage_lb_ == i * 8 / npaths);
        check(p[i].next_ == (i + 1 < npaths ? &p[i + 1] : nullptr));
        if (ok) {
            check(owners[i].gdn[0] == 72883);
            for (int q = (int) owners[i].s.qsa_ord0; q < owners[i].s.qsa_ord0 + owners[i].s.qsa_alloc; ++q) {
                check(*owners[i].qsa[q].idx_block_pos == 72883);
                for (int r = 0; r < 72883 / 256; ++r) check(owners[i].pooled[q][r] == 1);
            }
            check(owners[i].s.ple_prev[1] == 7);
        }
    }
    // Preflight declines must not launch paths or alter state.
    const int before = copies;
    check(!prefill::Prefill::run_dp(paths, tokens.data(), 8192, 256, 32768, 4224, err));
    check(copies == before);
    check(!prefill::Prefill::run_dp(paths, tokens.data(), tokens.size(), 257, 32768, 4224, err));
    check(copies == before);
}
'''

host_prefix = r'''
using namespace strata::core;
struct Cursor {
    uint8_t* p; uint64_t used = 0;
    template<class T> T* take(uint64_t n) {
        T* r = (T*) (p + used); used = (used + n * sizeof(T) + 15) & ~15ull; return r;
    }
};
uint64_t g_kv_host_bytes = 0;
uint8_t cpu_pool[8192], device_pool[4][8192];
int allocations = 0; bool fail_alloc = false, fail_map = false;
uint8_t* kv_host_alloc(uint64_t) { ++allocations; return fail_alloc ? nullptr : cpu_pool; }
int cudaHostGetDevicePointer(void** d, void* h, int) {
    check(h == cpu_pool); if (fail_map) return 1;
    *d = device_pool[device]; return 0;
}
uint64_t bind_host(QsaState& st, const QsaState* share_host) {
    struct { int mode = 1; int64_t pages = 4; } p;
    const auto s = kernels::Shapes{};
    const int64_t max_cells = st.max_cells, pages = p.pages;
    const uint64_t q4_row = 36;
'''

host_suffix = r'''
    return 1;
}
core::ModelGeometry draft_geometry;
struct Draft {
    int device_ = 2; const core::ModelGeometry* g_ = &draft_geometry; std::string rt_dir_ = "draft";
    uint8_t *dense_ = cpu_pool, *dense4_ = cpu_pool + 128, *experts_ = cpu_pool + 256;
    std::vector<int> tensors_{1};
    std::vector<std::pair<std::string, uint64_t>> q4_off_{{"q", 64}};
    bool owns_weights_ = true, owns_draft_head_ = true;
    bool borrow(const Draft* shared, std::string& err) {
        const int device_ = this->device_; const auto& g = *g_; const std::string rt_dir = rt_dir_;
'''

end = r'''
        return true;
    }
    ~Draft() { FREE_Q4 }
};
void ownership_test() {
    QsaState owner, borrower;
    device = 0; check(bind_host(owner, nullptr));
    check(allocations == 1 && owner.kv_host_base == cpu_pool);
    check((void*) owner.host.k_q != cpu_pool);
    const auto bytes = g_kv_host_bytes;
    for (device = 1; device < 4; ++device) {
        check(bind_host(borrower, &owner));
        check(borrower.kv_host_base == cpu_pool && borrower.host.k_q == (int8_t*) device_pool[device]);
        check(allocations == 1 && g_kv_host_bytes == bytes);
    }
    device = 0; fail_map = true; check(!bind_host(borrower, &owner)); fail_map = false;
    borrower.max_cells = 1; check(!bind_host(borrower, &owner)); borrower.max_cells = owner.max_cells;
    fail_alloc = true; check(!bind_host(borrower, nullptr)); fail_alloc = false;
    {
        Draft a;
        { Draft b; b.dense4_ = nullptr; b.q4_off_.clear(); std::string err;
          check(b.borrow(&a, err)); check(b.dense4_ == a.dense4_ && b.q4_off_ == a.q4_off_); }
        check(frees == 0);
    }
    check(frees == 1);
    // Partial/full DP cleanup frees staging and arena only, never the owner's host KV or RoPE.
    core::SessionState state_value;
    auto* state = &state_value;
    state->qsa_states = new QsaState[4];
    state->qsa_alloc = 4;
    for (int i = 0; i < 4; ++i) {
        state->qsa_states[i].host_step = (int32_t*) cpu_pool;
        state->qsa_states[i].kv_host_base = cpu_pool;
    }
    int dev = 0; void* p = cpu_pool;
    CLEANUP
    check(frees == 6 && state->qsa_states == nullptr && rope_releases == 0);
    // A later-stage RoPE owner is indexed by its global ordinal, not local row zero.
    core::SessionState later;
    later.qsa_states = new QsaState[4]; later.qsa_ord0 = 2; later.qsa_alloc = 1;
    later.qsa_states[2].owns_rope = true;
    core::session_release(later);
    check(rope_releases == 1 && !later.qsa_states[2].owns_rope);
    delete[] later.qsa_states;
}
int main() {
    ownership_test();
    scheduler_test(4, 0); scheduler_test(4, 1); scheduler_test(4, 2);
    check(live_events == 0);
    event_count = 0; fail_event = 3;
    std::string err;
    check(!prefill::Prefill::make_dp_link(8, {0, 1, 3, 2}, err));
    check(live_events == 0);
    std::puts("dp_port_test: fake-runtime ownership, aliases, DP4, indexer, failure and cleanup passed");
}
'''

# Exercise the actual group row constructor and verdict with four fake stages/slots.
verdict = block(gen, "auto pmtp_verdict =") + ";"
group = block(gen, "if (pmtp) {   // the group's slots:")
group = group[group.index("{") + 1:-1]
pmtp_fixture = r'''
constexpr int cudaMemcpyDeviceToDevice = 0;
int cudaMemcpyAsync(void* d, const void* s, size_t n, int, cudaStream_t) { std::memcpy(d, s, n); return 0; }
bool nonfinite = false;
namespace strata::core {
struct Verifier {
    int kept[4] = {}, stream_device = 2;
    float residual[8] = {};
    bool batch_commit_rows(const int*, int, int, const int* keep, const int32_t*, const int64_t*, std::string&) {
        for (int b = 0; b < 4; ++b) kept[b] = keep[b];
        return true;
    }
    bool nonfinite_in(int r) const { return nonfinite && r == 0; }
    const float* final_R_all() const { return residual; }
    cudaStream_t stream() { return &stream_device; }
};
}
struct SlotDraft {
    int stream_device = 2, calls = 0;
    int device() const { return 2; }
    cudaStream_t stream() { return &stream_device; }
    bool chain_launch(int, const int32_t*, int64_t, int, const int32_t*, int, int, int, std::string&) { ++calls; return true; }
};
struct TestSlot {
    bool active = true, stop = false, cached = false, draft_ready = true, img = false, tainted = false;
    int64_t p = 10, produced = 1, max_new = 10;
    int32_t x = 7;
    int n_drafts = 1;
    int32_t draft[2] = {8, 9};
    std::vector<int32_t> ids;
    std::vector<int> checks;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
};
struct TestGroup { int S = 0, active = 0, hbase = 0; bool inflight = false; int rows[8] = {}; int32_t tok[8] = {}; int64_t pos[8] = {}; };
void pmtp_test() {
    using Clock = std::chrono::steady_clock;
    using BSlot = TestSlot;
    using PGroup = TestGroup;
    const int n_pipe = 4, pm_drafts = 1, GS = 1, pmtp_rows_per_group = 2;
    struct { int hc = 1, n_embd = 1; } g;
    struct { std::vector<int64_t> eos_ids{2}; int prompt_cache = 1; int64_t max_context = 100000; } o;
    core::Verifier stages[4];
    auto stage_verifier = [&](int k) -> core::Verifier& { return stages[k]; };
    std::vector<TestSlot> bs(4);
    std::vector<std::unique_ptr<SlotDraft>> slot_mtp;
    std::vector<std::unique_ptr<float[]>> slot_mtp_rows;
    std::vector<cudaEvent_t> pm_copied;
    Event ev[4] = {{2}, {2}, {2}, {2}};
    std::vector<char> pm_drafting(4, 0);
    for (int b = 0; b < 4; ++b) {
        slot_mtp.push_back(std::make_unique<SlotDraft>());
        slot_mtp_rows.push_back(std::make_unique<float[]>(8));
        pm_copied.push_back(&ev[b]);
    }
    int64_t bt_accepted = 0, bt_tokens = 0, bt_rows = 0, pipe_rows = 0;
    bool nan_flush = false;
    std::string err;
    VERDICT
    auto make_group = [&](int pick) {
        PGroup G;
        for (int once = 0; once < 1; ++once) { GROUP }
        return G;
    };
    for (int b = 0; b < 4; ++b) {
        bs[b] = TestSlot{};
        auto G = make_group(b);
        check(G.S == 2 && G.rows[0] == b && G.rows[1] == b && G.hbase == 2 * b);
        const int32_t accept[2] = {8, 9};
        check(pmtp_verdict(G, stages[3], accept));
        for (auto& st : stages) check(st.kept[b] == 2);
        check(bs[b].produced == 3 && bs[b].ids.size() == 2 && slot_mtp[b]->calls == 1);
    }
    bs[0] = TestSlot{};
    auto G = make_group(0);
    const int32_t reject[2] = {6, 9};
    check(pmtp_verdict(G, stages[3], reject));
    check(stages[0].kept[0] == 1 && bs[0].ids.size() == 1 && bs[0].x == 6);
    bs[0] = TestSlot{};
    G = make_group(0);
    const int32_t eos[2] = {2, 9};
    check(pmtp_verdict(G, stages[3], eos));
    check(stages[0].kept[0] == 1 && !bs[0].active && bs[0].ids.size() == 1);
    bs[0] = TestSlot{}; bs[0].max_new = 2;
    G = make_group(0);
    check(G.S == 1); // No un-emittable speculative input is committed at the output cap.
    check(pmtp_verdict(G, stages[3], reject));
    check(stages[0].kept[0] == 1 && !bs[0].active && bs[0].produced == 2);
    bs[0] = TestSlot{}; G = make_group(0); bs[0].stop = true;
    check(pmtp_verdict(G, stages[3], reject));
    check(stages[0].kept[0] == 1 && !bs[0].active);
    bs[0] = TestSlot{}; G = make_group(0); nonfinite = true;
    check(pmtp_verdict(G, stages[3], reject));
    check(!bs[0].active && !bs[0].cached && bs[0].ids.empty() && nan_flush);
    nonfinite = false;
}
'''
end = end.replace('int main() {', pmtp_fixture.replace('VERDICT', verdict).replace('GROUP', group) + '\nint main() {\n    pmtp_test();')

output = prefix + link + "\n" + methods + suffix
output += "namespace strata::core {\n" + release + "\n}\n"
output += host_prefix + host + host_suffix + shared
output += end.replace("FREE_Q4", free_q4).replace("CLEANUP", cleanup)
if len(sys.argv) != 2:
    raise SystemExit("usage: dp_port_test.py OUTPUT.cpp")
pathlib.Path(sys.argv[1]).write_text(output)
