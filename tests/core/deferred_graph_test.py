"""Emit a CPU fake-runtime regression for the production deferred graph lifecycle.

Eviction, limit trimming, allocation retry, launch and commit methods are extracted
verbatim. Capture/math and CUDA calls are fakes, not GPU or native ABI validation.
"""
from pathlib import Path
import sys


def block(text, anchor):
    start = text.index(anchor)
    first = text.index("{", start)
    depth, end = 1, first + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


root = Path(__file__).resolve().parents[2]
source = (root / "src/core/verify.cpp").read_text()
header = (root / "include/strata/core/verify.hpp").read_text()
assert "if (!trim_batch_graphs(key, err)) return false;" in block(source, "bool Verifier::stage_batch(")
assert "instantiate_evicting(ex, graph, key, what, err)" in block(source, "bool Verifier::capture_deferred(")
assert "std::map<std::vector<int>, std::vector<int>> pending_bm_;" in header
methods = "\n\n".join(block(source, anchor) for anchor in (
    "bool Verifier::batch_graph_pending(",
    "std::set<std::vector<int>> Verifier::batch_graph_keys(",
    "bool Verifier::trim_batch_graphs(",
    "bool Verifier::evict_batch_graph(",
    "bool Verifier::instantiate_evicting(",
    "bool Verifier::batch_launch_rows(",
    "bool Verifier::batch_commit_rows(",
))

prefix = r'''
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

void check(bool ok) {
    if (!ok) { std::fputs("deferred_graph_test failed\n", stderr); std::abort(); }
}
struct FakeGraphExec;
using cudaGraphExec_t = FakeGraphExec*;
using cudaGraph_t = int;
using cudaStream_t = int;
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaErrorMemoryAllocation = 1, cudaErrorUnknown = 2;
std::set<cudaGraphExec_t> live;
std::vector<cudaGraphExec_t> queued;
int next_graph = 1, budget = 1024, destroyed = 0, synchronizations = 0;
int launch_attempts = 0, fail_launch_at = -1;
bool fail_sync = false;
int cudaGraphInstantiate(cudaGraphExec_t* ex, int, int) {
    if ((int) live.size() >= budget) return cudaErrorMemoryAllocation;
    *ex = reinterpret_cast<cudaGraphExec_t>(static_cast<uintptr_t>(next_graph++));
    live.insert(*ex);
    return cudaSuccess;
}
int cudaGraphExecDestroy(cudaGraphExec_t ex) {
    // Queued commits must finish before their graph is reclaimed.
    check(std::find(queued.begin(), queued.end(), ex) == queued.end());
    check(live.erase(ex) == 1);
    ++destroyed;
    return cudaSuccess;
}
int cudaGraphLaunch(cudaGraphExec_t ex, int) {
    check(live.count(ex) == 1);
    if (++launch_attempts == fail_launch_at) return cudaErrorUnknown;
    queued.push_back(ex);
    return cudaSuccess;
}
int cudaStreamSynchronize(int) {
    ++synchronizations;
    if (fail_sync) return cudaErrorUnknown;
    queued.clear();
    return cudaSuccess;
}
int cudaStreamQuery(int) { return cudaSuccess; }
int cudaGetLastError() { return cudaSuccess; }
const char* cudaGetErrorString(int) { return "fake allocation or launch error"; }

namespace strata::core {
struct OnDevice { explicit OnDevice(int) {} };
struct SessionState { int32_t ple_prev[2] = {-1, -1}; };
class Verifier {
public:
    std::map<std::vector<int>, cudaGraphExec_t> exec_bm_, commit_bm_, save_bm_, dcommit_bm_;
    std::map<std::vector<int>, uint64_t> bm_used_;
    std::map<std::vector<int>, std::vector<int>> pending_bm_;
    uint64_t bm_tick_ = 0;
    size_t batch_graph_limit_ = 0;
    int device_ = 0, dc_rows_ = 128, max_t_ = 8;
    int64_t lb_ = 0, le_ = 8, b_k_ = 0, b_steps_ = 0;
    bool ar_off_ = false, staging_deferred_ = false;
    std::atomic<bool> b_running_{false};
    cudaStream_t cs_ = 0;
    std::function<bool(cudaStream_t, std::string&)> pre_launch_;
    SessionState slot;
    std::vector<SessionState*> slots_{&slot};
    int32_t commit_storage[32] = {};
    int32_t* h_commitd_ = commit_storage;
    static std::vector<int> batch_key(const int* rows, int S, int hbase) {
        std::vector<int> key{hbase};
        for (int i = 0; i < S; ++i) key.push_back(rows[i]);
        return key;
    }
    std::vector<int> bkey(const int* rows, int S, int hbase) const {
        auto key = batch_key(rows, S, hbase);
        key.push_back(ar_off_ ? 1 : 0);
        return key;
    }
    bool ple_stage() const { return false; }
    void ring_now() {}
    bool batch_graph_pending(const std::vector<int>&) const;
    std::set<std::vector<int>> batch_graph_keys() const;
    bool trim_batch_graphs(const std::vector<int>&, std::string&);
    bool evict_batch_graph(const std::vector<int>&, bool&, std::string&);
    bool instantiate_evicting(cudaGraphExec_t&, cudaGraph_t, const std::vector<int>&, const char*, std::string&);
    bool batch_launch_rows(const int*, int, int, const int32_t*, const int64_t*, std::string&);
    bool batch_commit_rows(const int*, int, int, const int*, const int32_t*, const int64_t*, std::string&);
    bool make(std::map<std::vector<int>, cudaGraphExec_t>& graphs, const std::vector<int>& key, std::string& err) {
        auto& ex = graphs[key];
        return ex != 0 || instantiate_evicting(ex, 1, key, "fake capture: ", err);
    }
    bool stage_batch(const int* rows, int S, int base, const int32_t*, const int64_t*, std::string& err) {
        check(staging_deferred_);
        auto key = bkey(rows, S, base);
        if (!trim_batch_graphs(key, err)) return false;
        bm_used_[key] = ++bm_tick_;
        return make(exec_bm_, key, err);
    }
    bool capture_deferred(const int* rows, int S, int base, std::string& err) {
        check(!staging_deferred_);
        auto key = bkey(rows, S, base);
        return make(save_bm_, key, err) && make(dcommit_bm_, key, err);
    }
    ~Verifier() {
        check(cudaStreamSynchronize(cs_) == cudaSuccess);
        for (auto* graphs : {&exec_bm_, &commit_bm_, &save_bm_, &dcommit_bm_})
            for (auto& graph : *graphs) if (graph.second) cudaGraphExecDestroy(graph.second);
    }
};
'''

suffix = r'''
} // namespace strata::core
using strata::core::Verifier;
const int rows[1] = {0}, keep[1] = {1};
const int32_t tokens[1] = {7};
const int64_t positions[1] = {0};
std::vector<int> key(int base, bool variant = false) { return {base, 0, variant ? 1 : 0}; }
bool launch(Verifier& v, int base, std::string& err) {
    return v.batch_launch_rows(rows, 1, base, tokens, positions, err);
}
void finish_window(Verifier& v) {
    check(cudaStreamSynchronize(v.cs_) == cudaSuccess);
    v.b_running_ = false; // fake batch_poll completion; the verdict is still pending
}
void commit(Verifier& v, int base, std::string& err) {
    check(v.batch_commit_rows(rows, 1, base, keep, tokens, positions, err));
}
bool absent(Verifier& v, const std::vector<int>& k) {
    return !v.exec_bm_.count(k) && !v.commit_bm_.count(k) && !v.save_bm_.count(k) &&
           !v.dcommit_bm_.count(k) && !v.bm_used_.count(k);
}
void churn() {
    Verifier v;
    v.batch_graph_limit_ = 2;
    std::string err;
    for (int i = 0; i < 64; ++i) {
        check(launch(v, i, err));
        // Mixed normal/deferred variants of a completed layout must all be reclaimed.
        check(v.make(v.commit_bm_, key(i), err));
        finish_window(v);
        commit(v, i, err);
        check(v.pending_bm_.empty());
        check(v.batch_graph_keys().size() <= 2 && live.size() <= 8);
        if (i > 1) check(absent(v, key(i - 2)));
    }
    check(destroyed >= 62 * 4);
}
void pending_pressure() {
    Verifier v;
    v.batch_graph_limit_ = 1;
    std::string err;
    check(launch(v, 0, err));
    finish_window(v);
    const auto saved = v.save_bm_.at(key(0)), deferred = v.dcommit_bm_.at(key(0));
    // A pending window cannot be launched twice, even after its compute finished.
    check(!launch(v, 0, err));
    check(launch(v, 1, err));
    finish_window(v);
    check(v.batch_graph_keys().size() == 2); // only the bounded pinned working set exceeds the limit
    check(live.count(saved) && live.count(deferred));
    commit(v, 1, err);
    check(launch(v, 2, err));
    check(absent(v, key(1)) && live.count(deferred));
    finish_window(v);
    commit(v, 2, err);
    // Another group's admission may have changed the doorbell variant. Commit the captured one.
    v.ar_off_ = true;
    commit(v, 0, err);
    check(std::find(queued.begin(), queued.end(), deferred) != queued.end());
    check(launch(v, 3, err)); // eviction synchronizes queued commits before destroying their graphs
    check(absent(v, key(0)) && absent(v, key(2)));
    check(v.batch_graph_keys().size() == 1);
    finish_window(v);
    commit(v, 3, err);
}
void allocation_pressure() {
    Verifier v;
    std::string err;
    budget = 8;
    check(launch(v, 10, err));
    finish_window(v); // stays pinned
    const auto pending = v.dcommit_bm_.at(key(10));
    check(launch(v, 11, err));
    finish_window(v);
    commit(v, 11, err);
    // Window+save fit; deferred-commit instantiate hits OOM and must reclaim all eligible variants.
    check(launch(v, 12, err));
    check(absent(v, key(11)) && live.count(pending));
    finish_window(v);
    commit(v, 12, err);
    commit(v, 10, err);
    budget = 1024;
}
void partial_and_failure() {
    Verifier v;
    std::string err;
    budget = 2;
    check(!launch(v, 20, err)); // failed capture retains only its own partial graphs, no pending window
    check(v.pending_bm_.empty() && live.size() == 2);
    budget = 3;
    check(launch(v, 21, err));
    check(absent(v, key(20)) && live.size() == 3);
    finish_window(v);
    fail_launch_at = launch_attempts + 1;
    check(!v.batch_commit_rows(rows, 1, 21, keep, tokens, positions, err));
    check(v.batch_graph_pending(key(21))); // failed commit must not unpin the verdict
    fail_launch_at = -1;
    commit(v, 21, err);
    check(v.pending_bm_.empty());
    budget = 1024;
    check(v.make(v.save_bm_, key(22), err)); // orphan/partial deferred-only entries are also eligible
    bool evicted = false;
    check(v.evict_batch_graph(key(21), evicted, err) && evicted && absent(v, key(22)));
    fail_sync = true;
    const auto before = live;
    check(!v.evict_batch_graph(key(99), evicted, err) && live == before);
    fail_sync = false;
}
int main() {
    churn(); check(live.empty());
    pending_pressure(); check(live.empty());
    allocation_pressure(); check(live.empty());
    partial_and_failure(); check(live.empty());
    std::puts("deferred_graph_test: churn, OOM reclamation, pending verdicts and queued-commit safety passed");
}
'''

if len(sys.argv) != 2:
    raise SystemExit("usage: deferred_graph_test.py OUTPUT.cpp")
Path(sys.argv[1]).write_text(prefix + methods + suffix)
