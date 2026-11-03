// CPU policy/storage test only: this does not exercise native snapshot transfers.
#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_memory.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

static bool fail_alloc = false;
void* operator new(std::size_t n) {
    if (fail_alloc) throw std::bad_alloc();
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
using namespace strata::core;
namespace {
int checks = 0;
size_t payload_reused = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
ConversationCheckpoint cp(int32_t branch) {
    ConversationCheckpoint c; c.ids = {1, branch}; return c;
}
auto parts(ConversationKv& k) {
    return std::array<ConversationBuffer*, 5>{&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
}
SavedConversation image(int32_t branch, size_t stages = 0) {
    SavedConversation s;
    s.live.ids = {1, branch, 99}; s.live.gdn = {7, 8};
    s.checkpoints = {cp(branch)};
    s.kv.resize(2);
    for (auto& k : s.kv) {
        k.cells = 4; k.format = 1;
        for (auto* b : parts(k)) b->resize(4096, 0xa7);
    }
    for (size_t k = 0; k < stages; ++k) s.stage_images.push_back(image(branch));
    return s;
}
void populate(ConversationCache& c, size_t stages) {
    check(c.put(image(10, stages)) && c.put(image(11, stages)), "park two eligible victims");
    check(c.put(image(12)), "park sibling branch");
    auto no_checks = image(20); no_checks.checkpoints.clear();
    check(c.put(std::move(no_checks)), "park image without checkpoints");
    auto steering = image(10); steering.cvec = false;
    check(c.put(std::move(steering)), "park other cvec");
    auto picture = image(10); picture.checkpoints[0].imgs = {{0, 55}};
    check(c.put(std::move(picture)), "park different image identity");
    auto pin = image(13); pin.checkpoints[0].pinned = true;
    check(c.put(std::move(pin)), "park useful pinned branch");
    auto empty = image(30); empty.checkpoints[0].ids.clear();
    check(c.put(std::move(empty)), "park empty deepest checkpoint");
}
void same_cache(const ConversationCache& a, const ConversationCache& b) {
    check(a.size() == b.size() && a.bytes() == b.bytes() && a.superseded() == b.superseded() &&
          a.evictions() == b.evictions(), "baseline/optional-output logical accounting identical");
    for (int branch : {10, 11, 12, 13, 20, 30}) {
        const std::vector<int32_t> prompt{1, branch, 99, 100};
        const auto x = a.best(prompt, {}, true), y = b.best(prompt, {}, true);
        check(x.index == y.index && x.tokens == y.tokens && x.live == y.live,
              "surviving matches and LRU indices identical");
    }
}
void overwrite(ConversationKvReuse& r) {
    check(r.unchanged_tokens == 0 && r.ids.empty() && r.ids.capacity() == 0,
          "retired identity is absent and unchanged prefix is zero");
    // Equal-size resize plus a CPU full-write surrogate must allocate nothing.
    // Actual native keep=0/transfer behavior is a separate pending native test.
    for (auto& layer : r.kv) for (auto* b : parts(layer)) {
        const size_t before = b->bytes();
        uint8_t* original = nullptr;
        b->visit(0, b->size(), [&](uint8_t* p, size_t n, size_t) {
            original = p;
            for (size_t i = 0; i < n; ++i) check(p[i] == 0xa7, "moving does not zero live payload");
            return true;
        });
        fail_alloc = true;
        b->resize(b->size());
        b->visit(0, b->size(), [&](uint8_t* p, size_t n, size_t) {
            check(p == original, "equal-size destination retains its physical allocation");
            std::memset(p, 0x3c, n); return true;
        });
        fail_alloc = false;
        check(b->bytes() == before, "equal-size reuse preserves capacity/directory charge");
        b->visit(0, b->size(), [&](const uint8_t* p, size_t n, size_t) {
            for (size_t i = 0; i < n; ++i) check(p[i] == 0x3c, "all stale bytes overwritten by CPU surrogate");
            payload_reused += n; return true;
        });
    }
    for (auto& s : r.stages) overwrite(s);
}
}
int main() {
    const std::vector<int32_t> live{1, 10, 99}; // identical IDs are deliberately not content proof
    const std::vector<ConversationCheckpoint> chain{cp(10), cp(11)};
    for (size_t stages = 0; stages <= 4; ++stages) {
        ConversationCache baseline(8 << 20, 16), candidate(8 << 20, 16);
        populate(baseline, stages); populate(candidate, stages);
        // Snapshot the first victim's actual addresses before the cache move.
        // take/reinsert would alter LRU; use a separate fixture for pointer proof below.
        const size_t before = baseline.bytes();
        SavedConversation retired;
        check(baseline.drop_superseded(live, {}, chain, true) == 2 &&
              candidate.drop_superseded(live, {}, chain, true, &retired) == 2,
              "both paths drop exactly the same two eligible images");
        same_cache(baseline, candidate);
        check(before - baseline.bytes() == 2 * retired.bytes(), "bytes measured before moving victim");
        check(retired.live.ids == live && retired.stage_images.size() == stages,
              "first already-superseded victim selected, all stage counts supported");
        uint8_t* pointer = nullptr;
        retired.kv[0].k.visit(0, 1, [&](uint8_t* p, size_t, size_t) { pointer = p; return true; });
        const size_t capacity = retired.kv[0].k.bytes();
        auto r = conversation_retired_kv(std::move(retired), stages);
        r.kv[0].k.visit(0, 1, [&](uint8_t* p, size_t, size_t) {
            check(p == pointer && r.kv[0].k.bytes() == capacity, "conversion moves payload without allocation"); return true;
        });
        check(r.captured_tokens == 3 && r.stages.size() == stages, "extent and stages retained for prevalidation");
        overwrite(r);
        check(candidate.drop_superseded(live, {}, chain, true) == 0 && candidate.superseded() == 2,
              "later publication does not double-count extracted victim");
        same_cache(baseline, candidate);
        // Same make_room input yields identical LRU policy after dropping.
        check(baseline.make_room(4 << 20) == candidate.make_room(4 << 20), "admission policy unchanged");
        same_cache(baseline, candidate);
    }
    {
        ConversationCache c(1 << 20, 8);
        auto victim = image(10);
        uint8_t* ptr = nullptr;
        victim.kv[0].k.visit(0, 1, [&](uint8_t* p, size_t, size_t) { ptr = p; return true; });
        check(c.put(std::move(victim)), "park pointer fixture");
        SavedConversation retired;
        fail_alloc = true;
        const size_t n = c.drop_superseded(live, {}, chain, true, &retired);
        fail_alloc = false;
        check(n == 1 && c.bytes() == 0, "drop/extraction allocates nothing and uncharges all bytes");
        retired.kv[0].k.visit(0, 1, [&](uint8_t* p, size_t, size_t) {
            check(p == ptr, "cache-to-retired move retains actual allocation address"); return true;
        });
    }
    {
        auto wrong = conversation_retired_kv(image(10, 2), 3);
        check(wrong.kv.empty() && wrong.stages.empty(), "stage-count mismatch releases whole candidate");
        auto torn = image(10, 2); torn.stage_images[1].live.ids.pop_back();
        check(conversation_retired_kv(std::move(torn), 2).kv.empty(), "torn stage extent falls back fresh");
        ConversationCache c(1 << 20, 4);
        auto layers = image(10).kv;
        c.retain(std::move(layers), 3, {}, live);
        auto ordinary = c.take_reuse_for(live);
        auto retired = image(10, 2);
        // Exact park_source preference: do not convert retired storage when ordinary reuse exists.
        const bool storage_only = ordinary.kv.empty();
        if (storage_only) ordinary = conversation_retired_kv(std::move(retired), 2);
        retired = {};
        check(!storage_only && ordinary.unchanged_tokens == 3 && ordinary.stages.empty() && retired.bytes() == 0,
              "ordinary identity-based valid prefix wins and unused retired storage is released");
    }
    {
        // Failure after baseline's complete logical drop, not halfway through it.
        ConversationCache baseline(1 << 20, 8), candidate(1 << 20, 8);
        baseline.put(image(10, 2)); candidate.put(image(10, 2));
        baseline.put(image(11)); candidate.put(image(11));
        SavedConversation retired;
        baseline.drop_superseded(live, {}, chain, true);
        candidate.drop_superseded(live, {}, chain, true, &retired);
        auto active_checks = chain;
        bool failed = false;
        fail_alloc = true;
        try { auto r = conversation_retired_kv(std::move(retired), 2); (void) r; }
        catch (const std::bad_alloc&) { failed = true; }
        fail_alloc = false;
        check(failed && active_checks[0].ids == chain[0].ids && active_checks[1].ids == chain[1].ids,
              "new stage-directory allocation failure leaves active metadata untouched");
        same_cache(baseline, candidate);
        check(candidate.size() == 0 && candidate.superseded() == 2, "failure publishes no partial image");
    }
    {
        ConversationCache c(1 << 20, 4);
        ConversationCheckpoint root; root.ids = {1};
        auto richer = image(10); richer.checkpoints.insert(richer.checkpoints.begin(), root);
        c.put(std::move(richer));
        check(c.supersedes_richer(live, {}, {cp(10)}, true), "richer-chain guard still rejects poorer slot chain");
        check(!c.supersedes_richer(live, {}, {root, cp(10)}, true), "complete ancestry remains acceptable");
        SavedConversation unused;
        check(c.drop_superseded({1, 12, 99}, {}, {cp(12)}, true, &unused) == 0 && unused.bytes() == 0,
              "sibling does not supply storage or lose richer ancestry");
    }
    {
        ConversationBuffer b; b.resize(4096, 0xa7);
        const size_t retained = b.bytes();
        const size_t peak = b.allocation_peak(4096);
        ConversationBuffer fresh; const size_t fresh_peak = fresh.allocation_peak(4096);
        check(peak == retained && peak == fresh_peak, "equal-size storage includes physical payload and directory");
        const size_t metadata = 512, floor = 1000;
        check(conversation_memory_admit(metadata + floor, metadata + peak - retained, floor),
              "RAM gate subtracts only already-owned physical bytes from complete estimate");
        check(!conversation_memory_admit(metadata + floor - 1, metadata + peak - retained, floor),
              "RAM floor remains exact and fails closed below boundary");
        check(b.allocation_peak(2000) > fresh.allocation_peak(2000),
              "shrink capacity exceeding fresh peak is rejected by storage-only gate");
        check(b.allocation_peak(4097) > fresh.allocation_peak(4097),
              "geometric growth exceeding fresh peak falls back rather than extra eviction");
        b.resize(2000); b.resize(4096);
        b.visit(2000, 2096, [&](const uint8_t* p, size_t n, size_t) {
            for (size_t i = 0; i < n; ++i) check(p[i] == 0, "regrowth never exposes stale suffix bytes"); return true;
        });
        const size_t grown_peak = b.allocation_peak(8192);
        b.resize(8192);
        check(b.bytes() <= grown_peak, "growth payload and directory remain within peak accounting");
        b.visit(0, b.size(), [](uint8_t* p, size_t n, size_t) { std::memset(p, 0x5d, n); return true; });
        b.visit(0, b.size(), [&](const uint8_t* p, size_t n, size_t) {
            for (size_t i = 0; i < n; ++i) check(p[i] == 0x5d, "full CPU overwrite covers shrink/regrowth result"); return true;
        });
    }
    for (size_t incoming : {size_t(2000), size_t(4097)}) for (bool pinned : {false, true}) {
        // Near-budget grow/shrink: use the same peak-vs-fresh guard as park_source,
        // with both evictable and non-evictable useful siblings, plus an in-flight image.
        ConversationBuffer retired; retired.resize(4096, 7);
        ConversationBuffer fresh;
        const size_t fresh_peak = fresh.allocation_peak(incoming);
        const size_t recycled_peak = retired.allocation_peak(incoming);
        check(recycled_peak > fresh_peak, "near-budget fixture actually exercises worse reuse peak");
        auto sibling = image(12); sibling.checkpoints[0].pinned = pinned;
        const size_t metadata = 512, held = 123, floor = 1000;
        const size_t budget = sibling.bytes() + metadata + fresh_peak + held;
        ConversationCache baseline(budget, 4), candidate(budget, 4), unsafe(budget, 4);
        baseline.put(SavedConversation(sibling)); candidate.put(SavedConversation(sibling));
        unsafe.put(std::move(sibling));
        size_t estimate = metadata + recycled_peak;
        if (recycled_peak > fresh_peak) { retired = {}; estimate = metadata + fresh_peak; }
        check(retired.bytes() == 0, "worse retired capacity released BEFORE budget and RAM admission");
        check(baseline.make_room(metadata + fresh_peak, held) && candidate.make_room(estimate, held),
              "near-budget fresh fallback causes no extra refusal");
        check(candidate.size() == 1 && candidate.evictions() == 0,
              "near-budget fresh fallback preserves useful sibling, including pin");
        same_cache(baseline, candidate);
        const bool unsafe_fit = unsafe.make_room(metadata + recycled_peak, held);
        check(pinned ? !unsafe_fit : (!unsafe_fit || unsafe.evictions() != 0),
              "without fallback inflated reuse really would cause refusal or useful-image eviction");
        check(conversation_memory_admit(metadata + fresh_peak + floor, estimate, floor),
              "released fallback uses full fresh additional RAM, not stale retained credit");
        check(!conversation_memory_admit(metadata + fresh_peak + floor - 1, estimate, floor),
              "near-budget fallback does not bypass RAM floor");
    }
    std::printf("conversation_storage_reuse_test: %d checks passed; %zu live payload bytes physically reused/full-written (CPU surrogate)\n",
                checks, payload_reused);
}
