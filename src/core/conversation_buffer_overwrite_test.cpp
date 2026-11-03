// CPU-only real 16 MiB allocation tests; no accelerator headers or runtime.
#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_memory.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <limits>

static long fail_after = -1;
void* operator new(size_t n) {
    if (fail_after == 0) throw std::bad_alloc();
    if (fail_after > 0) --fail_after;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
using namespace strata::core;
static int checks = 0;
static void check(bool b, const char* s) {
    ++checks; if (!b) { std::fprintf(stderr,"FAIL: %s\n",s); std::exit(1); }
}
int main() {
    constexpr size_t M = ConversationBuffer::segment_bytes;
    size_t total_kept = 0, total_released = 0, total_new = 0;
    for (size_t old : {size_t(0), size_t(17), M-1, M, M+113, 2*M, 2*M+113, 3*M+113})
    for (size_t target : {size_t(0), size_t(17), size_t(19), M-1, M, M+111, M+113, 2*M, 2*M+113, 2*M+117, 3*M+113, 3*M+117}) {
        ConversationBuffer b; b.resize(old, 0xa7);
        std::vector<const uint8_t*> full, all;
        b.visit(0,b.size(),[&](const uint8_t* p,size_t n,size_t){ all.push_back(p); if(n==M) full.push_back(p); return true; });
        const auto plan = b.overwrite_plan(target);
        const auto fresh = ConversationBuffer{}.allocation_peak(target);
        check(plan.fresh == fresh, "fresh plan equals normal fresh capacity");
        if (!plan.fits) {
            check(plan.preparation_peak > fresh, "only explicit transient directory excess falls back");
            const auto before = b.bytes(); check(!b.prepare_overwrite(target) && before==b.bytes(), "fallback leaves scratch alone");
            continue;
        }
        check(b.prepare_overwrite(target) && b.prepared_for(target), "prepare valid destination extent");
        check(b.size()==plan.kept && b.overwrite_allocation_peak(target)==fresh, "prepared peak equals fresh, no geometric reserve");
        check(b.bytes() <= fresh && plan.preparation_peak <= fresh, "all true capacities and directory peaks counted");
        size_t at = 0;
        b.visit(0,b.size(),[&](const uint8_t* p,size_t n,size_t) {
            check(p==(plan.exact ? all[at] : full[at]) && (plan.exact || n==M), "exact-fit allocations or original full segments survive without relocation");
            ++at;
            check(p[0]==0xa7 && p[n-1]==0xa7, "preparation never zeros retained payload");
            return true;
        });
        const size_t held = b.bytes();
        check(b.resize_overwrite(target), "exact resize after simulated admission");
        check(b.bytes()==fresh && !b.prepared_for(target), "published buffer capacity exactly fresh");
        size_t segment_index=0;
        b.visit(0,b.size(),[&](uint8_t* p,size_t n,size_t offset) {
            if (offset < plan.kept) check(p==(plan.exact ? all[segment_index] : full[offset/M]) && p[0]==0xa7, "exact growth does not touch retained bytes");
            else for(size_t i=0;i<n;++i) if(p[i]) { check(false,"new missing tails/segments zero initialized"); break; }
            ++segment_index;
            std::memset(p,0x3c,n); return true;
        });
        total_kept += plan.kept; total_released += plan.released; total_new += fresh-held;
        // Near budget with held incoming image, pinned and unpinned siblings.
        for (bool pinned : {false,true}) {
            SavedConversation sibling; ConversationCheckpoint cp; cp.ids={1}; cp.pinned=pinned;
            sibling.checkpoints.push_back(cp); sibling.live.ids={1,2};
            const size_t held_incoming=123, metadata=512, floor=1000;
            ConversationCache base(sibling.bytes()+fresh+metadata+held_incoming,4), candidate(sibling.bytes()+fresh+metadata+held_incoming,4);
            base.put(SavedConversation(sibling)); candidate.put(std::move(sibling));
            check(base.make_room(fresh+metadata,held_incoming) && candidate.make_room(fresh+metadata,held_incoming), "no larger budget requirement");
            check(base.size()==candidate.size() && candidate.evictions()==0, "no useful sibling eviction including protected branches");
            const size_t additional=fresh+metadata-held;
            check(conversation_memory_admit(additional+floor,additional,floor), "exact RAM boundary admits only actual retained capacity credit");
            check(!conversation_memory_admit(additional+floor-1,additional,floor), "one byte below floor rejects");
        }
        std::printf("case old=%zu target=%zu kept=%zu kept_full=%zu released_payload=%zu new_allocation=%zu directory_allocation=%zu fresh=%zu preparation_peak=%zu exact=%d\n",
                    old,target,plan.kept,plan.kept_full,plan.released,fresh-held,plan.directory_allocation,fresh,plan.preparation_peak,plan.exact);
    }
    {
        ConversationBuffer b; b.resize(M+113,0xa7);
        // target has a different directory count, but enough missing payload to
        // cover the small old+new directory peak.
        auto p=b.overwrite_plan(2*M+117); check(p.fits && p.directory_allocation,"directory-failure fixture");
        bool failed=false; fail_after=0;
        try { b.prepare_overwrite(2*M+117); } catch(const std::bad_alloc&) { failed=true; }
        fail_after=-1;
        check(failed && b.size()==M,"reserve fails before retained payload moves; retired scratch still cleanable");
        check(b.prepare_overwrite(2*M+117),"prepare cleanable retired scratch after directory fault");
        const size_t before=b.bytes(); failed=false; fail_after=0;
        try { b.resize_overwrite(2*M+117); } catch(const std::bad_alloc&) { failed=true; }
        fail_after=-1;
        check(failed && before==b.bytes() && b.prepared_for(2*M+117),"new payload OOM cannot expose stale suffix or publish");
    }
    {
        ConversationBuffer b; b.resize(4096,7); auto peak=b.allocation_peak(8192); b.resize(8192);
        check(b.bytes()<=peak && b.bytes()>ConversationBuffer{}.allocation_peak(8192),"ordinary resize still uses original geometric growth");
        b.resize(2000); b.resize(4096);
        b.visit(2000,2096,[&](const uint8_t* p,size_t n,size_t){for(size_t i=0;i<n;++i) check(p[i]==0,"normal shrink/regrowth zeroes suffix"); return true;});
        check(!b.overwrite_plan(SIZE_MAX).fits,"overflow fails closed");
    }
    std::printf("conversation_buffer_overwrite_test: %d checks passed; aggregate sequential kept=%zu released=%zu newly_allocated=%zu (not simultaneous)\n",checks,total_kept,total_released,total_new);
}
