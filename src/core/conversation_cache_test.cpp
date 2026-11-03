#include "strata/core/conversation_cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
SavedConversation image(std::initializer_list<int32_t> ids, bool cvec = true) {
    SavedConversation s;
    s.live.ids = ids;
    s.live.gdn.resize(64, 7);
    s.cvec = cvec;
    return s;
}
}

int main() {
    {
        ConversationBuffer bytes;
        const size_t first = ConversationBuffer::segment_bytes + 17;
        const size_t peak = bytes.allocation_peak(first);
        bytes.resize(first, 7);
        check(bytes.bytes() <= peak, "segmented payload and directory fit admitted bytes");
        uint8_t* original = nullptr;
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) { original = p; return true; });
        const size_t grown_peak = bytes.allocation_peak(first + 99);
        bytes.resize(first + 99, 9);
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) {
            check(p == original, "appending preserves existing payload addresses"); return true;
        });
        check(bytes.bytes() <= grown_peak, "growth including transient directory fits admission");
        std::vector<uint8_t> tail(116);
        check(bytes.read(tail.data(), ConversationBuffer::segment_bytes, tail.size()), "read spans segment boundaries");
        check(std::all_of(tail.begin(), tail.begin()+17, [](auto v){return v==7;}) &&
              std::all_of(tail.begin()+17, tail.end(), [](auto v){return v==9;}), "growth preserves prefix and initializes only suffix");
        auto copy = bytes;
        bytes.resize(ConversationBuffer::segment_bytes + 10);
        bytes.resize(first, 7); bytes.resize(first + 99, 9);
        check(bytes == copy, "equality ignores differing segmentation after rewind and regrowth");
        check(!bytes.read(tail.data(), bytes.size()-1, 2), "range check rejects a truncated payload");
        check(bytes.allocation_peak(SIZE_MAX) == SIZE_MAX, "allocation estimate rejects overflow");
    }
    {
        ConversationBuffer bytes;
        for (size_t n=1;n<=4096;++n) {
            const size_t peak = bytes.allocation_peak(n*17);
            bytes.resize(n*17,7);
            check(bytes.bytes() <= peak,"small append stays within the predicted allocation peak");
        }
        size_t segments = 0;
        bytes.visit(0,bytes.size(),[&](const uint8_t*,size_t,size_t){++segments;return true;});
        check(segments <= 4,"thousands of small turns do not create thousands of restore transfers");
    }
    {
        ConversationKv layer;
        layer.k.resize(400);
        std::vector<ConversationKv> layers;
        layers.push_back(std::move(layer));
        const size_t retained = layers.capacity()*sizeof(ConversationKv) + layers[0].bytes();
        const size_t parked = image({1,2,3}).bytes();
        ConversationCache cache(retained + parked, 4);
        check(cache.put(image({1,2,3})), "park before retaining active storage");
        cache.retain(std::move(layers), 16);
        check(cache.bytes() == retained + parked && cache.size() == 1, "active retained storage consumes bytes but no parked slot");
        cache.limit_reuse(9); cache.limit_reuse(12);
        auto reuse = cache.take_reuse();
        check(reuse.unchanged_tokens == 9, "a later continuation cannot undo a rewind's dirty boundary");
        check(cache.bytes() == parked, "taking retained buffers releases their budget accounting");
        cache.retain(std::move(reuse.kv), 16);
        check(!cache.can_fit(parked),"retained storage is counted when checking a capture without eviction");
        check(cache.retained_bytes() == retained && cache.size() == 1 && cache.evictions() == 0,
              "optional reuse admission does not evict or release anything");
        check(!cache.can_fit(SIZE_MAX) && !cache.can_fit(1,SIZE_MAX),"non-mutating reservation rejects overflow");
        check(cache.make_room(parked), "reservation can discard optional active buffers");
        check(cache.retained_bytes() == 0 && cache.size() == 1 && cache.evictions() == 0,
              "pressure drops retained storage before evicting parked conversations");
    }
    {
        ConversationBuffer bytes;
        const size_t held = ConversationBuffer::segment_bytes;
        bytes.resize(held, 7);
        const size_t peak = bytes.allocation_peak(held + 1);
        bytes.resize(held + 1, 9);
        check(bytes.bytes() <= peak, "bounded reserve is included in the capture estimate");
        check(bytes.bytes() <= held + held / 8 + 1024, "a tiny append does not reserve another full segment");
        uint8_t ends[2] = {};
        check(bytes.read(ends, held - 1, 2) && ends[0] == 7 && ends[1] == 9,
              "bounded reserve preserves the old prefix and new suffix");
    }
    {
        auto layer = [] { ConversationKv k; k.k.resize(64, 7); return k; };
        ConversationCache cache(8192, 4);
        std::vector<ConversationKv> layers{layer()};
        std::vector<std::vector<ConversationKv>> stages{{layer()}};
        cache.retain(std::move(layers), 4, std::move(stages), {1, 2, 3, 4});
        auto partial = cache.take_reuse_for(std::vector<int32_t>{1, 2, 9, 8});
        check(partial.unchanged_tokens == 2 && partial.stages[0].unchanged_tokens == 2,
              "retained main and stage K/V stop at the actual conversation prefix");
        check(partial.ids.empty() && cache.retained_bytes() == 0, "identity metadata is released before capture admission");
        layers = {layer()};
        cache.retain(std::move(layers), 4, {}, {1, 2, 3, 4});
        check(cache.take_reuse_for(std::vector<int32_t>{9, 8, 7}).kv.empty(),
              "a batch switch cannot reuse another conversation's K/V");
        layers = {layer()};
        cache.retain(std::move(layers), 4, {}, {1, 2, 3, 4});
        cache.limit_reuse(2);
        check(cache.take_reuse_for(std::vector<int32_t>{1, 2, 3, 4, 5}).unchanged_tokens == 2,
              "identity matching does not undo a prior rewrite boundary");
        layers = {layer()};
        cache.retain(std::move(layers), 4);
        check(cache.take_reuse_for(std::vector<int32_t>{1, 2, 3, 4}).kv.empty(),
              "retained storage without a conversation identity is not reused");
        layers = {layer()};
        cache.retain(std::move(layers), 4, {}, {1, 2, 3, 4});
        cache.take_reuse();
        check(cache.take_reuse_for(std::vector<int32_t>{1, 2, 3, 4}).kv.empty(),
              "slot replacement discards retained storage even for identical token IDs");
    }
    {
        ConversationCheckpoint root, chosen, wrong, longer, picture;
        root.ids = {1}; root.pinned = true; root.stage_parts.resize(3); root.stage_parts[2].gdn = {7};
        chosen.ids = {1, 2}; wrong.ids = {9}; longer.ids = {1, 2, 3};
        picture.ids = {1}; picture.imgs = {{0, 99}};
        std::vector<ConversationCheckpoint> source{root, chosen, wrong, longer, picture}, restored;
        check(conversation_slot_checkpoints(restored, source, chosen.ids, &source[1]), "copy a slot's checkpoint chain");
        check(restored.size() == 2 && restored[0].ids == chosen.ids && restored[1].ids == root.ids,
              "chosen checkpoint is first and only compatible shorter checkpoints follow");
        check(restored[1].pinned && restored[1].stage_parts[2].gdn == std::vector<uint8_t>{7},
              "slot return preserves pinned and layer-split checkpoint state");
        restored.clear();
        check(conversation_slot_checkpoints(restored, source, std::vector<int32_t>{1, 2, 4}),
              "returning a live slot also restores its branch points");
        check(restored.size() == 2, "wrong branches, image checkpoints and longer states are excluded");
    }
    {
        ConversationCache cache(64 * 1024 * 1024, 1000);
        std::vector<std::vector<int32_t>> ids(4);
        std::vector<std::vector<ConversationCheckpoint>> slots(4);
        for (int s = 0; s < 4; ++s) ids[s] = {100 + s};
        for (int turn = 0; turn < 64; ++turn) for (int s = 0; s < 4; ++s) {
            std::vector<ConversationCheckpoint> chain;
            check(conversation_slot_checkpoints(chain, slots[s], ids[s]), "carry checkpoint chain on repeated slot return");
            ConversationCheckpoint next; next.ids = ids[s]; next.gdn.resize(32, 7);
            chain.push_back(next);
            ids[s].push_back(1000 + turn); ids[s].push_back(2000 + turn);
            SavedConversation saved; saved.live.ids = ids[s]; saved.checkpoints = chain;
            saved.kv.resize(1); saved.kv[0].k.resize(16384, 7);
            check(cache.put(std::move(saved)), "park an advancing interleaved conversation");
            slots[s] = {next};
            check(cache.size() <= 4, "successive turns replace obsolete images rather than accumulating full copies");
        }
        check(cache.size() == 4 && cache.superseded() == 252 && cache.evictions() == 0,
              "four working conversations remain bounded without LRU eviction");
    }
    const std::vector<int64_t> a = {1, 2, 3, 4}, b = {9, 8, 7, 6};
    {
        ConversationCache cache(1024, 2);
        check(cache.put(image({1, 2, 3})), "park A");
        check(cache.put(image({9, 8, 7})), "park B");
        auto match = cache.best(a, {}, true);
        check(match.tokens == 3 && match.live, "A/B/A: recover A");
        auto restored = cache.take(match.index);
        check(restored.live.ids == std::vector<int32_t>({1, 2, 3}), "taking selected A preserves identity");
        check(cache.size() == 1 && cache.best(b, {}, true).tokens == 3, "B remains parked");
        check(cache.bytes() == image({9,8,7}).bytes(), "byte accounting after take");
        check(cache.put(std::move(restored)), "park returned A as newest");
        check(cache.put(image({5, 6})), "evict oldest by slot limit");
        check(cache.best(b, {}, true).tokens == 0 && cache.best(a, {}, true).tokens == 3, "B evicted before A");
        check(cache.evictions() == 1, "eviction counter");
    }
    {
        // evict_oldest(): the physical-RAM admission gate in generate.cpp frees parked conversations
        // one at a time and re-checks the host's free memory, so this has to be exactly the oldest-first
        // step make_room() takes - same accounting, same counters - and it has to stop at empty.
        ConversationCache cache(4096, 4);
        cache.put(image({1, 2, 3}));
        cache.put(image({9, 8, 7}));
        check(cache.size() == 2 && cache.slots() == 4, "slots() reports the configured limit");
        const size_t both = cache.bytes();
        check(cache.evict_oldest(), "evict_oldest reports an eviction");
        check(cache.size() == 1 && cache.evictions() == 1, "evict_oldest drops exactly one, oldest first");
        check(cache.best(a, {}, true).tokens == 0 && cache.best(b, {}, true).tokens == 3,
              "the oldest conversation went, the newest stayed");
        check(cache.bytes() == both - image({1, 2, 3}).bytes(), "evict_oldest releases the entry's bytes");
        cache.evict_oldest();
        check(cache.size() == 0 && cache.bytes() == 0, "evicting the last parked conversation empties the cache");
        check(!cache.evict_oldest(), "the gate's loop can reach an empty cache: false");
        check(cache.size() == 0 && cache.evictions() == 2, "evict_oldest on an empty cache is a no-op");
    }
    {   // the RAM gate's eviction skips a conversation that holds a pinned shared prefix, and says so (false)
        ConversationCache cache(1 << 20, 4);
        auto pinned = image({1, 2, 3});
        pinned.checkpoints.push_back(ConversationCheckpoint{});
        pinned.checkpoints.back().ids = {1, 2};
        pinned.checkpoints.back().pinned = true;
        pinned.checkpoints.back().gdn.resize(64, 1);
        cache.put(std::move(pinned));
        cache.put(image({9, 8, 7}));
        check(cache.evict_oldest() && cache.size() == 1, "the oldest unpinned conversation went (the pinned one is older)");
        check(!cache.evict_oldest() && cache.size() == 1, "only a pinned one is left: nothing more can go");
        // #879: after non-finite logits every parked conversation goes, the pinned one too
        cache.put(image({4, 5, 6}));
        std::vector<ConversationKv> kv(1);
        kv[0].cells = 3;
        kv[0].k.resize(64, 1);
        cache.retain(std::move(kv), 3);
        check(cache.retained_bytes() > 0, "a retained K/V before clear()");
        check(cache.clear() == 2, "clear() reports how many were parked");
        check(cache.size() == 0 && cache.bytes() == 0 && cache.retained_bytes() == 0, "clear() leaves nothing parked");
        check(cache.best(std::vector<int64_t>{1, 2, 3, 4}, {}, true).tokens == 0, "a cleared cache matches nothing");
        check(cache.clear() == 0, "clear() on an empty cache");
    }
    {
        auto s = image({1, 2, 3});
        ConversationCheckpoint cp;
        cp.ids = {1, 2};
        s.checkpoints.push_back(cp);
        ConversationCache cache(4096, 4);
        cache.put(std::move(s));
        auto match = cache.best(std::vector<int64_t>{1, 2, 9, 4}, {}, true);
        check(match.tokens == 2 && !match.live, "edited suffix falls back to parked checkpoint");
        check(cache.best(a, {}, true).tokens == 3, "live prefix beats shorter checkpoint");
        check(cache.best(a, {}, false).tokens == 0, "steering mode is isolated");
        check(cache.best(std::vector<int64_t>{1, 2}, {}, true).tokens == 0, "equal-length checkpoint cannot consume last token");
        check(cache.best(std::vector<int64_t>{1}, {}, true).tokens == 0, "short prompt cannot match");
        check(cache.best(std::vector<int64_t>{}, {}, true).tokens == 0, "empty prompt cannot match");
        cache.put(image({1, 2, 3}));
        check(cache.best(a, {}, true).index == 1, "ties prefer most recently parked");
    }
    {
        auto s = image({1, 2, 3});
        s.live.imgs = {{1, 123}};
        ConversationCache cache(1024, 3);
        cache.put(std::move(s));
        check(cache.best(a, {{1,123}}, true).tokens == 3, "same image can resume");
        check(cache.best(a, {{1,124}}, true).tokens == 0, "different image pixels invalidate same pad tokens");
        check(cache.best(a, {}, true).tokens == 0, "missing image invalidates prefix");
        check(cache.best(a, {{1,123},{2,45}}, true).tokens == 0, "additional image in cached prefix invalidates");
        check(cache.best(a, {{1,123},{3,45}}, true).tokens == 3, "image after cached prefix does not invalidate");
    }
    {
        const size_t one = image({1,2,3}).bytes();
        ConversationCache cache(one*2, 8);
        cache.put(image({1,2,3})); cache.put(image({9,8,7}));
        check(cache.bytes() == one*2, "budget holds two exact-sized images");
        auto held = cache.take(cache.best(a, {}, true).index);
        check(cache.make_room(one, held.bytes()), "count in-flight image when reserving outgoing snapshot");
        check(cache.size() == 0, "in-flight reservation evicts otherwise fitting B");
        check(cache.put(image({5,6,7}), held.bytes()), "insert with in-flight accounting");
        check(cache.bytes()+held.bytes() <= one*2, "exchange obeys byte budget");
        check(!cache.make_room(one+1, one), "oversized exchange rejected");
        check(cache.size() == 1, "oversized snapshot does not evict useful entries");
        auto huge = image({1}); huge.live.gdn.resize(one*3);
        check(!cache.put(std::move(huge)), "oversized image rejected");
        check(cache.size() == 1, "oversized put leaves cache unchanged");
        check(!cache.make_room(0, one*2+1), "held larger than budget cannot underflow");
    }
    {
        // #342: MAIN -> SUB (9 turns) -> MAIN with 4 slots.  Every SUB turn parks the previous turn's live state
        // (its reply as generated, which the next request re-rendered) with a chain that holds that turn's
        // boundary checkpoint; the copy a turn back adds only that stale tail and is dropped, so MAIN survives.
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        auto with = [](std::vector<int32_t> prefix, std::initializer_list<int32_t> more) {
            prefix.insert(prefix.end(), more); return prefix;
        };
        const std::vector<int32_t> root = {1, 2, 3, 4};
        SavedConversation main = image({});
        main.live.ids = with(with(root, {10, 11, 12}), {13, 14});
        main.checkpoints = {cp(root), cp(with(root, {10, 11, 12}))};
        const size_t big = 1 << 20;
        ConversationCache cache(big, 4);
        check(cache.put(std::move(main)), "park MAIN");
        std::vector<int32_t> history = with(root, {20});         // SUB's conversation so far, re-rendered
        std::vector<ConversationCheckpoint> chain = {cp(root)};
        for (int turn = 1; turn <= 9; ++turn) {
            chain.push_back(cp(history));                        // the turn boundary the next request resumes from
            SavedConversation sub = image({});
            sub.live.ids = with(history, {900, (int32_t) turn});   // + the reply with its thinking (stale tail)
            sub.checkpoints = chain;
            check(cache.put(std::move(sub)), "park SUB turn");
            check(cache.size() <= 2, "one parked copy of SUB at a time");
            history = with(history, {30, (int32_t) turn});       // the reply as the next request renders it
        }
        check(cache.evictions() == 0 && cache.superseded() == 8, "8 superseded copies dropped, nothing evicted");
        const auto m = cache.best(with(with(root, {10, 11, 12}), {13, 14, 15}), {}, true);
        check(m.tokens == 9 && m.live, "MAIN still restores in full");
        const auto s = cache.best(with(history, {40}), {}, true);
        check(s.tokens == (int64_t) history.size() - 2, "SUB resumes from its last turn boundary");
    }
    {
        // what is NOT superseded: another conversation sharing only the root, an entry without checkpoints, the
        // other steering mode, a branch whose deepest checkpoint the new chain does not hold
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        ConversationCache cache(1 << 20, 8);
        SavedConversation other = image({1, 2, 3, 4, 50, 51, 52});
        other.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 50, 51})};
        cache.put(std::move(other));
        cache.put(image({1, 2, 3, 4, 7, 7}));                    // no checkpoints
        SavedConversation steered = image({1, 2, 3, 4, 60, 61}, false);
        steered.checkpoints = {cp({1, 2, 3, 4, 60})};
        cache.put(std::move(steered));
        SavedConversation branch = image({1, 2, 3, 4, 60, 70, 71});
        branch.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60, 70})};
        cache.put(std::move(branch));
        SavedConversation incoming = image({1, 2, 3, 4, 60, 80, 81});
        incoming.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        check(cache.put(std::move(incoming)), "park a conversation sharing roots with all of them");
        check(cache.size() == 5 && cache.superseded() == 0, "none of them is superseded");
        SavedConversation same_state = image({1, 2, 3, 4, 60, 70});   // live equal to the branch's deepest point
        same_state.checkpoints = {cp({1, 2, 3, 4})};
        check(cache.put(std::move(same_state)) && cache.superseded() == 1 && cache.size() == 5,
              "the live state equal to an entry's deepest checkpoint supersedes it");
        SavedConversation huge = image({1, 2, 3, 4, 60, 80});
        huge.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        huge.live.gdn.resize(2 << 20);
        check(!cache.put(std::move(huge)) && cache.size() == 5 && cache.superseded() == 1,
              "an oversized put drops nothing");
    }
    {
        ConversationCache disabled(0,4), no_slots(1024,0);
        check(!disabled.enabled() && !no_slots.enabled(), "both disable switches");
        check(!disabled.put(image({1,2,3})) && !no_slots.put(image({1,2,3})), "disabled cache stores nothing");
        check(disabled.best(a,{},true).tokens == 0, "disabled cache has no matches");
    }
    {
        // --conversation-cache-min-tokens: a conversation shorter than the minimum is not worth one of the slots.
        // Parking is a switch-time decision, so without this a client's short side requests park themselves and,
        // once the cache is full, evict the long conversation they were interleaved with (oldest first).
        ConversationCache any(1024, 4, 0), floor(1024, 4, 12288);
        check(any.enabled() && floor.enabled(), "the minimum does not disable the cache");
        check(any.wants(0) && any.wants(1) && any.wants(1 << 20), "0 parks every length, as before the flag");
        check(!floor.wants(0) && !floor.wants(1) && !floor.wants(12287), "below the minimum is not parked");
        check(floor.wants(12288) && floor.wants(12289), "the minimum itself and above are parked");
        // the CLI rejects a negative, but the predicate must not invert for one that reaches it another way
        ConversationCache negative(1024, 4, -1);
        check(negative.wants(0) && negative.wants(1), "a negative minimum parks everything, like 0");
        // wants() answers for the length alone; the gate is enabled() && wants(), so a disabled cache parks
        // nothing whatever the length - keep the two independent rather than folding one into the other
        ConversationCache disabled_min(0, 4, 12288);
        check(!disabled_min.enabled() && disabled_min.wants(1 << 20), "a disabled cache still answers for length");
    }
    {
        // layer-split parking: checkpoints moved apart into stage parts and put back, no running state copied
        auto running = [](std::initializer_list<int32_t> ids, uint8_t salt, size_t parts) {
            ConversationCheckpoint c;
            c.ids = ids;
            c.gdn.assign(256, salt);
            c.tails.assign(32, uint8_t(salt + 1));
            c.used = salt;
            for (size_t k = 0; k < parts; ++k) {
                ConversationCheckpoint p;
                p.gdn.assign(128, uint8_t(salt + 10 + k));
                c.stage_parts.push_back(std::move(p));
            }
            return c;
        };
        std::vector<ConversationCheckpoint> checks;
        checks.push_back(running({1, 2}, 3, 2));
        checks.push_back(running({1, 2, 3}, 5, 1));       // an incomplete set: kept aside, put back in place
        checks.push_back(running({1, 2, 3, 4}, 7, 2));
        const auto original = checks;
        const uint8_t* gdn0 = checks[0].gdn.data();
        const uint8_t* part1 = checks[2].stage_parts[1].gdn.data();
        auto split = conversation_checkpoints_split(std::move(checks), 2);
        check(checks.empty() && split.stage0.size() == 2 && split.rest.size() == 1 && split.parts.size() == 2 &&
              split.parts[0].size() == 2 && split.parts[1].size() == 2, "split: complete checkpoints apart, the other aside");
        check(split.stage0[0].stage_parts.empty() && split.parts[1][1].ids == original[2].ids &&
              split.parts[1][1].used == original[2].used && split.parts[0][0].gdn == original[0].stage_parts[0].gdn,
              "split: every stage part carries its checkpoint's ids and LRU stamp");
        check(split.stage0[0].gdn.data() == gdn0 && split.parts[1][1].gdn.data() == part1,
              "split moves the running state (same buffers, no copy)");
        std::vector<ConversationCheckpoint> back;
        check(conversation_checkpoints_merge(std::move(split), back) && back.size() == 3, "merge puts them back");
        bool same = true;
        for (size_t j = 0; j < back.size(); ++j) {
            same = same && back[j].ids == original[j].ids && back[j].gdn == original[j].gdn &&
                   back[j].tails == original[j].tails && back[j].used == original[j].used &&
                   back[j].stage_parts.size() == original[j].stage_parts.size();
            for (size_t k = 0; same && k < back[j].stage_parts.size(); ++k)
                same = back[j].stage_parts[k].gdn == original[j].stage_parts[k].gdn;
        }
        check(same, "merge(split(x)) == x, in the original order");
        check(back[0].gdn.data() == gdn0 && back[2].stage_parts[1].gdn.data() == part1, "merge moves them back too");

        auto torn = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2);
        torn.parts[1].pop_back();
        check(!conversation_checkpoints_merge(std::move(torn), back) && back.empty(), "a stage short of a part is refused");
        auto mixed = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2);
        mixed.parts[0][0].ids = {9, 9};
        check(!conversation_checkpoints_merge(std::move(mixed), back) && back.empty(), "a part of another checkpoint is refused");
        // a restored image: stage 0's checkpoints and each stage's, with no record of an incomplete one
        ConversationCheckpointSplit restored;
        restored.stage0.push_back(running({4, 5}, 11, 0));
        restored.parts.resize(2);
        for (auto& p : restored.parts) { auto c = running({4, 5}, 12, 0); p.push_back(std::move(c)); }
        check(conversation_checkpoints_merge(std::move(restored), back) && back.size() == 1 &&
              back[0].stage_parts.size() == 2, "a restored image's parts are all whole");
        auto none = conversation_checkpoints_split({}, 3);
        check(conversation_checkpoints_merge(std::move(none), back) && back.empty(), "no checkpoints: nothing to do");
    }
    {
        // the later stages' retained K/V, kept with the first stage's and limited together
        auto kv = [](size_t n) {
            std::vector<ConversationKv> v(2);
            for (auto& l : v) l.k.resize(n, 1);
            return v;
        };
        ConversationCache cache(1 << 20, 4);
        std::vector<std::vector<ConversationKv>> stage_kv;
        stage_kv.push_back(kv(1000));
        stage_kv.push_back(kv(2000));
        cache.retain(kv(500), 40, std::move(stage_kv));
        check(cache.retained_bytes() >= 2 * (500 + 1000 + 2000), "retained bytes count every stage");
        cache.limit_reuse(30);
        auto r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.size() == 2 && r.unchanged_tokens == 30 &&
              r.stages[0].unchanged_tokens == 30 && r.stages[1].unchanged_tokens == 30 &&
              r.stages[1].captured_tokens == 40, "every stage's reuse is limited to the first rewrite");
        check(cache.retained_bytes() == 0, "taken once");
        std::vector<std::vector<ConversationKv>> big;
        big.push_back(kv(1 << 20));
        cache.retain(kv(500), 40, std::move(big));
        check(cache.retained_bytes() == 0, "a stage's K/V over the budget drops the whole reuse");
        std::vector<std::vector<ConversationKv>> one;
        one.push_back(kv(100));
        cache.retain(kv(500), 40, std::move(one));
        cache.limit_reuse(0);
        check(cache.retained_bytes() == 0, "a rewrite from the start drops every stage's reuse");
        cache.retain(kv(500), 40);
        r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.empty(), "no layer split: no stage reuse, as before");
    }
    {   // pin=N: a parked conversation that holds a pinned shared prefix is not the one evicted
        const size_t one = image({1,2,3}).bytes();
        auto pinned = image({1,2,3});
        pinned.checkpoints.push_back(ConversationCheckpoint{});
        pinned.checkpoints.back().ids = {1,2};
        pinned.checkpoints.back().pinned = true;
        pinned.checkpoints.back().gdn.resize(64, 1);
        const size_t pinned_bytes = pinned.bytes();
        ConversationCache big(pinned_bytes + one * 2 + 64, 3);
        check(pinned.pinned() && !image({1,2,3}).pinned(), "a conversation is pinned when one of its checkpoints is");
        check(big.put(std::move(pinned)), "the pinned conversation is parked first (the oldest)");
        check(big.put(image({7,8,9})) && big.put(image({4,5,6})) && big.size() == 3, "two more fill the slots");
        check(big.put(image({10,11,12})) && big.size() == 3 && big.evictions() == 1,
              "a fourth evicts the oldest unpinned one");
        check(big.best(std::vector<int32_t>{1,2,9,9}, {}, true).tokens == 2, "the pinned prefix is still found");
        ConversationCache tiny(pinned_bytes + 8, 2);
        auto p2 = image({1,2,3});
        p2.checkpoints.push_back(ConversationCheckpoint{});
        p2.checkpoints.back().ids = {1,2};
        p2.checkpoints.back().pinned = true;
        p2.checkpoints.back().gdn.resize(64, 1);
        check(tiny.put(std::move(p2)), "a pinned one fits an empty cache");
        check(!tiny.put(image({7,8,9})) && tiny.size() == 1, "nothing else fits beside it: parking is refused, the pin stays");
    }
    {
        auto saved = image({1, 2, 3, 4, 5});
        ConversationCheckpoint root;
        root.ids = {1, 2, 3};
        saved.checkpoints.push_back(root);
        ConversationCache cache(4096, 4);
        cache.put(std::move(saved));
        check(cache.holds(std::vector<int32_t>{1,2,3,4,5}, {}, true), "parked live state covers a slot");
        check(cache.holds(std::vector<int32_t>{1,2}, {}, true), "a parked checkpoint covers its prefix");
        check(!cache.holds(std::vector<int32_t>{1,2,3,4,5,6}, {}, true), "a longer reply still needs parking");
        check(!cache.holds(std::vector<int32_t>{1,2,9}, {}, true), "another conversation is not covered");
        check(!cache.holds(std::vector<int32_t>{}, {}, true), "empty state is not covered");
        check(!cache.holds(std::vector<int32_t>{1,2}, {}, false), "coverage keeps steering isolated");
        auto pictures = image({1,2,3,4});
        pictures.live.imgs = {{1,77},{3,78}};
        cache.put(std::move(pictures));
        check(cache.holds(std::vector<int32_t>{1,2,3}, {{1,77}}, true), "coverage checks pictures below the prefix");
        check(!cache.holds(std::vector<int32_t>{1,2,3,4}, {{1,77}}, true), "missing pictures invalidate coverage");
    }
    {
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids=std::move(ids); return c; };
        auto main_copy = image({1,2,3,4,10,11,12});
        main_copy.checkpoints = {cp({1,2,3,4}),cp({1,2,3,4,10,11})};
        ConversationCache cache(1<<20,4);
        check(cache.put(std::move(main_copy)), "park a parent with its root and turn checkpoints");
        const std::vector<int32_t> slot_ids{1,2,3,4,10,11,12,20,21};
        const std::vector<ConversationCheckpoint> slot_chain{cp({1,2,3,4,10,11})};
        check(cache.supersedes_richer(slot_ids,{},slot_chain,true), "slot parking must preserve the richer parent's root");
        check(!cache.supersedes_richer(slot_ids,{}, {cp({1,2,3,4}),cp({1,2,3,4,10,11})},true), "a complete chain loses no root");
        check(!cache.supersedes_richer(slot_ids,{},slot_chain,false), "richer-chain protection isolates steering");
        check(!cache.supersedes_richer({1,2,3,4,30,31},{},{cp({1,2,3,4,30})},true), "a sibling supersedes no parent");
        check(cache.size()==1 && cache.superseded()==0, "richer-chain inspection changes no entries");
        check(cache.best(std::vector<int32_t>{1,2,3,4,40},{},true).tokens==4, "the shared root remains reusable");
    }
    {
        // Upstream's two-slot, four-conversation reply-retention replay.
        auto replay = [&](bool slot_parks, bool checks_back) {
            std::vector<std::vector<int32_t>> full(4);
            for (int c=0;c<4;++c) {
                for (int32_t t=1;t<=8;++t) full[c].push_back(c*100+t);
                for (int32_t t:{21,22,23,9,31,32,41,42,43,44,9,51,52}) full[c].push_back(c*100+t);
            }
            auto ids = [&](int c,size_t n) { return std::vector<int32_t>(full[c].begin(),full[c].begin()+n); };
            auto cp = [&](int c,size_t n) { ConversationCheckpoint k; k.ids=ids(c,n); return k; };
            auto park = [&](ConversationCache& cache,int c,size_t n,const std::vector<size_t>& chain) {
                auto saved=image({}); saved.live.ids=ids(c,n);
                for (size_t k:chain) saved.checkpoints.push_back(cp(c,k));
                check(cache.put(std::move(saved)),"reply replay stays within its budget");
            };
            auto restore = [&](ConversationCache& cache,int c,size_t n) {
                const auto match=cache.best(ids(c,n),{},true);
                if (match.tokens>0) cache.take(match.index);
                return match.tokens;
            };
            auto back = [&](const std::vector<size_t>& chain) { return checks_back?chain:std::vector<size_t>{}; };
            auto slot_park = [&](ConversationCache& cache,int c,size_t n,const std::vector<size_t>& chain) {
                if (slot_parks && !cache.holds(ids(c,n),{},true)) park(cache,c,n,chain);
            };
            enum {A,B,C,D};
            ConversationCache cache(1<<20,4);
            park(cache,B,9,{4}); park(cache,A,8,{4});
            auto chain_a=back({4});
            park(cache,A,12,chain_a);
            auto chain_b=back({4}); chain_b.push_back(12);
            check(restore(cache,A,14)==12,"second turn resumes its parked main state");
            park(cache,B,15,chain_b); chain_a.push_back(12);
            park(cache,A,14,chain_a); slot_park(cache,B,17,{12});
            park(cache,D,9,{4}); slot_park(cache,A,17,{12});
            park(cache,C,8,{4}); auto chain_c=back({4}); chain_c.push_back(12);
            park(cache,C,15,chain_c); auto chain_d=back({4}); chain_d.push_back(12);
            const auto found_a=restore(cache,A,21);
            park(cache,D,14,chain_d); slot_park(cache,C,17,{12}); park(cache,A,22,{12,20});
            return std::pair<int64_t,int64_t>(found_a,restore(cache,B,21));
        };
        check(replay(false,false)==std::make_pair<int64_t,int64_t>(14,0),"old paths lose reply state and one conversation");
        check(replay(true,false)==std::make_pair<int64_t,int64_t>(17,0),"slot parking alone cannot replace checkpoint preservation");
        check(replay(true,true)==std::make_pair<int64_t,int64_t>(17,17),"both upstream fixes retain both completed replies");
    }
    std::printf("conversation_cache_test: %d checks passed\n", checks);
}
