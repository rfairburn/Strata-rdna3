#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>

namespace strata::core {

// Snapshot-owned host storage. Appending never relocates existing payloads;
// only the small segment directory can move. No sharing or copy-on-write is
// needed because a parked image and its retained active buffers have one owner.
class ConversationBuffer {
public:
    static constexpr size_t segment_bytes = 16 * 1024 * 1024;
    ConversationBuffer() = default;
    ConversationBuffer(std::initializer_list<uint8_t> values) {
        resize(values.size());
        visit(0, size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, values.begin() + at, n);
            return true;
        });
    }
    ConversationBuffer(const ConversationBuffer&) = default;
    ConversationBuffer& operator=(const ConversationBuffer&) = default;
    ConversationBuffer(ConversationBuffer&& other) noexcept
        : segments_(std::move(other.segments_)), size_(std::exchange(other.size_, 0)),
          overwrite_target_(std::exchange(other.overwrite_target_, SIZE_MAX)),
          overwrite_exact_(std::exchange(other.overwrite_exact_, false)) {}
    ConversationBuffer& operator=(ConversationBuffer&& other) noexcept {
        if (this == &other) return *this;
        segments_ = std::move(other.segments_);
        size_ = std::exchange(other.size_, 0);
        overwrite_target_ = std::exchange(other.overwrite_target_, SIZE_MAX);
        overwrite_exact_ = std::exchange(other.overwrite_exact_, false);
        return *this;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_t bytes() const {
        size_t n = segments_.capacity() * sizeof(Segment);
        for (const auto& segment : segments_) n += segment.capacity();
        return n;
    }
    // Includes the old directory while reserve allocates its replacement.
    // Saturation makes an overflowing estimate fail admission before resize.
    size_t allocation_peak(size_t n) const {
        size_t total = bytes();
        if (n <= size_) return total;
        const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
        const size_t extra = n - size_ - std::min(n - size_, spare);
        const size_t full = extra / segment_bytes, tail = extra % segment_bytes;
        if (!add(total, full * segment_bytes)) return SIZE_MAX;
        if (tail && !add(total, segment_capacity(tail, full ? segment_bytes :
                segments_.empty() ? 0 : segments_.back().capacity(), size_ != 0, size_))) return SIZE_MAX;
        size_t count = segments_.size();
        if (!add(count, full + (tail != 0))) return SIZE_MAX;
        if (count > segments_.capacity() &&
            (count > SIZE_MAX / sizeof(Segment) || !add(total, count * sizeof(Segment)))) return SIZE_MAX;
        return total;
    }
    // Retired scratch only: exact-fit buffers keep ALL allocations (including
    // tails). Otherwise keep only exact full allocations, never mismatched tails
    // or spare capacity. Prepared storage is NOT valid snapshot content.
    struct OverwritePlan {
        size_t kept = 0, kept_full = 0, released = 0, directory_allocation = 0;
        size_t fresh = 0, preparation_peak = 0;
        bool fits = false, exact = false;
    };
    OverwritePlan overwrite_plan(size_t n) const {
        OverwritePlan p;
        const size_t count = n / segment_bytes + (n % segment_bytes != 0);
        if (count > SIZE_MAX / sizeof(Segment)) return p;
        p.fresh = n;
        if (!add(p.fresh, count * sizeof(Segment))) return p;
        if (size_ == n && segments_.capacity() == count && bytes() == p.fresh) {
            p.kept = n;
            for (const auto& s : segments_)
                if (s.size() == segment_bytes && s.capacity() == segment_bytes) p.kept_full += segment_bytes;
            p.preparation_peak = p.fresh;
            p.fits = p.exact = true;
            return p;
        }
        size_t payload = 0;
        for (const auto& s : segments_) {
            if (!add(payload, s.capacity())) return p;
            if (s.size() == segment_bytes && s.capacity() == segment_bytes &&
                p.kept / segment_bytes < n / segment_bytes) p.kept += segment_bytes;
        }
        p.kept_full = p.kept;
        p.released = payload - p.kept;
        const size_t olddir = segments_.capacity() * sizeof(Segment);
        const size_t newdir = count * sizeof(Segment);
        // When nothing survives, release the old directory before reserving.
        p.directory_allocation = segments_.capacity() == count ? 0 : newdir;
        p.preparation_peak = p.kept;
        if (!add(p.preparation_peak, p.kept ? olddir : 0) ||
            !add(p.preparation_peak, p.kept && !p.directory_allocation ? 0 : newdir)) return p;
        // Exact-multiple shrink with a larger old directory can conservatively
        // miss. Never hide a transient directory peak behind a capped estimate.
        p.fits = p.preparation_peak <= p.fresh;
        return p;
    }
    bool prepare_overwrite(size_t n) {
        const auto p = overwrite_plan(n);
        if (!p.fits) return false;
        if (p.exact) {
            overwrite_target_ = n;
            overwrite_exact_ = true;
            return true;
        }
        overwrite_exact_ = false;
        const size_t count = n / segment_bytes + (n % segment_bytes != 0);
        size_t kept = 0;
        for (size_t i = 0; i < segments_.size(); ++i) {
            if (kept == p.kept / segment_bytes) break;
            if (segments_[i].size() != segment_bytes || segments_[i].capacity() != segment_bytes) continue;
            if (kept != i) segments_[kept] = std::move(segments_[i]);
            ++kept;
        }
        segments_.resize(kept); // drop old partial tails/excess before directory allocation
        size_ = p.kept;
        if (segments_.capacity() != count) {
            if (!kept) std::vector<Segment>().swap(segments_);
            std::vector<Segment> directory;
            directory.reserve(count); // failure before any retained payload moves
            if (directory.capacity() != count) return false;
            for (auto& s : segments_) directory.push_back(std::move(s));
            segments_.swap(directory);
        }
        overwrite_target_ = n;
        return true;
    }
    bool prepared_for(size_t n) const {
        if (overwrite_exact_) {
            const size_t count = n / segment_bytes + (n % segment_bytes != 0);
            size_t fresh = n;
            return overwrite_target_ == n && size_ == n && count <= SIZE_MAX / sizeof(Segment) &&
                add(fresh, count * sizeof(Segment)) && segments_.capacity() == count && bytes() == fresh;
        }
        if (overwrite_target_ != n || size_ > n || size_ % segment_bytes ||
            segments_.size() != size_ / segment_bytes ||
            segments_.capacity() != n / segment_bytes + (n % segment_bytes != 0)) return false;
        for (const auto& s : segments_)
            if (s.size() != segment_bytes || s.capacity() != segment_bytes) return false;
        return true;
    }
    size_t overwrite_allocation_peak(size_t n) const {
        if (!prepared_for(n)) return SIZE_MAX;
        size_t total = n;
        return add(total, segments_.capacity() * sizeof(Segment)) ? total : SIZE_MAX;
    }
    bool resize_overwrite(size_t n) {
        if (!prepared_for(n)) return false;
        while (size_ < n) {
            const size_t count = std::min(segment_bytes, n - size_);
            Segment s;
            s.reserve(count); // exact growth, not ordinary append's geometry
            if (s.capacity() != count) return false;
            s.resize(count);
            segments_.push_back(std::move(s));
            size_ += count;
        }
        overwrite_target_ = SIZE_MAX;
        overwrite_exact_ = false;
        return true;
    }
    void resize(size_t n, uint8_t value = 0) {
        overwrite_target_ = SIZE_MAX;
        overwrite_exact_ = false;
        if (n > size_) {
            const size_t held = size_;
            const bool growing = held != 0;
            const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
            const size_t extend = std::min(n - size_, spare);
            const size_t extra = n - size_ - extend;
            segments_.reserve(segments_.size() + extra / segment_bytes + (extra % segment_bytes != 0));
            if (extend) {
                segments_.back().resize(segments_.back().size() + extend, value);
                size_ += extend;
            }
            while (size_ < n) {
                const size_t count = std::min(segment_bytes, n - size_);
                const size_t previous = segments_.empty() ? 0 : segments_.back().capacity();
                Segment segment;
                segment.reserve(segment_capacity(count, previous, growing, held));
                segment.resize(count, value);
                segments_.push_back(std::move(segment));
                size_ += count;
            }
        } else {
            while (!segments_.empty() && size_ - segments_.back().size() >= n) {
                size_ -= segments_.back().size();
                segments_.pop_back();
            }
            if (size_ > n) segments_.back().resize(segments_.back().size() - (size_ - n));
            size_ = n;
        }
    }
    void pop_back() { resize(size_ - 1); }

    // fn(pointer, count, absolute_offset); false aborts a transfer immediately.
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) {
        return visit_impl(*this, offset, count, fn);
    }
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) const {
        return visit_impl(*this, offset, count, fn);
    }
    bool read(void* target, size_t offset, size_t count) const {
        return visit(offset, count, [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(target) + at - offset, p, n);
            return true;
        });
    }
    bool operator==(const ConversationBuffer& other) const {
        if (size_ != other.size_) return false;
        size_t a = 0, b = 0, x = 0, y = 0;
        while (a < segments_.size() && b < other.segments_.size()) {
            const auto& left = segments_[a];
            const auto& right = other.segments_[b];
            const size_t n = std::min(left.size() - x, right.size() - y);
            if (std::memcmp(left.data() + x, right.data() + y, n)) return false;
            x += n; y += n;
            if (x == left.size()) { ++a; x = 0; }
            if (y == right.size()) { ++b; y = 0; }
        }
        return true;
    }

private:
    using Segment = std::vector<uint8_t>;
    // A fresh capture allocates exactly its payload. Later appends reserve
    // geometrically growing segments, bounded by 16 MiB. Small chat turns then
    // extend the final allocation instead of adding one transfer per turn.
    static size_t segment_capacity(size_t count, size_t previous, bool growing, size_t held) {
        if (!growing) return count;
        const size_t geometric = std::min(segment_bytes, std::max<size_t>(65536, std::min(previous, segment_bytes/2)*2));
        return std::max(count, std::min(geometric, std::max<size_t>(65536, held / 8)));
    }
    static bool add(size_t& n, size_t extra) {
        if (extra > SIZE_MAX - n) return false;
        n += extra;
        return true;
    }
    template<class Self, class Fn> static bool visit_impl(Self& self, size_t offset, size_t count, Fn fn) {
        if (offset > self.size_ || count > self.size_ - offset) return false;
        size_t begin = 0;
        for (auto& segment : self.segments_) {
            if (!count) break;
            const size_t end = begin + segment.size();
            if (offset < end) {
                const size_t n = std::min(count, end - offset);
                if (!fn(segment.data() + offset - begin, n, offset)) return false;
                offset += n; count -= n;
            }
            begin = end;
        }
        return count == 0;
    }
    std::vector<Segment> segments_;
    size_t size_ = 0;
    size_t overwrite_target_ = SIZE_MAX;
    bool overwrite_exact_ = false;
};

} // namespace strata::core
