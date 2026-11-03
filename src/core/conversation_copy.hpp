#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

namespace strata::core::conversation_detail {

// PRIVATE measurement overlay. Presence semantics match the existing parktrace:
// unset is off; even STRATA_PARK_TRACE=0 is on. No pointer inspection or error-state queries.
inline bool copy_trace_enabled() {
    static const bool enabled = std::getenv("STRATA_PARK_TRACE") != nullptr;
    return enabled;
}

struct CopyStamp {
    int64_t wall_ns = -1;
    int64_t cpu_ns = -1;
};
inline CopyStamp copy_stamp() noexcept {
    CopyStamp stamp;
    stamp.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
#if defined(__linux__)
    timespec cpu{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu) == 0)
        stamp.cpu_ns = int64_t(cpu.tv_sec) * 1000000000LL + cpu.tv_nsec;
#endif
    return stamp;
}
inline double copy_ms(int64_t a, int64_t b) noexcept {
    return a < 0 || b < 0 ? -1.0 : double(b - a) / 1000000.0;
}

// Labels are literals + public structural ordinals only, never raw pointers, IDs, keys or text.
// qsa is a global QSA ordinal, NOT a transformer block number; -1 means aggregate/unknown.
struct CopyPartTrace {
    bool on = copy_trace_enabled();
    const char *phase, *layer_kind, *part, *src, *dst;
    int64_t qsa;
    int src_mode, dst_mode;
    size_t chunks = 0, bytes = 0, slow_calls = 0;
    int device = -1, final_rc = 0;
    bool mixed_device = false;
    double wall[3] = {}, cpu[3] = {};
    bool cpu_known[3] = {true, true, true};
    size_t worst_chunk = 0, worst_offset = 0, worst_bytes = 0;
    int worst_device = -1;
    double worst_wall[3] = {-1, -1, -1}, worst_total = -1, worst_cpu = -1;
    CopyPartTrace(const char* ph, const char* lk, int64_t q, const char* p,
                  const char* s, const char* d, int sm, int dm)
        : phase(ph), layer_kind(lk), part(p), src(s), dst(d), qsa(q), src_mode(sm), dst_mode(dm) {}
    ~CopyPartTrace() noexcept {
        // One compact nonempty-part summary accounts for aggregate time even if every chunk is fast.
        if (!on || !chunks) return;
        std::fprintf(stderr, "strata parktrace: copy_part schema=1 phase=%s layer_kind=%s qsa=%lld part=%s src=%s dst=%s "
                             "src_mode=%d dst_mode=%d device=%d mixed_device=%d bytes=%zu chunks=%zu slow_calls=%zu final_rc=%d "
                             "acquire=%.3f ms enqueue=%.3f ms wait=%.3f ms total=%.3f ms "
                             "acquire_cpu=%.3f ms enqueue_cpu=%.3f ms wait_cpu=%.3f ms total_cpu=%.3f ms "
                             "worst_chunk=%zu worst_offset=%zu worst_bytes=%zu worst_device=%d "
                             "worst_acquire=%.3f ms worst_enqueue=%.3f ms worst_wait=%.3f ms worst_total=%.3f ms worst_cpu=%.3f ms\n",
                     phase, layer_kind, (long long) qsa, part, src, dst, src_mode, dst_mode, device, mixed_device ? 1 : 0,
                     bytes, chunks, slow_calls, final_rc, wall[0], wall[1], wall[2], wall[0] + wall[1] + wall[2],
                     cpu_known[0] ? cpu[0] : -1.0, cpu_known[1] ? cpu[1] : -1.0, cpu_known[2] ? cpu[2] : -1.0,
                     cpu_known[0] && cpu_known[1] && cpu_known[2] ? cpu[0] + cpu[1] + cpu[2] : -1.0,
                     worst_chunk, worst_offset, worst_bytes, worst_device,
                     worst_wall[0], worst_wall[1], worst_wall[2], worst_total, worst_cpu);
    }
};

struct CopyCallTrace {
    CopyPartTrace* part;
    size_t bytes, offset, chunk;
    CopyStamp stamps[4];
    int stages = 0, device = -1;
    int rc[3] = {-1, -1, -1}; // -1 = not attempted; raw runtime status otherwise
    bool created = false;
    CopyCallTrace(CopyPartTrace* p, size_t n, size_t at, size_t c) noexcept
        : part(p && p->on ? p : nullptr), bytes(n), offset(at), chunk(c) {
        if (part) stamps[0] = copy_stamp();
    }
    void acquired(cudaError_t status, int dev, bool made) noexcept {
        if (!part) return;
        device = dev; created = made; rc[0] = int(status);
        stamps[1] = copy_stamp(); stages = 1;
    }
    void enqueued(cudaError_t status) noexcept {
        if (!part) return;
        rc[1] = int(status); stamps[2] = copy_stamp(); stages = 2;
    }
    void waited(cudaError_t status) noexcept {
        if (!part) return;
        rc[2] = int(status); stamps[3] = copy_stamp(); stages = 3;
    }
    ~CopyCallTrace() noexcept {
        if (!part || !stages) return;
        double wall[3] = {-1, -1, -1}, cpu[3] = {-1, -1, -1};
        double total_cpu = 0;
        bool have_cpu = true;
        for (int i = 0; i < stages; ++i) {
            wall[i] = copy_ms(stamps[i].wall_ns, stamps[i + 1].wall_ns);
            cpu[i] = copy_ms(stamps[i].cpu_ns, stamps[i + 1].cpu_ns);
            part->wall[i] += wall[i];
            if (cpu[i] < 0) { part->cpu_known[i] = false; have_cpu = false; }
            else { part->cpu[i] += cpu[i]; total_cpu += cpu[i]; }
        }
        if (part->chunks && part->device != device) part->mixed_device = true;
        part->device = device; ++part->chunks; part->bytes += bytes;
        part->final_rc = rc[stages - 1];
        const double total = copy_ms(stamps[0].wall_ns, stamps[stages].wall_ns);
        // Account for EVERY chunk before the slow-call output cutoff. Ties keep the first chunk.
        if (total > part->worst_total) {
            part->worst_chunk = chunk; part->worst_offset = offset; part->worst_bytes = bytes;
            part->worst_device = device; part->worst_total = total;
            part->worst_cpu = have_cpu ? total_cpu : -1.0;
            for (int i = 0; i < 3; ++i) part->worst_wall[i] = wall[i];
        }
        if (total < 100.0) return;
        ++part->slow_calls;
        std::fprintf(stderr, "strata parktrace: copy_call schema=1 phase=%s layer_kind=%s qsa=%lld part=%s src=%s dst=%s "
                             "src_mode=%d dst_mode=%d device=%d bytes=%zu offset=%zu chunk=%zu created=%d "
                             "acquire_rc=%d enqueue_rc=%d wait_rc=%d final_rc=%d waited=%d "
                             "acquire_ns=%lld enqueue_ns=%lld wait_ns=%lld end_ns=%lld "
                             "acquire=%.3f ms enqueue=%.3f ms wait=%.3f ms total=%.3f ms "
                             "acquire_cpu=%.3f ms enqueue_cpu=%.3f ms wait_cpu=%.3f ms\n",
                     part->phase, part->layer_kind, (long long) part->qsa, part->part, part->src, part->dst,
                     part->src_mode, part->dst_mode, device, bytes, offset, chunk, created ? 1 : 0,
                     rc[0], rc[1], rc[2], rc[stages - 1], stages == 3 ? 1 : 0,
                     (long long) stamps[0].wall_ns, (long long) (rc[0] == 0 ? stamps[1].wall_ns : -1),
                     (long long) (stages >= 2 && rc[1] == 0 ? stamps[2].wall_ns : -1),
                     (long long) stamps[stages].wall_ns,
                     wall[0], wall[1], wall[2], total, cpu[0], cpu[1], cpu[2]);
    }
};

/// Conversation snapshot / checkpoint copies go through a private non-blocking stream of the current device
/// instead of the legacy (null) stream.  A plain cudaMemcpy is ordered against every blocking stream of the
/// device, and while another thread captures a graph on one (a drafter or a stage worker), HIP refuses it with
/// "operation would make the legacy stream depend on a capturing blocking stream".  The caller's contract is
/// unchanged: the device was synchronized before the copy (the copy itself is complete on return).
inline cudaError_t copy_nonblocking(void* dst, const void* src, size_t bytes,
                                    CopyPartTrace* part = nullptr, size_t offset = 0, size_t chunk = 0) {
    CopyCallTrace timing(part, bytes, offset, chunk);
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) { timing.acquired(e, -1, false); return e; }
    const int observed_device = dev; // exactly the existing cudaGetDevice result, before legacy normalization
    thread_local std::vector<cudaStream_t> streams;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= streams.size()) streams.resize((size_t) dev + 1, nullptr);   // never destroyed: process lifetime
    const bool created = !streams[(size_t) dev];
    if (!streams[(size_t) dev]) {
        e = cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking);
        if (e != cudaSuccess) { streams[(size_t) dev] = nullptr; timing.acquired(e, observed_device, created); return e; }
    }
    cudaStream_t s = streams[(size_t) dev];
    timing.acquired(cudaSuccess, observed_device, created);
    e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, s);
    timing.enqueued(e);
    if (e != cudaSuccess) return e; // preserve original enqueue error; do NOT wait or clear error state
    const cudaError_t result = cudaStreamSynchronize(s);
    timing.waited(result);
    return result;
}

} // namespace strata::core::conversation_detail
