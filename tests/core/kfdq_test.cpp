// Exercise the real interposer and watcher, with no GPU, device, sysfs or proc access.
#if !defined(__linux__) || !__has_include(<linux/kfd_ioctl.h>)
#error "KFD queue tests require Linux KFD headers; an empty production module is not a passing test"
#endif
#include <dirent.h>
#include <dlfcn.h>
#include <linux/kfd_ioctl.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cerrno>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(\
    std::string("line ") + std::to_string(__LINE__) + ": " #expr); } while (0)

namespace mock {
struct Stop {};
struct Step {
    useconds_t delay;
    long advance_ms;
    std::function<void()> before, after;
};
struct Call {
    int fd;
    unsigned long request;
    kfd_ioctl_update_queue_args update;
    int rc;
};
std::map<std::string, std::string> environment;
std::map<uint32_t, int> failures;
std::vector<Call> calls;
std::vector<Step> steps;
size_t next_step;
long time_ms;
int creates, detaches, directory_reads;
void* (*thread_fn)(void*);
uint64_t writes[128], reads[128];

char* getenv(const char* name) {
    auto it = environment.find(name);
    return it == environment.end() ? nullptr : it->second.data();
}
int ioctl(int fd, unsigned long request, ...) {
    va_list ap;
    va_start(ap, request);
    void* arg = va_arg(ap, void*);
    va_end(ap);
    Call call{fd, request, {}, 0};
    if (request == AMDKFD_IOC_CREATE_QUEUE) {
        auto* a = static_cast<kfd_ioctl_create_queue_args*>(arg);
        a->queue_id = static_cast<uint32_t>(calls.size() + 1);
    } else if (request == AMDKFD_IOC_UPDATE_QUEUE) {
        call.update = *static_cast<kfd_ioctl_update_queue_args*>(arg);
        int& remaining = failures[call.update.queue_id];
        if (remaining > 0) { --remaining; call.rc = -1; errno = EAGAIN; }
    } else {
        CHECK(request == AMDKFD_IOC_DESTROY_QUEUE);
    }
    calls.push_back(call);
    return call.rc;
}
void* dlsym(void* handle, const char* symbol) {
    CHECK(handle == RTLD_NEXT);
    CHECK(std::strcmp(symbol, "ioctl") == 0);
    return reinterpret_cast<void*>(&ioctl);
}
int clock_gettime(clockid_t id, timespec* t) {
    CHECK(id == CLOCK_MONOTONIC || id == CLOCK_REALTIME);
    t->tv_sec = time_ms / 1000;
    t->tv_nsec = (time_ms % 1000) * 1000000;
    return 0;
}
int usleep(useconds_t delay) {
    // Called before the next poll, after the previous poll released its mutex.
    if (next_step > 0) steps[next_step - 1].after();
    if (next_step == steps.size()) throw Stop{};
    auto& step = steps[next_step++];
    CHECK(delay == step.delay);
    time_ms += step.advance_ms;
    step.before();
    return 0;
}
int pthread_create(pthread_t* th, const pthread_attr_t*, void* (*fn)(void*), void* arg) {
    CHECK(arg == nullptr);
    ++creates;
    thread_fn = fn;
    *th = pthread_t{};
    return 0; // Capture only; never start a real thread.
}
int pthread_detach(pthread_t) { ++detaches; return 0; }
DIR* opendir(const char* path) {
    CHECK(std::strcmp(path, "/sys/class/kfd/kfd/proc") == 0);
    ++directory_reads;
    return nullptr; // Fake unavailable sysfs: use upstream's process-local queue count.
}
dirent* readdir(DIR*) { throw std::runtime_error("unexpected readdir"); }
int closedir(DIR*) { throw std::runtime_error("unexpected closedir"); }
FILE* fopen(const char*, const char*) { throw std::runtime_error("unexpected fopen"); }
char* fgets(char*, int, FILE*) { throw std::runtime_error("unexpected fgets"); }
int fclose(FILE*) { throw std::runtime_error("unexpected fclose"); }
pid_t getpid() { throw std::runtime_error("unexpected getpid"); }
} // namespace mock

// Headers have already been included: replace only the module's OS calls, not system declarations.
#define ioctl kfdq_test_ioctl
#define getenv mock::getenv
#define dlsym mock::dlsym
#define clock_gettime mock::clock_gettime
#define usleep mock::usleep
#define pthread_create mock::pthread_create
#define pthread_detach mock::pthread_detach
#define opendir mock::opendir
#define readdir mock::readdir
#define closedir mock::closedir
#define fopen mock::fopen
#define fgets mock::fgets
#define fclose mock::fclose
#define getpid mock::getpid
#include "../../src/core/kfdq.cpp"
#undef ioctl
#undef getenv
#undef dlsym
#undef clock_gettime
#undef usleep
#undef pthread_create
#undef pthread_detach
#undef opendir
#undef readdir
#undef closedir
#undef fopen
#undef fgets
#undef fclose
#undef getpid

namespace {
void reset() {
    for (auto& q : qs) q = Queue{};
    for (auto& g : util_gpu_seen) g = 0;
    n_util_gpu = th_started = logon = 0;
    enabled = -1;
    real_ioctl = nullptr;
    idle_ms = poll_us = 2000;
    slots = 8;
    mock::environment.clear();
    mock::failures.clear();
    mock::calls.clear();
    mock::steps.clear();
    mock::next_step = 0;
    mock::time_ms = 0;
    mock::creates = mock::detaches = mock::directory_reads = 0;
    mock::thread_fn = nullptr;
    for (auto& w : mock::writes) w = 0;
    for (auto& r : mock::reads) r = 0;
}
void create_queue(int index, uint32_t gpu, uint32_t pct, bool utility = false) {
    kfd_ioctl_create_queue_args a{};
    a.gpu_id = gpu;
    a.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
    a.ring_size = utility ? 4096 : 1048576;
    a.ring_base_address = 0x100000 + index * 1048576ULL;
    a.queue_percentage = pct;
    a.queue_priority = 7;
    a.write_pointer_address = reinterpret_cast<uintptr_t>(&mock::writes[index]);
    a.read_pointer_address = reinterpret_cast<uintptr_t>(&mock::reads[index]);
    CHECK(kfdq_test_ioctl(42, AMDKFD_IOC_CREATE_QUEUE, &a) == 0);
}
void default_queues(uint32_t pct = 73) {
    for (int i = 0; i < 9; ++i) create_queue(i, 11, pct, i == 0);
    CHECK(enabled == 1);
    CHECK(idle_ms == 2000 && poll_us == 2000 && slots == 8 && logon == 0);
    CHECK(mock::creates == 1 && mock::detaches == 1 && mock::thread_fn == watcher);
    CHECK(qs[0].util == 1);
    for (int i = 1; i < 9; ++i) CHECK(qs[i].util == 0);
    mock::calls.clear();
}
void advance_pointers() {
    // The submission is already consumed by the next observation. Only the first wake poll sees a delta.
    mock::writes[0] = mock::reads[0] = 1;
}
void check_update(size_t index, int queue_index, uint32_t pct, int rc) {
    CHECK(mock::calls.size() > index);
    const auto& c = mock::calls[index];
    const auto& q = qs[queue_index];
    CHECK(c.fd == q.fd && c.request == AMDKFD_IOC_UPDATE_QUEUE && c.rc == rc);
    CHECK(c.update.queue_id == q.a.queue_id);
    CHECK(c.update.ring_base_address == q.a.ring_base_address);
    CHECK(c.update.ring_size == q.a.ring_size);
    CHECK(c.update.queue_priority == q.a.queue_priority);
    CHECK(c.update.queue_percentage == pct);
}
void run(std::initializer_list<mock::Step> steps) {
    mock::steps = steps;
    mock::next_step = 0;
    try { watcher(nullptr); CHECK(false); }
    catch (const mock::Stop&) {}
    CHECK(mock::next_step == mock::steps.size());
    // The fake sleep stops the watcher only outside the lock; no real thread survives.
    CHECK(pthread_mutex_trylock(&mu) == 0);
    CHECK(pthread_mutex_unlock(&mu) == 0);
}
void default_and_success() {
    reset();
    default_queues();
    run({
        {20000, 1999, [] {}, [] { CHECK(mock::calls.empty() && mock::directory_reads == 0); }},
        {20000, 1, [] {}, [] {
            CHECK(qs[0].parked == 1 && mock::calls.size() == 1);
            check_update(0, 0, 0, 0);
        }},
        {2000, 2, advance_pointers, [] {
            CHECK(qs[0].parked == 0 && mock::calls.size() == 2);
            check_update(1, 0, 73, 0);
        }},
        {20000, 20, [] {}, [] { CHECK(mock::calls.size() == 2); }}
    });
}
void failed_wakes(int failures, uint32_t pct) {
    reset();
    default_queues(pct);
    // Park successfully first, then inject repeated wake failures through the real set_pct call.
    std::vector<mock::Step> polls{
        {20000, 2000, [] {}, [] { CHECK(qs[0].parked == 1); check_update(0, 0, 0, 0); }},
        {2000, 2, [failures] {
            advance_pointers();
            mock::failures[qs[0].a.queue_id] = failures;
        }, [] { CHECK(qs[0].parked == 1); check_update(1, 0, run_pct(qs[0]), -1); }}
    };
    for (int i = 1; i < failures; ++i) {
        polls.push_back({2000, 2, [] {}, [i] {
            CHECK(qs[0].parked == 1 && qs[0].last_w == mock::writes[0]);
            CHECK(mock::reads[0] == mock::writes[0]);
            CHECK(mock::calls.size() == static_cast<size_t>(i + 2));
            check_update(i + 1, 0, run_pct(qs[0]), -1);
        }});
    }
    polls.push_back({2000, 2, [] {}, [failures, pct] {
        CHECK(qs[0].parked == 0 && qs[0].last_w == mock::writes[0]);
        CHECK(mock::calls.size() == static_cast<size_t>(failures + 2));
        check_update(failures + 1, 0, pct ? pct : 100, 0);
        CHECK(qs[0].a.queue_percentage == pct); // Preserve original percentage, including upstream's zero fallback.
    }});
    polls.push_back({20000, 20, [] {}, [failures] {
        CHECK(mock::calls.size() == static_cast<size_t>(failures + 2));
    }});
    mock::steps = polls;
    mock::next_step = 0;
    try { watcher(nullptr); CHECK(false); }
    catch (const mock::Stop&) {}
    CHECK(mock::next_step == polls.size());
    CHECK(pthread_mutex_trylock(&mu) == 0);
    CHECK(pthread_mutex_unlock(&mu) == 0);
}
void mixed_wakes() {
    reset();
    mock::environment["STRATA_KFDQ"] = "1";
    mock::environment["STRATA_KFDQ_SLOTS"] = "1";
    create_queue(0, 11, 37, true);
    create_queue(1, 11, 37);
    create_queue(2, 22, 63, true);
    create_queue(3, 22, 63);
    CHECK(slots == 1 && qs[0].util && qs[2].util);
    mock::calls.clear();
    run({
        {20000, 2000, [] {}, [] {
            CHECK(qs[0].parked && qs[2].parked && mock::calls.size() == 2);
        }},
        {2000, 2, [] {
            advance_pointers();
            mock::failures[qs[0].a.queue_id] = 2;
        }, [] {
            CHECK(qs[0].parked == 1 && qs[2].parked == 0 && mock::calls.size() == 4);
            check_update(2, 0, 37, -1);
            check_update(3, 2, 63, 0);
        }},
        {2000, 2, [] {}, [] {
            CHECK(qs[0].parked == 1 && qs[2].parked == 0 && mock::calls.size() == 5);
            CHECK(qs[0].last_w == mock::writes[0] && mock::reads[0] == mock::writes[0]);
            check_update(4, 0, 37, -1);
        }},
        {2000, 2, [] {}, [] {
            CHECK(qs[0].parked == 0 && qs[2].parked == 0 && mock::calls.size() == 6);
            check_update(5, 0, 37, 0);
        }},
        {20000, 20, [] {}, [] { CHECK(mock::calls.size() == 6); }}
    });
}
void opt_out() {
    reset();
    mock::environment["STRATA_KFDQ"] = "0";
    create_queue(0, 11, 73, true);
    CHECK(enabled == 0);
    CHECK(mock::calls.size() == 1 && mock::calls[0].request == AMDKFD_IOC_CREATE_QUEUE);
    CHECK(mock::creates == 0 && mock::detaches == 0 && mock::directory_reads == 0);
    for (auto& q : qs) CHECK(q.used == 0);
}
} // namespace

int main() {
    try {
        default_and_success();
        failed_wakes(1, 73);
        failed_wakes(3, 73);
        failed_wakes(1, 0);
        mixed_wakes();
        opt_out();
        std::puts("kfdq: default-on, opt-out, successful wake, failed-then-successful wake, repeated failures, mixed wakes, unchanged pointers: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "kfdq: FAIL: %s\n", e.what());
        return 1;
    }
}
