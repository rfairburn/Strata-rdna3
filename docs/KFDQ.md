# Linux HIP queue parking

`src/core/kfdq.cpp` is enabled by default in the Linux HIP `strata` executable when
`linux/kfd_ioctl.h` is available at build time. Set `STRATA_KFDQ=0` before starting
the process to disable it. This does not change an already running process.
CUDA, SYCL and Windows builds do not gain this interposer.

The executable exports `ioctl` so ROCr's queue calls can be recorded and forwarded
to libc through `dlsym(RTLD_NEXT, "ioctl")`. A watcher checks the recorded queues'
write and read pointers. After an idle period it uses KFD's queue-update ioctl to
set selected queues' percentage to zero. On observed activity it restores their
recorded percentage (or 100 when the recorded percentage was zero).

Upstream defaults are preserved:

| Variable | Default | Purpose |
| --- | --- | --- |
| `STRATA_KFDQ` | enabled | A value starting with `0` disables parking. |
| `STRATA_KFDQ_SLOTS` | 8 | Assumed compute queue slots per GPU. |
| `STRATA_KFDQ_IDLE_MS` | 2000 | Idle interval before considering parking. |
| `STRATA_KFDQ_POLL_US` | 2000 | Watcher sleep while queues are parked. |
| `STRATA_KFDQ_LOG` | off | A value starting with `1` logs queue operations. |

The active watcher sleeps for 20 ms between checks. These are requested sleep
intervals, not hard wake-latency bounds. Scheduling and ioctl time can add delay;
a failed restore keeps the queue parked until a later restore succeeds.

## Upstream source and exact deviation

The module is based on
[`nicejunjie/Strata-rdna3`, commit `f557b5440fe345757cae69ede2ed34826da5e612`](https://github.com/nicejunjie/Strata-rdna3/blob/f557b5440fe345757cae69ede2ed34826da5e612/src/core/kfdq.cpp).
The pinned file's SHA-256 is
`2d3041b457b5d8560b9db7e31d0cd6fbfc400fdd3a5021c5bcd98d093fc1449d`.
It retains upstream's ioctl type, number, direction and minimum-size matching,
zero-initialized bounded create-argument copy, and PID-namespace own-queue
fallback. The latter subtracts locally recorded queues from the system total
when sysfs has no entry matching this process's namespace PID, clamped at zero.

Only the watcher's failed-wake bookkeeping differs from that file:

- Add a local `wake_pending` boolean, initially false.
- Enter the restore branch when parked and either activity or a pending wake is
  observed. Reset `wake_pending` before trying the remaining parked queues.
- Clear a queue's `parked` flag only when its restore ioctl returns zero;
  otherwise set `wake_pending`.
- Set the aggregate `parked` flag to `wake_pending` after the restore loop.

Thus a failed queue remains marked parked, and the aggregate state keeps polling
and retrying even when pointers stop changing. A successful queue is not restored
again during those retries. When all restores succeed, the watcher resumes its
original active polling behavior. No initialization, interposer, selection,
percentage, timing, thread lifecycle or teardown changes are included.

## Assumptions and limits

The module treats the first small AQL queue on each GPU (ring size at most 64 KiB)
as ROCr's utility queue. If this process has more queues than the slot limit, that
utility queue is eligible for parking. When KFD sysfs reports queues belonging to
other processes and the total exceeds the limit, all this process's queues on
that GPU are eligible instead. If sysfs cannot be opened, the local-count
heuristic still applies. Tracking is limited to 128 queues and 16 GPU identities,
as upstream. Queue pointer addresses must remain readable while tracked.

Pointer observation is a heuristic, not proof of kernel completion. Equal read
and write pointers are not a general kernel-completion fence. Queue-order,
ROCr/KFD ABI, dynamic symbol interposition and scheduling assumptions must be
checked for the runtime and driver in use. Initialization and runtime-driven
update/destroy handling remain exactly as upstream; this backport does not claim
to fix their concurrency or error-handling limits.

The measurements in the module's original header describe upstream's two-card
setup. They are not measurements of this backport, a hardware fix, a guaranteed
power saving, or evidence of four-GPU stability. CPU tests do not establish safe
hardware behavior or model-output equivalence.

## CPU regression tests

The standalone test includes the actual production module with a renamed
interposer. It substitutes queue storage, clocks, sleep, ioctl resolution and
calls, environment, filesystem calls and thread creation. It runs the watcher
synchronously for a bounded number of polls; no real device, sysfs, proc or GPU
operations occur, and no watcher thread is started.

From the source root, compile directly with the host C++ compiler; no CMake
changes, HIP libraries or dependency downloads are needed:

```sh
c++ -std=c++20 -O0 -g0 -pthread tests/core/kfdq_test.cpp -ldl -o /tmp/kfdq_test
timeout 10s /tmp/kfdq_test
```

Choose an unused output path. The executable is only the CPU mock fixture,
not the native engine.

These checks cover default-on and opt-out initialization, the idle threshold,
successful parking and wake, recorded percentage restoration (including the
upstream zero-to-100 fallback), failed-then-successful and repeated failed wakes,
and mixed successful/failed restores. Wake retries are observed with unchanged,
equal read/write pointers. The tests require Linux KFD headers and fail to
compile rather than silently exercising an empty module. The six scenario
invocations use unavailable-sysfs mocks and the host header's exact ioctl
requests. They do not exercise PID-namespace enumeration, alternate ioctl
sizes/directions, or real ROCr/KFD behavior; the portability code is retained,
not hardware-qualified by these tests.
