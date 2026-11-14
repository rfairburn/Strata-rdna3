// CPU regression for the opt-in resolver; no environment, GPU or model access.
#include "strata/program/batch_groups.hpp"

#include <cstdio>
#include <cstdlib>

namespace bg = strata::program::batch_groups;
namespace {
int checks = 0;
void check(bool ok) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "piped_mtp_groups_test: check %d failed\n", checks); std::exit(1); }
}

// The ordinary resolver before pipelined MTP: an independent compatibility oracle.
bg::Result ordinary(const bg::Input& in) {
    bg::Result r;
    r.groups = r.would_be = in.given;
    if (!in.auto_given && !(!in.set && in.later_stages > 0 && in.batch > 1)) return r;
    r.from_auto = true;
    r.groups = r.would_be = bg::auto_groups(in.later_stages, in.batch);
    if (!in.set && r.would_be > 1) {
        if (in.batch_mtp && in.mtp_serial) r.serial = bg::Serial::BatchMtp;
        else if (in.nv_serial_faster && in.later_stages == 1 && !in.batch_mtp && !in.adapt_async)
            r.serial = bg::Serial::TwoCardsSerialFaster;
        else if (in.adapt_async) r.serial = bg::Serial::AdaptAsync;
        r.mtp_dropped = in.batch_mtp && r.serial == bg::Serial::None;
        if (r.serial != bg::Serial::None) r.groups = 1;
    }
    return r;
}
}

int main() {
    for (int stages = 0; stages <= 7; ++stages)
        for (int batch = 0; batch <= 16; ++batch)
            for (int mode = 0; mode < 4; ++mode)
                for (int flags = 0; flags < 16; ++flags) {
                    bg::Input in;
                    in.later_stages = stages; in.batch = batch;
                    in.set = mode != 0;
                    in.auto_given = mode == 1;
                    in.given = mode == 3 ? 2 : 1;
                    in.batch_mtp = (flags & 1) != 0;
                    in.mtp_serial = (flags & 2) != 0;
                    in.adapt_async = (flags & 4) != 0;
                    in.nv_serial_faster = (flags & 8) != 0;
                    const auto expected = ordinary(in), off = bg::resolve(in);
                    check(off.groups == expected.groups && off.would_be == expected.would_be &&
                          off.from_auto == expected.from_auto && off.serial == expected.serial &&
                          off.mtp_dropped == expected.mtp_dropped && !off.mtp_piped);
                    in.mtp_piped = true;
                    const auto on = bg::resolve(in);
                    // Opt-in changes draft admission, never ordinary group selection or serial precedence.
                    check(on.groups == off.groups && on.would_be == off.would_be &&
                          on.from_auto == off.from_auto && on.serial == off.serial);
                    check(on.mtp_piped == (in.batch_mtp && on.groups > 1 && on.serial == bg::Serial::None));
                    check(on.mtp_dropped == (off.mtp_dropped && !in.mtp_piped));
                }
    bg::Input four;
    four.batch = 4; four.later_stages = 3; four.batch_mtp = true;
    check(bg::resolve(four).groups == 4 && bg::resolve(four).mtp_dropped);
    four.mtp_piped = true;
    check(bg::resolve(four).groups == 4 && bg::resolve(four).mtp_piped && !bg::resolve(four).mtp_dropped);
    four.mtp_serial = true;
    check(bg::resolve(four).groups == 1 && !bg::resolve(four).mtp_piped &&
          bg::resolve(four).serial == bg::Serial::BatchMtp);
    four.set = four.auto_given = true;
    check(bg::resolve(four).groups == 4 && bg::resolve(four).mtp_piped);
    std::printf("piped_mtp_groups_test: %d checks passed\n", checks);
}
