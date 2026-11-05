#include "strata/program/batch_groups.hpp"
#include <cstdio>
#include <cstdlib>

namespace bg = strata::program::batch_groups;
void check(bool v) { if (!v) { std::fputs("piped_mtp_groups_test failed\n", stderr); std::exit(1); } }

int main() {
    bg::Input in;
    in.batch = 4;
    in.later_stages = 3;
    in.batch_mtp = true;
    // Without the new opt-in, preserve Nice's pipeline-without-slot-drafts default.
    auto r = bg::resolve(in);
    check(r.groups == 4 && r.mtp_dropped && !r.mtp_piped);
    in.mtp_piped = true;
    r = bg::resolve(in);
    check(r.groups == 4 && !r.mtp_dropped && r.mtp_piped);
    in.mtp_serial = true;
    r = bg::resolve(in);
    check(r.groups == 1 && !r.mtp_piped);
    in.mtp_serial = false;
    in.set = true;
    in.given = 4;
    check(bg::resolve(in).mtp_piped);
    in.given = 1;
    check(!bg::resolve(in).mtp_piped);
    in.set = false;
    in.adapt_async = true;
    check(bg::resolve(in).groups == 1 && !bg::resolve(in).mtp_piped);
    in.adapt_async = false;
    in.later_stages = 1;
    check(bg::resolve(in).groups == 2 && bg::resolve(in).mtp_piped);
    in.later_stages = 0;
    check(!bg::resolve(in).mtp_piped);
    std::puts("piped_mtp_groups_test: opt-in groups and unchanged defaults passed");
}
