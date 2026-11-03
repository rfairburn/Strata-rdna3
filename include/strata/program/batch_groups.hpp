// include/strata/program/batch_groups.hpp - how many pipelined slot groups --batch runs with.
//
// Explicit groups are honored. Without STRATA_PIPED_MTP, keep the fork's serial MTP default.
#pragma once

namespace strata::program::batch_groups {

enum class Serial { None, BatchMtp, AdaptAsync };

struct Input {
    bool set = false;
    bool auto_given = false;
    int given = 1;
    int batch = 0;
    int later_stages = 0;
    bool batch_mtp = false;
    bool adapt_async = false;
    bool mtp_serial = false;
    bool mtp_piped = false;
};

struct Result {
    int groups = 1;
    bool from_auto = false;
    Serial serial = Serial::None;
    int would_be = 1;
    bool mtp_dropped = false;
    bool mtp_piped = false;
};

inline int auto_groups(int later_stages, int batch) {
    int best = 1;
    if (later_stages <= 0) return best;
    for (int d = 2; d <= later_stages + 1 && d <= batch; ++d)
        if (batch % d == 0) best = d;
    return best;
}

inline Result resolve(const Input& in) {
    Result r;
    r.groups = in.given;
    r.would_be = in.given;
    const bool defaulted = !in.set && in.later_stages > 0 && in.batch > 1;
    if (in.auto_given || defaulted) {
        r.from_auto = true;
        r.would_be = auto_groups(in.later_stages, in.batch);
        r.groups = r.would_be;
        if (!in.set && r.would_be > 1) {
            if (in.batch_mtp && (in.mtp_serial || !in.mtp_piped)) r.serial = Serial::BatchMtp;
            else if (in.adapt_async) r.serial = Serial::AdaptAsync;
            if (r.serial != Serial::None) r.groups = 1;
        }
    }
    r.mtp_piped = in.batch_mtp && in.mtp_piped && r.serial == Serial::None && r.groups > 1;
    return r;
}

inline const char* serial_what(Serial s) {
    return s == Serial::BatchMtp ? "--batch-mtp" : s == Serial::AdaptAsync ? "--adapt-async 1" : "";
}

}  // namespace strata::program::batch_groups
