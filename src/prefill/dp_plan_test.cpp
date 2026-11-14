#include "strata/prefill/dp_plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
int checks = 0;
void check(bool ok) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL dp_plan_test at check %d\n", checks); std::exit(1); }
}
}

int main() {
    namespace dp = strata::prefill::dp;
    constexpr int64_t min_last = 4224;
    for (int64_t cap : {512, 4096, 6144, 8192, 16384, 28672, 32768}) {
        for (int64_t n = 1; n <= 262144; n += 127) {
            int64_t k = std::max<int64_t>(2, (n + cap - 1) / cap);
            if (k & 1) ++k;
            const int64_t old_step = ((n + k - 1) / k + 255) / 256 * 256;
            const int64_t full = (n + old_step - 1) / old_step;
            const bool old_takes = full >= 2 && n - (full - 1) * old_step >= min_last;
            check(dp::step_for(n, cap, 2) == old_step);
            check(dp::takes(n, cap, 2, min_last) == old_takes);
        }
    }
    for (int paths : {2, 3, 4, 8}) {
        for (int64_t cap : {4096, 8192, 16384, 28672, 32768}) {
            for (int64_t n = 1; n <= 262144; n += 997) {
                const auto plan = dp::plan(n, cap, paths, min_last);
                check(!plan.empty() == dp::takes(n, cap, paths, min_last));
                if (plan.empty()) continue;
                int64_t pos = 0;
                std::vector<int> count((size_t) paths, 0);
                std::vector<std::vector<bool>> seen((size_t) paths, std::vector<bool>(plan.size(), false));
                for (size_t i = 0; i < plan.size(); ++i) {
                    const auto& c = plan[i];
                    check(c.c0 == pos && c.c0 % 256 == 0);
                    check(c.T > 0 && c.T <= cap && c.k == (int64_t) i);
                    check(c.who >= 0 && c.who < paths);
                    if (i > 0) check(c.who == (plan[i - 1].who + 1) % paths);
                    ++count[(size_t) c.who];
                    for (size_t j = 0; j < i; ++j) check(seen[(size_t) c.who][j]);
                    seen[(size_t) c.who][i] = true;
                    const int64_t begin = dp::handoff_begin(plan, (int64_t) i, paths);
                    if (paths == 2) check(begin == c.c0);
                    for (size_t j = 0; j <= i; ++j)
                        if (plan[j].c0 >= begin) seen[(size_t) ((c.who + 1) % paths)][j] = seen[(size_t) c.who][j];
                    pos += c.T;
                }
                check(pos == n && plan.back().who == paths - 1 && plan.back().T >= min_last);
                check(*std::min_element(count.begin(), count.end()) >= 1);
                check(*std::max_element(count.begin(), count.end()) - *std::min_element(count.begin(), count.end()) <= 1);
            }
        }
    }
    const auto p = dp::plan(72627, 32768, 4, min_last);
    check(p.size() == 4 && p[0].who == 0 && p.back().who == 3);
    check(p[0].T == 18176 && p.back().T == 18099);
    check(dp::plan(8192, 32768, 4, min_last).empty());
    check(dp::plan(72627, 0, 4, min_last).empty());
    check(dp::plan(72627, 32768, 1, min_last).empty());
    check(dp::step_for(-1, 32768, 4) == 0);
    check(dp::step_for(std::numeric_limits<int64_t>::max(), 1, 4) == 0);
    std::printf("dp_plan_test: %d checks passed\n", checks);
}
