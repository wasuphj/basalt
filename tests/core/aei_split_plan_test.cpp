#include "basalt/core/aei_split_plan.hpp"
#include <cstdio>
using namespace basalt::core::aei;
#define CHECK(x) do { if(!(x)) {std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);return 1;} } while(0)
int main() {
    const auto baseline = plan_staged_split(0, 40000, 16, 2);
    for (int m = 0; m <= 128; ++m) CHECK(baseline.staged[m] == 0);
    const auto gpu_fast = plan_staged_split(250000, 10000, 16, 3);
    const auto gpu_slow = plan_staged_split(20000, 200000, 16, 3);
    for (int m = 0; m <= 128; ++m) {
        CHECK(gpu_fast.staged[m] >= 0 && gpu_fast.staged[m] <= m && gpu_fast.staged[m] <= 16);
        CHECK(gpu_slow.staged[m] >= 0 && gpu_slow.staged[m] <= m && gpu_slow.staged[m] <= 16);
        if (m < 3) CHECK(gpu_fast.staged[m] == 0);
    }
    CHECK(gpu_fast.staged[32] > gpu_slow.staged[32]);
    const auto limited = plan_staged_split(250000, 10000, 2, 1);
    CHECK(limited.staged[128] <= 2);
    const auto no_gpu_samples = plan_staged_split(250000,10000,16,1,0);
    const auto expensive_gpu = plan_staged_split(250000,10000,16,1,400000);
    CHECK(expensive_gpu.staged[32] < no_gpu_samples.staged[32]);
    const auto cheap_gpu = plan_staged_split(250000,10000,16,1,1000);
    CHECK(cheap_gpu.staged[32] >= expensive_gpu.staged[32]);
    std::puts("aei_split_plan_test: OK");
}
