#include "basalt/core/aei_scheduler.hpp"
#include <cstdio>
#include <cstdint>
#include <limits>
using namespace basalt::core::aei;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\\n", __LINE__, #x); return 1; } } while (0)
int main() {
    ShadowScheduler s;
    CHECK(s.max_metadata_bytes() < 4096);
    CHECK(s.candidate(16) == Placement::baseline);
    s.observe({Device::cpu, 0, 100, 0, 0});
    s.observe({Device::gpu, 16, 0, 0, 0});
    CHECK(s.estimate(Device::cpu, 16).observations == 0);
    for (unsigned i = 0; i < 8; ++i) {
        s.observe({Device::cpu, 16, 100, 0, 50});
        s.observe({Device::gpu, 16, 60, 20, 0});
    }
    CHECK(s.candidate(16) == Placement::gpu);
    CHECK(s.candidate(16, 70) == Placement::baseline);
    CHECK(s.candidate(1) == Placement::baseline);
    for (unsigned i = 0; i < 32; ++i)
        s.observe({Device::gpu, 16, 300, 40, 0});
    CHECK(s.candidate(16) == Placement::cpu);
    const auto before = s.estimate(Device::cpu, 16);
    s.observe({Device::cpu, 16, UINT64_MAX, UINT64_MAX, UINT64_MAX});
    CHECK(s.estimate(Device::cpu, 16).observations == before.observations + 1);
    CHECK(s.candidate(0) == Placement::baseline);
    std::puts("aei_scheduler_test: OK");
}
