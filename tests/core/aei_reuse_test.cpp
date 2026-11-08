#include "basalt/core/aei_reuse.hpp"
#include <cmath>
#include <cstdio>
#include <thread>
#include <array>
using namespace basalt::core::aei;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); return 1; } } while (0)
int main() {
    ExpertReuseIndex r(128);
    CHECK(r.metadata_bytes() <= 128 * 16 + 128);
    CHECK(r.priority_multiplier(5) == 1.0f);
    r.observe(5); r.observe(2); r.observe(5); // establishes reuse interval
    auto v = r.priority_multiplier(5);
    CHECK(std::isfinite(v) && v >= 0.5f && v <= 1.4f);
    CHECK(r.priority_multiplier(130) == 1.0f);
    std::array<std::thread, 4> workers;
    for (auto& worker : workers)
        worker = std::thread([&] {
            for (size_t i = 0; i < 10000; ++i) r.observe(i % 128);
        });
    for (auto& worker : workers) worker.join();
    for (size_t i = 0; i < 128; ++i) {
        v = r.priority_multiplier(i);
        CHECK(std::isfinite(v) && v >= 0.5f && v <= 1.4f);
    }
    bool invalid = false;
    try { ExpertReuseIndex oversized((1u << 20) + 1); }
    catch (const std::invalid_argument&) { invalid = true; }
    CHECK(invalid);
    std::puts("aei_reuse_test: OK");
}
