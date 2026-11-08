#include "basalt/core/aei_cost_rank.hpp"
#include <cstdio>
#include <thread>
#include <array>
using namespace basalt::core::aei;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); return 1; } } while(0)
int main() {
    LayerCostRank model;
    CHECK(model.metadata_bytes() < 4096);
    CHECK(model.reference_cost_ns(4) == 0);
    for (int i = 0; i < 8; ++i) {
        model.observe(0, 2, 1000);  // 500 ns/CPU entry
        model.observe(1, 2, 4000);  // 2000 ns/CPU entry
    }
    const auto reference = model.reference_cost_ns(4);
    CHECK(reference != 0);
    CHECK(model.multiplier(0, reference) <= 1.0f);
    CHECK(model.multiplier(1, reference) >= 1.0f);
    CHECK(model.multiplier(2, reference) == 1.0f);
    CHECK(model.multiplier(500, reference) == 1.0f);
    std::array<std::thread, 4> threads;
    for (auto& t : threads)
        t = std::thread([&model] {
            for (int i = 0; i < 1000; ++i)
                model.observe(0, 10, 10000);
        });
    for (auto& t : threads) t.join();
    CHECK(model.samples(0) == 4008);
    CHECK(model.multiplier(0, reference) >= 0.5f);
    CHECK(model.multiplier(0, reference) <= 2.0f);
    std::puts("aei_cost_rank_test: OK");
}
