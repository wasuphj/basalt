#include "basalt/core/aei_transitions.hpp"
#include <cstdio>
#include <cstdint>
using namespace basalt::core::aei;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\\n",__LINE__,#x); return 1; } } while (0)
int main() {
    SparseTransitions t;
    CHECK(t.metadata_bytes() <= 16 * 1024 * 1024);
    CHECK(t.predict(7)[0].count == 0);
    for (unsigned i = 0; i < 100; ++i) t.observe(7, 12);
    for (unsigned i = 0; i < 20; ++i) t.observe(7, 13);
    auto p = t.predict(7);
    CHECK(p[0].expert == 12 && p[0].count == 100);
    CHECK(p[1].expert == 13 && p[1].count == 20);
    t.observe(7, 7);
    CHECK(t.predict(7)[0].count == 100);
    CHECK(t.predict(8)[0].count == 0);
    for (uint32_t i = 0; i < 100000; ++i) t.observe(91, 91 + (i % 8) + 1);
    p = t.predict(91);
    CHECK(p[0].count > 0);
    for (size_t i = 1; i < p.size(); ++i) CHECK(p[i-1].count >= p[i].count);
    for (unsigned i = 0; i < 150000; ++i) t.observe(42, 43);
    CHECK(t.predict(42)[0].count > 0);
    CHECK(t.predict(42)[0].count <= UINT16_MAX);
    std::puts("aei_transitions_test: OK");
}
