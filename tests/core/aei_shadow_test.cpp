#include "basalt/core/aei_shadow.hpp"
#include <cstdio>
using namespace basalt::core::aei;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\\n",__LINE__,#x); return 1; } } while (0)
int main() {
    TransitionShadow s;
    CHECK(s.metadata_bytes() < 16 * 1024 * 1024);
    s.record(1);
    CHECK(s.metrics().opportunities == 0);
    s.record(2); // First 1 -> 2 transition: unavailable before learning.
    CHECK(s.metrics().no_prediction == 1);
    s.record(1);
    s.record(2); // 1 -> 2 is now seen and should be a hit.
    const auto m = s.metrics();
    CHECK(m.observed == 4);
    CHECK(m.opportunities == 3);
    CHECK(m.top1_hits == 1 && m.top4_hits == 1);
    CHECK(m.no_prediction == 2);
    CHECK(m.top4_hits <= m.opportunities);
    CHECK(m.top1_hits <= m.top4_hits);
    s.reset();
    CHECK(s.metrics().observed == 0 && s.metrics().top1_hits == 0);
    s.record(2);
    CHECK(s.metrics().opportunities == 0);
    std::puts("aei_shadow_test: OK");
}
