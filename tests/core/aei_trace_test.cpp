#include "basalt/core/aei_trace.hpp"
#include <cstdio>
#include <limits>
using namespace basalt::core::aei;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\\n", __LINE__,#x); return 1; } } while(0)
int main() {
    HistogramStats h(3);
    const uint32_t first[] = {1, 9, 2};
    CHECK(h.merge({first, 3, 5}));
    CHECK(h.count(1) == 9 && h.count(2) == 2 && h.tokens() == 5);
    CHECK(!h.merge({first, 2, 7}));
    CHECK(!h.merge({nullptr, 3, 7}));
    CHECK(h.count(1) == 9 && h.tokens() == 5);
    CHECK(h.metadata_bytes() <= 3 * sizeof(uint64_t));
    OrderedRouterGuard g;
    CHECK(g.accept({0, 2, 0, 10}) && g.changed_stream());
    CHECK(g.accept({0, 1, 1, 10}) && !g.changed_stream());
    CHECK(!g.accept({0, 1, 1, 10}));
    CHECK(!g.accept({0, 1, 0, 10}));
    CHECK(g.accept({0, 2, 0, 11}) && g.changed_stream());
    CHECK(g.accept({0, 3, 1, 11}) && !g.changed_stream());
    std::puts("aei_trace_test: OK");
}
