#include "basalt/core/aei_multistream.hpp"
#include <cstdio>
using namespace basalt::core::aei;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %d: %s\\n",__LINE__,#x); return 1; } } while (0)
int main() {
    MultiStreamShadow m(4, 2);
    CHECK(m.metadata_bytes() <= 16 * 1024 * 1024);
    CHECK(m.consume({0, 1, 0, 111}));
    CHECK(m.consume({0, 3, 0, 222}));
    CHECK(m.consume({0, 2, 1, 111})); // learn 1 -> 2, not 3 -> 2
    CHECK(m.consume({0, 1, 1, 222})); // learn 3 -> 1
    CHECK(m.consume({0, 1, 2, 111})); // learn 2 -> 1
    CHECK(m.consume({0, 2, 3, 111})); // 1 -> 2 hit
    CHECK(m.predictions().top1_hits == 1);
    CHECK(m.predictions().opportunities == 4);
    CHECK(!m.consume({0, 2, 3, 111}));
    CHECK(m.ingestion().rejected_events == 1);
    auto before = m.predictions();
    CHECK(m.consume({0, 2, 7, 222})); // one request loses events; other unaffected
    CHECK(m.ingestion().missing_events == 5);
    CHECK(m.predictions().opportunities == before.opportunities);
    CHECK(m.consume({0, 1, 4, 111}));
    CHECK(m.predictions().opportunities == before.opportunities + 1);
    for (uint64_t stream = 1000; stream < 1040; ++stream)
        CHECK(m.consume({0, 1, 0, stream}));
    CHECK(m.ingestion().boundaries >= 1);
    CHECK(m.predictions().top1_hits <= m.predictions().top4_hits);
    CHECK(m.predictions().top4_hits <= m.predictions().opportunities);
    std::puts("aei_multistream_test: OK");
}
