#include "basalt/core/aei_ingest.hpp"
#include <cstdio>
using namespace basalt::core::aei;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\\n",__LINE__,#x); return 1; } } while (0)
int main() {
    OrderedShadowIngest s(4, 2);
    CHECK(s.metadata_bytes() <= 16 * 1024 * 1024);
    CHECK(s.consume({0, 1, 1, 10}));
    CHECK(s.consume({0, 2, 2, 10})); // learn 1 -> 2
    CHECK(s.consume({0, 1, 3, 10}));
    CHECK(s.consume({0, 2, 4, 10})); // causal hit
    CHECK(s.predictions().top1_hits == 1);
    const auto before = s.predictions();
    CHECK(!s.consume({0, 2, 4, 10})); // duplicate
    CHECK(!s.consume({0, 2, 3, 10})); // out of order
    CHECK(!s.consume({0, 4, 5, 10})); // invalid expert
    CHECK(!s.consume({2, 1, 5, 10})); // invalid layer
    CHECK(s.predictions().observed == before.observed);
    CHECK(s.ingestion().rejected_events == 4);
    CHECK(s.consume({0, 2, 8, 10})); // lost 5,6,7: must break continuity
    CHECK(s.ingestion().missing_events == 3);
    CHECK(s.ingestion().boundaries == 1);
    CHECK(s.predictions().opportunities == before.opportunities);
    CHECK(s.consume({0, 1, 1, 11})); // new request boundary
    CHECK(s.ingestion().boundaries == 2);
    CHECK(s.predictions().opportunities == before.opportunities);
    CHECK(s.consume({0, 2, 2, 11})); // persistent learned 1->2, no cross-request edge
    CHECK(s.predictions().top1_hits == 2);
    OrderedShadowIngest invalid(0, 2);
    CHECK(!invalid.consume({0, 0, 1, 1}));
    std::puts("aei_ingest_test: OK");
}
