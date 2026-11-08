// CPU-only policy replay invariants; does not assert GPU performance.
#include "basalt/core/aei_replay.hpp"
#include <cassert>
#undef assert
#define assert(expr) do { if (!(expr)) { std::fprintf(stderr, "FAIL %s:%d: %s\\n", __FILE__, __LINE__, #expr); return 1; } } while (0)
#include <cstdio>
#include <stdexcept>
using namespace basalt::core::aei;

int main() {
    for (Policy p : {Policy::lru, Policy::frequency, Policy::cost_aware, Policy::least_stale}) {
        Replay r(4, 2, p);
        assert(r.metadata_bytes() <= 4 * 64);
        for (uint32_t id : {0u, 1u, 0u, 2u, 0u, 3u, 3u}) {
            auto v = r.access({id, 100, 10});
            assert(v.hits + v.misses == v.accesses);
            assert(r.resident_count() <= 2);
            assert(v.transfers <= v.misses);
        }
        assert(r.result().accesses == 7);
        assert(r.result().hits >= 2);
    }
    Replay empty(2, 0, Policy::lru);
    empty.access({0, 50, 20});
    assert(empty.result().hits == 0 && empty.result().transfers == 0);
    assert(empty.result().estimated_ns == 50);
    Replay capped(2, 99, Policy::lru);
    capped.access({0, 50, 20});
    capped.access({1, 50, 20});
    assert(capped.resident_count() == 2);
    bool threw = false;
    try { capped.access({2, 50, 20}); } catch (const std::out_of_range&) { threw = true; }
    assert(threw);
    std::puts("aei_replay_test: OK");
}
