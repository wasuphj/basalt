#include "basalt/core/aei_replay.hpp"
#include "basalt/core/aei_event_buffer.hpp"
#include "basalt/core/aei_multistream.hpp"
#include "basalt/core/aei_scheduler.hpp"
#include "basalt/core/aei_trace.hpp"
#include <cstdint>
#include <cstdio>

using namespace basalt::core::aei;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); return 1; } } while (0)
int main() {
    constexpr uint64_t limit = 64ull * 1024ull * 1024ull;
    // Conservative maximum 1M flattened experts, one replay and one histogram.
    constexpr size_t experts = 1u << 20;
    Replay replay(experts, 4096, Policy::cost_aware);
    HistogramStats histogram(experts);
    const uint64_t total = replay.metadata_bytes() + histogram.metadata_bytes() +
                           sizeof(RouterEventBuffer) + sizeof(MultiStreamShadow) +
                           sizeof(ShadowScheduler);
    CHECK(total <= limit);
    CHECK(replay.resident_count() == 0);
    std::printf("aei_memory_budget_test: metadata=%llu limit=%llu\n",
                (unsigned long long)total, (unsigned long long)limit);
}
