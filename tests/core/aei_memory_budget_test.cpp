#include "basalt/core/aei_replay.hpp"
#include "basalt/core/aei_event_buffer.hpp"
#include "basalt/core/aei_multistream.hpp"
#include "basalt/core/aei_scheduler.hpp"
#include "basalt/core/aei_trace.hpp"
#include "basalt/core/aei_reuse.hpp"
#include "basalt/core/aei_cost_rank.hpp"
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
    const uint64_t offline = replay.metadata_bytes() + histogram.metadata_bytes() +
                             sizeof(RouterEventBuffer) + sizeof(MultiStreamShadow) +
                             sizeof(ShadowScheduler);
    CHECK(offline <= limit);
    CHECK(replay.resident_count() == 0);
    ExpertReuseIndex reuse(experts);
    const uint64_t live = reuse.metadata_bytes() + sizeof(RouterEventBuffer) +
                          sizeof(MultiStreamShadow) + sizeof(ShadowScheduler) +
                          sizeof(LayerCostRank);
    CHECK(live <= limit);
    std::printf("aei_memory_budget_test: offline=%llu live=%llu limit=%llu\n",
                (unsigned long long)offline, (unsigned long long)live,
                (unsigned long long)limit);
}
