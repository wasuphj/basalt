#include "basalt/core/aei_next_prefetch.hpp"
#include <cstdio>
using namespace basalt::core::aei;
#define CHECK(x) do { if(!(x)) {std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);return 1;}} while(0)
int main() {
    NextExpertPrefetch p(8);
    CHECK(p.metadata_bytes() < 1024 * 1024);
    CHECK(p.boost(2) == 0.0f);
    for (int i = 0; i < 6; ++i) {
        p.observe(1, 2);
        p.observe(2, 3);
        p.observe(3, 1);
    }
    // Learned 2->3, so after 1->2, expert 3 becomes a next candidate.
    p.observe(1, 2);
    CHECK(p.boost(3) > 0.0f);
    CHECK(p.boost(99) == 0.0f);
    for (uint64_t i = 0; i < NextExpertPrefetch::kFreshEvents + 10; ++i)
        p.observe(6, 7);
    CHECK(p.boost(3) == 0.0f);
    // Restart preserves learned conditional transitions, but never an
    // unconsumed old prediction timestamp.
    const std::string snapshot = p.checkpoint();
    NextExpertPrefetch restored(8);
    CHECK(restored.restore_checkpoint(snapshot));
    CHECK(restored.boost(3) == 0.0f);
    restored.observe(1, 2);
    CHECK(restored.boost(3) > 0.0f);
    CHECK(!restored.restore_checkpoint(snapshot.substr(0, snapshot.size()-1)));
    CHECK(!restored.restore_checkpoint(std::string(snapshot.size(), '\0')));
    NextExpertPrefetch unpredictable(64);
    // Train one source with a uniformly distributed successor set.
    // The posterior top-two is too uncertain for proactive hints.
    for (int pass = 0; pass < 100; ++pass)
        for (uint32_t target = 10; target < 18; ++target)
            unpredictable.observe(2, target);
    CHECK(unpredictable.evaluated() >= 128);
    CHECK(unpredictable.correct() < unpredictable.evaluated());
    CHECK(unpredictable.hit_ema_percent() <= 100);
    CHECK(p.evaluated() > 0);
    CHECK(p.correct() > 0);
    std::puts("aei_next_prefetch_test: OK");
}
