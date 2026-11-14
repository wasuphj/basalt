#include "basalt/core/aei_event_buffer.hpp"
#include <array>
#include <cstdio>
using namespace basalt::core::aei;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\\n",__LINE__,#x); return 1; } } while(0)
int main() {
    RouterEventBuffer b;
    CHECK(b.metadata_bytes() < 1024 * 1024);
    std::array<RoutedExpertEvent, RouterEventBuffer::kCapacity> batch{};
    CHECK(b.drain(batch.data(), 10) == 0);
    CHECK(b.drain(nullptr, 10) == 0);
    for (size_t i = 0; i < RouterEventBuffer::kCapacity; ++i)
        CHECK(b.try_record({0, static_cast<uint32_t>(i), i, 42}));
    CHECK(!b.try_record({0, 0, 5000, 42}));
    CHECK(b.dropped_capacity() == 1);
    CHECK(b.drain(batch.data(), 3) == 3);
    CHECK(batch[0].sequence == 0 && batch[2].sequence == 2);
    for (unsigned i = 0; i < 3; ++i)
        CHECK(b.try_record({1, i, 5000 + i, 42}));
    CHECK(b.drain(batch.data(), batch.size()) == batch.size());
    CHECK(batch[0].sequence == 3);
    CHECK(batch.back().sequence == 5002);
    CHECK(b.drain(batch.data(), batch.size()) == 0);
    CHECK(b.dropped_contention() == 0);
    std::puts("aei_event_buffer_test: OK");
}
