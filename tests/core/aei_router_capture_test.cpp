#include "basalt/core/aei_router_capture.hpp"
#include <array>
#include <cstdio>
using namespace basalt::core::aei;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\\n",__LINE__,#x); return 1; } } while(0)
int main() {
    RouterEventBuffer buffer;
    VerifierRouterCapture capture;
    const int32_t ids[] = {3, -1, 2, 99, 1, 0};
    capture.capture(2, ids, 2, 3, 8); // disabled
    CHECK(capture.next_sequence() == 0);
    capture.bind(&buffer, 44);
    capture.capture(2, ids, 2, 3, 8);
    CHECK(capture.next_sequence() == 6);
    std::array<RoutedExpertEvent, 8> out{};
    CHECK(buffer.drain(out.data(), out.size()) == 4);
    CHECK(out[0].stream == 44 && out[0].layer == 2 && out[0].expert == 3 && out[0].sequence == 0);
    CHECK(out[1].expert == 2 && out[1].sequence == 2);
    CHECK(out[2].expert == 1 && out[2].sequence == 4);
    CHECK(out[3].expert == 0 && out[3].sequence == 5);
    capture.bind(nullptr, 45);
    capture.capture(0, ids, 2, 3, 8);
    CHECK(capture.next_sequence() == 0);
    CHECK(buffer.drain(out.data(), out.size()) == 0);
    std::puts("aei_router_capture_test: OK");
}
