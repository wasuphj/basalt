#include "basalt/core/aei_auto_policy.hpp"
#include <cstdio>
#include <cmath>
using namespace basalt::core::aei;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); return 1; } } while (0)

int main() {
    AutoPolicyController fsm;
    CHECK(!fsm.enabled());
    CHECK(fsm.active() == AutoArm::baseline);
    constexpr uint32_t allowed = (1u << (unsigned)AutoArm::baseline) |
                                 (1u << (unsigned)AutoArm::prefetch);
    fsm.enable(allowed);
    CHECK(fsm.enabled());
    CHECK(fsm.supports(AutoArm::prefetch));
    CHECK(!fsm.supports(AutoArm::split));
    for (uint32_t i = 0; i < AutoPolicyController::kExploreSamples; ++i)
        fsm.observe(128, 1280.0, true);
    CHECK(fsm.active() == AutoArm::prefetch);
    for (uint32_t i = 0; i < AutoPolicyController::kExploreSamples; ++i)
        fsm.observe(128, 850.0, true);
    CHECK(fsm.phase() == AutoPhase::exploit);
    CHECK(fsm.active() == AutoArm::prefetch);
    CHECK(fsm.samples(AutoArm::baseline, 1) == AutoPolicyController::kExploreSamples);
    CHECK(fsm.samples(AutoArm::prefetch, 1) == AutoPolicyController::kExploreSamples);
    CHECK(fsm.completed() == 2 * AutoPolicyController::kExploreSamples);
    fsm.observe(128, NAN, true);
    CHECK(fsm.phase() == AutoPhase::cooldown);
    CHECK(fsm.active() == AutoArm::baseline);
    CHECK(fsm.rejected() == 1);
    for (int i = 0; i < 8; ++i) fsm.observe(128, 1280.0, true);
    CHECK(fsm.phase() == AutoPhase::explore);
    CHECK(fsm.active() == AutoArm::baseline);
    CHECK(fsm.samples(AutoArm::baseline, 1) == 0);
    fsm.disable();
    CHECK(!fsm.enabled());
    CHECK(fsm.active() == AutoArm::baseline);
    fsm.enable(1);
    for (uint32_t i = 0; i < AutoPolicyController::kExploreSamples; ++i)
        fsm.observe(128, 1280.0, true);
    CHECK(fsm.phase() == AutoPhase::exploit);
    CHECK(fsm.active() == AutoArm::baseline);
    CHECK(fsm.metadata_bytes() < 4096);
    std::puts("aei_auto_policy_test: OK");
}
