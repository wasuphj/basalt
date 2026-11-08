#include "basalt/core/aei_auto_policy.hpp"
#include <cstdio>
#include <string>
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

    AutoPolicyController pp_guard;
    pp_guard.enable(allowed);
    for (uint32_t i = 0; i < AutoPolicyController::kExploreSamples; ++i)
        pp_guard.observe(128, 1280.0, true, 128, 1000.0);
    for (uint32_t i = 0; i < AutoPolicyController::kExploreSamples; ++i)
        pp_guard.observe(128, 850.0, true, 128, 1250.0);
    CHECK(pp_guard.prefill_samples(AutoArm::baseline, 1) == AutoPolicyController::kExploreSamples);
    CHECK(pp_guard.active() == AutoArm::baseline);  // TP gain cannot hide PP regression
    pp_guard.enable(allowed);
    for (uint32_t i = 0; i < AutoPolicyController::kExploreSamples; ++i)
        pp_guard.observe(128, 1280.0, true, 128, 1000.0);
    for (uint32_t i = 0; i < AutoPolicyController::kExploreSamples; ++i)
        pp_guard.observe(128, 850.0, true, 128, 990.0);
    CHECK(pp_guard.active() == AutoArm::prefetch);
    AutoPolicyController audit;
    audit.enable(allowed);
    for (uint32_t i = 0; i < AutoPolicyController::kExploreSamples; ++i)
        audit.observe(128, 1280.0, true);
    for (uint32_t i = 0; i < AutoPolicyController::kExploreSamples; ++i)
        audit.observe(128, 850.0, true);
    CHECK(audit.active() == AutoArm::prefetch);
    for (uint32_t i = 0; i < AutoPolicyController::kAuditEvery; ++i)
        audit.observe(128, 850.0, true);
    CHECK(audit.phase() == AutoPhase::audit);
    CHECK(audit.active() == AutoArm::baseline);
    for (uint32_t i = 0; i < AutoPolicyController::kAuditSamples; ++i)
        audit.observe(128, 1280.0, true);
    CHECK(audit.phase() == AutoPhase::exploit);
    CHECK(audit.active() == AutoArm::prefetch);
    CHECK(audit.audits() == 1);
    // Temporary baseline audit samples never mutate the original
    // exploration baseline or candidate histories.
    CHECK(audit.samples(AutoArm::baseline, 1) ==
          AutoPolicyController::kExploreSamples);
    CHECK(audit.samples(AutoArm::prefetch, 1) ==
          AutoPolicyController::kExploreSamples + AutoPolicyController::kAuditEvery);
    for (uint32_t i = 0; i < AutoPolicyController::kAuditEvery; ++i)
        audit.observe(128, 850.0, true);
    CHECK(audit.phase() == AutoPhase::audit);
    for (uint32_t i = 0; i < AutoPolicyController::kAuditSamples; ++i)
        audit.observe(128, 700.0, true); // baseline now wins
    CHECK(audit.phase() == AutoPhase::cooldown);
    CHECK(audit.active() == AutoArm::baseline);
    CHECK(audit.audits() == 2);
    AutoPolicyController combo;
    constexpr uint32_t mix = (1u << (unsigned)AutoArm::baseline) |
                             (1u << (unsigned)AutoArm::prefetch_split);
    combo.enable(mix);
    CHECK(combo.supports(AutoArm::prefetch_split));
    for (uint32_t i=0; i<AutoPolicyController::kExploreSamples; ++i)
        combo.observe(128, 1280.0, true, 128, 1000.0);
    CHECK(combo.active() == AutoArm::prefetch_split);
    for (uint32_t i=0; i<AutoPolicyController::kExploreSamples; ++i)
        combo.observe(128, 780.0, true, 128, 980.0);
    CHECK(combo.phase() == AutoPhase::exploit);
    CHECK(combo.active() == AutoArm::prefetch_split);
    CHECK(std::string(auto_arm_name(AutoArm::prefetch_split)) == "prefetch+split");
    std::puts("aei_auto_policy_test: OK");
}
