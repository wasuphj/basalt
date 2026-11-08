#include "basalt/core/aei_auto_policy.hpp"
#include "basalt/platform/aei_policy_store.hpp"
#include <cstdio>
#include <string>
using basalt::core::aei::AutoPolicyController;
using basalt::core::aei::AutoArm;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); return 1; } } while (0)
int main() {
    constexpr uint32_t supported = (1u << (unsigned)AutoArm::baseline) |
                                   (1u << (unsigned)AutoArm::prefetch);
    AutoPolicyController first;
    first.enable(supported);
    for (int i=0; i<AutoPolicyController::kExploreSamples; ++i)
        first.observe(128,1280.0,true,256,1600.0);
    CHECK(first.samples(AutoArm::baseline,1)==AutoPolicyController::kExploreSamples);
    const std::string payload=first.checkpoint();
    CHECK(payload.size() < 4096);
    AutoPolicyController second;
    CHECK(second.restore_checkpoint(payload,supported));
    CHECK(second.active()==AutoArm::baseline);
    CHECK(second.samples(AutoArm::baseline,1)==AutoPolicyController::kExploreSamples);
    CHECK(second.prefill_samples(AutoArm::baseline,2)==AutoPolicyController::kExploreSamples);
    CHECK(!second.restore_checkpoint("AEI-AUTO-1 invalid",supported));
    const std::string path = basalt::platform::temp_dir() +
        "/basalt-aei-policy-checkpoint-" + std::to_string((uintptr_t)&main);
    const std::string env = "model=stable-hash;gpu=stable-hardware";
    CHECK(basalt::platform::aei_save_policy_checkpoint(path,env,payload));
    std::string reloaded;
    CHECK(!basalt::platform::aei_load_policy_checkpoint(path,"model=different",reloaded));
    CHECK(basalt::platform::aei_load_policy_checkpoint(path,env,reloaded));
    CHECK(reloaded==payload);
    CHECK(second.restore_checkpoint(reloaded,supported));
    CHECK(basalt::platform::aei_save_policy_checkpoint(path,env,payload+" "));
    CHECK(basalt::platform::aei_load_policy_checkpoint(path,env,reloaded));
    CHECK(!second.restore_checkpoint(reloaded,supported) ||
          second.active()==AutoArm::baseline);
    (void)basalt::platform::remove_file(path);
    std::puts("aei_policy_checkpoint_test: OK");
}
