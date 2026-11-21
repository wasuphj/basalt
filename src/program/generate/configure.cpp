#include "run.hpp"
#include "session_model.hpp"
#include "../cli/gpu.hpp"
#include "../cli/startup.hpp"

#include "basalt/core/host_pool.hpp"
#include "basalt/log.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/models/builtin.hpp"
#include "basalt/program/gpu_select.hpp"
#include "basalt/program/options.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

namespace basalt::program::generate {

namespace cli = basalt::program::cli;

int run(int argc, char** argv) {
    basalt::models::register_builtin_models();
    std::unique_ptr<basalt::program::CommonOptions> options;
    const basalt::model::ModelAdapter* adapter = nullptr;
    if (const auto code = cli::start(argc, argv, options, adapter)) return *code;
    basalt::program::CommonOptions& common = *options;
    // ---- the GPUs: chosen before any CUDA call, which then sees only them, the main card as device 0
    std::vector<basalt::program::GpuCard> chosen_cards;
    const auto layer_stage = [&](double main_gbps, double peer_gbps) {
        return adapter != nullptr && adapter->second_card_runs_layers(common, main_gbps, peer_gbps);
    };
    if (const auto code = cli::select_gpus(common, chosen_cards, (adapter != nullptr ? adapter->max_expert_cards() : basalt::model::max_expert_cards()) + 1,
                                         adapter != nullptr && adapter->second_card_reads_prompts(), layer_stage)) return *code;
    // ---- the fixed host pool: opened before any host buffer of the model is made, and refused (never worked around) when it cannot be
    if (common.host_pool_bytes > 0) {
        if (!common.serve || common.batch > 0 || chosen_cards.size() != 1 || common.prompt_cache <= 0 || common.conversation_cache_disk_mib <= 0) {
            basalt::log::error("basalt", "--host-pool-gib needs single-GPU solo --serve, checkpoints and a disk conversation cache");
            return 1;
        }
        basalt::platform::HostMemory memory;
        if (!basalt::platform::host_available_memory(memory)) {
            basalt::log::error("basalt", "host pool cannot read RAM/cgroup availability");
            return 1;
        }
        const uint64_t permitted = std::min(memory.available, basalt::platform::total_physical_memory());
        std::string error;
        if (!basalt::core::open_host_pool(common.host_pool_bytes, permitted, error)) {
            basalt::log::error("basalt", "%s", error.c_str());
            return 1;
        }
    }
    // ---- where every loader reads from: the sections of the .basalt file
    if (const auto code = cli::prepare_model_process(common)) return *code;
    if (const std::optional<int> code = adapter->run(common, chosen_cards)) return *code;
    return run_session_model(common, *adapter);
}

}
