// src/program/serve/runtime.cpp - lifecycle of the persistent engine.
#include "state.hpp"
#include <algorithm>

namespace basalt::program::serve {

int Runtime::execute() {
    if (const int code = init_prompt()) return code;
    if (const int code = init_verify()) return code;
    if (const int code = init_conversations()) return code;
    if (const int code = init_tiers()) return code;
    if (const int code = init_protocol()) return code;
    if (const int code = announce()) return code;
    if (const int code = init_memory()) return code;
    if (const int code = capture_windows()) return code;
    if (const int code = init_batch()) return code;
    if (o.aei_auto) {
        std::string auto_error;
        if (!aei_enable_automatic(auto_error))
            basalt::log::warn("basalt serve", "AEI auto startup unavailable: %s",
                              auto_error.c_str());
    }
    return run_requests();
}

int run(Context& context) {
    Runtime runtime(context);
    return runtime.execute();
}

void Runtime::stop_aei_trace() {
    sp.set_aei_router_trace(nullptr);
    ver.set_aei_router_trace(nullptr, 0);
    ver_b.set_aei_router_trace(nullptr, 0);
    // A live writer owns the ring: always drain, flush and join first.
    if (aei_trace_writer) {
        aei_trace_writer->stop();
        const uint64_t overflow = aei_trace ? aei_trace->dropped_capacity() : UINT64_MAX;
        const uint64_t contention = aei_trace ? aei_trace->dropped_contention() : UINT64_MAX;
        const uint64_t io_errors = aei_trace_writer->io_errors();
        const bool complete = overflow == 0 && contention == 0 && io_errors == 0;
        std::printf("PROBE trace file end written=%llu dropped_capacity=%llu dropped_contention=%llu io_errors=%llu complete=%d\n",
                    (unsigned long long)aei_trace_writer->written(),
                    (unsigned long long)overflow, (unsigned long long)contention,
                    (unsigned long long)io_errors, complete ? 1 : 0);
        std::fflush(stdout);
        aei_trace_writer.reset();
    }
    aei_trace.reset();
}

bool Runtime::aei_enable_automatic(std::string& err) {
    if (!aei_apply_auto_policy(basalt::core::aei::AutoArm::baseline, err))
        return false;
    using basalt::core::aei::AutoArm;
    uint32_t supported = 1u << (unsigned)AutoArm::baseline;
    const uint64_t total = g.n_layers > 0 && g.n_expert > 0
        ? (uint64_t)g.n_layers * (uint64_t)g.n_expert : 0;
    const bool rank_ready = o.adapt_swaps > 0 && !host_res.empty() &&
                            !drive.d.usage.empty();
    if (rank_ready && xq_across &&
        g.n_layers <= (int64_t)basalt::core::aei::LayerCostRank::kMaxLayers)
        supported |= (1u << (unsigned)AutoArm::cost);
    if (rank_ready && xq_across && total > 0 &&
        total <= basalt::core::aei::ExpertReuseIndex::kMaximumExperts)
        supported |= (1u << (unsigned)AutoArm::reuse);
    if (rank_ready && total > 0 &&
        total <= basalt::core::aei::NextExpertPrefetch::kMaxExperts)
        supported |= (1u << (unsigned)AutoArm::prefetch);
    if (g.n_layers > 0 &&
        g.n_layers <= (int64_t)basalt::core::aei::LayerCostRank::kMaxLayers &&
        (ver.aei_staged_split_available() || ver_b.aei_staged_split_available()))
        supported |= (1u << (unsigned)AutoArm::split);
    aei_auto.enable(supported);
    basalt::log::info("basalt serve", "AEI auto policy enabled, candidate mask 0x%x", supported);
    return true;
}

bool Runtime::aei_apply_auto_policy(basalt::core::aei::AutoArm arm, std::string& err) {
    using basalt::core::aei::AutoArm;
    if (batch_on() || (piped && pipe_inflight()) || ba_job) {
        err = "AEI auto policy: execution still in flight";
        return false;
    }
    const uint64_t total = g.n_layers > 0 && g.n_expert > 0
        ? (uint64_t)g.n_layers * (uint64_t)g.n_expert : 0;
    // Validate before deactivating the previous policy.
    if (arm == AutoArm::cost || arm == AutoArm::split) {
        if (g.n_layers <= 0 ||
            g.n_layers > (int64_t)basalt::core::aei::LayerCostRank::kMaxLayers) {
            err = "AEI auto policy: too many layers for cost model";
            return false;
        }
    }
    if ((arm == AutoArm::reuse || arm == AutoArm::prefetch) &&
        (total == 0 || total > basalt::core::aei::NextExpertPrefetch::kMaxExperts)) {
        err = "AEI auto policy: expert geometry outside bounded index";
        return false;
    }
    if (arm == AutoArm::split &&
        !ver.aei_staged_split_available() && !ver_b.aei_staged_split_available()) {
        err = "AEI auto policy: no GPU staged-share engine";
        return false;
    }

    // All changes are allowed only at idle request boundaries. Removing a
    // callback's observer always precedes destroying its memory.
    ver.set_aei_cost_placement_probe(nullptr);
    ver_b.set_aei_cost_placement_probe(nullptr);
    ver.set_aei_reuse_probe(nullptr, 0);
    ver_b.set_aei_reuse_probe(nullptr, 0);
    ver.set_aei_predictive_prefetch(nullptr, 0);
    ver_b.set_aei_predictive_prefetch(nullptr, 0);
    sp.set_aei_predictive_prefetch(nullptr);
    if (aei_split_costs) {
        std::string ignored;
        if (ver.aei_staged_split_available() &&
            !ver.aei_update_staged_split(nullptr, true, ignored)) {
            err = "AEI auto policy: cannot restore primary GPU split: " + ignored;
            return false;
        }
        if (ver_b.aei_staged_split_available() &&
            !ver_b.aei_update_staged_split(nullptr, true, ignored)) {
            err = "AEI auto policy: cannot restore secondary GPU split: " + ignored;
            return false;
        }
        aei_split_last_cpu_ns = 0;
    }
    aei_rank_costs.reset();
    aei_reuse_index.reset();
    aei_prefetch.reset();
    aei_split_costs.reset();
    switch (arm) {
        case AutoArm::baseline: break;
        case AutoArm::cost:
            aei_rank_costs = std::make_unique<basalt::core::aei::LayerCostRank>();
            ver.set_aei_cost_placement_probe(aei_rank_costs.get());
            ver_b.set_aei_cost_placement_probe(aei_rank_costs.get());
            break;
        case AutoArm::reuse:
            aei_reuse_index = std::make_unique<basalt::core::aei::ExpertReuseIndex>((size_t)total);
            ver.set_aei_reuse_probe(aei_reuse_index.get(), (uint32_t)g.n_expert);
            ver_b.set_aei_reuse_probe(aei_reuse_index.get(), (uint32_t)g.n_expert);
            break;
        case AutoArm::prefetch:
            aei_prefetch = std::make_unique<basalt::core::aei::NextExpertPrefetch>((size_t)total);
            ver.set_aei_predictive_prefetch(aei_prefetch.get(), (uint32_t)g.n_expert);
            ver_b.set_aei_predictive_prefetch(aei_prefetch.get(), (uint32_t)g.n_expert);
            sp.set_aei_predictive_prefetch(aei_prefetch.get());
            break;
        case AutoArm::split:
            aei_split_costs = std::make_unique<basalt::core::aei::LayerCostRank>();
            ver.set_aei_cost_placement_probe(aei_split_costs.get());
            ver_b.set_aei_cost_placement_probe(aei_split_costs.get());
            break;
        default:
            err = "AEI auto policy: invalid policy";
            return false;
    }
    aei_auto_applied = arm;
    basalt::log::info("basalt serve", "AEI auto policy: %s",
                      basalt::core::aei::auto_arm_name(arm));
    return true;
}

void Runtime::aei_record_auto_request(int64_t generated, double decode_ms,
                                      int64_t fresh_prompt, double prompt_ms, bool healthy) {
    if (!aei_auto.enabled()) return;
    (void)aei_auto.observe(generated, decode_ms, healthy, fresh_prompt, prompt_ms);
}

int Runtime::run_requests() {
    for (;;) {
        const Flow flow = dispatch();
        // Refit the device's CPU/staging split between completed requests,
        // never in the graph or inside the CPU pool callback. Updates are
        // rate limited to >=10% measured CPU cost movement.
        if (flow == Flow::next && aei_split_costs && !batch_on() &&
            !ba_job && (!piped || !pipe_inflight())) {
            const uint64_t current = aei_split_costs->reference_cost_ns((size_t)g.n_layers);
            const uint64_t previous = aei_split_last_cpu_ns;
            const uint64_t delta = current >= previous ? current - previous : previous - current;
            if (current && (!previous || delta >= std::max<uint64_t>(1, previous / 10))) {
                const bool has_main = ver.aei_staged_split_available();
                const bool has_alt = ver_b.aei_staged_split_available();
                std::string first, second, ignored;
                const bool ok_main = !has_main ||
                    ver.aei_update_staged_split(aei_split_costs.get(), false, first);
                const bool ok_alt = !has_alt ||
                    ver_b.aei_update_staged_split(aei_split_costs.get(), false, second);
                if ((has_main || has_alt) && ok_main && ok_alt) {
                    aei_split_last_cpu_ns = current;
                } else {
                    if (has_main && ok_main)
                        (void)ver.aei_update_staged_split(nullptr, true, ignored);
                    if (has_alt && ok_alt)
                        (void)ver_b.aei_update_staged_split(nullptr, true, ignored);
                }
            }
        }
        if (flow == Flow::error) {
            stop_aei_trace();
            return 1;
        }
        if (flow == Flow::quit) return shutdown();
    }
}

int Runtime::shutdown() {
    // Final trace receipt must be emitted even on QUIT or stdin EOF.
    stop_aei_trace();
    // the conversations this process holds go to the disk tier before it ends
    if (disk.enabled()) {
        const auto t0 = Clock::now();
        const size_t written = spill_to_disk(true, SIZE_MAX);
        const auto st = disk.stats();
        basalt::log::info("basalt serve", "conversation cache: %zu conversations written to disk at exit in %.0f ms; "
                          "on disk %zu (%.2f GiB); since the start written %zu (%.2f GiB, %.2f GiB with the shared "
                          "files they reuse), read back %zu (%.2f GiB), evicted %zu, dropped %zu", written,
                          std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), st.files,
                          (double) st.bytes / 1073741824.0, st.written, (double) st.written_bytes / 1073741824.0,
                          (double) st.whole_bytes / 1073741824.0, st.restored, (double) st.read_bytes / 1073741824.0,
                          st.evicted, st.dropped);
    }
    for (int st = 0; st < std::max(n_stages, 1); ++st)
        for (const std::string& r : stage_ver(st).route_report()) basalt::log::info("basalt serve", "%s", r.c_str());
    save_profile("exit");   // QUIT, or the server closed stdin
    return 0;
}

}  // namespace basalt::program::serve
