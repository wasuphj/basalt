// src/program/serve/runtime.cpp - lifecycle of the persistent engine.
#include "state.hpp"
#include "basalt/platform/aei_policy_store.hpp"
#include <algorithm>
#include <charconv>
#include <string_view>

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

std::string Runtime::aei_checkpoint_identity() const {
    if (!model_file || !chosen_cards || chosen_cards->empty()) return {};
    const std::string model = model_file->expert_profile_identity();
    if (model.empty()) return {};
    std::string key = "AEI-EXEC-ENV-1\n" + model;
    key += "gpu=" + o.gpu + "\npeer=" + std::to_string(o.peer_device) +
           "\nsplit=" + o.split_device + "\n";
    key += "geometry=" + std::to_string(g.n_layers) + ":" +
           std::to_string(g.n_expert) + "\n";
    key += "adapt=" + std::to_string(o.adapt_every) + ":" +
           std::to_string(o.adapt_swaps) + ":" +
           std::to_string(o.adapt_decay) + "\n";
    for (const auto& card : *chosen_cards) {
        key += card.token + ":" + card.name + ":" +
               std::to_string(card.cc_major) + "." +
               std::to_string(card.cc_minor) + ":" +
               std::to_string(card.vram) + "\n";
    }
    return basalt::platform::aei_sha256_hex(key);
}

bool Runtime::aei_save_checkpoint() {
    if (!aei_auto.enabled() || o.aei_state_path.empty()) return true;
    const std::string id = aei_checkpoint_identity();
    if (id.empty()) return false;
    const std::string policy = aei_auto.checkpoint();
    const std::string learned = aei_prefetch ? aei_prefetch->checkpoint() : std::string{};
    const std::array<std::string,5> parts{
        policy, learned,
        aei_reuse_index ? aei_reuse_index->checkpoint() : std::string{},
        aei_rank_costs ? aei_rank_costs->checkpoint() : std::string{},
        aei_split_costs ? aei_split_costs->checkpoint() : std::string{}
    };
    std::string bundle = "AEI-BUNDLE-3\n";
    for (const auto& part : parts)
        bundle += std::to_string(part.size()) + "\n";
    for (const auto& part : parts) bundle += part;
    if (!basalt::platform::aei_save_policy_checkpoint(
            o.aei_state_path, id, bundle)) return false;
    aei_checkpoint_saved_at = aei_auto.completed();
    return true;
}

bool Runtime::aei_enable_automatic(std::string& err) {
    // Enabling anew is a new exploration session. Release old observers,
    // not simply the selector statistics; avoid carrying stale model costs.
    aei_auto.disable();
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
    if ((supported & (1u << (unsigned)AutoArm::prefetch)) &&
        (supported & (1u << (unsigned)AutoArm::split)))
        supported |= (1u << (unsigned)AutoArm::prefetch_split);
    aei_auto.enable(supported);
    if (!o.aei_state_path.empty()) {
        std::string payload;
        const std::string identity = aei_checkpoint_identity();
        bool resumed = false;
        if (!identity.empty() &&
            basalt::platform::aei_load_policy_checkpoint(
                o.aei_state_path, identity, payload)) {
            // Bundles are length-prefixed: the sparse table is binary and
            // must never be parsed as a newline-delimited text file.
            constexpr std::string_view v3 = "AEI-BUNDLE-3\n";
            if (payload.compare(0,v3.size(),v3)==0) {
                std::array<size_t,5> sizes{};
                std::array<std::string_view,5> pieces{};
                size_t pos=v3.size();
                bool valid=true;
                auto decimal = [](std::string_view value, size_t& output) {
                    const auto result = std::from_chars(
                        value.data(), value.data() + value.size(), output);
                    return result.ec == std::errc{} &&
                           result.ptr == value.data() + value.size();
                };
                for (auto& size : sizes) {
                    const auto end=payload.find('\n',pos);
                    if (end==std::string::npos ||
                        !decimal(std::string_view(payload).substr(pos,end-pos),size)) {
                        valid=false;
                        break;
                    }
                    pos=end+1;
                }
                const std::array<size_t,5> limits{
                    4096, 256u*1024u, 12u*1024u*1024u+12, 4096, 4096
                };
                if (valid) {
                    size_t needed=0;
                    for (size_t i=0;i<sizes.size();++i) {
                        if (sizes[i]>limits[i]) {valid=false;break;}
                        needed+=sizes[i];
                    }
                    valid=valid && needed==payload.size()-pos;
                }
                if (valid) {
                    for (size_t i=0;i<sizes.size();++i) {
                        pieces[i]=std::string_view(payload.data()+pos,sizes[i]);
                        pos+=sizes[i];
                    }
                    resumed=aei_auto.restore_checkpoint(pieces[0],supported);
                    if (resumed && !pieces[1].empty()) {
                        resumed=(supported & (1u << (unsigned)AutoArm::prefetch)) && total>0 &&
                                total<=basalt::core::aei::NextExpertPrefetch::kMaxExperts;
                        if (resumed) {
                            aei_prefetch=std::make_unique<
                                basalt::core::aei::NextExpertPrefetch>((size_t)total);
                            resumed=aei_prefetch->restore_checkpoint(pieces[1]);
                        }
                    }
                    if (resumed && !pieces[2].empty()) {
                        resumed=(supported & (1u << (unsigned)AutoArm::reuse)) && total>0 &&
                                total<=basalt::core::aei::ExpertReuseIndex::kMaximumExperts;
                        if (resumed) {
                            aei_reuse_index=std::make_unique<
                                basalt::core::aei::ExpertReuseIndex>((size_t)total);
                            resumed=aei_reuse_index->restore_checkpoint(pieces[2]);
                        }
                    }
                    if (resumed && !pieces[3].empty()) {
                        aei_rank_costs=std::make_unique<basalt::core::aei::LayerCostRank>();
                        resumed=aei_rank_costs->restore_checkpoint(pieces[3]);
                    }
                    if (resumed && !pieces[4].empty()) {
                        aei_split_costs=std::make_unique<basalt::core::aei::LayerCostRank>();
                        resumed=aei_split_costs->restore_checkpoint(pieces[4]);
                    }
                    if (!resumed) {
                        aei_prefetch.reset();
                        aei_reuse_index.reset();
                        aei_rank_costs.reset();
                        aei_split_costs.reset();
                        aei_auto.enable(supported);
                    }
                }
            } else {
            constexpr std::string_view prefix = "AEI-BUNDLE-2\n";
            if (payload.compare(0, prefix.size(), prefix) == 0) {
                const auto p1 = payload.find('\n', prefix.size());
                const auto p2 = p1 == std::string::npos ?
                    std::string::npos : payload.find('\n', p1 + 1);
                size_t policy_len = 0, index_len = 0;
                auto decimal = [](std::string_view value, size_t& output) {
                    const auto result = std::from_chars(
                        value.data(), value.data() + value.size(), output);
                    return result.ec == std::errc{} &&
                           result.ptr == value.data() + value.size();
                };
                if (p1 != std::string::npos && p2 != std::string::npos &&
                    decimal(std::string_view(payload).substr(prefix.size(),
                            p1 - prefix.size()), policy_len) &&
                    decimal(std::string_view(payload).substr(p1 + 1,
                            p2 - p1 - 1), index_len) &&
                    policy_len <= 4096 && index_len <= 256u * 1024u &&
                    payload.size() - (p2 + 1) == policy_len + index_len) {
                    const std::string_view policy(payload.data() + p2 + 1, policy_len);
                    const std::string_view sparse(payload.data() + p2 + 1 + policy_len,
                                                   index_len);
                    resumed = aei_auto.restore_checkpoint(policy, supported);
                    if (resumed && index_len > 0) {
                        if (!(supported & (1u << (unsigned)AutoArm::prefetch)) ||
                            total == 0 ||
                            total > basalt::core::aei::NextExpertPrefetch::kMaxExperts)
                            resumed = false;
                        else {
                            if (!aei_prefetch)
                                aei_prefetch = std::make_unique<
                                    basalt::core::aei::NextExpertPrefetch>((size_t)total);
                            resumed = aei_prefetch->restore_checkpoint(sparse);
                        }
                        if (!resumed) {
                            aei_prefetch.reset();
                            aei_auto.enable(supported);
                        }
                    }
                }
            } else {
                // Earlier statistics-only state.
                resumed = aei_auto.restore_checkpoint(payload, supported);
            }
            } // End legacy v2/raw checkpoint format.
        }
        if (resumed) {
            aei_checkpoint_saved_at = aei_auto.completed();
            basalt::log::info("basalt serve", "AEI restored model-bound policy and Expert transitions from %s",
                              o.aei_state_path.c_str());
        }
    }
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
    if (arm == AutoArm::cost || arm == AutoArm::split ||
        arm == AutoArm::prefetch_split) {
        if (g.n_layers <= 0 ||
            g.n_layers > (int64_t)basalt::core::aei::LayerCostRank::kMaxLayers) {
            err = "AEI auto policy: too many layers for cost model";
            return false;
        }
    }
    if ((arm == AutoArm::reuse || arm == AutoArm::prefetch ||
         arm == AutoArm::prefetch_split) &&
        (total == 0 || total > basalt::core::aei::NextExpertPrefetch::kMaxExperts)) {
        err = "AEI auto policy: expert geometry outside bounded index";
        return false;
    }
    if ((arm == AutoArm::split || arm == AutoArm::prefetch_split) &&
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
    if (aei_split_costs &&
        (!aei_auto.enabled() || (aei_auto_applied == AutoArm::split ||
          aei_auto_applied == AutoArm::prefetch_split))) {
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
    // During automatic exploration retain observer histories between
    // arms. Reset only at an explicit stop; a baseline audit should not
    // destroy the winning policy's learned transitions or CPU estimates.
    const bool retain = aei_auto.enabled();
    if (!retain) {
        aei_rank_costs.reset();
        aei_reuse_index.reset();
        aei_prefetch.reset();
        aei_split_costs.reset();
    }
    switch (arm) {
        case AutoArm::baseline: break;
        case AutoArm::cost:
            if (!aei_rank_costs)
                aei_rank_costs = std::make_unique<basalt::core::aei::LayerCostRank>();
            ver.set_aei_cost_placement_probe(aei_rank_costs.get());
            ver_b.set_aei_cost_placement_probe(aei_rank_costs.get());
            break;
        case AutoArm::reuse:
            if (!aei_reuse_index)
                aei_reuse_index = std::make_unique<basalt::core::aei::ExpertReuseIndex>((size_t)total);
            ver.set_aei_reuse_probe(aei_reuse_index.get(), (uint32_t)g.n_expert);
            ver_b.set_aei_reuse_probe(aei_reuse_index.get(), (uint32_t)g.n_expert);
            break;
        case AutoArm::prefetch:
            if (!aei_prefetch)
                aei_prefetch = std::make_unique<basalt::core::aei::NextExpertPrefetch>((size_t)total);
            ver.set_aei_predictive_prefetch(aei_prefetch.get(), (uint32_t)g.n_expert);
            ver_b.set_aei_predictive_prefetch(aei_prefetch.get(), (uint32_t)g.n_expert);
            sp.set_aei_predictive_prefetch(aei_prefetch.get());
            break;
        case AutoArm::split:
            if (!aei_split_costs)
                aei_split_costs = std::make_unique<basalt::core::aei::LayerCostRank>();
            ver.set_aei_cost_placement_probe(aei_split_costs.get());
            ver_b.set_aei_cost_placement_probe(aei_split_costs.get());
            break;
        case AutoArm::prefetch_split:
            if (!aei_prefetch)
                aei_prefetch = std::make_unique<basalt::core::aei::NextExpertPrefetch>((size_t)total);
            if (!aei_split_costs)
                aei_split_costs = std::make_unique<basalt::core::aei::LayerCostRank>();
            ver.set_aei_predictive_prefetch(aei_prefetch.get(), (uint32_t)g.n_expert);
            ver_b.set_aei_predictive_prefetch(aei_prefetch.get(), (uint32_t)g.n_expert);
            sp.set_aei_predictive_prefetch(aei_prefetch.get());
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
        if (flow == Flow::next && aei_split_costs &&
            (!aei_auto.enabled() ||
             aei_auto_applied == basalt::core::aei::AutoArm::split ||
             aei_auto_applied == basalt::core::aei::AutoArm::prefetch_split) &&
            !batch_on() &&
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
        if (flow == Flow::next && aei_auto.enabled() &&
            !o.aei_state_path.empty() &&
            aei_auto.completed() - aei_checkpoint_saved_at >= 64) {
            if (!aei_save_checkpoint())
                basalt::log::warn("basalt serve", "AEI policy evidence checkpoint write failed");
        }
        if (flow == Flow::error) {
            stop_aei_trace();
            return 1;
        }
        if (flow == Flow::quit) return shutdown();
    }
}

int Runtime::shutdown() {
    if (aei_auto.enabled() && !aei_save_checkpoint())
        basalt::log::warn("basalt serve", "AEI policy evidence checkpoint write failed");
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
