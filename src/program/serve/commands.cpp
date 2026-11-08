// src/program/serve/commands.cpp - scheduling and protocol command dispatch.
#include "request.hpp"

namespace basalt::program::serve {

namespace {
// the busy clock runs while a request or a batch window does
struct BusyClock {
    double& total;
    bool& on;
    Clock::time_point t0;
    BusyClock(double& total_, bool& on_, Clock::time_point& from) : total(total_), on(on_), t0(Clock::now()) {
        from = t0;
        on = true;
    }
    ~BusyClock() {
        total += std::chrono::duration<double>(Clock::now() - t0).count();
        on = false;
    }
};
}  // namespace

Flow Runtime::dispatch() {
    if (o.host_pool_bytes > 0) relieve_host_pressure();
    if (batch_on() || (piped && pipe_inflight())) {
        if (!try_next_line(line)) {
            const BusyClock busy(busy_s, busy_on, busy_from);
            if (!(piped ? pump(true) : batch_step())) return Flow::error;
            return Flow::next;
        }
        // a request reads its prompt through every stage: the groups in flight finish first
        if (piped && line.rfind("BSTOP ", 0) != 0 && !pipe_drain()) return Flow::error;
    } else {
        // idle: once no line has come for a second (a client's next turn usually comes sooner), what a crash would
        // lose goes to disk, then the idle conversations whose time is up leave VRAM, one at a time while no line
        // waits
        bool got = false, parking = true;
        std::string pe;
        while (!got && (disk.enabled() || parking)) {
            const int r = next_line_within(line, 1000);
            if (r < 0) return Flow::quit;
            got = r == 1;
            if (!got) relieve_host_pressure();
            while (!got && !(got = try_next_line(line)) && disk.enabled() && spill_to_disk(false, 1) > 0) {}
            if (got) break;
            parking = parking && park_idle_engine(pe);
            if (!pe.empty()) { std::printf("ERR %s\n", pe.c_str()); return Flow::error; }
            if (!parking && o.host_pool_bytes == 0) break;
        }
        if (!got && !next_line(line)) return Flow::quit;
    }
    // --batch: BSTOP <slot> ends that slot at its next batch window; BGEN <slot> <max_new> ... reads the
    // request's prompt and first token as a GEN 1, then continues it in that slot
    if (line.rfind("BSTOP ", 0) == 0) {
        const int b = std::atoi(line.c_str() + 6);
        if (b >= 0 && b < (int) bs.size()) bs[(size_t) b].stop = true;
        return Flow::next;
    }
    if (line.rfind("BSCRIPT ", 0) == 0) {   // BSCRIPT <slot> <seed> <id,id,...>: before the slot's BGEN
        char* e1 = nullptr;
        const long b = std::strtol(line.c_str() + 8, &e1, 10);
        char* e2 = nullptr;
        const unsigned long long seed = std::strtoull(e1, &e2, 10);
        std::vector<int64_t> sids;
        std::string se;
        if (b < 0 || b >= (long) bs.size() || e2 == e1 || !cli::parse_i64_list(e2, sids, se)) {
            std::printf("ERR BSCRIPT: bad slot or list\n");
            std::fflush(stdout);
            return Flow::next;
        }
        bscript[(size_t) b].first.assign(sids.begin(), sids.end());
        bscript[(size_t) b].second = seed;
        return Flow::next;
    }
    if (line.rfind("BYIELD", 0) == 0) return Flow::next;   // for a prompt read that has ended meanwhile
    if (line.rfind("ENC ", 0) == 0) {   // the in-engine image encoder, between requests and windows
        enc_command(line);
        return Flow::next;
    }
    if (line.rfind("PROBE", 0) == 0) {   // between requests, not a request (basalt-probe's commands)
        std::string perr;
        if (line == "PROBE policy split on") {
            if (batch_on() || (piped && pipe_inflight())) perr = "PROBE split: inference in flight";
            else if (aei_rank_costs || aei_reuse_index || aei_prefetch) {
                perr = "PROBE split: return to baseline before enabling a different policy";
            } else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                if (g.n_layers <= 0 ||
                    g.n_layers > (int64_t)basalt::core::aei::LayerCostRank::kMaxLayers) {
                    perr = "PROBE split: layer count outside bounded cost model";
                } else {
                    if (!aei_split_costs)
                        aei_split_costs = std::make_unique<basalt::core::aei::LayerCostRank>();
                    ver.set_aei_cost_placement_probe(aei_split_costs.get());
                    ver_b.set_aei_cost_placement_probe(aei_split_costs.get());
                    std::printf("PROBE policy split on: measuring CPU misses\n");
                }
            }
        } else if (line == "PROBE policy split refresh") {
            if (batch_on() || (piped && pipe_inflight())) perr = "PROBE split: inference in flight";
            else if (!aei_split_costs) perr = "PROBE split: not enabled";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                std::string e0, e1;
                const bool have0 = ver.aei_staged_split_available();
                const bool have1 = ver_b.aei_staged_split_available();
                const bool ok0 = !have0 || ver.aei_update_staged_split(aei_split_costs.get(), false, e0);
                const bool ok1 = !have1 || ver_b.aei_update_staged_split(aei_split_costs.get(), false, e1);
                if ((!have0 && !have1) || !ok0 || !ok1) {
                    std::string ignored;
                    if (have0 && ok0) (void)ver.aei_update_staged_split(nullptr, true, ignored);
                    if (have1 && ok1) (void)ver_b.aei_update_staged_split(nullptr, true, ignored);
                    perr = !have0 && !have1 ? "PROBE split: no available staging path"
                        : "PROBE split: " + (ok0 ? e1 : e0);
                } else {
                    aei_split_last_cpu_ns = aei_split_costs->reference_cost_ns((size_t)g.n_layers);
                    std::printf("PROBE policy split updated cpu_ns=%llu dma_ns=%llu/%llu\n",
                        (unsigned long long)aei_split_costs->reference_cost_ns((size_t)g.n_layers),
                        (unsigned long long)ver.aei_staged_dma_ns(),
                        (unsigned long long)ver_b.aei_staged_dma_ns());
                }
            }
        } else if (line == "PROBE policy prefetch on") {
            if (aei_split_costs) perr = "PROBE policy: return split to baseline first";
            else
            if (batch_on() || (piped && pipe_inflight())) perr = "PROBE policy: inference in flight";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                const uint64_t total = (uint64_t)g.n_layers * (uint64_t)g.n_expert;
                if (!total || total > basalt::core::aei::NextExpertPrefetch::kMaxExperts) {
                    perr = "PROBE policy: model geometry exceeds bounded predictor";
                } else {
                    ver.set_aei_cost_placement_probe(nullptr);
                    ver_b.set_aei_cost_placement_probe(nullptr);
                    aei_rank_costs.reset();
                    ver.set_aei_reuse_probe(nullptr, 0);
                    ver_b.set_aei_reuse_probe(nullptr, 0);
                    aei_reuse_index.reset();
                    if (!aei_prefetch)
                        aei_prefetch = std::make_unique<basalt::core::aei::NextExpertPrefetch>((size_t)total);
                    ver.set_aei_predictive_prefetch(aei_prefetch.get(), (uint32_t)g.n_expert);
                    ver_b.set_aei_predictive_prefetch(aei_prefetch.get(), (uint32_t)g.n_expert);
                    std::printf("PROBE policy prefetch on metadata_bytes=%zu\n",
                                aei_prefetch->metadata_bytes());
                }
            }
        } else if (line == "PROBE policy reuse on") {
            if (aei_split_costs) perr = "PROBE policy: return split to baseline first";
            else
            if (batch_on() || (piped && pipe_inflight())) perr = "PROBE policy: inference in flight";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                const uint64_t total = (uint64_t)g.n_layers * (uint64_t)g.n_expert;
                if (!total || total > basalt::core::aei::ExpertReuseIndex::kMaximumExperts)
                    perr = "PROBE policy: model geometry exceeds bounded reuse index";
                else {
                    ver.set_aei_predictive_prefetch(nullptr, 0);
                    ver_b.set_aei_predictive_prefetch(nullptr, 0);
                    aei_prefetch.reset();
                    // Policy choices are mutually exclusive; keep an unambiguous A/B.
                    ver.set_aei_cost_placement_probe(nullptr);
                    ver_b.set_aei_cost_placement_probe(nullptr);
                    aei_rank_costs.reset();
                    if (!aei_reuse_index)
                        aei_reuse_index = std::make_unique<basalt::core::aei::ExpertReuseIndex>((size_t)total);
                    ver.set_aei_reuse_probe(aei_reuse_index.get(), (uint32_t)g.n_expert);
                    ver_b.set_aei_reuse_probe(aei_reuse_index.get(), (uint32_t)g.n_expert);
                    std::printf("PROBE policy reuse on metadata_bytes=%zu\n",
                                aei_reuse_index->metadata_bytes());
                }
            }
        } else if (line == "PROBE policy cost on") {
            if (aei_split_costs) perr = "PROBE policy: return split to baseline first";
            else
            if (batch_on() || (piped && pipe_inflight())) perr = "PROBE policy: inference in flight";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                if (g.n_layers <= 0 || g.n_layers > (int64_t)basalt::core::aei::LayerCostRank::kMaxLayers)
                    perr = "PROBE policy: layer count exceeds bounded model";
                else {
                    ver.set_aei_reuse_probe(nullptr, 0);
                    ver_b.set_aei_reuse_probe(nullptr, 0);
                    aei_reuse_index.reset();
                    ver.set_aei_predictive_prefetch(nullptr, 0);
                    ver_b.set_aei_predictive_prefetch(nullptr, 0);
                    aei_prefetch.reset();
                    if (!aei_rank_costs)
                        aei_rank_costs = std::make_unique<basalt::core::aei::LayerCostRank>();
                    ver.set_aei_cost_placement_probe(aei_rank_costs.get());
                    ver_b.set_aei_cost_placement_probe(aei_rank_costs.get());
                    std::printf("PROBE policy cost on reference_ns=%llu\n",
                        (unsigned long long)aei_rank_costs->reference_cost_ns((size_t)g.n_layers));
                }
            }
        } else if (line == "PROBE policy baseline") {
            if (batch_on() || (piped && pipe_inflight())) perr = "PROBE policy: inference in flight";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                ver.set_aei_cost_placement_probe(nullptr);
                ver_b.set_aei_cost_placement_probe(nullptr);
                aei_rank_costs.reset();
                ver.set_aei_reuse_probe(nullptr, 0);
                ver_b.set_aei_reuse_probe(nullptr, 0);
                aei_reuse_index.reset();
                ver.set_aei_predictive_prefetch(nullptr, 0);
                ver_b.set_aei_predictive_prefetch(nullptr, 0);
                aei_prefetch.reset();
                if (aei_split_costs) {
                    std::string ignored;
                    if (ver.aei_staged_split_available())
                        (void)ver.aei_update_staged_split(nullptr, true, ignored);
                    if (ver_b.aei_staged_split_available())
                        (void)ver_b.aei_update_staged_split(nullptr, true, ignored);
                    ver.set_aei_cost_placement_probe(nullptr);
                    ver_b.set_aei_cost_placement_probe(nullptr);
                    aei_split_costs.reset();
                    aei_split_last_cpu_ns = 0;
                }
                std::printf("PROBE policy baseline\n");
            }
        } else if (line == "PROBE policy stats") {
            if (aei_split_costs) {
                std::printf("PROBE policy split cpu_ns=%llu dma_ns=%llu/%llu\n",
                            (unsigned long long)aei_split_costs->reference_cost_ns((size_t)g.n_layers),
                            (unsigned long long)ver.aei_staged_dma_ns(),
                            (unsigned long long)ver_b.aei_staged_dma_ns());
            } else if (aei_prefetch) {
                std::printf("PROBE policy prefetch metadata_bytes=%zu dropped_samples=%llu\n",
                            aei_prefetch->metadata_bytes(),
                            (unsigned long long)aei_prefetch->dropped_samples());
            } else if (aei_reuse_index) {
                std::printf("PROBE policy reuse metadata_bytes=%zu\n", aei_reuse_index->metadata_bytes());
            } else if (!aei_rank_costs) perr = "PROBE policy: baseline";
            else {
                std::printf("PROBE policy cost reference_ns=%llu\n",
                    (unsigned long long)aei_rank_costs->reference_cost_ns((size_t)g.n_layers));
            }
        } else if (line == "PROBE cost on") {
            if (batch_on() || (piped && pipe_inflight())) perr = "PROBE cost: inference in flight";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                if (!aei_cpu_costs) aei_cpu_costs = std::make_unique<basalt::core::aei::ShadowScheduler>();
                ver.set_aei_cpu_cost_probe(aei_cpu_costs.get());
                ver_b.set_aei_cpu_cost_probe(aei_cpu_costs.get());
                std::printf("PROBE cost on\n");
            }
        } else if (line == "PROBE cost off") {
            if (batch_on() || (piped && pipe_inflight())) perr = "PROBE cost: inference in flight";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                ver.set_aei_cpu_cost_probe(nullptr);
                ver_b.set_aei_cpu_cost_probe(nullptr);
                aei_cpu_costs.reset();
                std::printf("PROBE cost off\n");
            }
        } else if (line == "PROBE cost report") {
            if (batch_on() || (piped && pipe_inflight())) perr = "PROBE cost: inference in flight";
            else if (!aei_cpu_costs) perr = "PROBE cost: not enabled";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                for (size_t b = 0; b < basalt::core::aei::ShadowScheduler::kBuckets; ++b) {
                    const uint32_t n = (uint32_t) 1 << b;
                    const auto cpu = aei_cpu_costs->estimate(basalt::core::aei::Device::cpu, n);
                    std::printf("AEI_COST cpu %u %u %llu\n", n, cpu.observations,
                                (unsigned long long)cpu.execution_ns);
                }
                std::printf("PROBE cost end dropped_samples=%llu\n",
                            (unsigned long long)aei_cpu_costs->dropped_samples());
            }
        } else if (line.rfind("PROBE trace file ", 0) == 0) {
            if (batch_on() || (piped && pipe_inflight())) perr = "PROBE trace: inference in flight";
            else if (aei_trace_writer) perr = "PROBE trace: file writer already active";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                const std::string path = line.substr(17);
                if (path.empty()) perr = "PROBE trace file: missing output path";
                else {
                    if (!aei_trace) aei_trace = std::make_unique<basalt::core::aei::RouterEventBuffer>();
                    auto writer = std::make_unique<basalt::platform::AeiTraceWriter>();
                    if (!writer->start(path, *aei_trace, perr)) {
                        // Preserve any prior ring/diagnostics on startup error.
                    } else {
                        aei_trace_writer = std::move(writer);
                        std::printf("PROBE trace file on capacity=%zu\n",
                                    basalt::core::aei::RouterEventBuffer::kCapacity);
                    }
                }
            }
        } else if (line == "PROBE trace stats") {
            if (!aei_trace) perr = "PROBE trace: not enabled";
            else {
                std::printf("PROBE trace stats dropped_capacity=%llu dropped_contention=%llu written=%llu io_errors=%llu\n",
                            (unsigned long long)aei_trace->dropped_capacity(),
                            (unsigned long long)aei_trace->dropped_contention(),
                            (unsigned long long)(aei_trace_writer ? aei_trace_writer->written() : 0),
                            (unsigned long long)(aei_trace_writer ? aei_trace_writer->io_errors() : 0));
            }
        } else if (line == "PROBE trace on") {
            if (batch_on() || (piped && pipe_inflight())) perr = "PROBE trace: inference in flight";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                if (!aei_trace) aei_trace = std::make_unique<basalt::core::aei::RouterEventBuffer>();
                std::printf("PROBE trace on capacity=%zu\n",
                            basalt::core::aei::RouterEventBuffer::kCapacity);
            }
        } else if (line == "PROBE trace off") {
            if (batch_on() || (piped && pipe_inflight())) perr = "PROBE trace: inference in flight";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                stop_aei_trace();
                std::printf("PROBE trace off\n");
            }
        } else if (line == "PROBE trace drain") {
            if (batch_on() || (piped && pipe_inflight())) perr = "PROBE trace: inference in flight";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                if (!aei_trace) perr = "PROBE trace: not enabled";
                else if (aei_trace_writer) perr = "PROBE trace: file writer owns the drain";
                else {
                    std::array<basalt::core::aei::RoutedExpertEvent, 128> events{};
                    const size_t count = aei_trace->drain(events.data(), events.size());
                    for (size_t i = 0; i < count; ++i) {
                        const auto& e = events[i];
                        std::printf("AEI %llu %llu %u %u\n",
                                    (unsigned long long)e.stream,
                                    (unsigned long long)e.sequence,
                                    (unsigned)e.layer, (unsigned)e.expert);
                    }
                    std::printf("PROBE trace end count=%zu dropped_capacity=%llu dropped_contention=%llu\n",
                                count, (unsigned long long)aei_trace->dropped_capacity(),
                                (unsigned long long)aei_trace->dropped_contention());
                }
            }
        } else if (line.rfind("PROBE set ", 0) == 0) {   // PROBE set workers=K share=F: live values for the next requests
            if (batch_on()) perr = "PROBE: not while batch slots are decoding";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                std::istringstream kvs(line.substr(10));
                for (std::string kv; kvs >> kv;) {
                    const size_t eq = kv.find('=');
                    const std::string key = eq == std::string::npos ? kv : kv.substr(0, eq);
                    const double v = eq == std::string::npos ? -1.0 : std::atof(kv.c_str() + eq + 1);
                    if (key == "workers" && v >= 1) pool.set_active_workers((int) v);
                    else if (key == "share" && v >= 0 && v <= 1) sp.set_cpu_share(v);
                    else perr = "PROBE: bad setting " + kv;
                }
                if (perr.empty()) std::printf("PROBE end workers=%d\n", pool.active_workers());
            }
        } else if (line == "PROBE adapt" || line.rfind("PROBE adapt ", 0) == 0) {
            if (batch_on()) perr = "PROBE: not while batch slots are decoding";
            else {
                if (ba_job) { ba_thr.join(); ba_job = false; }
                (void) probe_adapt(line, perr);
            }
        } else if (line != "PROBE rates") perr = "PROBE: expected rates, set, adapt, trace, cost or policy";
        else if (batch_on()) perr = "PROBE: not while batch slots are decoding";
        else {
            if (ba_job) { ba_thr.join(); ba_job = false; }
            basalt::program::ProbeEnv penv;
            penv.pool = &pool;
            penv.src = srcp;
            penv.peer = peer.valid() ? &peer : nullptr;
            penv.n_layers = g.n_layers;
            penv.n_expert = g.n_expert;
            basalt::program::probe_rates(penv, stdout, perr);
        }
        if (!perr.empty()) std::printf("ERR %s\n", perr.c_str());
        std::fflush(stdout);
        return Flow::next;
    }
    if (line.rfind("VRAM", 0) == 0) {   // between requests, not a request
        std::string verr;
        if (batch_on()) verr = "VRAM: not while batch slots are decoding";
        else vram_command(line, verr, false);
        if (!verr.empty()) std::printf("ERR %s\n", verr.c_str());
        std::fflush(stdout);
        return Flow::next;
    }
    // SLEEP / WAKE, between requests: the expert cache gives its VRAM back down to its smallest size (every
    // expert keeps a RAM copy and the slot it held is remembered), so the process can be suspended with little
    // VRAM left to save; WAKE maps the cache again and puts each expert back in its slot.  Reply: a VRAM line.
    if (line == "SLEEP" || line == "WAKE") {
        std::string verr;
        const bool down = line == "SLEEP";
        if (batch_on()) {
            verr = line + ": not while batch slots are decoding";
        } else {
            // A peer card's tier goes too.  SLEEP then drops the RAM copy's registration with the driver (a process
            // checkpoint would map the copy into every context again on restore); WAKE registers it again while
            // both caches refill from the ranges already back.  A tier or a registration that cannot come back
            // fails WAKE: the server then starts the engine again.
            std::string perr, reply, uerr;
            const auto tp = Clock::now();
            const bool repin = !down && src.complement_unpinned();
            std::function<bool(const void*, uint64_t)> ready;
            if (repin) {
                src.repin_start(4);
                ready = [this](const void* p, uint64_t n) { return src.repin_wait(p, n); };
            }
            const bool peer_ok = !peer.valid() || (down ? peer.sleep(perr) : peer.wake(perr, ready));
            vram_command(down ? "VRAM " + std::to_string(1ll << 30) : std::string("VRAM"), verr, true, &reply);
            if (repin) {
                if (!src.repin_finish(uerr) && verr.empty()) verr = "WAKE: " + uerr;
                basalt::log::info("basalt serve", "WAKE: the RAM copy registered again beside the caches' refill "
                                  "in %.0f ms", std::chrono::duration<double, std::milli>(Clock::now() - tp).count());
            }
            if (down && verr.empty() && src.complement_ready()) {
                const auto tu = Clock::now();
                if (src.unpin_complement(uerr))
                    basalt::log::info("basalt serve", "SLEEP: the RAM copy's registration dropped in %.0f ms",
                                      std::chrono::duration<double, std::milli>(Clock::now() - tu).count());
                else
                    basalt::log::info("basalt serve", "SLEEP: the RAM copy stays registered (%s)", uerr.c_str());
            }
            if (!down && peer.valid() && peer_ok && !peer.wake_wait(perr)) {
                if (verr.empty()) verr = "WAKE: the peer card's tier could not come back (" + perr + ")";
            } else if (!down && !peer_ok) {
                if (verr.empty()) verr = "WAKE: the peer card's tier could not come back (" + perr + ")";
            } else if (!peer_ok) {
                basalt::log::warn("basalt serve", "SLEEP: the peer tier stays as it is (%s)", perr.c_str());
            } else if (peer.valid()) {
                basalt::log::info("basalt serve", "%s: the peer tier %s; %.0f ms with the main cache",
                                  line.c_str(), down ? "gave its VRAM back" : "is back",
                                  std::chrono::duration<double, std::milli>(Clock::now() - tp).count());
            }
            if (verr.empty()) std::fputs(reply.c_str(), stdout);
        }
        // asleep, the driver holds the engine's GPU state in RAM until the wake: the RAM guard judges no reading
        // then (no shed, no admission wait from it), and judges a fresh one at the wake
        if (down && verr.empty()) basalt::platform::ram_guard_pause(true);
        if (!down) basalt::platform::ram_guard_pause(false);
        if (!verr.empty()) std::printf("ERR %s\n", verr.c_str());
        std::fflush(stdout);
        return Flow::next;
    }
    if (line.rfind("ERR_GENT ", 0) == 0) {
        std::printf("ERR %s\n", line.c_str() + 9);
        std::fflush(stdout);
        return Flow::next;
    }
    basalt::platform::ram_guard_pause(false);   // a request without a WAKE first: the guard judges again
    admit_slot = -1;
    solo_slot = -1;
    if (line.rfind("SGEN ", 0) == 0 || line.rfind("SGENI ", 0) == 0) {   // SGEN <slot> <max_new> ...: a GEN in that slot
        const bool simg = line.rfind("SGENI ", 0) == 0;
        char* e1 = nullptr;
        const long b = std::strtol(line.c_str() + (simg ? 6 : 5), &e1, 10);
        if (!slot0_is_solo || bs.empty() || b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || e1 == line.c_str() + (simg ? 6 : 5)) {
            std::printf("ERR SGEN: no such free slot or slot 0 is not the solo session\n");
            std::fflush(stdout);
            return Flow::next;
        }
        solo_slot = (int) b;
        line = std::string(simg ? "GENI" : "GEN") + e1;
    }
    if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
        const bool bimg = line.rfind("BGENI ", 0) == 0;
        char* e1 = nullptr;
        const long b = std::strtol(line.c_str() + (bimg ? 6 : 5), &e1, 10);
        char* e2 = nullptr;
        const long long mn = std::strtoll(e1, &e2, 10);
        if (bs.empty() || b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || mn < 1 || e2 == e1) {
            std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", o.batch);
            std::fflush(stdout);
            return Flow::next;
        }
        admit_slot = (int) b;
        admit_max_new = mn;
        line = std::string(bimg ? "GENI 1" : "GEN 1") + e2;
    }
    // every --expert-profile-save-every minutes, before the next request (at QUIT: after the loop)
    if (!heat.empty() && line != "QUIT" && o.expert_profile_save_min > 0 &&
        Clock::now() - profile_saved_at >= std::chrono::duration<double>(o.expert_profile_save_min * 60.0))
        save_profile("periodic");
    if (ba_job) { ba_thr.join(); ba_job = false; }   // (a request runs the same copy steps itself)
    if (line == "QUIT") return Flow::quit;
    // the watchdog watches a request from here until this iteration ends, whichever way it ends
    struct BusyScope {
        BusyScope() { basalt::core::progress().busy.store(true); basalt::core::progress_at("request"); }
        ~BusyScope() { basalt::core::progress().busy.store(false); basalt::core::progress_at("idle"); }
    } busy_scope;
    const BusyClock busy(busy_s, busy_on, busy_from);
    Request request(*this);
    return request.execute();
}

}  // namespace basalt::program::serve
