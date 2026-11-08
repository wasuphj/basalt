// src/program/serve/runtime.cpp - lifecycle of the persistent engine.
#include "state.hpp"

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
    return run_requests();
}

int run(Context& context) {
    Runtime runtime(context);
    return runtime.execute();
}

void Runtime::stop_aei_trace() {
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

int Runtime::run_requests() {
    for (;;) {
        const Flow flow = dispatch();
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
