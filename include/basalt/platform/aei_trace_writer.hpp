// Host OS boundary: the AEI trace file is written by a separate polling worker.
#pragma once
#include "basalt/core/aei_event_buffer.hpp"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>

namespace basalt::platform {

class AeiTraceWriter {
public:
    AeiTraceWriter() = default;
    ~AeiTraceWriter() { stop(); }
    AeiTraceWriter(const AeiTraceWriter&) = delete;
    AeiTraceWriter& operator=(const AeiTraceWriter&) = delete;

    bool start(const std::string& utf8_path, basalt::core::aei::RouterEventBuffer& source,
               std::string& error);
    void stop() noexcept;

    uint64_t written() const noexcept { return written_.load(std::memory_order_relaxed); }
    uint64_t io_errors() const noexcept { return io_errors_.load(std::memory_order_relaxed); }
    bool running() const noexcept { return active_.load(std::memory_order_relaxed); }

private:
    void work() noexcept;
    std::FILE* file_ = nullptr;
    basalt::core::aei::RouterEventBuffer* source_ = nullptr;
    std::thread worker_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> active_{false};
    std::atomic<uint64_t> written_{0};
    std::atomic<uint64_t> io_errors_{0};
};

} // namespace basalt::platform
