// Host-only AEI background trace file writer. All path I/O stays in platform.
#include "basalt/platform/aei_trace_writer.hpp"
#include "basalt/platform/file.hpp"
#include <array>
#include <chrono>
#include <exception>
#include <thread>

namespace basalt::platform {

bool AeiTraceWriter::start(const std::string& path,
                           basalt::core::aei::RouterEventBuffer& source,
                           std::string& error) {
    if (file_ || worker_.joinable()) { error = "AEI trace writer already started"; return false; }
    if (path.empty()) { error = "AEI trace writer path is empty"; return false; }
    file_ = open_file_utf8(path, "wb");
    if (!file_) { error = "AEI trace output could not be opened"; return false; }
    if (std::fputs("stream,sequence,layer,expert\n", file_) < 0) {
        error = "AEI trace header could not be written";
        std::fclose(file_);
        file_ = nullptr;
        return false;
    }
    source_ = &source;
    stop_.store(false, std::memory_order_relaxed);
    written_.store(0, std::memory_order_relaxed);
    io_errors_.store(0, std::memory_order_relaxed);
    // Publish running before the worker is scheduled; otherwise a fast
    // completion could flip running=false and then be overwritten by start().
    active_.store(true, std::memory_order_release);
    try {
        worker_ = std::thread([this] { work(); });
    } catch (const std::exception&) {
        active_.store(false, std::memory_order_release);
        error = "AEI trace worker could not start";
        std::fclose(file_);
        file_ = nullptr;
        source_ = nullptr;
        return false;
    }
    return true;
}

void AeiTraceWriter::work() noexcept {
    std::array<basalt::core::aei::RoutedExpertEvent, 256> batch{};
    bool failed = false;
    for (;;) {
        const size_t n = source_->drain(batch.data(), batch.size());
        for (size_t i = 0; i < n; ++i) {
            const auto& e = batch[i];
            if (!failed && std::fprintf(file_, "%llu,%llu,%u,%u\n",
                     (unsigned long long)e.stream, (unsigned long long)e.sequence,
                     (unsigned)e.layer, (unsigned)e.expert) < 0) {
                io_errors_.fetch_add(1, std::memory_order_relaxed);
                failed = true;
            }
            if (!failed) written_.fetch_add(1, std::memory_order_relaxed);
        }
        if (n == 0) {
            if (std::fflush(file_) != 0 && !failed) {
                io_errors_.fetch_add(1, std::memory_order_relaxed);
                failed = true;
            }
            if (stop_.load(std::memory_order_relaxed)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        // After a failed write the ring is still drained to avoid blocking inference.
    }
    active_.store(false, std::memory_order_relaxed);
}

void AeiTraceWriter::stop() noexcept {
    stop_.store(true, std::memory_order_relaxed);
    if (worker_.joinable()) worker_.join();
    if (file_) {
        if (std::fclose(file_) != 0) io_errors_.fetch_add(1, std::memory_order_relaxed);
        file_ = nullptr;
    }
    source_ = nullptr;
    active_.store(false, std::memory_order_relaxed);
}

} // namespace basalt::platform
