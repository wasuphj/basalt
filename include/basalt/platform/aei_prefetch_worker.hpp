// Platform worker: predicted page touches never run on the GPU router path.
#pragma once
#include "basalt/core/aei_prefetch_queue.hpp"
#include <atomic>
#include <cstdint>
#include <thread>
namespace basalt::core { class ExpertSource; }
namespace basalt::platform {
class AeiPrefetchWorker {
public:
    AeiPrefetchWorker() = default;
    ~AeiPrefetchWorker() { stop(); }
    AeiPrefetchWorker(const AeiPrefetchWorker&) = delete;
    AeiPrefetchWorker& operator=(const AeiPrefetchWorker&) = delete;
    bool start(basalt::core::ExpertSource& source,
               basalt::core::aei::PrefetchHintQueue& queue,
               uint32_t experts_per_layer);
    void stop() noexcept;
    uint64_t hints() const noexcept {return hints_.load(std::memory_order_relaxed);}
    uint64_t errors() const noexcept {return errors_.load(std::memory_order_relaxed);}
private:
    void run() noexcept;
    basalt::core::ExpertSource* source_ = nullptr;
    basalt::core::aei::PrefetchHintQueue* queue_ = nullptr;
    uint32_t experts_per_layer_ = 0;
    std::thread worker_;
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> hints_{0}, errors_{0};
};
}
