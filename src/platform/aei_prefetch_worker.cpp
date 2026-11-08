#include "basalt/platform/aei_prefetch_worker.hpp"
#include "basalt/core/expert_source.hpp"
#include <array>
#include <chrono>
#include <thread>
namespace basalt::platform {
bool AeiPrefetchWorker::start(basalt::core::ExpertSource& source,
                              basalt::core::aei::PrefetchHintQueue& queue,
                              uint32_t experts_per_layer) {
    if (!experts_per_layer || worker_.joinable()) return false;
    source_ = &source;
    queue_ = &queue;
    experts_per_layer_ = experts_per_layer;
    stop_.store(false, std::memory_order_relaxed);
    try {
        worker_ = std::thread([this] {run();});
    } catch (...) {
        source_ = nullptr;
        queue_ = nullptr;
        return false;
    }
    return true;
}
void AeiPrefetchWorker::stop() noexcept {
    stop_.store(true, std::memory_order_release);
    if (worker_.joinable()) worker_.join();
    if (queue_) queue_->clear();
    source_ = nullptr;
    queue_ = nullptr;
}
void AeiPrefetchWorker::run() noexcept {
    std::array<uint32_t,64> batch{};
    for (;;) {
        if (stop_.load(std::memory_order_acquire)) break;
        const size_t count = queue_->drain(batch.data(),batch.size());
        if (!count) {
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
            continue;
        }
        for (size_t i=0;i<count;++i) {
            if (stop_.load(std::memory_order_acquire)) break;
            bool repeat=false;
            for (size_t j=0;j<i;++j)
                if (batch[j]==batch[i]) {repeat=true;break;}
            if (repeat) continue;
            const uint32_t flat=batch[i];
            try {
                source_->prefetch(flat/experts_per_layer_,flat%experts_per_layer_);
                hints_.fetch_add(1,std::memory_order_relaxed);
            } catch (...) {
                errors_.fetch_add(1,std::memory_order_relaxed);
            }
        }
    }
}
}
