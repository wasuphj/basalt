// Nonblocking router-to-host read-ahead hint mailbox.
#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace basalt::core::aei {

class PrefetchHintQueue {
public:
    static constexpr size_t kCapacity = 512;
    bool try_push(uint32_t flattened) noexcept {
        if (!lock_.try_lock()) {
            rejected_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const bool accepted = used_ < kCapacity;
        if (accepted) {
            data_[(head_ + used_) % kCapacity] = flattened;
            ++used_;
        } else rejected_.fetch_add(1, std::memory_order_relaxed);
        lock_.unlock();
        return accepted;
    }
    size_t drain(uint32_t* output, size_t limit) noexcept {
        if (!output || !limit) return 0;
        std::lock_guard<std::mutex> guard(lock_);
        const size_t take = limit < used_ ? limit : used_;
        for (size_t i=0; i<take; ++i)
            output[i] = data_[(head_ + i) % kCapacity];
        head_ = (head_ + take) % kCapacity;
        used_ -= take;
        return take;
    }
    void clear() noexcept {
        std::lock_guard<std::mutex> guard(lock_);
        head_ = used_ = 0;
    }
    uint64_t rejected() const noexcept {
        return rejected_.load(std::memory_order_relaxed);
    }
private:
    std::mutex lock_;
    std::array<uint32_t,kCapacity> data_{};
    size_t head_ = 0, used_ = 0;
    std::atomic<uint64_t> rejected_{0};
};

} // namespace basalt::core::aei
