// Optional bounded router event collector. No file IO and no waiting on the inference producer.
#pragma once
#include "basalt/core/aei_trace.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace basalt::core::aei {

class RouterEventBuffer {
public:
    static constexpr size_t kCapacity = 4096;

    // Fast-fail rather than waiting for a lock or allocating during inference.
    bool try_record(RoutedExpertEvent event) noexcept {
        if (!mutex_.try_lock()) {
            dropped_contention_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (used_ == kCapacity) {
            mutex_.unlock();
            dropped_capacity_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        entries_[tail_] = event;
        tail_ = (tail_ + 1) % kCapacity;
        ++used_;
        mutex_.unlock();
        return true;
    }

    // Called by a separate worker, never on the inference hot path.
    // Bounded batch copy; sink handles persistence after the buffer lock is released.
    size_t drain(RoutedExpertEvent* out, size_t limit) noexcept {
        if (!out || limit == 0) return 0;
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t take = limit < used_ ? limit : used_;
        for (size_t i = 0; i < take; ++i) {
            out[i] = entries_[head_];
            head_ = (head_ + 1) % kCapacity;
        }
        used_ -= take;
        return take;
    }

    uint64_t dropped_capacity() const noexcept {
        return dropped_capacity_.load(std::memory_order_relaxed);
    }
    uint64_t dropped_contention() const noexcept {
        return dropped_contention_.load(std::memory_order_relaxed);
    }
    static constexpr size_t metadata_bytes() noexcept { return sizeof(RouterEventBuffer); }

private:
    std::array<RoutedExpertEvent, kCapacity> entries_{};
    mutable std::mutex mutex_;
    size_t head_ = 0;
    size_t tail_ = 0;
    size_t used_ = 0;
    std::atomic<uint64_t> dropped_capacity_{0};
    std::atomic<uint64_t> dropped_contention_{0};
};

static_assert(sizeof(RouterEventBuffer) < 1024 * 1024,
              "AEI router trace buffer exceeds its metadata bound");
} // namespace basalt::core::aei
