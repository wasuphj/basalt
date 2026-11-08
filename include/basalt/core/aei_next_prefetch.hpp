// Optional next-expert advisor. Predictions affect cache-fill ranking only.
#pragma once
#include "basalt/core/aei_transitions.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

namespace basalt::core::aei {

class NextExpertPrefetch {
public:
    static constexpr size_t kMaxExperts = 1u << 20;
    static constexpr uint64_t kFreshEvents = 8192;

    explicit NextExpertPrefetch(size_t experts)
        : size_(checked(experts)), predicted_at_(new std::atomic<uint64_t>[size_]) {
        for (size_t i = 0; i < size_; ++i)
            predicted_at_[i].store(0, std::memory_order_relaxed);
    }

    // The sender guarantees from/to belong to the same ordered request stream.
    // Under contention skip learning rather than waiting on inference.
    void observe(uint32_t from, uint32_t to) noexcept {
        if (from >= size_ || to >= size_) return;
        const uint64_t tick = seen_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (tick == 0 || !lock_.try_lock()) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        index_.observe(from, to);
        // A prediction is evaluated for the NEXT transition, after 'to'.
        const auto candidates = index_.predict(to);
        for (size_t i = 0; i < 2 && i < candidates.size(); ++i) {
            if (candidates[i].count < 3 || candidates[i].expert >= size_) continue;
            predicted_at_[candidates[i].expert].store(tick, std::memory_order_relaxed);
        }
        lock_.unlock();
    }

    float boost(size_t flattened) const noexcept {
        if (flattened >= size_) return 0.0f;
        const auto timestamp = predicted_at_[flattened].load(std::memory_order_relaxed);
        const auto now = seen_.load(std::memory_order_relaxed);
        if (!timestamp || timestamp > now || now - timestamp > kFreshEvents)
            return 0.0f;
        // An explicit prediction gets a short-lived rank uplift. It is not
        // residency ownership: existing swap limits and CUDA copy stay valid.
        return 4.0f;
    }

    // Called outside inference; snapshot the bounded learned transition
    // table without blocking the realtime producer on file I/O.
    std::string checkpoint() const {
        std::lock_guard<std::mutex> guard(lock_);
        return index_.checkpoint();
    }
    bool restore_checkpoint(std::string_view binary) {
        std::lock_guard<std::mutex> guard(lock_);
        if (!index_.restore_checkpoint(binary, static_cast<uint32_t>(size_)))
            return false;
        // No old predictions remain fresh across a process restart.
        seen_.store(0, std::memory_order_relaxed);
        dropped_.store(0, std::memory_order_relaxed);
        for (size_t i = 0; i < size_; ++i)
            predicted_at_[i].store(0, std::memory_order_relaxed);
        return true;
    }

    uint64_t dropped_samples() const noexcept {
        return dropped_.load(std::memory_order_relaxed);
    }
    size_t metadata_bytes() const noexcept {
        return sizeof(*this) + size_ * sizeof(std::atomic<uint64_t>);
    }

private:
    static size_t checked(size_t n) {
        if (!n || n > kMaxExperts) throw std::invalid_argument("AEI predictor geometry too large");
        return n;
    }
    size_t size_;
    std::unique_ptr<std::atomic<uint64_t>[]> predicted_at_;
    mutable std::mutex lock_;
    SparseTransitions index_{};
    std::atomic<uint64_t> seen_{0};
    std::atomic<uint64_t> dropped_{0};
};

} // namespace basalt::core::aei
