// Optional next-expert advisor. Predictions affect cache-fill ranking only.
#pragma once
#include "basalt/core/aei_transitions.hpp"
#include "basalt/core/aei_prefetch_queue.hpp"
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

    void set_hint_queue(PrefetchHintQueue* queue) noexcept {
        std::lock_guard<std::mutex> guard(lock_);
        hint_queue_ = queue;
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
        // Predict BEFORE learning this edge. This is a causal online
        // hit rate and cannot credit an observation for its own prediction.
        const auto previous = index_.predict(from);
        uint32_t total_prior = 0;
        for (const auto& candidate : previous) total_prior += candidate.count;
        bool eligible = false, hit = false;
        if (total_prior >= 3) {
            for (size_t i = 0; i < 2; ++i) {
                const auto& choice = previous[i];
                if (choice.count >= 3 && 4u * choice.count >= total_prior) {
                    eligible = true;
                    hit |= choice.expert == to;
                }
            }
        }
        if (eligible) {
            evaluated_.fetch_add(1, std::memory_order_relaxed);
            if (hit) correct_.fetch_add(1, std::memory_order_relaxed);
            const uint32_t sample = hit ? 256u : 0u;
            const uint32_t old = ema_q8_;
            ema_q8_ = sample >= old ? old + (sample - old + 15) / 16
                                    : old - (old - sample + 15) / 16;
        }
        index_.observe(from, to);
        // A prediction is evaluated for the NEXT transition, after 'to'.
        const auto candidates = index_.predict(to);
        uint32_t total = 0;
        for (const auto& candidate : candidates) total += candidate.count;
        const bool reliable = evaluated_.load(std::memory_order_relaxed) < 128 ||
                              ema_q8_ >= 64; // 25% top-2 online precision
        if (!reliable) {
            lock_.unlock();
            return;
        }
        for (size_t i = 0; i < 2 && i < candidates.size(); ++i) {
            if (candidates[i].count < 3 || candidates[i].expert >= size_ ||
                total < 3 || 4u * candidates[i].count < total) continue;
            const auto expert = candidates[i].expert;
            const uint64_t previous =
                predicted_at_[expert].exchange(tick,std::memory_order_relaxed);
            // Hints are sent before the swap round, at the moment routing
            // predicts a future Expert. Never block on page read-ahead.
            if (hint_queue_ && (!previous || tick < previous || tick-previous >= 32))
                (void)hint_queue_->try_push(expert);
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
        evaluated_.store(0, std::memory_order_relaxed);
        correct_.store(0, std::memory_order_relaxed);
        ema_q8_ = 256;
        for (size_t i = 0; i < size_; ++i)
            predicted_at_[i].store(0, std::memory_order_relaxed);
        return true;
    }

    uint64_t dropped_samples() const noexcept {
        return dropped_.load(std::memory_order_relaxed);
    }
    uint64_t evaluated() const noexcept {
        return evaluated_.load(std::memory_order_relaxed);
    }
    uint64_t correct() const noexcept {
        return correct_.load(std::memory_order_relaxed);
    }
    uint32_t hit_ema_percent() const noexcept {
        std::lock_guard<std::mutex> guard(lock_);
        return (ema_q8_ * 100u) / 256u;
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
    PrefetchHintQueue* hint_queue_ = nullptr;
    std::atomic<uint64_t> seen_{0};
    std::atomic<uint64_t> dropped_{0};
    std::atomic<uint64_t> evaluated_{0};
    std::atomic<uint64_t> correct_{0};
    uint32_t ema_q8_ = 256;
};

} // namespace basalt::core::aei
