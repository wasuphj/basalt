// Optional per-expert reuse interval heuristic for live adaptive residency.
#pragma once
#include <atomic>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace basalt::core::aei {

class ExpertReuseIndex {
public:
    static constexpr size_t kMaximumExperts = 1u << 20;

    explicit ExpertReuseIndex(size_t experts)
        : size_(validate(experts)), cells_(new Cell[size_]) {}

    void observe(size_t flattened) noexcept {
        if (flattened >= size_) return;
        const uint64_t now = clock_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (now == 0) return;
        auto& c = cells_[flattened];
        // Writers can interleave after obtaining a global clock tick.
        // Never let a late writer move last-seen time backwards.
        uint64_t previous = c.last.load(std::memory_order_relaxed);
        while (previous < now && !c.last.compare_exchange_weak(
                   previous, now, std::memory_order_relaxed, std::memory_order_relaxed)) {}
        if (previous == 0 || previous >= now) return;
        const uint64_t distance = now - previous;
        auto old = c.interval.load(std::memory_order_relaxed);
        for (;;) {
            const uint64_t next = old == 0 ? distance :
                distance > old ? old + (distance - old) / 4 :
                                 old - (old - distance) / 4;
            if (c.interval.compare_exchange_weak(old, next,
                    std::memory_order_relaxed, std::memory_order_relaxed)) break;
        }
    }

    float priority_multiplier(size_t flattened) const noexcept {
        if (flattened >= size_) return 1.0f;
        const auto& c = cells_[flattened];
        const uint64_t last = c.last.load(std::memory_order_relaxed);
        const uint64_t interval = c.interval.load(std::memory_order_relaxed);
        const uint64_t now = clock_.load(std::memory_order_relaxed);
        if (!last || !interval || last > now) return 1.0f;
        const double phase = (double)(now - last) / (double)interval;
        // Early since last reuse: lower urgency; expected soon: higher.
        // Very overdue: decay, avoiding indefinite residency for cold experts.
        const double weight = phase < 1.0 ? 0.7 + 0.7 * phase
            : phase < 2.0 ? 1.4 - 0.4 * (phase - 1.0)
            : std::max(0.5, 1.0 - 0.15 * (phase - 2.0));
        return static_cast<float>(weight);
    }

    size_t metadata_bytes() const noexcept { return sizeof(*this) + size_ * sizeof(Cell); }
    size_t size() const noexcept { return size_; }

private:
    static size_t validate(size_t experts) {
        if (experts == 0 || experts > kMaximumExperts)
            throw std::invalid_argument("AEI reuse index geometry outside bounded range");
        return experts;
    }
    struct Cell {
        std::atomic<uint64_t> last{0};
        std::atomic<uint64_t> interval{0};
    };
    size_t size_ = 0;
    std::unique_ptr<Cell[]> cells_;
    std::atomic<uint64_t> clock_{0};
};

} // namespace basalt::core::aei
