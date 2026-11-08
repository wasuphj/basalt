// Optional per-expert reuse interval heuristic for live adaptive residency.
#pragma once
#include <atomic>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

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

    // Only interval estimates are durable. Old recency timestamps
    // would be meaningless on a fresh request clock and are not saved.
    std::string checkpoint() const {
        std::string result = "AER1";
        auto append = [&](uint64_t value, int bytes) {
            for (int i=0;i<bytes;++i)
                result.push_back(static_cast<char>(value >> (i*8)));
        };
        append(size_, 4);
        const size_t count_at=result.size();
        append(0,4);
        uint32_t records=0;
        for (size_t i=0;i<size_;++i) {
            const uint64_t interval = cells_[i].interval.load(std::memory_order_relaxed);
            if (!interval) continue;
            append(i,4);
            append(interval,8);
            ++records;
        }
        for (int i=0; i<4; ++i)
            result[count_at+(size_t)i]=static_cast<char>(records >> (i*8));
        return result;
    }
    bool restore_checkpoint(std::string_view data) noexcept {
        if (data.size() < 12 || data.substr(0,4) != "AER1") return false;
        size_t pos=4;
        auto read = [&](int bytes) {
            uint64_t value=0;
            for (int i=0;i<bytes;++i)
                value |= uint64_t(static_cast<uint8_t>(data[pos++])) << (8*i);
            return value;
        };
        const uint64_t size=read(4), count=read(4);
        if (size != size_ || count > size_ ||
            data.size() != 12+count*12) return false;
        size_t cursor=12;
        uint64_t prev=0;
        for (uint64_t j=0; j<count; ++j) {
            pos=cursor;
            const uint64_t index=read(4), interval=read(8);
            if (index >= size_ || !interval ||
                (j && index <= prev)) return false;
            prev=index;
            cursor=pos;
        }
        // Commit only after all records were validated.
        for (size_t i=0;i<size_;++i) {
            cells_[i].last.store(0, std::memory_order_relaxed);
            cells_[i].interval.store(0, std::memory_order_relaxed);
        }
        clock_.store(0, std::memory_order_relaxed);
        pos=12;
        for (uint64_t j=0;j<count;++j) {
            const auto index=static_cast<size_t>(read(4));
            const uint64_t interval=read(8);
            cells_[index].interval.store(interval, std::memory_order_relaxed);
        }
        return true;
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
