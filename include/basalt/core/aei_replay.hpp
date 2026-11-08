// Bounded CPU-only replay of expert cache policies. Not the production adaptive tier.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace basalt::core::aei {

enum class Policy : uint8_t { lru, frequency, cost_aware, least_stale };

struct Access {
    uint32_t expert = 0;       // flattened layer * experts_per_layer + expert
    uint64_t cpu_ns = 0;       // observed or calibrated critical-path miss cost
    uint64_t transfer_ns = 0;  // measured host/device transfer cost
};

struct ReplayResult {
    uint64_t accesses = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t estimated_ns = 0; // additive proxy only; not wall time or predicted TG
    uint64_t transfers = 0;
};

class Replay {
public:
    Replay(size_t experts, size_t slots, Policy policy)
        : cells_(experts), slots_(std::min(experts, slots)), policy_(policy) {
        if (experts > (1u << 20)) throw std::invalid_argument("too many experts for bounded replay");
    }

    ReplayResult access(const Access& a) {
        if (a.expert >= cells_.size()) throw std::out_of_range("expert ID");
        if (clock_ == std::numeric_limits<uint64_t>::max()) throw std::overflow_error("replay clock");
        const uint64_t now = ++clock_;
        auto& c = cells_[a.expert];
        ++result_.accesses;
        if (c.last != 0) {
            const uint64_t distance = now - c.last;
            // Compute the EWMA without a potentially overflowing 3*x sum.
            if (c.interval == 0) c.interval = distance;
            else if (distance >= c.interval) c.interval += (distance - c.interval) / 4;
            else c.interval -= (c.interval - distance) / 4;
        }
        c.last = now;
        if (c.frequency < 65535) ++c.frequency;
        c.cpu_ns = c.cpu_ns == 0 ? a.cpu_ns : (7 * c.cpu_ns + a.cpu_ns) / 8;
        c.transfer_ns = c.transfer_ns == 0 ? a.transfer_ns : (7 * c.transfer_ns + a.transfer_ns) / 8;
        if (c.resident) {
            ++result_.hits;
            return result_;
        }
        ++result_.misses;
        result_.estimated_ns = saturating_add(result_.estimated_ns, a.cpu_ns);
        if (slots_ == 0) return result_;
        if (residents_ < slots_) {
            c.resident = true;
            ++residents_;
            ++result_.transfers;
            result_.estimated_ns = saturating_add(result_.estimated_ns, a.transfer_ns);
            return result_;
        }
        size_t victim = cells_.size();
        uint64_t worst = std::numeric_limits<uint64_t>::max();
        for (size_t i = 0; i < cells_.size(); ++i) {
            const auto& e = cells_[i];
            if (!e.resident) continue;
            const uint64_t score = priority(e, now);
            if (score < worst) { worst = score; victim = i; }
        }
        if (victim == cells_.size()) throw std::logic_error("missing cache victim");
        // Same-slot replacement, with no extra residency or VRAM growth.
        cells_[victim].resident = false;
        c.resident = true;
        ++result_.transfers;
        result_.estimated_ns = saturating_add(result_.estimated_ns, a.transfer_ns);
        return result_;
    }

    ReplayResult result() const { return result_; }
    size_t metadata_bytes() const { return cells_.capacity() * sizeof(Cell); }
    size_t resident_count() const { return residents_; }

private:
    struct Cell {
        uint64_t last = 0;
        uint64_t interval = 0;
        uint64_t cpu_ns = 0;
        uint64_t transfer_ns = 0;
        uint16_t frequency = 0;
        bool resident = false;
    };
    static uint64_t saturating_add(uint64_t a, uint64_t b) {
        return a > std::numeric_limits<uint64_t>::max() - b
            ? std::numeric_limits<uint64_t>::max() : a + b;
    }
    uint64_t priority(const Cell& c, uint64_t now) const {
        switch (policy_) {
            case Policy::lru: return c.last;
            case Policy::frequency: return c.frequency;
            case Policy::cost_aware:
                // Frequency-weighted avoided CPU cost, minus movement cost.
                return c.cpu_ns > c.transfer_ns
                    ? saturating_mul(c.frequency, c.cpu_ns - c.transfer_ns) : 0;
            case Policy::least_stale: {
                // Next-use heuristic from EWMA reuse interval (not SpecMD's exact policy).
                const uint64_t age = now - c.last;
                const uint64_t remaining = c.interval > age ? c.interval - age : 0;
                // Higher priority means the expert is expected sooner.
                // The eviction loop removes the lowest priority, thus the
                // expert with the farthest predicted reuse is displaced.
                return std::numeric_limits<uint64_t>::max() - remaining;
            }
        }
        return 0;
    }
    static uint64_t saturating_mul(uint64_t a, uint64_t b) {
        return b != 0 && a > std::numeric_limits<uint64_t>::max() / b
            ? std::numeric_limits<uint64_t>::max() : a * b;
    }
    std::vector<Cell> cells_;
    size_t slots_;
    Policy policy_;
    uint64_t clock_ = 0;
    size_t residents_ = 0;
    ReplayResult result_;
};

} // namespace basalt::core::aei
