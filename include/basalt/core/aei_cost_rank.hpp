// Bounded, optional relative cost model for adaptive expert placement.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace basalt::core::aei {

class LayerCostRank {
public:
    static constexpr size_t kMaxLayers = 128;
    static constexpr uint32_t kMinimumSamples = 8;

    void observe(uint32_t layer, uint32_t cpu_entries, uint64_t callback_ns) noexcept {
        if (layer >= kMaxLayers || !cpu_entries || !callback_ns) return;
        const uint64_t sample = callback_ns / cpu_entries;
        if (!sample) return;
        auto& cell = cells_[layer];
        auto old = cell.ns_per_miss.load(std::memory_order_relaxed);
        for (;;) {
            const uint64_t next = old == 0 ? sample :
                (sample >= old ? old + (sample - old) / 8 : old - (old - sample) / 8);
            if (cell.ns_per_miss.compare_exchange_weak(
                    old, next, std::memory_order_relaxed, std::memory_order_relaxed)) break;
        }
        auto count = cell.samples.load(std::memory_order_relaxed);
        while (count != UINT32_MAX &&
               !cell.samples.compare_exchange_weak(
                   count, count + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {}
    }

    // Called between adaptation rounds, never from a CUDA kernel.
    uint64_t reference_cost_ns(size_t layers) const {
        std::vector<uint64_t> samples;
        const size_t n = std::min(layers, kMaxLayers);
        samples.reserve(n);
        for (size_t l = 0; l < n; ++l) {
            if (cells_[l].samples.load(std::memory_order_relaxed) >= kMinimumSamples) {
                const auto ns = cells_[l].ns_per_miss.load(std::memory_order_relaxed);
                if (ns) samples.push_back(ns);
            }
        }
        if (samples.size() < std::max<size_t>(2, n / 4)) return 0;
        const auto middle = samples.begin() + (ptrdiff_t)(samples.size() / 2);
        std::nth_element(samples.begin(), middle, samples.end());
        return *middle;
    }

    float multiplier(uint32_t layer, uint64_t reference) const noexcept {
        if (layer >= kMaxLayers || !reference ||
            cells_[layer].samples.load(std::memory_order_relaxed) < kMinimumSamples)
            return 1.0f;
        const auto cost = cells_[layer].ns_per_miss.load(std::memory_order_relaxed);
        return std::clamp((float)((double)cost / (double)reference), 0.5f, 2.0f);
    }

    uint32_t samples(uint32_t layer) const noexcept {
        return layer < kMaxLayers ? cells_[layer].samples.load(std::memory_order_relaxed) : 0;
    }
    size_t metadata_bytes() const noexcept { return sizeof(*this); }

private:
    struct Cell {
        std::atomic<uint64_t> ns_per_miss{0};
        std::atomic<uint32_t> samples{0};
    };
    std::array<Cell, kMaxLayers> cells_{};
};
static_assert(sizeof(LayerCostRank) < 4096, "AEI layer cost model exceeded static budget");

} // namespace basalt::core::aei
