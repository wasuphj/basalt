// Replan existing CUDA staged-CPU-miss share without changing routing bits.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace basalt::core::aei {

struct StagedSplitPlan {
    static constexpr int kMaxEntries = 128;
    std::array<int32_t, kMaxEntries + 1> staged{};
};

// Compute the overlap-minimizing staged count for each number of CPU
// misses, but never exceed the staging allocation or leave CPU entries
// uncovered. baseline_min keeps the original host-side staging threshold.
inline StagedSplitPlan plan_staged_split(
        uint64_t cpu_ns_per_miss, uint64_t dma_ns_per_blob,
        int slots, int baseline_min) noexcept {
    StagedSplitPlan out{};
    if (!cpu_ns_per_miss || !dma_ns_per_blob || slots < 1) return out;
    slots = std::min(slots, 16);
    baseline_min = std::clamp(baseline_min, 1, StagedSplitPlan::kMaxEntries);
    constexpr double host_fixed = 15000.0, gpu_fixed = 35000.0;
    const double cpu = (double) cpu_ns_per_miss;
    const double gpu = (double) dma_ns_per_blob;
    for (int m = baseline_min; m <= StagedSplitPlan::kMaxEntries; ++m) {
        double best = host_fixed + m * cpu;
        for (int k = 1; k <= std::min(m, slots); ++k) {
            const double cost = std::max(host_fixed + (m-k)*cpu, gpu_fixed + k*gpu);
            if (cost < best) {best = cost; out.staged[(size_t)m] = k;}
        }
    }
    return out;
}

} // namespace basalt::core::aei
