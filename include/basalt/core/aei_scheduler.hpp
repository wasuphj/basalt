// AEI cost observations and shadow-only CPU/GPU scheduler. No execution-path changes.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace basalt::core::aei {

enum class Device : uint8_t { cpu, gpu };
enum class Placement : uint8_t { baseline, cpu, gpu };

struct CostSample {
    Device device;
    uint32_t work_items;
    uint64_t execution_ns;
    uint64_t transfer_ns;
    uint64_t sync_wait_ns;
};

struct CostEstimate {
    uint64_t execution_ns = 0;
    uint64_t transfer_ns = 0;
    uint64_t sync_wait_ns = 0;
    uint32_t observations = 0;
};

class ShadowScheduler {
public:
    // 2 devices * 8 size buckets * 32 bytes (maximum size measured by test).
    static constexpr size_t kBuckets = 8;
    static constexpr uint32_t kMinimumSamples = 8;

    void observe(const CostSample& sample) noexcept {
        if (sample.work_items == 0 || sample.execution_ns == 0) return;
        auto& cell = cells_[static_cast<size_t>(sample.device)][bucket(sample.work_items)];
        cell.execution_ns = smooth(cell.execution_ns, sample.execution_ns);
        cell.transfer_ns = smooth(cell.transfer_ns, sample.transfer_ns);
        cell.sync_wait_ns = smooth(cell.sync_wait_ns, sample.sync_wait_ns);
        if (cell.observations < std::numeric_limits<uint32_t>::max()) ++cell.observations;
    }

    CostEstimate estimate(Device device, uint32_t work_items) const noexcept {
        if (work_items == 0) return {};
        return cells_[static_cast<size_t>(device)][bucket(work_items)];
    }

    // Neither placement nor router selection is changed. The caller must only log
    // this candidate and separately validate critical-path times before activation.
    Placement candidate(uint32_t work_items, uint64_t safety_margin_ns = 0) const noexcept {
        const auto cpu = estimate(Device::cpu, work_items);
        const auto gpu = estimate(Device::gpu, work_items);
        if (cpu.observations < kMinimumSamples || gpu.observations < kMinimumSamples)
            return Placement::baseline;
        const auto cpu_total = add(cpu.execution_ns, cpu.sync_wait_ns);
        const auto gpu_total = add(add(gpu.execution_ns, gpu.transfer_ns), gpu.sync_wait_ns);
        if (add(cpu_total, safety_margin_ns) < gpu_total) return Placement::cpu;
        if (add(gpu_total, safety_margin_ns) < cpu_total) return Placement::gpu;
        return Placement::baseline;
    }

    static constexpr size_t max_metadata_bytes() noexcept {
        return sizeof(std::array<std::array<CostEstimate, kBuckets>, 2>);
    }

private:
    static size_t bucket(uint32_t n) noexcept {
        size_t i = 0;
        while (n > 1 && i + 1 < kBuckets) { n >>= 1; ++i; }
        return i;
    }
    static uint64_t add(uint64_t a, uint64_t b) noexcept {
        return a > UINT64_MAX - b ? UINT64_MAX : a + b;
    }
    static uint64_t smooth(uint64_t previous, uint64_t incoming) noexcept {
        if (previous == 0) return incoming;
        // alpha = 1/8, avoids overflow by calculating the signed difference.
        return incoming >= previous
            ? previous + (incoming - previous) / 8
            : previous - (previous - incoming) / 8;
    }
    std::array<std::array<CostEstimate, kBuckets>, 2> cells_{};
};

} // namespace basalt::core::aei
