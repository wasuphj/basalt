// Fixed-memory sparse transition predictor; shadow-only, router decisions remain authoritative.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace basalt::core::aei {

struct Prediction {
    uint32_t expert = 0;
    uint16_t count = 0;
};

class SparseTransitions {
public:
    static constexpr size_t kRows = 4096;
    static constexpr size_t kTop = 4;

    void observe(uint32_t from, uint32_t to) noexcept {
        Row& row = rows_[mix(from) % kRows];
        if (!row.used || row.source != from) {
            row = Row{};
            row.used = true;
            row.source = from;
        }
        for (auto& edge : row.edges) {
            if (edge.count != 0 && edge.expert == to) {
                if (edge.count == std::numeric_limits<uint16_t>::max()) {
                    for (auto& e : row.edges) e.count = static_cast<uint16_t>((e.count + 1) / 2);
                }
                ++edge.count;
                return;
            }
        }
        for (auto& edge : row.edges) {
            if (edge.count == 0) { edge = {to, 1}; return; }
        }
        // Space-saving replacement: retain only probable frequent successors.
        size_t victim = 0;
        for (size_t i = 1; i < kTop; ++i)
            if (row.edges[i].count < row.edges[victim].count) victim = i;
        if (row.edges[victim].count == std::numeric_limits<uint16_t>::max()) {
            for (auto& edge : row.edges) edge.count = static_cast<uint16_t>((edge.count + 1u) / 2u);
        }
        row.edges[victim] = {to, static_cast<uint16_t>(row.edges[victim].count + 1u)};
    }

    std::array<Prediction, kTop> predict(uint32_t from) const noexcept {
        const auto& row = rows_[mix(from) % kRows];
        if (!row.used || row.source != from) return {};
        auto out = row.edges;
        for (size_t i = 0; i < kTop; ++i)
            for (size_t j = i + 1; j < kTop; ++j)
                if (out[j].count > out[i].count) {
                    auto tmp = out[i]; out[i] = out[j]; out[j] = tmp;
                }
        return out;
    }

    size_t metadata_bytes() const noexcept { return sizeof(rows_); }

private:
    struct Row {
        uint32_t source = 0;
        bool used = false;
        std::array<Prediction, kTop> edges{};
    };
    static uint32_t mix(uint32_t x) noexcept {
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }
    std::array<Row, kRows> rows_{};
};

static_assert(sizeof(SparseTransitions) <= 16 * 1024 * 1024,
              "AEI transition index exceeds 16 MiB budget");

} // namespace basalt::core::aei
