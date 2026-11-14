// Fixed-memory sparse transition predictor; shadow-only, router decisions remain authoritative.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

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

    // Portable, bounded little-endian checkpoint of only the transition
    // table. On restart there is no claim that an old prediction is fresh.
    std::string checkpoint() const {
        std::string out = "AET1";
        out.reserve(4 + kRows * (1 + 4 + kTop * 6));
        auto u32 = [&](uint32_t value) {
            for (unsigned n = 0; n < 4; ++n)
                out.push_back(static_cast<char>(value >> (n * 8)));
        };
        for (const auto& row : rows_) {
            out.push_back(row.used ? char(1) : char(0));
            u32(row.used ? row.source : 0);
            for (const auto& edge : row.edges) {
                u32(row.used && edge.count ? edge.expert : 0);
                const uint16_t count = row.used ? edge.count : 0;
                out.push_back(static_cast<char>(count));
                out.push_back(static_cast<char>(count >> 8));
            }
        }
        return out;
    }

    bool restore_checkpoint(std::string_view data, uint32_t experts) {
        constexpr size_t kRowSize = 1 + 4 + kTop * 6;
        if (!experts || data.size() != 4 + kRows * kRowSize ||
            data.substr(0, 4) != "AET1") return false;
        SparseTransitions candidate;
        size_t at = 4;
        auto u32 = [&]() {
            uint32_t result = 0;
            for (unsigned n = 0; n < 4; ++n)
                result |= uint32_t(static_cast<uint8_t>(data[at++])) << (n * 8);
            return result;
        };
        for (size_t i = 0; i < kRows; ++i) {
            const unsigned used = static_cast<uint8_t>(data[at++]);
            const uint32_t source = u32();
            if (used > 1 || (used && (source >= experts || mix(source) % kRows != i)) ||
                (!used && source != 0)) return false;
            Row& row = candidate.rows_[i];
            row.used = used == 1;
            row.source = source;
            for (auto& edge : row.edges) {
                edge.expert = u32();
                edge.count = static_cast<uint8_t>(data[at++]);
                edge.count |= uint16_t(static_cast<uint8_t>(data[at++])) << 8;
                if ((!edge.count && edge.expert) ||
                    (edge.count && (!used || edge.expert >= experts)))
                    return false;
            }
        }
        *this = candidate;
        return true;
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
