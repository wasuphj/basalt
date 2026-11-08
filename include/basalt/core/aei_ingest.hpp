// Causal event ingestion: never infer temporal order from frequency histograms.
#pragma once
#include "basalt/core/aei_shadow.hpp"
#include "basalt/core/aei_trace.hpp"
#include <cstdint>
#include <limits>

namespace basalt::core::aei {

struct IngestionMetrics {
    uint64_t boundaries = 0;
    uint64_t missing_events = 0;
    uint64_t rejected_events = 0;
};

class OrderedShadowIngest {
public:
    explicit OrderedShadowIngest(uint32_t experts_per_layer, uint32_t layers)
        : per_layer_(experts_per_layer), layers_(layers) {}

    bool consume(const RoutedExpertEvent& event) noexcept {
        if (per_layer_ == 0 || layers_ == 0 || event.layer >= layers_ ||
            event.expert >= per_layer_ ||
            static_cast<uint64_t>(event.layer) * per_layer_ + event.expert > UINT32_MAX) {
            ++ingestion_.rejected_events;
            return false;
        }
        if (active_ && event.stream == stream_) {
            if (event.sequence <= last_sequence_) {
                ++ingestion_.rejected_events;
                return false;
            }
            if (event.sequence - last_sequence_ != 1) {
                shadow_.boundary();
                ++ingestion_.boundaries;
                ingestion_.missing_events += event.sequence - last_sequence_ - 1;
            }
        } else {
            if (active_) {
                shadow_.boundary();
                ++ingestion_.boundaries;
            }
        }
        shadow_.record(static_cast<uint32_t>(static_cast<uint64_t>(event.layer) * per_layer_ +
                                             event.expert));
        active_ = true;
        stream_ = event.stream;
        last_sequence_ = event.sequence;
        return true;
    }

    ShadowMetrics predictions() const noexcept { return shadow_.metrics(); }
    IngestionMetrics ingestion() const noexcept { return ingestion_; }
    size_t metadata_bytes() const noexcept { return sizeof(*this); }

private:
    TransitionShadow shadow_{};
    IngestionMetrics ingestion_{};
    uint64_t stream_ = 0;
    uint64_t last_sequence_ = 0;
    uint32_t per_layer_;
    uint32_t layers_;
    bool active_ = false;
};

} // namespace basalt::core::aei
