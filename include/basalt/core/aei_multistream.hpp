// Bounded interleaved-request shadow evaluator. Shared learning, isolated ordered cursors.
#pragma once
#include "basalt/core/aei_ingest.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace basalt::core::aei {

class MultiStreamShadow {
public:
    static constexpr size_t kStreams = 32;

    MultiStreamShadow(uint32_t experts_per_layer, uint32_t layers)
        : per_layer_(experts_per_layer), layers_(layers) {}

    bool consume(const RoutedExpertEvent& e) noexcept {
        if (per_layer_ == 0 || layers_ == 0 || e.layer >= layers_ || e.expert >= per_layer_ ||
            static_cast<uint64_t>(e.layer) * per_layer_ + e.expert > UINT32_MAX) {
            ++ingestion_.rejected_events;
            return false;
        }

        Stream* stream = nullptr;
        for (auto& s : streams_)
            if (s.active && s.id == e.stream) { stream = &s; break; }
        if (stream && e.sequence <= stream->sequence) {
            ++ingestion_.rejected_events;
            return false;
        }

        if (!stream) {
            // Prefer an empty cursor. Otherwise replace the oldest stream,
            // never block or grow memory for new concurrent requests.
            for (auto& s : streams_) if (!s.active) { stream = &s; break; }
            if (!stream) {
                stream = &streams_[0];
                for (auto& s : streams_) if (s.seen < stream->seen) stream = &s;
                ++ingestion_.boundaries;
            }
            *stream = Stream{};
            stream->active = true;
            stream->id = e.stream;
        }

        const uint32_t flattened = static_cast<uint32_t>(
            static_cast<uint64_t>(e.layer) * per_layer_ + e.expert);
        ++metrics_.observed;
        if (stream->previous) {
            const uint64_t gap = e.sequence - stream->sequence;
            if (gap == 1) {
                ++metrics_.opportunities;
                const auto predictions = index_.predict(stream->expert);
                if (predictions[0].count == 0) ++metrics_.no_prediction;
                for (size_t i = 0; i < predictions.size(); ++i) {
                    if (predictions[i].count && predictions[i].expert == flattened) {
                        if (i == 0) ++metrics_.top1_hits;
                        ++metrics_.top4_hits;
                        break;
                    }
                }
                index_.observe(stream->expert, flattened);
            } else {
                ++ingestion_.boundaries;
                ingestion_.missing_events = add(ingestion_.missing_events, gap - 1);
            }
        }
        stream->expert = flattened;
        stream->sequence = e.sequence;
        stream->previous = true;
        stream->seen = ++clock_;
        return true;
    }

    ShadowMetrics predictions() const noexcept { return metrics_; }
    IngestionMetrics ingestion() const noexcept { return ingestion_; }
    size_t metadata_bytes() const noexcept { return sizeof(*this); }

private:
    struct Stream {
        uint64_t id = 0;
        uint64_t sequence = 0;
        uint64_t seen = 0;
        uint32_t expert = 0;
        bool active = false;
        bool previous = false;
    };
    static uint64_t add(uint64_t a, uint64_t b) noexcept {
        return a > UINT64_MAX - b ? UINT64_MAX : a + b;
    }
    SparseTransitions index_{};
    std::array<Stream, kStreams> streams_{};
    ShadowMetrics metrics_{};
    IngestionMetrics ingestion_{};
    uint64_t clock_ = 0;
    uint32_t per_layer_ = 0;
    uint32_t layers_ = 0;
};

static_assert(sizeof(MultiStreamShadow) <= 16 * 1024 * 1024,
              "AEI multi-stream shadow exceeds transition-index budget");
} // namespace basalt::core::aei
