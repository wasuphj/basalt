// AEI transition-prediction evaluator. Observe outcomes after predicting, never replace the router.
#pragma once
#include "basalt/core/aei_transitions.hpp"
#include <cstdint>
#include <limits>

namespace basalt::core::aei {
struct ShadowMetrics {
    uint64_t observed = 0;
    uint64_t opportunities = 0;
    uint64_t top1_hits = 0;
    uint64_t top4_hits = 0;
    uint64_t no_prediction = 0;
};

class TransitionShadow {
public:
    void record(uint32_t expert) noexcept {
        if (have_previous_) {
            ++metrics_.opportunities;
            const auto predictions = index_.predict(previous_);
            if (predictions[0].count == 0) ++metrics_.no_prediction;
            bool found = false;
            for (size_t i = 0; i < predictions.size(); ++i) {
                if (predictions[i].count != 0 && predictions[i].expert == expert) {
                    if (i == 0) ++metrics_.top1_hits;
                    ++metrics_.top4_hits;
                    found = true;
                    break;
                }
            }
            (void) found;
            // Learn only after evaluating this observed transition.
            index_.observe(previous_, expert);
        }
        previous_ = expert;
        have_previous_ = true;
        ++metrics_.observed;
    }
    ShadowMetrics metrics() const noexcept { return metrics_; }
    size_t metadata_bytes() const noexcept { return sizeof(*this); }
    void reset() noexcept {
        index_ = SparseTransitions{};
        metrics_ = {};
        previous_ = 0;
        have_previous_ = false;
    }

private:
    SparseTransitions index_;
    ShadowMetrics metrics_{};
    uint32_t previous_ = 0;
    bool have_previous_ = false;
};
} // namespace basalt::core::aei
