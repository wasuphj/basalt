// AEI trace contracts distinguish unordered histograms from ordered routing events.
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace basalt::core::aei {

struct ExpertHistogram {
    const uint32_t* counts = nullptr;
    size_t experts = 0;
    uint64_t sample_tokens = 0;
};

// Counts come from completed router decisions; histogram order is expert ID,
// not time order. Never train transition predictors from this representation.
class HistogramStats {
public:
    explicit HistogramStats(size_t experts) : counts_(experts, 0) {}

    bool merge(const ExpertHistogram& input) noexcept {
        if (input.experts != counts_.size() || (!input.counts && input.experts != 0))
            return false;
        // Validate first: a failed merge must not partially change counters.
        for (size_t i = 0; i < input.experts; ++i)
            if (input.counts[i] > std::numeric_limits<uint64_t>::max() - counts_[i])
                return false;
        for (size_t i = 0; i < input.experts; ++i)
            counts_[i] += input.counts[i];
        if (input.sample_tokens > std::numeric_limits<uint64_t>::max() - tokens_)
            tokens_ = std::numeric_limits<uint64_t>::max();
        else tokens_ += input.sample_tokens;
        return true;
    }

    uint64_t count(size_t expert) const noexcept {
        return expert < counts_.size() ? counts_[expert] : 0;
    }
    uint64_t tokens() const noexcept { return tokens_; }
    size_t metadata_bytes() const noexcept { return counts_.capacity() * sizeof(uint64_t); }

private:
    std::vector<uint64_t> counts_;
    uint64_t tokens_ = 0;
};

struct RoutedExpertEvent {
    uint32_t layer;
    uint32_t expert;
    uint64_t sequence; // monotonic within a stream; rejected if out of order
    uint64_t stream;   // separate unrelated requests, never learn across streams
};

class OrderedRouterGuard {
public:
    bool accept(const RoutedExpertEvent& event) noexcept {
        if (have_ && event.stream == stream_ && event.sequence <= sequence_) return false;
        if (!have_ || event.stream != stream_) {
            // Stream transitions reset the predecessor; caller must reset the learner too.
            changed_stream_ = true;
        } else {
            changed_stream_ = false;
        }
        stream_ = event.stream;
        sequence_ = event.sequence;
        have_ = true;
        return true;
    }
    bool changed_stream() const noexcept { return changed_stream_; }

private:
    uint64_t stream_ = 0;
    uint64_t sequence_ = 0;
    bool have_ = false;
    bool changed_stream_ = false;
};

} // namespace basalt::core::aei
