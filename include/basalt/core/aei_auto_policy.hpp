// Deterministic, bounded request-boundary AEI policy state machine.
// It never mutates a CUDA allocation and never makes execution decisions.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace basalt::core::aei {

enum class AutoArm : uint8_t { baseline, cost, reuse, prefetch, split, count };
enum class AutoPhase : uint8_t { disabled, explore, exploit, cooldown };
inline constexpr const char* auto_arm_name(AutoArm arm) noexcept {
    switch (arm) {
        case AutoArm::baseline: return "baseline";
        case AutoArm::cost: return "cost";
        case AutoArm::reuse: return "reuse";
        case AutoArm::prefetch: return "prefetch";
        case AutoArm::split: return "split";
        default: return "invalid";
    }
}

class AutoPolicyController {
public:
    static constexpr uint32_t kExploreSamples = 12;
    static constexpr uint32_t kMinBucketSamples = 4;
    static constexpr uint32_t kReevaluateAfter = 96;
    static constexpr double kMinRelativeGain = 0.08;
    static constexpr size_t kArms = static_cast<size_t>(AutoArm::count);
    static constexpr size_t kBuckets = 3;
    void enable(uint32_t supported_arms = (1u << kArms) - 1) noexcept {
        clear();
        supported_ = supported_arms | 1u; // baseline always available
        phase_ = AutoPhase::explore;
    }
    bool supports(AutoArm arm) const noexcept {
        return ((supported_ >> static_cast<uint8_t>(arm)) & 1u) != 0;
    }
    void disable() noexcept { clear(); }
    bool enabled() const noexcept { return phase_ != AutoPhase::disabled; }
    AutoPhase phase() const noexcept { return phase_; }
    AutoArm active() const noexcept { return active_; }
    uint32_t completed() const noexcept { return completed_; }
    uint32_t rejected() const noexcept { return rejected_; }
    uint32_t transitions() const noexcept { return transitions_; }
    size_t metadata_bytes() const noexcept { return sizeof(*this); }
    uint32_t samples(AutoArm arm, size_t bucket) const noexcept {
        return bucket < kBuckets ? sample_[static_cast<size_t>(arm)][bucket].count : 0;
    }
    static size_t bucket(int64_t output_tokens) noexcept {
        return output_tokens < 64 ? 0 : output_tokens < 256 ? 1 : 2;
    }

    // A failed execution or an unusable measurement never improves an arm.
    // Only eligible, completed decode requests contribute to progression.
    AutoArm observe(int64_t generated, double decode_ms, bool healthy) noexcept {
        if (!enabled()) return AutoArm::baseline;
        if (!healthy || !(decode_ms > 0.0) || !std::isfinite(decode_ms)) {
            rejected_++;
            phase_ = AutoPhase::cooldown;
            cooldown_left_ = 8;
            active_ = AutoArm::baseline;
            transitions_++;
            return active_;
        }
        if (generated < 32 || generated > (1ll << 20)) return active_;
        const double cost = decode_ms / static_cast<double>(generated);
        if (!(cost > 0.0) || !std::isfinite(cost)) return active_;
        const size_t b = bucket(generated);
        auto& current = sample_[static_cast<size_t>(active_)][b];
        // Online mean only, no unbounded trace history.
        if (current.count < 65535) {
            ++current.count;
            current.mean += (cost - current.mean) / current.count;
        }
        ++completed_;
        if (phase_ == AutoPhase::cooldown) {
            if (cooldown_left_ && --cooldown_left_ == 0) {
                reset_samples();
                phase_ = AutoPhase::explore;
                set_active(AutoArm::baseline);
            }
            return active_;
        }
        if (phase_ == AutoPhase::explore) {
            if (++segment_samples_ >= kExploreSamples) {
                segment_samples_ = 0;
                size_t next = static_cast<size_t>(active_) + 1;
                while (next < kArms && !supports(static_cast<AutoArm>(next))) ++next;
                if (next < kArms) {
                    set_active(static_cast<AutoArm>(next));
                } else {
                    phase_ = AutoPhase::exploit;
                    choose_best();
                }
            }
        } else if (phase_ == AutoPhase::exploit) {
            if (++exploit_samples_ >= kReevaluateAfter) {
                reset_samples();
                phase_ = AutoPhase::explore;
                exploit_samples_ = 0;
                set_active(AutoArm::baseline);
            }
        }
        return active_;
    }
    void failed_transition() noexcept {
        if (!enabled()) return;
        ++rejected_;
        phase_ = AutoPhase::cooldown;
        cooldown_left_ = 8;
        set_active(AutoArm::baseline);
    }
private:
    struct Sample {
        uint32_t count = 0;
        double mean = 0;
    };
    std::array<std::array<Sample, kBuckets>, kArms> sample_{};
    AutoArm active_ = AutoArm::baseline;
    AutoPhase phase_ = AutoPhase::disabled;
    uint32_t completed_ = 0;
    uint32_t rejected_ = 0;
    uint32_t transitions_ = 0;
    uint32_t segment_samples_ = 0;
    uint32_t exploit_samples_ = 0;
    uint32_t cooldown_left_ = 0;
    uint32_t supported_ = (1u << kArms) - 1;

    void set_active(AutoArm policy) noexcept {
        if (active_ != policy) { active_ = policy; ++transitions_; }
    }
    void reset_samples() noexcept {
        sample_ = {};
        segment_samples_ = 0;
    }
    void clear() noexcept {
        reset_samples();
        active_ = AutoArm::baseline;
        phase_ = AutoPhase::disabled;
        completed_ = rejected_ = transitions_ = exploit_samples_ = cooldown_left_ = 0;
    }
    void choose_best() noexcept {
        AutoArm best = AutoArm::baseline;
        double best_gain = kMinRelativeGain;
        // Select only arms with >=4 apples-to-apples observations against
        // baseline in a common decode-length bucket.
        for (size_t a = 1; a < kArms; ++a) {
            if (!supports(static_cast<AutoArm>(a))) continue;
            for (size_t b = 0; b < kBuckets; ++b) {
                const auto& base = sample_[0][b];
                const auto& candidate = sample_[a][b];
                if (base.count < kMinBucketSamples || candidate.count < kMinBucketSamples ||
                    !(base.mean > 0.0) || !(candidate.mean > 0.0)) continue;
                const double gain = 1.0 - candidate.mean / base.mean;
                if (gain >= best_gain) { best_gain = gain; best = static_cast<AutoArm>(a); }
            }
        }
        set_active(best);
    }
};

} // namespace basalt::core::aei
