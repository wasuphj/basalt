// Deterministic, bounded request-boundary AEI policy state machine.
// It never mutates a CUDA allocation and never makes execution decisions.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace basalt::core::aei {

enum class AutoArm : uint8_t { baseline, cost, reuse, prefetch, split, prefetch_split, count };
enum class AutoPhase : uint8_t { disabled, explore, exploit, audit, cooldown };
inline constexpr const char* auto_arm_name(AutoArm arm) noexcept {
    switch (arm) {
        case AutoArm::baseline: return "baseline";
        case AutoArm::cost: return "cost";
        case AutoArm::reuse: return "reuse";
        case AutoArm::prefetch: return "prefetch";
        case AutoArm::split: return "split";
        case AutoArm::prefetch_split: return "prefetch+split";
        default: return "invalid";
    }
}

class AutoPolicyController {
public:
    static constexpr uint32_t kExploreSamples = 12;
    static constexpr uint32_t kMinBucketSamples = 4;
    static constexpr uint32_t kReevaluateAfter = 96;
    static constexpr uint32_t kAuditEvery = 32;
    static constexpr uint32_t kAuditSamples = 8;
    static constexpr double kAuditMinRelativeGain = 0.05;
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
    AutoArm selected() const noexcept { return winner_; }
    uint32_t audits() const noexcept { return audits_; }
    uint32_t completed() const noexcept { return completed_; }
    uint32_t rejected() const noexcept { return rejected_; }
    uint32_t transitions() const noexcept { return transitions_; }
    size_t metadata_bytes() const noexcept { return sizeof(*this); }
    uint32_t samples(AutoArm arm, size_t bucket) const noexcept {
        return bucket < kBuckets ? sample_[static_cast<size_t>(arm)][bucket].count : 0;
    }
    uint32_t prefill_samples(AutoArm arm, size_t bucket) const noexcept {
        return bucket < kBuckets ? prefill_[static_cast<size_t>(arm)][bucket].count : 0;
    }
    static size_t bucket(int64_t output_tokens) noexcept {
        return output_tokens < 64 ? 0 : output_tokens < 256 ? 1 : 2;
    }

    // A failed execution or an unusable measurement never improves an arm.
    // Only eligible, completed decode requests contribute to progression.
    AutoArm observe(int64_t generated, double decode_ms, bool healthy,
                    int64_t prompt_tokens = 0, double prompt_ms = 0.0) noexcept {
        if (!enabled()) return AutoArm::baseline;
        if (!healthy || !(decode_ms > 0.0) || !std::isfinite(decode_ms)) {
            rejected_++;
            phase_ = AutoPhase::cooldown;
            cooldown_left_ = 8;
            winner_ = AutoArm::baseline;
            set_active(AutoArm::baseline);
            return active_;
        }
        if (generated < 32 || generated > (1ll << 20)) return active_;
        const double cost = decode_ms / static_cast<double>(generated);
        if (!(cost > 0.0) || !std::isfinite(cost)) return active_;
        const size_t b = bucket(generated);
        ++completed_;
        // Audit and cooldown are baseline observations only. Keep them
        // separate from the exploration baseline and winner estimates:
        // otherwise a temporary drift probe silently rewrites history.
        if (phase_ == AutoPhase::audit) {
            add(audit_decode_[b], cost);
            if (prompt_tokens > 0 && prompt_tokens <= (1ll << 20) &&
                prompt_ms > 0.0 && std::isfinite(prompt_ms))
                add(audit_prefill_[bucket(prompt_tokens)],
                    prompt_ms / static_cast<double>(prompt_tokens));
            if (++audit_samples_ >= kAuditSamples) {
                ++audits_;
                if (audit_benefit_survives()) {
                    phase_ = AutoPhase::exploit;
                    set_active(winner_);
                } else {
                    winner_ = AutoArm::baseline;
                    phase_ = AutoPhase::cooldown;
                    cooldown_left_ = 8;
                    set_active(AutoArm::baseline);
                }
                audit_samples_ = 0;
                since_audit_ = 0;
                audit_decode_ = {};
                audit_prefill_ = {};
            }
            return active_;
        }
        if (phase_ == AutoPhase::cooldown) {
            if (cooldown_left_ && --cooldown_left_ == 0) {
                reset_samples();
                phase_ = AutoPhase::explore;
                set_active(AutoArm::baseline);
            }
            return active_;
        }
        add(sample_[static_cast<size_t>(active_)][b], cost);
        if (prompt_tokens > 0 && prompt_tokens <= (1ll << 20) &&
            prompt_ms > 0.0 && std::isfinite(prompt_ms)) {
            add(prefill_[static_cast<size_t>(active_)][bucket(prompt_tokens)],
                prompt_ms / static_cast<double>(prompt_tokens));
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
                winner_ = AutoArm::baseline;
                exploit_samples_ = since_audit_ = 0;
                set_active(AutoArm::baseline);
            } else if (winner_ != AutoArm::baseline &&
                       ++since_audit_ >= kAuditEvery) {
                phase_ = AutoPhase::audit;
                audit_samples_ = 0;
                audit_decode_ = {};
                audit_prefill_ = {};
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
        winner_ = AutoArm::baseline;
        set_active(AutoArm::baseline);
    }
private:
    struct Sample {
        uint32_t count = 0;
        double mean = 0;
    };
    std::array<std::array<Sample, kBuckets>, kArms> sample_{};
    std::array<std::array<Sample, kBuckets>, kArms> prefill_{};
    AutoArm active_ = AutoArm::baseline;
    AutoArm winner_ = AutoArm::baseline;
    AutoPhase phase_ = AutoPhase::disabled;
    uint32_t completed_ = 0;
    uint32_t rejected_ = 0;
    uint32_t transitions_ = 0;
    uint32_t segment_samples_ = 0;
    uint32_t exploit_samples_ = 0;
    uint32_t since_audit_ = 0;
    uint32_t audit_samples_ = 0;
    uint32_t audits_ = 0;
    std::array<Sample, kBuckets> audit_decode_{};
    std::array<Sample, kBuckets> audit_prefill_{};
    uint32_t cooldown_left_ = 0;
    uint32_t supported_ = (1u << kArms) - 1;

    static void add(Sample& sample, double value) noexcept {
        if (!(value > 0.0) || !std::isfinite(value) || sample.count >= 65535) return;
        ++sample.count;
        sample.mean += (value - sample.mean) / sample.count;
    }
    bool audit_benefit_survives() const noexcept {
        if (winner_ == AutoArm::baseline) return false;
        const size_t arm = static_cast<size_t>(winner_);
        bool improved = false;
        for (size_t b = 0; b < kBuckets; ++b) {
            const auto& base = audit_decode_[b];
            const auto& candidate = sample_[arm][b];
            if (base.count < kMinBucketSamples ||
                candidate.count < kMinBucketSamples || !(base.mean > 0.0))
                continue;
            if (candidate.mean <= base.mean * (1.0 - kAuditMinRelativeGain))
                improved = true;
        }
        if (!improved) return false;
        // Block restoration if either measured Prefill latency regresses
        // or there is too little comparable current Prefill traffic.
        bool some_pp = false, compared = false;
        for (size_t b = 0; b < kBuckets; ++b) {
            const auto& base = audit_prefill_[b];
            const auto& candidate = prefill_[arm][b];
            some_pp |= base.count != 0 || candidate.count != 0;
            if (base.count < kMinBucketSamples ||
                candidate.count < kMinBucketSamples || !(base.mean > 0.0))
                continue;
            compared = true;
            if (candidate.mean > base.mean * 1.02) return false;
        }
        return !some_pp || compared;
    }
    void set_active(AutoArm policy) noexcept {
        if (active_ != policy) { active_ = policy; ++transitions_; }
    }
    void reset_samples() noexcept {
        sample_ = {};
        prefill_ = {};
        segment_samples_ = 0;
    }
    void clear() noexcept {
        reset_samples();
        active_ = AutoArm::baseline;
        phase_ = AutoPhase::disabled;
        winner_ = AutoArm::baseline;
        audit_decode_ = {};
        audit_prefill_ = {};
        completed_ = rejected_ = transitions_ = exploit_samples_ = cooldown_left_ = 0;
        since_audit_ = audit_samples_ = audits_ = 0;
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
                if (gain < best_gain) continue;
                // A policy improving TG but degrading prompt processing by
                // more than 2% is never promoted. Match only prompt-size
                // buckets having observations in both baseline and arm.
                bool any_prefill = false, comparable = false, regressed = false;
                for (size_t p = 0; p < kBuckets; ++p) {
                    const auto& pp0 = prefill_[0][p];
                    const auto& pp1 = prefill_[a][p];
                    any_prefill |= pp0.count != 0 || pp1.count != 0;
                    if (pp0.count < kMinBucketSamples ||
                        pp1.count < kMinBucketSamples || !(pp0.mean > 0.0))
                        continue;
                    comparable = true;
                    if (pp1.mean > pp0.mean * 1.02) {
                        regressed = true;
                        break;
                    }
                }
                if (regressed || (any_prefill && !comparable)) continue;
                best_gain = gain;
                best = static_cast<AutoArm>(a);
            }
        }
        winner_ = best;
        set_active(best);
    }
};

} // namespace basalt::core::aei
