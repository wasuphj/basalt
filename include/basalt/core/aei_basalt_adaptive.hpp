// Exact standard single-GPU, no-peer per-layer adaptive ranking replay.
// The weights and swap thresholds mirror Runtime::rank_swaps(..., false).
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace basalt::core::aei {

struct BasaltAdaptiveStats {
    uint64_t windows = 0, accesses = 0, hits = 0, misses = 0, swaps = 0;
};

class BasaltAdaptiveBaseline {
public:
    BasaltAdaptiveBaseline(size_t layers, size_t experts_per_layer,
                           size_t swaps_per_round, uint32_t every_windows,
                           float decay, float promote_margin = 0.0f)
        : layers_(validated_layers(layers,experts_per_layer,every_windows,decay,promote_margin)),
          experts_(experts_per_layer),
          swaps_(swaps_per_round), every_(every_windows),
          decay_(decay), margin_(promote_margin),
          usage_(layers_ * experts_, 0.0f),
          resident_(layers_ * experts_, 0) {}

    void add_initial(size_t layer, size_t expert) {
        if (sealed_) throw std::logic_error("initial residency already sealed");
        const size_t id = flat(layer, expert);
        if (resident_[id]) throw std::invalid_argument("duplicate initial resident");
        resident_[id] = 1;
        ++initial_;
    }
    void seal() {
        if (!initial_ || initial_ >= resident_.size())
            throw std::invalid_argument("initial resident snapshot is missing or complete");
        sealed_ = true;
    }
    void access(size_t layer, size_t expert) {
        if (!sealed_) throw std::logic_error("initial residency snapshot not loaded");
        const size_t id = flat(layer, expert);
        ++stats_.accesses;
        if (resident_[id]) ++stats_.hits;
        else ++stats_.misses;
        usage_[id] += 1.0f;
    }
    void end_window() {
        if (!sealed_) throw std::logic_error("initial residency snapshot not loaded");
        if (++stats_.windows % every_ != 0) return;
        struct Ent {float score; uint32_t layer, expert;};
        struct Swap {float gain; uint32_t layer, in, out;};
        std::vector<Swap> candidates;
        for (uint32_t layer = 0; layer < layers_; ++layer) {
            std::vector<Ent> incoming, victims;
            for (uint32_t expert = 0; expert < experts_; ++expert) {
                const size_t id = flat(layer, expert);
                const float hot = usage_[id];
                if (resident_[id]) victims.push_back({hot, layer, expert});
                else if (hot >= 2.0f) incoming.push_back({hot - margin_, layer, expert});
            }
            if (incoming.empty() || victims.empty()) continue;
            std::sort(incoming.begin(), incoming.end(), [](const Ent& a,const Ent& b) {
                return a.score > b.score;
            });
            const size_t n = std::min(incoming.size(), victims.size());
            std::partial_sort(victims.begin(), victims.begin() + (ptrdiff_t)n,
                              victims.end(), [](const Ent& a,const Ent& b) {
                                  return a.score < b.score;
                              });
            for (size_t i = 0; i < n; ++i) {
                if (incoming[i].score < victims[i].score + 1.5f) break;
                candidates.push_back({incoming[i].score - victims[i].score,
                                      layer, incoming[i].expert, victims[i].expert});
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Swap& a,const Swap& b) {return a.gain > b.gain;});
        if (candidates.size() > swaps_) candidates.resize(swaps_);
        // Do not select one victim or replacement Expert twice.
        // The no-peer per-layer planner returns disjoint pairs.
        for (const auto& s : candidates) {
            const size_t in = flat(s.layer,s.in);
            const size_t out = flat(s.layer,s.out);
            if (!resident_[out] || resident_[in])
                throw std::logic_error("invalid baseline swap");
            resident_[out] = 0;
            resident_[in] = 1;
            ++stats_.swaps;
        }
        for (auto& usage : usage_) usage *= decay_;
    }
    const BasaltAdaptiveStats& stats() const noexcept {return stats_;}
    size_t metadata_bytes() const noexcept {
        return usage_.capacity() * sizeof(float) + resident_.capacity();
    }
    bool resident(size_t layer,size_t expert) const {
        return resident_[flat(layer,expert)] != 0;
    }
private:
    static size_t validated_layers(size_t layers, size_t experts, uint32_t every,
                                   float decay, float margin) {
        if (!layers || !experts || layers > 128 ||
            experts > 65535 || layers > (1u<<20)/experts ||
            !every || !std::isfinite(decay) || decay < 0 || decay > 1 ||
            !std::isfinite(margin) || margin < 0)
            throw std::invalid_argument("invalid baseline geometry or adaptation policy");
        return layers;
    }
    size_t flat(size_t layer,size_t expert) const {
        if (layer >= layers_ || expert >= experts_)
            throw std::out_of_range("baseline expert geometry");
        return layer * experts_ + expert;
    }
    size_t layers_, experts_, swaps_;
    uint32_t every_;
    float decay_, margin_;
    std::vector<float> usage_;
    std::vector<uint8_t> resident_;
    size_t initial_ = 0;
    bool sealed_ = false;
    BasaltAdaptiveStats stats_{};
};
} // namespace basalt::core::aei
