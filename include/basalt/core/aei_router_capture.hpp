// Optional host-visible verifier router trace; no allocations or file I/O in capture.
#pragma once
#include "basalt/core/aei_event_buffer.hpp"
#include "basalt/core/aei_reuse.hpp"
#include "basalt/core/aei_next_prefetch.hpp"
#include <cstddef>
#include <cstdint>

namespace basalt::core::aei {

class VerifierRouterCapture {
public:
    // Set only at a request boundary, never while a verifier window is executing.
    void bind(RouterEventBuffer* buffer, uint64_t stream) noexcept {
        buffer_ = buffer;
        stream_ = stream;
        sequence_ = 0;
        have_previous_ = false;
    }

    void set_reuse_index(ExpertReuseIndex* index, uint32_t n_expert) noexcept {
        reuse_ = index;
        reuse_n_expert_ = n_expert;
    }
    void set_prefetch_index(NextExpertPrefetch* predictor, uint32_t n_expert) noexcept {
        prefetch_ = predictor;
        prefetch_n_expert_ = n_expert;
        have_previous_ = false;
    }
    void capture(uint32_t layer, const int32_t* ids, size_t tokens, size_t topk,
                 uint32_t n_expert) noexcept {
        if ((!buffer_ && !reuse_ && !prefetch_) || !ids || !topk ||
            !n_expert || tokens > SIZE_MAX / topk) return;
        const size_t count = tokens * topk;
        for (size_t i = 0; i < count; ++i) {
            // Advance even for invalid router entries and dropped events:
            // consumers can detect any resulting sequence discontinuity.
            if (sequence_ == UINT64_MAX) return;
            const uint64_t seq = sequence_++;
            const int32_t id = ids[i];
            if (id >= 0 && static_cast<uint32_t>(id) < n_expert) {
                if (reuse_ && reuse_n_expert_ == n_expert) {
                    const uint64_t flat = (uint64_t)layer * n_expert + (uint32_t)id;
                    if (flat < reuse_->size()) reuse_->observe((size_t)flat);
                }
                if (prefetch_ && prefetch_n_expert_ == n_expert) {
                    const uint64_t flat = (uint64_t)layer * n_expert + (uint32_t)id;
                    if (flat <= UINT32_MAX) {
                        if (have_previous_) prefetch_->observe(previous_, (uint32_t)flat);
                        previous_ = (uint32_t)flat;
                        have_previous_ = true;
                    } else have_previous_ = false;
                }
                if (buffer_)
                    (void) buffer_->try_record({layer, static_cast<uint32_t>(id), seq, stream_});
            } else {
                have_previous_ = false;
            }
        }
    }

    // Preserve temporal gaps if a host-visible ring cannot be authenticated.
    void skip(size_t count) noexcept {
        if (!buffer_ && !reuse_ && !prefetch_) return;
        have_previous_ = false;
        if (count > UINT64_MAX - sequence_) sequence_ = UINT64_MAX;
        else sequence_ += (uint64_t)count;
    }

    bool enabled() const noexcept { return buffer_ != nullptr || reuse_ != nullptr || prefetch_ != nullptr; }
    uint64_t next_sequence() const noexcept { return sequence_; }

private:
    RouterEventBuffer* buffer_ = nullptr;
    ExpertReuseIndex* reuse_ = nullptr;
    uint32_t reuse_n_expert_ = 0;
    NextExpertPrefetch* prefetch_ = nullptr;
    uint32_t prefetch_n_expert_ = 0;
    uint32_t previous_ = 0;
    bool have_previous_ = false;
    uint64_t stream_ = 0;
    uint64_t sequence_ = 0;
};

} // namespace basalt::core::aei
