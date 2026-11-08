// Deterministic count of GPU-staged distinct Experts in a verified ring.
#pragma once
#include <cstddef>
#include <cstdint>

namespace basalt::core::aei {
inline uint32_t staged_expert_groups(const int32_t* ids, const int32_t* kinds,
                                     size_t entries, uint32_t experts) noexcept {
    if (!ids || !kinds || !experts || entries > 128) return 0;
    uint32_t groups = 0;
    for (size_t i = 0; i < entries; ++i) {
        if (kinds[i] != 1 || ids[i] < 0 || (uint32_t)ids[i] >= experts) continue;
        bool first = true;
        for (size_t j=0; j<i; ++j) {
            if (kinds[j] == 1 && ids[j] == ids[i]) {first = false;break;}
        }
        if (first) ++groups;
    }
    return groups;
}
}
