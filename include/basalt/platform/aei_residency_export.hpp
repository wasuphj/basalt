// Atomic snapshot of actual Expert residency for offline Basalt replay.
#pragma once
#include "basalt/platform/file.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace basalt::platform {
inline bool aei_export_residents(const std::string& path,
                                const std::vector<int32_t>& slots,
                                size_t layers, size_t experts) {
    if(path.empty() || !layers || !experts ||
       layers > (1u<<20)/experts || slots.size()!=layers*experts)
        return false;
    const std::string temporary=path+".tmp";
    std::FILE* f=open_file_utf8(temporary,"wb");
    if(!f)return false;
    bool ok=std::fputs("layer,expert\n",f)>=0;
    size_t count=0;
    for(size_t l=0;ok && l<layers;++l)
        for(size_t e=0;ok && e<experts;++e)
            if(slots[l*experts+e]>=0) {
                ok=std::fprintf(f,"%zu,%zu\n",l,e)>0;
                ++count;
            }
    ok=ok && count!=0 && std::fflush(f)==0;
    const bool closed=std::fclose(f)==0;
    if(!ok || !closed || !rename_replace(temporary,path)) {
        remove_file(temporary);
        return false;
    }
    return true;
}
} // namespace basalt::platform
