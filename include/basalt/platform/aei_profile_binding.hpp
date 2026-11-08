// Strict sidecar binding for learned expert profiles.
// A plain profile is never accepted as learned for an unrelated model.
#pragma once
#include "basalt/platform/file.hpp"
#include <cstdio>
#include <string>

namespace basalt::platform {

inline bool aei_profile_identity_matches(const std::string& profile_path,
                                         const std::string& identity) {
    if (identity.empty()) return false;
    std::FILE* f = open_file_utf8(profile_path + ".model-id", "rb");
    if (!f) return false;
    std::string contents;
    char buf[512];
    while (!std::feof(f) && contents.size() <= 1024) {
        const size_t n = std::fread(buf, 1, sizeof buf, f);
        contents.append(buf, n);
        if (std::ferror(f)) break;
    }
    const bool valid = !std::ferror(f) && contents.size() <= 1024;
    const bool closed = std::fclose(f) == 0;
    return valid && closed && contents == identity;
}

inline bool aei_bind_profile_identity(const std::string& profile_path,
                                      const std::string& identity) {
    if (profile_path.empty() || identity.empty() || identity.size() > 1024)
        return false;
    const std::string temporary = profile_path + ".model-id.tmp";
    std::FILE* f = open_file_utf8(temporary, "wb");
    if (!f) return false;
    const bool written = std::fwrite(identity.data(), 1, identity.size(), f) == identity.size();
    const bool closed = std::fclose(f) == 0;
    if (!written || !closed || !rename_replace(temporary, profile_path + ".model-id")) {
        remove_file(temporary);
        return false;
    }
    return true;
}

} // namespace basalt::platform
