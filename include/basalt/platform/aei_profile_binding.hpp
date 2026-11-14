// Fail-closed binding of learned expert ranks to a packed model and
// the exact persisted profile bytes. A failed two-file update is safe:
// a stale identity receipt never validates a newly written rank file.
#pragma once
#include "basalt/core/aei_sha256.hpp"
#include "basalt/platform/file.hpp"
#include <array>
#include <cstdio>
#include <string>

namespace basalt::platform {

inline bool aei_profile_file_hash(const std::string& path, std::string& digest) {
    std::FILE* file = open_file_utf8(path, "rb");
    if (!file) return false;
    basalt::core::aei::Sha256 sha;
    std::array<char, 64 * 1024> block{};
    uint64_t total = 0;
    bool valid = true;
    for (;;) {
        const size_t n = std::fread(block.data(), 1, block.size(), file);
        if (n) {
            total += n;
            // Profile ranks are tiny compared with model tensors. Reject
            // pathological files rather than hashing unbounded contents.
            if (total > 64ull * 1024 * 1024) { valid = false; break; }
            sha.update(block.data(), n);
        }
        if (n < block.size()) {
            if (std::ferror(file)) valid = false;
            break;
        }
    }
    if (std::fclose(file) != 0) valid = false;
    if (!valid || total == 0) return false;
    digest = basalt::core::aei::Sha256::hex(sha.finish());
    return true;
}

inline std::string aei_profile_binding_receipt(const std::string& model_id,
                                                const std::string& profile_hash) {
    if (model_id.empty() || model_id.size() > 256 ||
        model_id.back() != '\n' || profile_hash.size() != 64) return {};
    return "basalt-aei-profile-v2\n" + model_id +
           "profile-sha256:" + profile_hash + "\n";
}

inline bool aei_profile_identity_matches(const std::string& profile_path,
                                         const std::string& identity) {
    if (identity.empty()) return false;
    std::string hash;
    if (!aei_profile_file_hash(profile_path, hash)) return false;
    const std::string expected = aei_profile_binding_receipt(identity, hash);
    if (expected.empty()) return false;
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
    return valid && closed && contents == expected;
}

inline bool aei_bind_profile_identity(const std::string& profile_path,
                                      const std::string& identity) {
    if (profile_path.empty() || identity.empty()) return false;
    std::string hash;
    if (!aei_profile_file_hash(profile_path, hash)) return false;
    const std::string contents = aei_profile_binding_receipt(identity, hash);
    if (contents.empty()) return false;
    const std::string temporary = profile_path + ".model-id.tmp";
    std::FILE* f = open_file_utf8(temporary, "wb");
    if (!f) return false;
    const bool written = std::fwrite(contents.data(), 1, contents.size(), f) ==
                         contents.size();
    const bool closed = std::fclose(f) == 0;
    if (!written || !closed || !rename_replace(temporary, profile_path + ".model-id")) {
        remove_file(temporary);
        return false;
    }
    return true;
}

} // namespace basalt::platform
