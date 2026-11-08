// Platform-owned atomic storage for bounded policy evidence.
#pragma once
#include "basalt/core/aei_sha256.hpp"
#include "basalt/platform/file.hpp"
#include <array>
#include <cstdio>
#include <string>

namespace basalt::platform {

inline std::string aei_sha256_hex(const std::string& text) {
    basalt::core::aei::Sha256 sha;
    sha.update(text);
    return basalt::core::aei::Sha256::hex(sha.finish());
}

inline bool aei_save_policy_checkpoint(const std::string& path,
                                       const std::string& environment,
                                       const std::string& payload) {
    if (path.empty() || environment.empty() || payload.empty() ||
        environment.size() > 8192 || payload.size() > 4096) return false;
    const std::string output = "AEI-POLICY-STORE-1\n" +
        aei_sha256_hex(environment) + "\n" +
        aei_sha256_hex(payload) + "\n" + payload;
    const std::string temporary = path + ".tmp";
    std::FILE* file = open_file_utf8(temporary, "wb");
    if (!file) return false;
    const bool written = std::fwrite(output.data(), 1, output.size(), file) == output.size();
    const bool flushed = std::fflush(file) == 0;
    const bool closed = std::fclose(file) == 0;
    if (!written || !flushed || !closed || !rename_replace(temporary, path)) {
        remove_file(temporary);
        return false;
    }
    return true;
}

inline bool aei_load_policy_checkpoint(const std::string& path,
                                       const std::string& environment,
                                       std::string& payload) {
    if (path.empty() || environment.empty() || environment.size() > 8192)
        return false;
    std::FILE* file = open_file_utf8(path, "rb");
    if (!file) return false;
    std::array<char, 8193> bytes{};
    const size_t count = std::fread(bytes.data(), 1, bytes.size(), file);
    const bool ok = !std::ferror(file) && std::feof(file) && std::fclose(file) == 0;
    if (!ok || count == bytes.size()) return false;
    const std::string contents(bytes.data(), count);
    const auto a = contents.find('\n');
    if (a == std::string::npos || contents.substr(0,a) != "AEI-POLICY-STORE-1")
        return false;
    const auto b = contents.find('\n', a+1);
    const auto c = contents.find('\n', b == std::string::npos ? a+1 : b+1);
    if (b == std::string::npos || c == std::string::npos) return false;
    if (contents.substr(a+1,b-a-1) != aei_sha256_hex(environment)) return false;
    const std::string candidate = contents.substr(c+1);
    if (candidate.empty() || candidate.size() > 4096 ||
        contents.substr(b+1,c-b-1) != aei_sha256_hex(candidate)) return false;
    payload = candidate;
    return true;
}

} // namespace basalt::platform
