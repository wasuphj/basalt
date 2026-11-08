#include "basalt/platform/aei_profile_binding.hpp"
#include "basalt/platform/file.hpp"
#include <cstdio>
#include <string>
using namespace basalt::platform;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return 1; } } while(0)
int main() {
    const std::string base = temp_dir() + "/basalt-aei-profile-binding-fixture-" +
                             std::to_string((uintptr_t)&main);
    const std::string identity = "basalt-model-sha256-v3:" + std::string(64, 'a') + "\n";
    {
        auto* f = open_file_utf8(base, "wb");
        CHECK(f != nullptr);
        CHECK(std::fputs("expert 1 3 5\n", f) >= 0);
        CHECK(std::fclose(f) == 0);
    }
    CHECK(!aei_profile_identity_matches(base, identity));
    CHECK(aei_bind_profile_identity(base, identity));
    CHECK(aei_profile_identity_matches(base, identity));
    CHECK(!aei_profile_identity_matches(base, "basalt-model-sha256-v3:" + std::string(64, 'b') + "\n"));
    {
        auto* f = open_file_utf8(base, "ab");
        CHECK(f != nullptr);
        CHECK(std::fputc('X', f) == 'X');
        CHECK(std::fclose(f) == 0);
    }
    // A rank update without a matching sidecar is rejected even if
    // the old sidecar still names this exact packed model.
    CHECK(!aei_profile_identity_matches(base, identity));
    CHECK(aei_bind_profile_identity(base, identity));
    CHECK(aei_profile_identity_matches(base, identity));
    (void)remove_file(base + ".model-id");
    (void)remove_file(base);
    std::puts("aei_profile_binding_test: OK");
}
