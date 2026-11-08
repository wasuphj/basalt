#include "basalt/core/aei_sha256.hpp"
#include <cstdio>
#include <string>
using basalt::core::aei::Sha256;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return 1; } } while(0)
int main() {
    Sha256 empty;
    CHECK(Sha256::hex(empty.finish()) ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    Sha256 abc;
    abc.update("a", 1);
    abc.update("bc", 2);
    CHECK(Sha256::hex(abc.finish()) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    Sha256 long_input;
    const std::string chunk(1000, 'a');
    for (int i = 0; i < 1000; ++i) long_input.update(chunk);
    CHECK(Sha256::hex(long_input.finish()) ==
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    std::puts("aei_sha256_test: OK");
}
