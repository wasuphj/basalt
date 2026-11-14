// Small standalone SHA-256 for AEI metadata/file integrity. No dependencies.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace basalt::core::aei {

class Sha256 {
public:
    Sha256() = default;
    void update(const void* data, size_t n) noexcept {
        const auto* p = static_cast<const uint8_t*>(data);
        if (!p && n) return;
        while (n) {
            const size_t part = n < 64 - used_ ? n : 64 - used_;
            std::memcpy(block_.data() + used_, p, part);
            p += part;
            n -= part;
            used_ += part;
            bytes_ += part;
            if (used_ == 64) {
                compress();
                used_ = 0;
            }
        }
    }
    void update(const std::string& data) noexcept { update(data.data(), data.size()); }
    std::array<uint8_t, 32> finish() noexcept {
        const uint64_t bit_length = bytes_ * 8;
        const uint8_t one = 0x80;
        update(&one, 1);
        const uint8_t zero = 0;
        while (used_ != 56) update(&zero, 1);
        uint8_t tail[8];
        for (int i = 7; i >= 0; --i)
            tail[7 - i] = static_cast<uint8_t>(bit_length >> (i * 8));
        update(tail, sizeof tail);
        std::array<uint8_t, 32> result{};
        for (size_t i = 0; i < 8; ++i) {
            for (size_t j = 0; j < 4; ++j)
                result[i * 4 + j] = static_cast<uint8_t>(state_[i] >> (24 - 8*j));
        }
        return result;
    }
    static std::string hex(const std::array<uint8_t, 32>& bytes) {
        static constexpr char digits[] = "0123456789abcdef";
        std::string text;
        text.reserve(64);
        for (uint8_t b : bytes) {
            text.push_back(digits[b >> 4]);
            text.push_back(digits[b & 15]);
        }
        return text;
    }
private:
    static uint32_t ror(uint32_t v, int n) noexcept {
        return (v >> n) | (v << (32 - n));
    }
    void compress() noexcept {
        static constexpr uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
        };
        uint32_t w[64]{};
        for (size_t i = 0; i < 16; ++i) {
            const size_t o = 4*i;
            w[i] = (uint32_t(block_[o]) << 24) | (uint32_t(block_[o+1]) << 16) |
                   (uint32_t(block_[o+2]) << 8) | uint32_t(block_[o+3]);
        }
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t a = w[i-15], b = w[i-2];
            const uint32_t s0 = ror(a,7) ^ ror(a,18) ^ (a >> 3);
            const uint32_t s1 = ror(b,17) ^ ror(b,19) ^ (b >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=state_[0],b=state_[1],c=state_[2],d=state_[3],
                 e=state_[4],f=state_[5],g=state_[6],h=state_[7];
        for (size_t i=0; i<64; ++i) {
            const uint32_t S1 = ror(e,6) ^ ror(e,11) ^ ror(e,25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t temp1 = h + S1 + ch + k[i] + w[i];
            const uint32_t S0 = ror(a,2) ^ ror(a,13) ^ ror(a,22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = S0 + maj;
            h=g;g=f;f=e;e=d+temp1;d=c;c=b;b=a;a=temp1+temp2;
        }
        const uint32_t words[8] = {a,b,c,d,e,f,g,h};
        for (size_t i=0; i<8; ++i) state_[i] += words[i];
    }
    std::array<uint32_t,8> state_{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                  0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::array<uint8_t,64> block_{};
    size_t used_ = 0;
    uint64_t bytes_ = 0;
};

} // namespace basalt::core::aei
