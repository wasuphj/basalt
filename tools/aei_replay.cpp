// Streaming CSV trace replay. Reads one access at a time; no whole-trace allocation.
#include "basalt/core/aei_replay.hpp"
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

using basalt::core::aei::Access;
using basalt::core::aei::Policy;
using basalt::core::aei::Replay;

namespace {
bool number(std::string_view s, uint64_t& value) {
    if (s.empty()) return false;
    auto r = std::from_chars(s.data(), s.data() + s.size(), value);
    return r.ec == std::errc{} && r.ptr == s.data() + s.size();
}
bool record(std::string_view s, Access& a) {
    const size_t p = s.find(',');
    if (p == std::string_view::npos) return false;
    const size_t q = s.find(',', p + 1);
    if (q == std::string_view::npos || s.find(',', q + 1) != std::string_view::npos) return false;
    uint64_t id = 0, cpu = 0, transfer = 0;
    if (!number(s.substr(0, p), id) || !number(s.substr(p + 1, q - p - 1), cpu) ||
        !number(s.substr(q + 1), transfer) || id > UINT32_MAX) return false;
    a = {static_cast<uint32_t>(id), cpu, transfer};
    return true;
}
bool policy(std::string_view s, Policy& p) {
    if (s == "lru") p = Policy::lru;
    else if (s == "frequency") p = Policy::frequency;
    else if (s == "cost") p = Policy::cost_aware;
    else if (s == "least-stale") p = Policy::least_stale;
    else return false;
    return true;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: aei-replay POLICY EXPERTS SLOTS trace.csv\n";
        return 2;
    }
    Policy p{};
    uint64_t experts = 0, slots = 0;
    if (!policy(argv[1], p) || !number(argv[2], experts) || !number(argv[3], slots) ||
        experts == 0 || experts > (1u << 20) || slots > (1u << 20)) {
        std::cerr << "invalid policy or expert/slot count\n";
        return 2;
    }
    std::ifstream input(argv[4]);
    if (!input) { std::cerr << "cannot open trace\n"; return 2; }
    Replay replay(static_cast<size_t>(experts), static_cast<size_t>(slots), p);
    std::string line;
    uint64_t row = 0;
    try {
        while (std::getline(input, line)) {
            ++row;
            if (line == "expert,cpu_ns,transfer_ns" && row == 1) continue;
            if (line.size() > 128) { std::cerr << "oversized trace row " << row << "\n"; return 2; }
            Access a;
            if (!record(line, a)) { std::cerr << "bad trace row " << row << "\n"; return 2; }
            replay.access(a);
        }
    } catch (const std::exception& ex) {
        std::cerr << "trace row " << row << ": " << ex.what() << "\n";
        return 2;
    }
    if (input.bad()) { std::cerr << "trace read error\n"; return 2; }
    const auto r = replay.result();
    std::cout << "accesses,hits,misses,transfers,additive_proxy_ns,metadata_bytes\n"
              << r.accesses << ',' << r.hits << ',' << r.misses << ',' << r.transfers << ','
              << r.estimated_ns << ',' << replay.metadata_bytes() << '\n';
}
