// Evaluate causal transition predictions from an existing routing trace, no GPU.
#include "basalt/core/aei_shadow.hpp"
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: aei-shadow routing.csv\n";
        return 2;
    }
    std::ifstream input(argv[1]);
    if (!input) { std::cerr << "cannot open trace\n"; return 2; }
    basalt::core::aei::TransitionShadow shadow;
    uint64_t row = 0;
    std::string line;
    while (std::getline(input, line)) {
        ++row;
        if (row == 1 && line == "expert,cpu_ns,transfer_ns") continue;
        if (line.size() > 128) { std::cerr << "oversized row " << row << "\n"; return 2; }
        std::string_view value(line);
        const size_t comma = value.find(',');
        if (comma == std::string_view::npos || value.find(',', comma + 1) == std::string_view::npos) {
            std::cerr << "bad row " << row << "\n"; return 2;
        }
        // Validate all three CSV fields, not just the expert column.
        const size_t comma2 = value.find(',', comma + 1);
        if (comma2 == std::string_view::npos ||
            value.find(',', comma2 + 1) != std::string_view::npos) {
            std::cerr << "bad field count at row " << row << "\\n"; return 2;
        }
        auto valid_number = [](std::string_view field) {
            if (field.empty()) return false;
            uint64_t number = 0;
            const auto parsed = std::from_chars(field.data(), field.data() + field.size(), number);
            return parsed.ec == std::errc{} && parsed.ptr == field.data() + field.size();
        };
        if (!valid_number(value.substr(comma + 1, comma2 - comma - 1)) ||
            !valid_number(value.substr(comma2 + 1))) {
            std::cerr << "invalid cost at row " << row << "\\n"; return 2;
        }
        value = value.substr(0, comma);
        uint32_t expert = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), expert);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
            std::cerr << "invalid expert at row " << row << "\n"; return 2;
        }
        shadow.record(expert);
    }
    if (input.bad()) { std::cerr << "trace read error\n"; return 2; }
    const auto m = shadow.metrics();
    std::cout << "observed,opportunities,top1_hits,top4_hits,no_prediction,metadata_bytes\n"
              << m.observed << ',' << m.opportunities << ',' << m.top1_hits << ','
              << m.top4_hits << ',' << m.no_prediction << ',' << shadow.metadata_bytes() << '\n';
}
