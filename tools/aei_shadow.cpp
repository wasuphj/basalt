// Ordered router-event shadow evaluation. No GPU, no route substitutions.
#include "basalt/core/aei_ingest.hpp"
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace {
bool parse(std::string_view raw, uint64_t& value) {
    if (raw.empty()) return false;
    const auto r = std::from_chars(raw.data(), raw.data() + raw.size(), value);
    return r.ec == std::errc{} && r.ptr == raw.data() + raw.size();
}
bool event_row(std::string_view line, basalt::core::aei::RoutedExpertEvent& e) {
    uint64_t parts[4]{};
    for (size_t i = 0; i < 4; ++i) {
        const size_t comma = line.find(',');
        const auto segment = comma == std::string_view::npos ? line : line.substr(0, comma);
        if (!parse(segment, parts[i])) return false;
        if (i == 3) {
            if (comma != std::string_view::npos) return false;
        } else {
            if (comma == std::string_view::npos) return false;
            line.remove_prefix(comma + 1);
        }
    }
    if (parts[2] > UINT32_MAX || parts[3] > UINT32_MAX) return false;
    e = {static_cast<uint32_t>(parts[2]), static_cast<uint32_t>(parts[3]),
         parts[1], parts[0]};
    return true;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: aei-shadow EXPERTS_PER_LAYER LAYERS ordered-events.csv\n";
        return 2;
    }
    uint64_t experts = 0, layers = 0;
    if (!parse(argv[1], experts) || !parse(argv[2], layers) ||
        experts == 0 || layers == 0 || experts > (1u << 20) ||
        layers > (1u << 20) / experts) {
        std::cerr << "invalid model geometry\n";
        return 2;
    }
    std::ifstream input(argv[3]);
    if (!input) { std::cerr << "cannot open trace\n"; return 2; }
    basalt::core::aei::OrderedShadowIngest shadow(
        static_cast<uint32_t>(experts), static_cast<uint32_t>(layers));
    std::string line;
    uint64_t row = 0;
    while (std::getline(input, line)) {
        ++row;
        if (row == 1 && line == "stream,sequence,layer,expert") continue;
        if (line.size() > 128) { std::cerr << "oversized row " << row << "\n"; return 2; }
        basalt::core::aei::RoutedExpertEvent e{};
        if (!event_row(line, e) || !shadow.consume(e)) {
            std::cerr << "invalid or out-of-order event row " << row << "\n";
            return 2;
        }
    }
    if (input.bad()) { std::cerr << "trace read error\n"; return 2; }
    const auto p = shadow.predictions();
    const auto m = shadow.ingestion();
    std::cout << "observed,opportunities,top1_hits,top4_hits,no_prediction,boundaries,"
                 "missing_events,rejected_events,metadata_bytes\n"
              << p.observed << ',' << p.opportunities << ',' << p.top1_hits << ','
              << p.top4_hits << ',' << p.no_prediction << ',' << m.boundaries << ','
              << m.missing_events << ',' << m.rejected_events << ','
              << shadow.metadata_bytes() << '\n';
}
