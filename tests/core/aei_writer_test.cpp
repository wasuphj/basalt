#include "basalt/platform/aei_trace_writer.hpp"
#include "basalt/platform/file.hpp"
#include <chrono>
#include <cstdio>
#include <string>

using basalt::core::aei::RouterEventBuffer;
using basalt::platform::AeiTraceWriter;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main() {
    RouterEventBuffer buffer;
    AeiTraceWriter writer;
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string path = basalt::platform::temp_dir() + "/basalt-aei-" + std::to_string(unique) + ".csv";
    std::string error;
    CHECK(writer.start(path, buffer, error));
    CHECK(!writer.start(path, buffer, error)); // can't restart live writer
    CHECK(buffer.try_record({0, 2, 0, 8}));
    CHECK(buffer.try_record({0, 3, 1, 8}));
    CHECK(buffer.try_record({1, 9, 2, 8}));
    writer.stop();
    CHECK(writer.written() == 3 && writer.io_errors() == 0);
    CHECK(!writer.running());
    writer.stop(); // idempotent
    std::FILE* in = basalt::platform::open_file_utf8(path, "rb");
    CHECK(in != nullptr);
    char row[128]{};
    CHECK(std::fgets(row, sizeof row, in) && std::string(row) == "stream,sequence,layer,expert\n");
    CHECK(std::fgets(row, sizeof row, in) && std::string(row) == "8,0,0,2\n");
    CHECK(std::fgets(row, sizeof row, in) && std::string(row) == "8,1,0,3\n");
    CHECK(std::fgets(row, sizeof row, in) && std::string(row) == "8,2,1,9\n");
    CHECK(std::fgets(row, sizeof row, in) == nullptr);
    CHECK(std::fclose(in) == 0);
    CHECK(std::remove(path.c_str()) == 0);
    std::puts("aei_writer_test: OK");
}
