#include "basalt/core/aei_basalt_adaptive.hpp"
#include <cstdio>
using basalt::core::aei::BasaltAdaptiveBaseline;
#define CHECK(x) do {if(!(x)){std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);return 1;}}while(0)
int main() {
    BasaltAdaptiveBaseline sim(2,4,2,2,0.9f);
    sim.add_initial(0,0);
    sim.add_initial(1,0);
    sim.seal();
    CHECK(sim.resident(0,0));
    CHECK(sim.resident(1,0));
    sim.access(0,1);sim.access(0,1);
    sim.access(1,0);
    sim.end_window();
    CHECK(sim.stats().swaps==0);
    sim.access(0,1);sim.access(0,1);
    sim.access(0,2);
    sim.end_window();
    CHECK(sim.stats().windows==2);
    CHECK(sim.stats().swaps==1);
    CHECK(sim.resident(0,1));
    CHECK(!sim.resident(0,0));
    CHECK(sim.resident(1,0));
    CHECK(sim.stats().accesses==6);
    CHECK(sim.stats().hits==1);
    CHECK(sim.stats().misses==5);
    CHECK(sim.metadata_bytes() < 1024);
    bool error=false;
    try { BasaltAdaptiveBaseline bad(0,1,1,1,0.9f); }
    catch(const std::invalid_argument&) { error=true; }
    CHECK(error);
    std::puts("aei_basalt_baseline_test: OK");
}
