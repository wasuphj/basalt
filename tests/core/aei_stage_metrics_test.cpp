#include "basalt/core/aei_stage_metrics.hpp"
#include <cstdio>
using basalt::core::aei::staged_expert_groups;
#define CHECK(x) do {if(!(x)){std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);return 1;}}while(0)
int main() {
    constexpr int32_t ids[]{3,9,3,2,9,4,5,3};
    constexpr int32_t kinds[]{1,-1,1,1,1,0,1,1};
    CHECK(staged_expert_groups(ids,kinds,8,16)==4);
    CHECK(staged_expert_groups(ids,kinds,0,16)==0);
    CHECK(staged_expert_groups(ids,kinds,129,16)==0);
    CHECK(staged_expert_groups(nullptr,kinds,8,16)==0);
    CHECK(staged_expert_groups(ids,nullptr,8,16)==0);
    constexpr int32_t invalid[]{-1,99,3,3};
    constexpr int32_t staged[]{1,1,1,1};
    CHECK(staged_expert_groups(invalid,staged,4,8)==1);
    std::puts("aei_stage_metrics_test: OK");
}
