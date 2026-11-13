#include "basalt/core/aei_prefetch_queue.hpp"
#include "basalt/core/aei_next_prefetch.hpp"
#include <array>
#include <cstdio>
#include <thread>
using namespace basalt::core::aei;
#define CHECK(x) do {if(!(x)){std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);return 1;}}while(0)
int main() {
    PrefetchHintQueue queue;
    NextExpertPrefetch model(8);
    model.set_hint_queue(&queue);
    for (int i=0;i<8;++i) {
        model.observe(1,2);
        model.observe(2,3);
        model.observe(3,1);
    }
    model.observe(1,2);
    std::array<uint32_t,PrefetchHintQueue::kCapacity> out{};
    const size_t n=queue.drain(out.data(),out.size());
    bool found=false;
    for (size_t i=0;i<n;++i) found|=out[i]==3;
    CHECK(found);
    CHECK(queue.drain(out.data(),out.size())==0);
    model.set_hint_queue(nullptr);
    for(int i=0;i<100;++i) model.observe(1,2);
    CHECK(queue.drain(out.data(),out.size())==0);
    std::array<std::thread,4> producers;
    for(auto& t:producers)
        t=std::thread([&]{for(uint32_t i=0;i<5000;++i)(void)queue.try_push(i);});
    for(auto& t:producers)t.join();
    CHECK(queue.drain(out.data(),out.size())<=PrefetchHintQueue::kCapacity);
    queue.clear();
    CHECK(queue.drain(out.data(),out.size())==0);
    std::puts("aei_prefetch_queue_test: OK");
}
