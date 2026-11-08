// Offline single-stream comparison with the original per-layer adaptive tier.
// The initial snapshot is REQUIRED: guessing residents invalidates a baseline.
#include "basalt/core/aei_basalt_adaptive.hpp"
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

using basalt::core::aei::BasaltAdaptiveBaseline;
namespace {
bool number(std::string_view text,uint64_t& value) {
    if(text.empty())return false;
    const auto result=std::from_chars(text.data(),text.data()+text.size(),value);
    return result.ec==std::errc{} && result.ptr==text.data()+text.size();
}
bool pair(std::string_view line,uint64_t& a,uint64_t& b) {
    const auto i=line.find(',');
    return i!=std::string_view::npos && line.find(',',i+1)==std::string_view::npos &&
           number(line.substr(0,i),a) && number(line.substr(i+1),b);
}
bool quartet(std::string_view line,uint64_t (&out)[4]) {
    for(int i=0;i<4;++i) {
        const auto sep=line.find(',');
        if(i<3 && sep==std::string_view::npos)return false;
        if(i==3 && sep!=std::string_view::npos)return false;
        if(!number(sep==std::string_view::npos?line:line.substr(0,sep),out[i]))
            return false;
        if(i<3)line.remove_prefix(sep+1);
    }
    return true;
}
// Never store an unbounded untrusted CSV line.
bool line(std::istream& in,std::string& output,bool& eof) {
    output.clear();
    eof=false;
    for(;;) {
        const int c=in.get();
        if(c==std::char_traits<char>::eof()) {
            if(in.bad())return false;
            eof=output.empty();
            return true;
        }
        if(c=='\n')return true;
        if(output.size()>=128)return false;
        if(c!='\r')output.push_back(static_cast<char>(c));
    }
}
}
int main(int argc,char**argv) {
    if(argc!=8) {
        std::cerr<<"usage: aei-basalt-baseline LAYERS EXPERTS_PER_LAYER SWAPS_PER_ROUND EVERY_WINDOWS DECAY initial_residents.csv router.csv\n";
        return 2;
    }
    uint64_t layers=0,experts=0,swaps=0,every=0;
    if(!number(argv[1],layers) || !number(argv[2],experts) ||
       !number(argv[3],swaps) || !number(argv[4],every))return 2;
    try {
        const float decay=std::stof(argv[5]);
        BasaltAdaptiveBaseline sim(layers,experts,swaps,(uint32_t)every,decay);
        std::ifstream initial(argv[6]);
        std::ifstream trace(argv[7]);
        if(!initial || !trace){std::cerr<<"missing initial snapshot or route trace\n";return 2;}
        std::string row;
        bool eof=false;
        uint64_t lineno=0;
        while(true) {
            if(!line(initial,row,eof)){std::cerr<<"oversized initial row\n";return 2;}
            if(eof)break;
            if(++lineno==1 && row=="layer,expert")continue;
            uint64_t l=0,e=0;
            if(!pair(row,l,e))return 2;
            sim.add_initial(l,e);
        }
        sim.seal();
        uint64_t stream=UINT64_MAX,expected=0;
        int64_t previous_layer=-1;
        lineno=0;
        while(true) {
            if(!line(trace,row,eof)){std::cerr<<"oversized router row\n";return 2;}
            if(eof)break;
            if(++lineno==1 && row=="stream,sequence,layer,expert")continue;
            uint64_t v[4]{};
            if(!quartet(row,v) || v[0] >= (1ull<<63) || v[2]>=layers || v[3]>=experts) {
                std::cerr<<"router geometry or sequence parse failure: "<<lineno<<"\n";
                return 2;
            }
            if(stream==UINT64_MAX)stream=v[0];
            if(v[0]!=stream || v[1]!=expected) {
                std::cerr<<"single-stream contiguous Decode trace required\n";
                return 2;
            }
            ++expected;
            if(previous_layer>=0 && v[2]<(uint64_t)previous_layer)
                sim.end_window();
            sim.access(v[2],v[3]);
            previous_layer=(int64_t)v[2];
        }
        if(previous_layer>=0)sim.end_window();
        const auto result=sim.stats();
        std::cout<<"windows,accesses,hits,misses,swaps,metadata_bytes\n"
                 <<result.windows<<','<<result.accesses<<','<<result.hits<<','
                 <<result.misses<<','<<result.swaps<<','<<sim.metadata_bytes()<<'\n';
        return 0;
    }catch(const std::exception&ex){
        std::cerr<<"baseline replay: "<<ex.what()<<"\n";return 2;
    }
}
