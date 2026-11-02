#include "mpsc.hpp"
#include "protocol.hpp"
#include "model.hpp"
#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include <thread>
using namespace ti::server;
struct Life {
  static std::atomic<int> alive;
  int producer,sequence;
  Life(int p,int s):producer(p),sequence(s) { ++alive; }
  ~Life() { --alive; }
};
std::atomic<int> Life::alive{0};
static void contention(bool smoke) {
  const int producers=smoke?4:8,count=smoke?500:10000;
  for(int round=0;round<(smoke?1:3);++round) {
    MpscQueue<std::unique_ptr<Life>> queue;
    std::atomic<bool> start{false};std::vector<std::thread> threads;
    for(int p=0;p<producers;++p) threads.emplace_back([&,p]{
      while(!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for(int i=0;i<count;++i) { queue.push(std::make_unique<Life>(p,i)); if(i%17==0) std::this_thread::yield(); }
    });
    start.store(true,std::memory_order_release);
    std::vector<int> expected(producers,0);int received=0;std::unique_ptr<Life> out;
    while(received<producers*count) {
      if(queue.try_pop(out)) { assert(out->sequence==expected.at(out->producer)++); ++received; out.reset(); }
      else std::this_thread::yield();
    }
    for(auto& thread:threads) thread.join();
    assert(!queue.try_pop(out));assert(Life::alive.load()==0);
    // Remaining nodes/payloads must also be released when the queue is drained
    // by its destructor after all producers have joined.
    for(int i=0;i<19;++i) queue.push(std::make_unique<Life>(0,i));
  }
  assert(Life::alive.load()==0);
}
static Bytes request() {
  Bytes b;append_be(b,version,2);append_be(b,generate,2);append_be(b,71,8);append_be(b,2,4);append_be(b,3,4);
  for(int i:{5,9,1})append_be(b,i,4);
  Bytes frame;append_be(frame,b.size(),4);frame.insert(frame.end(),b.begin(),b.end());return frame;
}
static void protocol() {
  auto frame=request(); int seen=0;
  Framer framer(4096);
  for(uint8_t byte:frame)framer.feed(&byte,1,[&](Bytes b){auto r=parse_request(b,1024,128);assert(r.id==71&&r.tokens.size()==3&&r.max_new==2);++seen;});
  Bytes combined=frame;combined.insert(combined.end(),frame.begin(),frame.end());
  framer.feed(combined.data(),combined.size(),[&](Bytes b){parse_request(b,1024,128);++seen;});assert(seen==3);
  bool rejected=false;try { Bytes bad{0xff,0xff,0xff,0xff};framer.feed(bad.data(),bad.size(),[](Bytes){}); }catch(const std::exception&){rejected=true;}assert(rejected);
  auto body=Bytes(frame.begin()+4,frame.end());body[20]=0xff;
  rejected=false;try{parse_request(body,1024,128);}catch(const std::exception&){rejected=true;}assert(rejected);
  Model cpu(false,"",2,10,0,0,0);auto first=cpu.forward({0,1},{{4,5,6},{8}});assert(first[0]==10&&first[1]==10);
  auto second=cpu.forward({0,1},{{first[0]},{first[1]}});assert(second[0]==15&&second[1]==13);
  cpu.reset(0);auto isolated=cpu.forward({0},{{1}});assert(isolated[0]==3);
}
int main(int argc,char** argv){
  const bool smoke=argc==2&&std::string(argv[1])=="--smoke";
  contention(smoke);protocol();std::cout<<(smoke?"smoke: 4 producers x 500":"contention: 3 x 8 producers x 10000")<<"; queue/protocol/CPU-model checks passed\n";
}
