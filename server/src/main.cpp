#include "model.hpp"
#include "mpsc.hpp"
#include "protocol.hpp"
#include "transport.hpp"
#include <arpa/inet.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

using namespace ti::server;
using Clock=std::chrono::steady_clock;
static uint64_t ns(Clock::duration d) { return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count(); }
static volatile std::sig_atomic_t stop_signal=0;
static void on_signal(int) { stop_signal=1; }
struct Options {
  int port=8091,batch=8,wait_us=1000,capacity=1024,max_new=128,max_pending=256,max_connections=256;
  int per_connection=32,max_output=1048576,max_ids=100000,send_chunk=65536,device=0,gemm=0,attention=0;
  bool native=false; std::string manifest,host="127.0.0.1";
};
static Options options(int argc,char** argv) {
  Options o;
  std::unordered_map<std::string,int*> fields{{"port",&o.port},{"max-batch",&o.batch},{"max-wait-us",&o.wait_us},
    {"capacity",&o.capacity},{"max-new-tokens",&o.max_new},{"max-pending",&o.max_pending},
    {"max-connections",&o.max_connections},{"per-connection",&o.per_connection},{"max-output-bytes",&o.max_output},
    {"max-ids",&o.max_ids},{"send-chunk-bytes",&o.send_chunk},{"device",&o.device},{"gemm-config",&o.gemm},{"attention-config",&o.attention}};
  for(int i=1;i<argc;i+=2) {
    if(i+1>=argc||std::string(argv[i]).rfind("--",0)!=0) throw std::runtime_error("expected option/value pairs");
    std::string key=std::string(argv[i]).substr(2),v=argv[i+1];
    if(key=="model") { if(v!="cpu"&&v!="native") throw std::runtime_error("model must be cpu/native"); o.native=v=="native"; }
    else if(key=="manifest") o.manifest=v;
    else if(key=="host") o.host=v;
    else if(fields.count(key)) { size_t used=0; *fields.at(key)=std::stoi(v,&used); if(used!=v.size()) throw std::runtime_error("noninteger option"); }
    else throw std::runtime_error("unknown option: "+key);
  }
  if(o.port<1||o.port>65535||o.batch<1||o.batch>32||o.wait_us<0||o.wait_us>1000000||
     o.capacity<2||o.capacity>1024||o.max_new<1||o.max_new>=o.capacity||o.max_pending<1||o.max_pending>100000||
     o.max_connections<1||o.max_connections>100000||o.per_connection<1||o.per_connection>o.max_pending||
     o.max_output<64||o.max_output>64*1024*1024||o.max_ids<1||o.max_ids>1000000||
     o.send_chunk<1||o.send_chunk>1048576||
     o.device<0||o.gemm<0||o.gemm>11||o.attention<0||o.attention>1||
     (o.native&&o.manifest.empty())) throw std::runtime_error("option range");
  return o;
}
struct Request {
  int fd; uint64_t connection; WireRequest wire; Clock::time_point accepted;
  std::atomic<bool> cancel{false};
};
struct Completion { std::shared_ptr<Request> request; Response response; };
struct Client {
  uint64_t connection; Framer framer; std::deque<Bytes> output; size_t offset=0,output_bytes=0;
  bool backpressure=false;
  std::unordered_map<uint64_t,std::shared_ptr<Request>> active;
  std::unordered_set<uint64_t> seen;
  Client(uint64_t n,uint32_t frame):connection(n),framer(frame) {}
};
class Server {
  Options o_; int listener_=-1,epoll_=-1,event_=-1;
  uint64_t listener_token_=0,event_token_=0; size_t outstanding_=0;
  Registrations registrations_;
  std::unordered_map<int,std::unique_ptr<Client>> clients_;
  MpscQueue<std::shared_ptr<Request>> requests_; MpscQueue<Completion> completions_;
  std::atomic<size_t> queued_{0}; std::atomic<bool> stopping_{false};
  std::mutex wake_mutex_; std::condition_variable wake_; std::thread worker_;
  std::unique_ptr<Model> model_;
  void signal_worker() { std::lock_guard<std::mutex> guard(wake_mutex_); wake_.notify_one(); }
  void complete(const std::shared_ptr<Request>& r,Response response) {
    response.id=r->wire.id; completions_.push({r,std::move(response)});
    uint64_t one=1; ssize_t wrote;
    do { wrote=write(event_,&one,sizeof(one)); } while(wrote<0&&errno==EINTR);
    // EAGAIN means the nonblocking eventfd already contains an outstanding wake.
  }
  bool take(std::shared_ptr<Request>& r) {
    if(!queued_.load(std::memory_order_acquire)||!requests_.try_pop(r)) return false;
    queued_.fetch_sub(1,std::memory_order_acq_rel); return true;
  }
  void run_worker() noexcept {
    bool poisoned=false;
    for(;;) {
      std::shared_ptr<Request> first;
      if(!take(first)) {
        if(stopping_.load()&&!queued_.load()) break;
        std::unique_lock<std::mutex> lock(wake_mutex_);
        wake_.wait_for(lock,std::chrono::milliseconds(1),[&]{return stopping_.load()||queued_.load()>0;});
        continue;
      }
      std::vector<std::shared_ptr<Request>> batch{first};
      auto deadline=first->accepted+std::chrono::microseconds(o_.wait_us);
      while(batch.size()<static_cast<size_t>(o_.batch)) {
        std::shared_ptr<Request> r;
        if(take(r)) { batch.push_back(std::move(r)); continue; }
        if(stopping_.load()||Clock::now()>=deadline) break;
        std::unique_lock<std::mutex> lock(wake_mutex_);
        wake_.wait_until(lock,deadline,[&]{return stopping_.load()||queued_.load()>0;});
      }
      const auto begin=Clock::now();
      std::vector<Response> results(batch.size()); std::vector<bool> done(batch.size(),false);
      std::vector<std::vector<int32_t>> generated(batch.size());
      for(size_t i=0;i<batch.size();++i) {
        results[i].batch_size=batch.size(); results[i].queue_ns=ns(begin-batch[i]->accepted);
      }
      auto finish=[&](size_t i,Status status) {
        if(done[i]) return;
        auto& response=results[i]; response.kind=final_response; response.status=status;
        response.infer_ns=ns(Clock::now()-begin); response.server_ns=ns(Clock::now()-batch[i]->accepted);
        response.tokens=status==ok?generated[i]:std::vector<int32_t>{};
        complete(batch[i],response); done[i]=true;
      };
      try {
        for(size_t i=0;i<batch.size();++i) {
          if(poisoned) finish(i,internal);
          else if(batch[i]->cancel.load()||stopping_.load()) finish(i,cancelled);
          else model_->reset(i);
        }
        for(;;) {
          std::vector<int32_t> slots; std::vector<std::vector<int32_t>> rows;
          for(size_t i=0;i<batch.size();++i) if(!done[i]) {
            if(batch[i]->cancel.load()||stopping_.load()) { model_->reset(i); finish(i,cancelled); continue; }
            slots.push_back(i);
            rows.push_back(generated[i].empty()?batch[i]->wire.tokens:std::vector<int32_t>{generated[i].back()});
          }
          if(slots.empty()) break;
          auto next=model_->forward(slots,rows);
          const auto returned=Clock::now();
          for(size_t j=0;j<slots.size();++j) {
            const size_t i=slots[j];
            if(batch[i]->cancel.load()||stopping_.load()) { model_->reset(i); finish(i,cancelled); continue; }
            generated[i].push_back(next[j]);
            if(generated[i].size()==1) {
              results[i].first_ns=ns(returned-begin);
              auto first_event=results[i]; first_event.kind=first_response; first_event.tokens={next[j]};
              first_event.infer_ns=ns(returned-begin); first_event.server_ns=ns(returned-batch[i]->accepted);
              complete(batch[i],std::move(first_event));
            }
            if(generated[i].size()==batch[i]->wire.max_new) { model_->reset(i); finish(i,ok); }
          }
        }
      } catch(const std::exception& error) {
        // A partial native execution must never be reused. Existing outstanding
        // work receives explicit internal errors; the process stays observable.
        std::cerr<<"model_poisoned: "<<error.what()<<'\n'; poisoned=true;
        for(size_t i=0;i<batch.size();++i) finish(i,internal);
      }
    }
  }
  void interest(int fd,Client& c) {
    if(!registrations_.matches(c.connection,fd,EventKind::client)) throw std::runtime_error("retired client registration");
    epoll_event e{}; e.data.u64=c.connection; e.events=EPOLLIN|EPOLLRDHUP;
    if(!c.output.empty()) e.events|=EPOLLOUT;
    if(epoll_ctl(epoll_,EPOLL_CTL_MOD,fd,&e)<0) throw std::runtime_error("epoll MOD");
  }
  void close_client(int fd,uint64_t token) {
    auto found=clients_.find(fd);
    if(found==clients_.end()||found->second->connection!=token||!registrations_.matches(token,fd,EventKind::client)) return;
    for(auto& entry:found->second->active) entry.second->cancel.store(true);
    registrations_.remove(token); // Retire before fd can be closed/reused.
    epoll_ctl(epoll_,EPOLL_CTL_DEL,fd,nullptr); close(fd); clients_.erase(found);
  }
  bool enqueue(int fd,Response r) {
    auto found=clients_.find(fd); if(found==clients_.end()) return false;
    auto& c=*found->second; auto bytes=encode_response(r);
    if(c.output_bytes+bytes.size()>static_cast<size_t>(o_.max_output)) { c.backpressure=true; return false; }
    c.output_bytes+=bytes.size(); c.output.push_back(std::move(bytes)); interest(fd,c); return true;
  }
  void on_frame(int fd,Bytes body) {
    auto found=clients_.find(fd); if(found==clients_.end()) return;
    auto& c=*found->second; if(c.backpressure) return; auto wire=parse_request(body,o_.capacity,o_.max_new);
    if(wire.kind==cancel) {
      auto entry=c.active.find(wire.id);
      if(entry!=c.active.end()) entry->second->cancel.store(true);
      // Unknown/already completed cancellations are idempotently ignored.
      signal_worker(); return;
    }
    if(c.seen.count(wire.id)||c.seen.size()>=static_cast<size_t>(o_.max_ids)) throw std::runtime_error("duplicate/request-ID limit");
    c.seen.insert(wire.id);
    if(outstanding_>=static_cast<size_t>(o_.max_pending)||c.active.size()>=static_cast<size_t>(o_.per_connection)) {
      Response r; r.id=wire.id; r.status=overloaded; enqueue(fd,std::move(r)); return;
    }
    auto r=std::make_shared<Request>(); r->fd=fd; r->connection=c.connection; r->wire=std::move(wire); r->accepted=Clock::now();
    c.active.emplace(r->wire.id,r); ++outstanding_;
    requests_.push(std::move(r)); queued_.fetch_add(1,std::memory_order_release); signal_worker();
  }
  void read_client(int fd,uint64_t token) {
    const auto result=bounded_read(65536,
      [fd](uint8_t* buffer,size_t count) { return recv(fd,buffer,count,0); },
      [&](const uint8_t* buffer,size_t count) {
        auto found=clients_.find(fd);
        if(found==clients_.end()||found->second->connection!=token) return false;
        found->second->framer.feed(buffer,count,[&](Bytes b){on_frame(fd,std::move(b));});
        found=clients_.find(fd);
        if(found==clients_.end()||found->second->connection!=token) return false;
        if(found->second->backpressure) { close_client(fd,token); return false; }
        return true;
      });
    if(result.reason==ReadStop::closed||result.reason==ReadStop::error) close_client(fd,token);
  }
  void write_client(int fd,uint64_t token) {
    auto found=clients_.find(fd); if(found==clients_.end()||found->second->connection!=token) return; auto& c=*found->second;
    size_t budget=65536;
    while(!c.output.empty()&&budget) {
      auto& b=c.output.front();size_t count=std::min({b.size()-c.offset,static_cast<size_t>(o_.send_chunk),budget});
      const ssize_t sent=send(fd,b.data()+c.offset,count,MSG_NOSIGNAL);
      if(sent>0) { budget-=sent; c.offset+=sent; c.output_bytes-=sent; if(c.offset==b.size()) { c.output.pop_front(); c.offset=0; } }
      else if(sent<0&&errno==EINTR) continue;
      else if(sent<0&&(errno==EAGAIN||errno==EWOULDBLOCK)) break;
      else { close_client(fd,token); return; }
    }
    interest(fd,c);
  }
  void accept_clients() {
    for(int accepted=0;accepted<64;++accepted) {
      int fd=accept4(listener_,nullptr,nullptr,SOCK_NONBLOCK|SOCK_CLOEXEC);
      if(fd<0) { if(errno==EINTR) continue; if(errno==EAGAIN||errno==EWOULDBLOCK) return; throw std::runtime_error("accept"); }
      if(clients_.size()>=static_cast<size_t>(o_.max_connections)) { close(fd); continue; }
      uint64_t token=0;
      try {
        token=registrations_.add(fd,EventKind::client);
        clients_.emplace(fd,std::make_unique<Client>(token,20+4*o_.capacity));
        epoll_event e{};e.data.u64=token;e.events=EPOLLIN|EPOLLRDHUP;
        if(epoll_ctl(epoll_,EPOLL_CTL_ADD,fd,&e)<0) throw std::runtime_error("epoll ADD client");
      } catch(...) {
        if(clients_.count(fd)) close_client(fd,token);
        else { registrations_.remove(token); close(fd); }
        throw;
      }
    }
  }
  void drain_completions() {
    uint64_t count; while(read(event_,&count,sizeof(count))<0&&errno==EINTR) {}
    Completion completion;
    while(completions_.try_pop(completion)) {
      auto& r=completion.request; const bool terminal=completion.response.kind==final_response;
      if(terminal) --outstanding_;
      auto found=clients_.find(r->fd);
      if(found==clients_.end()||found->second->connection!=r->connection) continue;
      if(terminal) found->second->active.erase(r->wire.id);
      if(!enqueue(r->fd,std::move(completion.response))) close_client(r->fd,r->connection);
    }
  }
public:
  explicit Server(Options o):o_(std::move(o)) {
    // Load before listening; a failed model startup cannot advertise readiness.
    model_=std::make_unique<Model>(o_.native,o_.manifest,o_.batch,o_.capacity,o_.device,o_.gemm,o_.attention);
  }
  ~Server() {
    stopping_.store(true); for(auto& c:clients_) for(auto& r:c.second->active) r.second->cancel.store(true);
    signal_worker(); if(worker_.joinable()) worker_.join();
    model_->report_trace(std::cerr);
    for(auto& c:clients_) close(c.first);
    if(listener_>=0) close(listener_);
    if(event_>=0) close(event_);
    if(epoll_>=0) close(epoll_);
  }
  void run() {
    listener_=socket(AF_INET,SOCK_STREAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0); if(listener_<0) throw std::runtime_error("socket");
    int one=1;setsockopt(listener_,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
    sockaddr_in address{};address.sin_family=AF_INET;address.sin_port=htons(o_.port);
    if(inet_pton(AF_INET,o_.host.c_str(),&address.sin_addr)!=1||bind(listener_,reinterpret_cast<sockaddr*>(&address),sizeof(address))<0||listen(listener_,128)<0)
      throw std::runtime_error("bind/listen");
    epoll_=epoll_create1(EPOLL_CLOEXEC);event_=eventfd(0,EFD_NONBLOCK|EFD_CLOEXEC);
    if(epoll_<0||event_<0) throw std::runtime_error("epoll/eventfd");
    listener_token_=registrations_.add(listener_,EventKind::listener);
    event_token_=registrations_.add(event_,EventKind::completions);
    for(auto token:{listener_token_,event_token_}) {
      const auto target=registrations_.resolve(token);
      epoll_event e{};e.data.u64=token;e.events=EPOLLIN;
      if(epoll_ctl(epoll_,EPOLL_CTL_ADD,target->fd,&e)<0) throw std::runtime_error("epoll ADD");
    }
    worker_=std::thread([this]{run_worker();});
    std::cout<<"{\"ready\":true,\"model\":\""<<(o_.native?"native":"cpu_oracle")<<"\",\"port\":"<<o_.port<<"}"<<std::endl;
    epoll_event events[64];
    while(!stop_signal) {
      int count=epoll_wait(epoll_,events,64,50); if(count<0) { if(errno==EINTR) continue; throw std::runtime_error("epoll wait"); }
      for(int i=0;i<count;++i) {
        const uint64_t token=events[i].data.u64;const auto flags=events[i].events;
        const auto target=registrations_.resolve(token);
        if(!target) continue; // Old epoll_wait batch entry, already retired.
        const int fd=target->fd;
        if(target->kind==EventKind::listener) {
          if(token==listener_token_&&fd==listener_) accept_clients();
          continue;
        }
        if(target->kind==EventKind::completions) {
          if(token==event_token_&&fd==event_) drain_completions();
          continue;
        }
        try {
          dispatch_client_event(registrations_,token,flags&EPOLLIN,flags&EPOLLOUT,flags&(EPOLLERR|EPOLLHUP|EPOLLRDHUP),
            [&](int current_fd){read_client(current_fd,token);},
            [&](int current_fd){write_client(current_fd,token);},
            [&](int current_fd){close_client(current_fd,token);});
        } catch(const std::exception&) { close_client(fd,token); }
      }
    }
  }
};
int main(int argc,char** argv) {
  std::signal(SIGINT,on_signal);std::signal(SIGTERM,on_signal);
  try { Server server(options(argc,argv));server.run();return 0; }
  catch(const std::exception& e) { std::cerr<<"server_failed: "<<e.what()<<'\n';return 1; }
}
