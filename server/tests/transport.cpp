#include "transport.hpp"
#include <sys/socket.h>
#include <unistd.h>
#include <cassert>
#include <iostream>
#include <vector>
using namespace ti::server;

static void bounded_real_receive_after_unaligned_short_read() {
  int pair[2];
  assert(socketpair(AF_UNIX,SOCK_STREAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0,pair)==0);
  int send_buffer=262144;
  assert(setsockopt(pair[1],SOL_SOCKET,SO_SNDBUF,&send_buffer,sizeof(send_buffer))==0);
  uint8_t first=7;
  assert(send(pair[1],&first,1,MSG_NOSIGNAL)==1);
  const std::vector<uint8_t> ready(65536+64,9);
  std::vector<size_t> requested;
  size_t delivered=0;bool seeded=false;
  const auto result=bounded_read(65536,
    [&](uint8_t* destination,size_t count) {
      requested.push_back(count);
      const ssize_t got=recv(pair[0],destination,count,0);
      if(!seeded) {
        // Force a genuine one-byte first recv, then make more than a full
        // remaining allowance ready before the production loop receives again.
        assert(got==1);seeded=true;
        size_t offset=0;
        while(offset<ready.size()) {
          const auto sent=send(pair[1],ready.data()+offset,ready.size()-offset,MSG_NOSIGNAL);
          assert(sent>0);offset+=static_cast<size_t>(sent);
        }
      }
      return got;
    },
    [&](const uint8_t*,size_t count) { delivered+=count;return true; });
  assert(result.reason==ReadStop::exhausted&&result.bytes==65536&&delivered==65536);
  assert(requested.size()==5&&requested.front()==16384&&requested.back()==16383);
  // The bound is exact, not an accidental EAGAIN: unread bytes remain ready.
  uint8_t unread[128];
  const auto left=recv(pair[0],unread,sizeof(unread),MSG_DONTWAIT);
  assert(left==65);
  close(pair[0]);close(pair[1]);
}

static void stale_event_cannot_operate_on_reused_fd() {
  Registrations registrations;
  const auto listener=registrations.add(3,EventKind::listener);
  const auto completions=registrations.add(4,EventKind::completions);
  const auto old=registrations.add(9,EventKind::client);
  const auto collected_old_event=old; // An earlier epoll_wait batch entry.
  registrations.remove(old);
  const auto replacement=registrations.add(9,EventKind::client); // accept reused9.
  assert(old!=replacement&&!registrations.resolve(old));
  int reads=0,writes=0,closes=0;
  auto read=[&](int fd){assert(fd==9);++reads;};
  auto write=[&](int fd){assert(fd==9);++writes;};
  auto close=[&](int fd){assert(fd==9);++closes;};
  for(auto token:{collected_old_event,listener,completions})
    dispatch_client_event(registrations,token,true,true,true,read,write,close);
  assert(reads==0&&writes==0&&closes==0);
  assert(registrations.matches(replacement,9,EventKind::client));
  assert(registrations.matches(listener,3,EventKind::listener));
  assert(registrations.matches(completions,4,EventKind::completions));
  dispatch_client_event(registrations,replacement,true,true,true,read,write,close);
  assert(reads==1&&writes==1&&closes==1);
}

static void identity_is_rechecked_between_event_actions() {
  Registrations registrations;
  auto old=registrations.add(11,EventKind::client);uint64_t replacement=0;
  int reads=0,writes=0,closes=0;
  dispatch_client_event(registrations,old,true,true,true,
    [&](int fd){++reads;registrations.remove(old);replacement=registrations.add(fd,EventKind::client);},
    [&](int){++writes;},[&](int){++closes;});
  assert(reads==1&&writes==0&&closes==0);
  assert(registrations.matches(replacement,11,EventKind::client));
  const auto intermediate=replacement;
  dispatch_client_event(registrations,intermediate,false,true,true,
    [&](int){++reads;},
    [&](int fd){++writes;registrations.remove(intermediate);replacement=registrations.add(fd,EventKind::client);},
    [&](int){++closes;});
  assert(reads==1&&writes==1&&closes==0);
  assert(registrations.matches(replacement,11,EventKind::client));
}

int main() {
  bounded_real_receive_after_unaligned_short_read();
  stale_event_cannot_operate_on_reused_fd();
  identity_is_rechecked_between_event_actions();
  std::cout<<"bounded actual socket receive and stale registration regressions passed\n";
}
