#pragma once
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace ti::server {
enum class ReadStop { exhausted, would_block, closed, stopped, error };
struct ReadResult { ReadStop reason; size_t bytes; int error=0; };

// The exact receive loop used by the epoll frontend. Receiver has recv's
// return/errno contract; consume returns false if it closed/stopped the client.
// Each syscall is bounded by the *remaining* allowance, including after short
// reads. The defensive return-size check also prevents unsigned underflow.
template<size_t BufferSize=16384,class Receive,class Consume>
ReadResult bounded_read(size_t allowance,Receive&& receive,Consume&& consume) {
  static_assert(BufferSize>0,"read buffer must be nonempty");
  std::array<uint8_t,BufferSize> buffer;
  size_t remaining=allowance;
  while(remaining) {
    const size_t count=std::min(buffer.size(),remaining);
    const std::ptrdiff_t got=receive(buffer.data(),count);
    if(got>0) {
      if(static_cast<size_t>(got)>count) throw std::logic_error("receiver exceeded requested byte count");
      remaining-=static_cast<size_t>(got);
      if(!consume(buffer.data(),static_cast<size_t>(got))) return {ReadStop::stopped,allowance-remaining};
    } else if(got==0) return {ReadStop::closed,allowance-remaining};
    else {
      const int error=errno;
      if(error==EINTR) continue;
      if(error==EAGAIN||error==EWOULDBLOCK) return {ReadStop::would_block,allowance-remaining};
      return {ReadStop::error,allowance-remaining,error};
    }
  }
  return {ReadStop::exhausted,allowance};
}

enum class EventKind { listener,completions,client };
struct EventTarget { int fd; EventKind kind; };

// epoll carries a monotonically allocated value, never a Client pointer or a
// bare descriptor. Removing a registration makes already collected old events
// inert even if the OS immediately reuses its fd. Exhaustion fails before wrap.
class Registrations {
  uint64_t next_=0;
  std::unordered_map<uint64_t,EventTarget> live_;
public:
  uint64_t add(int fd,EventKind kind) {
    if(fd<0) throw std::invalid_argument("negative event descriptor");
    if(next_==std::numeric_limits<uint64_t>::max()) throw std::overflow_error("epoll identity exhausted");
    const uint64_t token=++next_;
    live_.emplace(token,EventTarget{fd,kind});
    return token;
  }
  void remove(uint64_t token) { live_.erase(token); }
  std::optional<EventTarget> resolve(uint64_t token) const {
    const auto found=live_.find(token);
    if(found==live_.end()) return std::nullopt;
    return found->second; // Copy: no pointer/iterator survives a callback.
  }
  bool matches(uint64_t token,int fd,EventKind kind) const {
    const auto target=resolve(token);
    return target&&target->fd==fd&&target->kind==kind;
  }
};

// Revalidate before every action: an earlier read callback can retire the
// registration. A replacement using the same fd cannot receive remaining
// write/hangup actions from the old event, including within one dispatch.
template<class Read,class Write,class Close>
void dispatch_client_event(const Registrations& registrations,uint64_t token,
                           bool readable,bool writable,bool closed,
                           Read&& read,Write&& write,Close&& close) {
  const auto target=registrations.resolve(token);
  if(!target||target->kind!=EventKind::client) return;
  const int fd=target->fd;
  if(readable&&registrations.matches(token,fd,EventKind::client)) read(fd);
  if(writable&&registrations.matches(token,fd,EventKind::client)) write(fd);
  if(closed&&registrations.matches(token,fd,EventKind::client)) close(fd);
}
}
