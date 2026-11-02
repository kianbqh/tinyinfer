#pragma once
#include <atomic>
#include <optional>
#include <utility>

namespace ti::server {
// Unbounded MPSC linked queue. Admission limits live outside this queue.
// Producers initialize owned payload, exchange tail, then release-publish link.
// Only the sole consumer touches head/payload. It deletes the previous head
// only after acquiring its successor: a producer can no longer reference it.
// A delayed producer may cause transient false from try_pop; retry after wake.
// Destruction requires every producer/consumer to have stopped and joined.
template<class T> class MpscQueue {
  struct Node {
    std::atomic<Node*> next{nullptr};
    std::optional<T> value;
    Node() = default;
    explicit Node(T&& v): value(std::move(v)) {}
  };
  static_assert(std::atomic<Node*>::is_always_lock_free,"MPSC requires lock-free pointer atomics");
  alignas(64) std::atomic<Node*> tail_;
  alignas(64) Node* head_;
public:
  MpscQueue(): tail_(new Node), head_(tail_.load()) {}
  MpscQueue(const MpscQueue&) = delete;
  MpscQueue& operator=(const MpscQueue&) = delete;
  ~MpscQueue() {
    Node* p=head_;
    while(p) { Node* next=p->next.load(std::memory_order_relaxed); delete p; p=next; }
  }
  void push(T value) {
    Node* n=new Node(std::move(value));
    Node* prev=tail_.exchange(n,std::memory_order_acq_rel);
    prev->next.store(n,std::memory_order_release);
  }
  bool try_pop(T& out) {
    Node* next=head_->next.load(std::memory_order_acquire);
    if(!next) return false;
    out=std::move(*next->value);
    next->value.reset();
    delete head_;
    head_=next;
    return true;
  }
};
}
