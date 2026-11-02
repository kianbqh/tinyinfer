#include "artifact.hpp"
#include "input.hpp"
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data,size_t size) {
  if(size>ti_fuzz::max_input)return 0;
  try {
    const std::string text(reinterpret_cast<const char*>(data),size);
    auto root=ti_llm_detail::Parser(text).run();
    std::vector<const ti_llm_detail::Json*> todo{&root};
    // Parsed nodes cannot exceed the bounded input length plus its root.
    for(size_t visited=0;!todo.empty() && visited<=size;++visited) {
      const auto* node=todo.back();todo.pop_back();
      if(node->kind==ti_llm_detail::Json::Number) {
        try{(void)node->integer();}catch(const ti_llm_detail::Invalid&){}
        try{(void)node->real();}catch(const ti_llm_detail::Invalid&){}
        catch(const std::invalid_argument&){}catch(const std::out_of_range&){}
      }
      for(const auto& item:node->array)todo.push_back(&item);
      for(const auto& pair:node->object)todo.push_back(&pair.second);
    }
  } catch(const ti_llm_detail::Invalid&) {} // ordinary parser rejection
  return 0;
}
