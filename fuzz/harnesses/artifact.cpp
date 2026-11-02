#include "artifact.hpp"
#include "input.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data,size_t size) {
  if(size>ti_fuzz::max_input || size<4)return 0;
  ti_fuzz::Input input{data,size};
  const size_t manifest_size=std::min(size_t(input.u32()),size-4);
  const size_t weight_size=size-4-manifest_size;
  if(weight_size>ti_fuzz::max_weights)return 0;
  const std::string manifest(reinterpret_cast<const char*>(data+4),manifest_size);
  std::vector<uint8_t> weights(data+4+manifest_size,data+size);
  try {
    auto artifact=ti_llm_detail::Artifact::from_memory(manifest,std::move(weights));
    for(const auto& pair:artifact.tensors)
      (void)artifact.require(pair.first,pair.second.shape,pair.second.dtype);
    try{(void)artifact.require("missing-fuzz-tensor",{1},"float16");}
    catch(const ti_llm_detail::Invalid&){}
  } catch(const ti_llm_detail::Invalid&) {}
  catch(const std::invalid_argument&) {}catch(const std::out_of_range&) {}
  return 0;
}
