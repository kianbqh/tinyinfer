#include "artifact.hpp"
#include <iostream>
#include <stdexcept>
using namespace ti_llm_detail;
void require(bool okay){if(!okay)throw std::runtime_error("artifact CPU check failed");}
int main(){
  require(Sha256::digest("",0)=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  require(Sha256::digest("abc",3)=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  std::string million(1000000,'a');require(Sha256::digest(million.data(),million.size())=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  Sha256 split;for(char c:std::string("abc"))split.update(&c,1);require(split.finish()==Sha256::digest("abc",3));
  auto j=Parser(std::string("{\"s\":\"a\\nb\",\"n\":18446744073709551615,\"a\":[1,2],\"e\":1e-5}")).run();
  require(j.at("s").str()=="a\nb");require(j.at("n").integer()==UINT64_MAX);require(j.at("a").array.size()==2);require(j.at("e").real()==1e-5);
  for(std::string bad:{"{\"x\":1,\"x\":2}","[01]","[1,]","{\"x\":NaN}","{} trailing","{\"x\":\"\\u1234\"}"}){
    bool rejected=false;try{Parser(bad).run();}catch(const Invalid&){rejected=true;}require(rejected);
  }
  bool overflow=false;try{Parser(std::string("18446744073709551616")).run().integer();}catch(const Invalid&){overflow=true;}require(overflow);
  std::cout<<"artifact JSON/SHA256 CPU checks passed\n";
}
