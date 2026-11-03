#ifndef TINYINFER_LLM_H
#define TINYINFER_LLM_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Synchronous HOST API. All token/logit/output pointers below are host pointers.
 * One context owns its CUDA stream, weights, slots and scratch; no libtorch/Python.
 * Calls on one context must be serialized, including destruction. Different
 * contexts own independent resources. Errors poison only execution failures;
 * validation failures enqueue nothing and leave slot lengths unchanged. */
typedef struct ti_llm_model ti_llm_model;
enum ti_llm_status { TI_LLM_OK=0, TI_LLM_INVALID=1, TI_LLM_IO=2,
                     TI_LLM_CUDA=3, TI_LLM_INTERNAL=4 };
typedef struct ti_llm_options {
  uint32_t struct_size; /* sizeof(ti_llm_options), checked for ABI compatibility */
  int32_t device, max_batch, cache_slots, capacity;
  int32_t gemm_config, attention_config;
} ti_llm_options;
typedef struct ti_llm_trace {
  uint64_t forward_calls, embedding, gemm_fp16, gemm_int8, layernorm;
  uint64_t gelu, attention, residual, kv_scatter_gather, lm_head, argmax, nll;
} ti_llm_trace;
int ti_llm_abi_version(void);
const char* ti_llm_last_error(void); /* host-thread local; next API call invalidates */
int ti_llm_create(const char* manifest_path,const ti_llm_options*,ti_llm_model** out);
int ti_llm_destroy(ti_llm_model*);
int ti_llm_reset(ti_llm_model*,int32_t slot);
int ti_llm_lengths(ti_llm_model*,int32_t* out,int32_t count);
int ti_llm_get_trace(ti_llm_model*,ti_llm_trace* out);
/* Append tokens to distinct slots. tokens[B,token_stride], lengths[B] positive;
 * padding is ignored. Absolute positions start at each slot's current length.
 * last_logits optionally receives FP32[B,50257]. next_tokens optionally receives
 * exact greedy argmax[B] (lowest ID on ties); both may be null for prefill only.
 * On success, lengths advance by the corresponding input lengths. */
int ti_llm_forward_batch(ti_llm_model*,const int32_t* slots,const int32_t* tokens,
                         const int32_t* lengths,int32_t batch,int32_t token_stride,
                         float* last_logits,int32_t* next_tokens);
/* Start from empty distinct slots. prompt[B,prompt_stride]; each prompt has its
 * own length and max_new_tokens. output[B,output_stride] is padded with -1.
 * Every request generates exactly max_new_tokens (EOS does not stop this ABI).
 * Last generated token is returned but is NOT appended to its cache yet.
 * capacity must cover prompt_length + max_new_tokens (conservative bound).
 * output_lengths[B], ttft_ms[B] (completion of first argmax), total_ms[B] are
 * optional. Timers cover H2D, native dispatch, synchronization, D2H token copy;
 * exclude model load/tokenization and external queueing. */
int ti_llm_greedy_batch(ti_llm_model*,const int32_t* slots,const int32_t* prompts,
                        const int32_t* prompt_lengths,const int32_t* max_new_tokens,
                        int32_t batch,int32_t prompt_stride,int32_t* output,
                        int32_t output_stride,int32_t* output_lengths,
                        double* ttft_ms,double* total_ms);
/* Exact shifted teacher forcing for one window. tokens[T], mask[T] with
 * mask[0]==0 and mask[t] in {0,1}; score token[t] from hidden[t-1].
 * Resets/uses an internal slot that server clients cannot address. NLL is
 * reduced on-device from <=16 vocabulary rows at a time; no T*V host tensor.
 * loss_by_position optional FP64[T] (zero for excluded positions), sum_nll
 * FP64 and target_count are mandatory. Does not mutate public cache slots. */
int ti_llm_score_window(ti_llm_model*,const int32_t* tokens,const uint8_t* mask,
                        int32_t length,double* sum_nll,uint64_t* target_count,
                        double* loss_by_position);
#ifdef __cplusplus
}
#endif
#endif
