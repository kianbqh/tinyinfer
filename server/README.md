# Native serving runtime

This module connects a bounded TCP frontend to the same native GPT-2 runtime used by `llm.runtime.NativeModel`. One C++ worker owns the model and CUDA stream. The epoll thread handles sockets and owns connection state; owned requests cross an MPSC queue and completions return through another queue plus eventfd.

**Status:** private implementation candidate. A prior revision passed a bounded WSL CPU compile/protocol/queue/socket check. Subsequent review fixed remaining-byte accounting after short reads and stale epoll events after descriptor reuse; this revised transport and its focused native regressions are **not compiled/run yet**. Native GPT-2 serving, full multicore contention, ThreadSanitizer, Poisson load sweeps and GPU performance are **not run**. No performance claim follows from the CPU oracle.

## Build and checks

Root applies `CMAKE_ADDITIVE.cmake` after the decoder/kernel additions. Existing targets/jobs stay unchanged. A standalone CPU build needs only Linux, a C++17 compiler, CMake and Python stdlib:

```bash
cmake -S server -B "$M2_CPU_BUILD" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$M2_CPU_BUILD" --parallel 1
ctest --test-dir "$M2_CPU_BUILD" --output-on-failure
python -m unittest discover -s server/tests -p 'test_*.py'
```

Choose a unique G-backed WSL scratch path for `M2_CPU_BUILD`. For the root native build enable `TINYC_CUDA=ON`, `TINYC_V2_CUDA=ON`, `TINYC_LLM=ON`, `TINYC_SERVER=ON`, `TI_SERVER_NATIVE=ON`. Link `tinyinfer_llm` directly; CPU-only binaries reject native mode. Root owns actual toolkit paths/flags and records the actual build commands. `server/build_receipt.py` records an existing successful CMake build, binary/library hashes, source hashes, actual compile commands and linkage. It does not build or infer successful execution.

```bash
python -m server.build_receipt --binary "$SERVER_BINARY" --build-directory "$M2_BUILD" \
  --build-log "$BUILD_LOG" --command-record "$BUILD_ARGV_JSON" --exit-code 0 \
  --native-library "$M2_BUILD/libtinyinfer_llm.so" --backend-library "$M2_BUILD/libtinyinfer_backend.so" \
  --output "$SERVER_RECEIPT"
python -m server.run --config "$SERVER_CONFIG" --binary "$SERVER_BINARY" \
  --build-receipt "$SERVER_RECEIPT" --output "results/v2/$UNIQUE_SERVER_RUN"
```

`BUILD_ARGV_JSON` is an actual recorded argument array, not a reconstructed shell string. Omit both native-library flags for CPU-only builds. Start from `config_cpu.json`; native mode requires a local exported `manifest.json`, matching model capacity, and the successful native build receipt. The launcher retains exact command/config/hash/environment/PID/stdout/stderr and forwards termination to its child. The server emits readiness only after native model loading succeeds, and emits native operation counts at graceful shutdown. Validate the model's numerical parity before any serving benchmark.

`ci_additive.yml` is a candidate new job, not an installed workflow. It runs normal CPU tests and a separate `-DTI_SERVER_TSAN=ON` build with real producer threads. A platform/TSAN mapping failure is a failure to record, never a successful sanitizer result. `ti_server_unit --smoke` is a small functional preparation gate; its default mode is three rounds of eight producers ×10,000 payloads, checking exact-once delivery, per-producer order and destruction. Queue allocation itself is not promised to be lock-free; the exchange/link queue algorithm uses lock-free pointer atomics. Destroy only after producers/consumer have joined.

## Protocol and limits

Every frame starts with a network-order u32 **body** byte length. Version1 request body:

| Field | Bytes |
|---|---:|
| version=1, kind (generate1 / cancel2) | 2 + 2 |
| nonzero request ID | 8 |
| maximum new tokens, prompt token count | 4 + 4 |
| token IDs, each in [0,50256] | 4 × count |

Generate prompts are nonempty, output counts positive, and prompt+output fits the configured capacity (at most1024). Cancel has zero count/output fields; it targets an active ID on the same connection. Unknown/completed cancels are idempotently ignored. IDs cannot repeat during a connection. Tokenization and model weights are outside the wire protocol.

Response body is `!HHQIIIIQQQQ`: version, kind (FIRST0x8002 / FINAL0x8001), ID, status, token count, batch size, reserved0, queue_ns, first_ns, infer_ns, server_ns, followed by u32 tokens. FIRST contains the first returned token; FINAL contains the complete generated sequence (including that token). Status0 succeeds;1 invalid;2 overloaded;3 cancelled;4 model/internal failure. Malformed frames/duplicate IDs close the connection. An overload gets a terminal response when output capacity permits; slow/disconnected clients are closed and active work is cancelled.

Limits cover frame/context/tokens, per-connection/global pending requests, connections, IDs and queued response bytes. Incremental reads and offset writes handle fragmentation/coalescing, EINTR, EAGAIN and partial sends. Every receive request uses the minimum of buffer capacity and the remaining per-event byte budget, including after an unaligned short read. `send_chunk_bytes` caps one send call (65536 normally,7 in socket tests to exercise offset writes). Exceeding the output bound closes a connection, so configure it to hold the intended largest framed response.

Each epoll registration carries a monotonically allocated uint64 ID in `data.u64`. The registry stores descriptor/kind values and never Client pointers. Closing retires that ID before releasing the descriptor. An already collected event for the old ID is ignored even if a later accept reuses its descriptor; identity is checked again before each read/write/hangup action. Listener and completion-eventfd registrations have their own exact IDs/kinds. Model completions use the same connection identity. The IDs never wrap into reuse: exhaustion fails explicitly. `v2_server_transport_regressions` stages an actual socketpair short-read budget check and focused stale-ID/replacement-dispatch cases; these new native checks remain to be executed.

The worker waits at most `max_wait_us` from the oldest request's acceptance and packs at most `max_batch` requests. Each batch slot owns separate KV state. Different prompt/output lengths are supported. Prefill returns the first greedy token; each decode step appends the previous returned token. EOS does not stop generation; exactly the requested count is returned. Cancellation is observed before/between synchronous native calls and after a call returns, then releases its slot. An in-flight native call is not interrupted. A native execution error poisons the context: outstanding/new work fails explicitly until process restart, and that context is never called again.

## Offered-load experiment

Run off/on sequentially with the same binary/model and `config_load.json`; change only server `max_batch/max_wait_us` (plus an optional endpoint port). Off requires batch1/wait0. Stop the first server and verify it exited before loading the second, so another model is not left resident on the GPU.

```bash
python -m server.loadgen --config "$LOAD_CONFIG" --server-record "$SERVER_RUN/server.json" \
  --mode off --output "results/v2/$UNIQUE_LOAD_OFF"
# Stop/unload, start batching-on with the matching server configuration.
python -m server.loadgen --config "$LOAD_CONFIG" --server-record "$SERVER_RUN_ON/server.json" \
  --mode on --output "results/v2/$UNIQUE_LOAD_ON"
python -m server.compare --off "results/v2/$UNIQUE_LOAD_OFF/load.json" \
  --on "results/v2/$UNIQUE_LOAD_ON/load.json" --output "results/v2/$UNIQUE_COMPARISON"
```

One seeded exponential-arrival stream determines due times, IDs, prompts and output counts independently of responses. Both runs retain byte-identical workload hashes. Outstanding work never shifts future arrivals; late arrivals are dropped with their original due times, rather than sent in catchup bursts. Client-capacity drops, scheduling drops, overloads, socket errors, timeouts and drain failures all remain in raw records. Full drain is awaited after the fixed offered window. Each request uses its own TCP connection; connect cost is included. Warmups0, one offered window per rate, and the configured outstanding-request cap are explicit. First calls remain in samples; no hidden warmups are claimed.

Raw JSON includes offered/completed/dropped/errors, actual offered rate, completions per offer-window second **and** per complete drain-window second. Every completed request retains due-to-final latency, socket TTFT, send-to-first/final and server queue/first/worker-service/accepted-to-complete times. Percentiles interpolate `(n-1)*p/100`; p50/p99/p99.9 and histogram edges/counts/overflow are saved for each field. These are individual request samples, never batch averages. Completed-only percentiles must be read beside all failure/drop counts. `socket_residual_ms` is the unsynchronized-duration residual of send-to-final minus server accepted-to-complete; it includes framing/eventfd/epoll/network overhead and is not a pure network timer. Negative residuals remain raw.

Native worker service includes packing, native synchronous H2D/dispatch/synchronization/D2H and host bookkeeping. It is not a CUDA-event kernel timer. Queue time includes dynamic-batch waiting. Socket TTFT ends when FIRST is received, not when a full response finishes. No result exists yet for the Poisson experiment.
