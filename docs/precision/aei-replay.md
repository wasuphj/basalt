# AEI cache policy replay (experimental)

This is an offline CPU-only comparison tool. It does not change Basalt's
production adaptive cache, route selection, CUDA kernels or default settings.

Build with the normal CMake build and run:

```sh
bin/aei-replay lru 24576 8000 routing.csv
bin/aei-replay frequency 24576 8000 routing.csv
bin/aei-replay cost 24576 8000 routing.csv
bin/aei-replay least-stale 24576 8000 routing.csv
```

The expert count and slot count above are examples, **not** automatic
detections of the running model. Use the model's actual flattened expert
count and the real cache slot budget. The first CSV row may be
`expert,cpu_ns,transfer_ns`, followed by unsigned decimal integers.
Expert IDs are zero-based `layer * experts_per_layer + expert`.
Only measured or separately labelled calibrated costs should populate
`cpu_ns` and `transfer_ns`; invented example costs must never be reported
as observations. Each record is one expert lookup, in original router order.

The output is hits, misses, transfers, additive proxy nanoseconds and
metadata bytes. It is **not** estimated GPU wall-clock latency, throughput,
TTFT, or a substitute for actual CUDA A/B. LRU is a simplified reference,
not Basalt's existing adaptive placement. The least-stale arm is a
lightweight independently designed EWMA interval heuristic, not an exact
implementation of a published algorithm.

The parser streams rows, rejects malformed records and limits each line
to 128 bytes. A replay stores one bounded metadata cell per possible expert
and never creates a second VRAM cache. The maximum accepted expert count
is 1,048,576; this cap is a safety bound, not a recommended model size.

## Required before a production verdict

1. Capture original Basalt router decisions with bounded sampling; do not
   block inference on trace file I/O.
2. Calibrate CPU expert, GPU and H2D timing using the actual engine, not
   synthetic constants. Account for overlapped execution and contention.
3. Replay the same held-out traces at the same *actual* VRAM slot budget.
4. Compare candidates with the existing Basalt adaptive tier, not just LRU.
5. Apply a candidate behind an off-by-default switch and measure real
   5090 A/B, total completion time, TTFT, RSS, VRAM and output quality.
6. Promote only when the measured production workload improves without
   violating the correctness and memory gates.

The cumulative proxy deliberately counts CPU miss and admission transfer
cost independently; it does not model transfer overlap, latency hiding or
the GPU scheduling critical path. Treat it only as an exploration signal.

## Causal shadow evaluation

`aei-shadow` accepts **only** ordered events, not histogram rows or
the 3-column policy replay input. It requires a verified stream identifier
and consecutive sequence numbers from the real router event producer:

```csv
stream,sequence,layer,expert
15,0,0,3
15,1,0,7
15,2,1,4
16,0,0,3
```

```sh
bin/aei-shadow 512 48 ordered-events.csv
```

The geometry arguments are **examples**, not model autodetection.
The predictor evaluates each successor before learning that transition.
A changed stream resets only the predecessor, retaining prior statistics;
a missing sequence interval also resets the predecessor and increments
`missing_events`. Duplicate or reversed sequence numbers are rejected.
The CLI accepts interleaved request streams and tracks up to 32 active
stream cursors with one shared predictor. A cursor lost to capacity-based
LRU replacement starts fresh when its stream returns; no fabricated
predecessor is used. Events must still be ordered **within** each stream,
and once a stream's cursor is evicted its previous ordering cannot be
authenticated without an external durable sequence ledger. Do not
reconstruct temporal sequences by iterating `route_hist_take()` counts:
those are unordered aggregates.

The output's Top-1/Top-4 hits are causal offline prediction metrics.
They do not establish cache residency, timely prefetch, wall-clock
speedup or quality preservation. The live ordered event producer and
its asynchronous persistence consumer are still pending integration.

## Host-visible verifier capture seam

The non-device-planned verifier now has an **opt-in** sink setter,
`Verifier::set_aei_router_trace(buffer, stream_id)`. After the per-layer
GPU doorbell, the verifier copies its already mapped host-side Top-K IDs
into the bounded in-memory event collector without file I/O or a new
device transfer. Invalid IDs and full-buffer drops advance the sequence
and therefore produce detectable gaps. A downstream worker must drain
and persist events. An opt-in control is now provided through the
engine's existing idle-only probe protocol. There is still no background
persistence worker or production-enabled tracing flag.

This seam does **not** capture the device-planned MoE ring: its host IDs
cannot yet be certified as the complete original routing decisions.
It also does not cover prompt prefill routing. Neither source should be
filled in from frequency histograms. The sink must be bound only when
the verifier is idle; the request stream ID must not be reused across
different requests, and each verifier's stream must remain unique.

## Interactive observation control

At an idle Basalt engine command boundary, `PROBE trace on` allocates
one bounded 4096-event host buffer. Each subsequently decoded request
binds distinct stream IDs to its two verifier chains. `PROBE trace
drain` emits at most 128 records as `AEI stream sequence layer expert`
followed by a summary showing overflow and contention drop counters.
Run drain repeatedly to consume a larger buffer. `PROBE trace off`
unbinds both verifiers and discards any unconsumed records. These
commands are experimental and remain inaccessible from the regular
HTTP inference API unless the corresponding engine protocol is exposed
by its host.

The ring has no disk I/O on the decode path; drain is a separate
idle command. A 4096-entry buffer can overflow quickly when a model
selects many experts per token. Gaps are visible through the event
sequence or drop counters. No synthetic replacement events should be
used in prediction accuracy calculations. This currently covers
host-visible non-device-planned verify windows only. The default
execution path does not allocate the buffer or activate the hook.

## From live diagnostics to the shadow evaluator

Capture the diagnostic output of `PROBE trace drain` in a log. Convert
the ordered records (other engine log lines are ignored) with:

```sh
python tools/aei_trace_to_csv.py --require-lossless < engine-trace.log > ordered.csv
bin/aei-shadow 512 48 ordered.csv
```

The model geometry shown is an example. The converter rejects malformed
events, duplicate or regressing per-stream sequence values, inconsistent
drain counts, counter resets, and any observed loss if
`--require-lossless` is selected. It validates at most 65,536 distinct
stream cursors and consumes input incrementally. Check exit status
**before** using `ordered.csv`: shell redirection may leave an incomplete
file on validation failure. Input should cover the full tracing session,
including drain summaries. After `trace off`, start a new conversion
session because loss counters reset with a newly allocated buffer.

An event gap from a dropped or invalid ID also remains detectable by
the downstream shadow evaluator's sequence checks. Since verifier
windows can include rejected speculative drafts, these measurements
describe executed router activity, **not** necessarily accepted output
tokens. They are not production wall-clock performance evidence.

## CPU expert service timing (experimental)

At an idle command boundary, `PROBE cost on` enables the existing
bounded AEI scheduling-cost model and attaches it to both verifier
chains. `PROBE cost report` emits `AEI_COST cpu bucket observations
ewma_ns` for each of eight workload-size buckets. `PROBE cost off`
detaches the observer and discards its measurements. Inference remains
unmodified: all scheduling decisions continue through Basalt's existing
policy. There is no additional device allocation.

The sample spans the host's expert-pool callback inside
`Verifier::serve_poll` in the non-device-planned path. This is a
**CPU service wall-time**, not a clean per-expert kernel duration, GPU
execution time, PCIe DMA time or critical-path latency. A candidate
CPU/GPU placement must remain `baseline` until separately measured GPU
observations are available. The probe is disabled by default and needs
on-device benchmarking to quantify timestamp overhead.

## Asynchronous disk capture

At an idle engine command boundary, `PROBE trace file /absolute/path/events.csv`
starts a platform-owned low-priority polling writer. Each event is enqueued by
the existing nonblocking producer; file opening, writing and flushing happen
outside inference. `PROBE trace stats` prints dropped events, written rows
and I/O errors. `PROBE trace off` detaches the verifier, joins the writer,
drains pending events, closes the file and reports final counts.

The supplied path is opened with `wb` (existing file truncated); choose
a new output filename for each session. Avoid placing this on a congested
disk, and check `io_errors=0`, `dropped_capacity=0` and
`dropped_contention=0` at shutdown before considering a trace lossless.
Only files with a clean session end should be evaluated. The writer
continues draining after a write failure to keep inference nonblocking,
so I/O errors invalidate completeness. A very slow filesystem can cause
ring overflow despite asynchronous output.
