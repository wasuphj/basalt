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
