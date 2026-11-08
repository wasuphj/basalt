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
