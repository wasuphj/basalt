# Basalt Adaptive single-GPU baseline replay

The `aei-basalt-baseline` tool replays **the original standard
per-layer adaptive swap ranking**, not an LRU or frequency surrogate.
It mirrors the production `rank_swaps(swaps,false)` thresholds:
incoming usage >=2, candidate/eviction score margin 1.5, ordering
by descending gain, a maximum swaps-per-round cap, and usage decay
after each adaptation. CPU/GPU residence changes at these rounds
are replayed from the actual observed Expert choices.

It requires two independently captured inputs:

1. At an idle engine boundary on one GPU with no peer, remote tier
   or cross-layer XQ policy, run
   `PROBE baseline residents /path/residents.csv`.
   It exports the *actual* resident Expert identities atomically.
2. Use a lossless ordered Decode-only router CSV with **one request
   stream**, collected from the same engine/model.
   `stream,sequence,layer,expert` is the expected header, with
   consecutive per-stream sequences and ascending layer visits
   within each window. A wrap to an earlier layer ends a window.

Then run:

```sh
aei-basalt-baseline 48 128 16 4 0.9 0 residents.csv router.csv
```

The numeric arguments are `LAYERS`, `EXPERTS_PER_LAYER`,
`SWAPS_PER_ROUND`, `EVERY_WINDOWS`, `ADAPT_DECAY`, and
`PROMOTE_MARGIN`. **The values above are examples**; use those
of the real engine and loaded model. Output contains windows,
accesses, hits, misses, swaps and metadata bytes. Evaluate other
policies against the *same* held-out trace, memory geometry and
initial residency.

Scope is deliberately strict: single GPU, no peer/remote, no
cross-layer accelerated XQ policy, no prompt-seeding of usage during
the sampled window, and no speculative staging semantics. As the
current router CSV has no explicit window identifier, a trace that
does not show a decreasing layer between successive windows cannot
be used to infer accurate window boundaries. Do not call this
full fidelity for batch-interleaved, multi-stream, or all-one-layer
MoE workloads. The CLI rejects multiple streams and damaged
sequences rather than mixing them into one baseline.

This baseline uses the same ranking mechanics as ordinary per-layer
Basalt adaptation, but it does not emulate all asynchronous slot
refills, peer tier ownership or VRAM/PCIe overlap. These remain
separate from the exact engine's runtime, and measured TG, TTFT,
quality and memory decisions still require real engine runs.
