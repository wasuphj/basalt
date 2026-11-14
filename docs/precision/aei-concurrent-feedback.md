# Automatic AEI policy feedback for concurrent requests

The single-request path records committed Decode output after the final
GPU commit, as before. Requests that merely admit a prompt into a
`BGEN` batch slot are excluded from single-request FSM observations:
an admission mini-decode is not representative of multi-token TG
throughput.

The normal batch loop and pipelined batch loop now each record a
completed epoch when all active batch slots and in-flight pipeline
groups become idle. The sample uses cumulative emitted output tokens
and elapsed batch-epoch wall time, including device completion and
the concurrent workload rather than multiplying a single slot's
performance by the number of slots.

The automatic FSM sees these as larger output-token workload buckets,
and may adjust the selected policy only before the next idle request.
No GPU resident cache or speculative group is switched while any
batch slot or stage is running. Epoch stats reset after sampling.

This is end-to-end concurrent throughput feedback at the configured
batch geometry; it is not equivalent to an isolated single-request
latency SLO. Prompt-processing cost for BGEN admission is not yet
attributed per batch epoch, so the separate PP safeguard is meaningful
only for comparable requests having prompt metrics.
