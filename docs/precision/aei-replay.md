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

The device-planned MoE path is now included: CUDA's
`dm_route_kernel` publishes all Top-K IDs in `DmRing::ids`, followed
by a system fence and host doorbell. The host checks the layer index,
token count and `kDmCap` bound before reading. Invalid ring headers
advance the sequence to mark missing observations rather than invent
routing decisions. **Prompt Prefill routing is also captured by its own host-visible Top-K grouping path.**
Never reconstruct sequences from frequency histograms. The sink must be bound only when
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
used in prediction accuracy calculations. This currently covers both the host-visible and device-planned
verify windows plus each Prefill run/stage. The default
execution path does not allocate the buffer or activate the hook.

## From live diagnostics to the shadow evaluator

Capture the diagnostic output of `PROBE trace drain` in a log. Convert
the ordered records (other engine log lines are ignored) with:

```sh
python tools/aei_trace_to_csv.py --require-lossless --output ordered.csv < engine-trace.log
bin/aei-shadow 512 48 ordered.csv
```

The model geometry shown is an example. The converter rejects malformed
events, duplicate or regressing per-stream sequence values, inconsistent
drain counts, counter resets, and any observed loss if
`--require-lossless` is selected. It validates at most 65,536 distinct
stream cursors and consumes input incrementally. The recommended `--output` mode writes to a sibling temporary file,
flushes and fsyncs before atomically replacing `ordered.csv` on
successful validation. On loss or parse failure an earlier verified
output is preserved and the incomplete temporary file is removed.
Plain stdout streaming remains supported for pipelines, but a failed
pipeline can leave incomplete redirected output. Check exit status. Input should cover the full tracing session,
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

## Benchmark promotion gate (run on the target GPU)

Use the **existing** `basalt-bench -o json` executable for the same
model, driver, GPU/power limit, fixtures and three or more repetitions
in each arm. Capture separate JSON files for the baseline and the
candidate. The benchmark should include prefill and decode fixtures and
a completed PPL/KLD quality run. Compare the actual outputs with:

```sh
python tools/aei_ab_gate.py baseline.json candidate.json
```

Exit code 0 means the provided benchmark passes the default experimental
thresholds: at least +10% median TG in every decode fixture, no more
than 2% median PP regression, quality KLD p99 <= 0.1, top-1 agreement
>= 99%, and absolute log PPL ratio <= 0.02. Exit 1 means a gate failure;
exit 2 means **indeterminate** (missing/incompatible metrics or
insufficient runs). These thresholds are a starting research policy,
not a validated Basalt production SLO. All test identities and
repetition sets must match; GPU identity, drivers and power limits
must also match. `engine_args` may intentionally differ for the
candidate but should be reviewed and preserved with artifacts.

Even a PASS is not a proven improvement of all inference use cases:
measure TTFT, complete request latency, VRAM, host RSS, concurrency
scaling and application-level quality independently. Never use
fabricated JSON or synthetic toy fixtures as performance evidence.

## Experimental measured-cost adaptive placement

After attaching a live DM-router capture, idle commands support
`PROBE policy cost on`, `PROBE policy stats`, and
`PROBE policy baseline`. Cost policy observes the actual CPU pool
callback wall-time divided by the ring's CPU-tier miss entries, with
eight samples minimum per layer. If fewer than 25% of the layers
have enough observations (at least two layers), cross-layer adaptive
ranking uses the exact previous scoring. Once enough samples exist,
only cross-layer candidate and victim hotness gets a bounded
0.5x–2x relative CPU-cost weight. The existing swap caps, async copy
path, residency invariants and GPU kernels remain unchanged.

This is a **measured CPU-miss ranking heuristic**, not fully calibrated
CPU+GPU+DMA critical-path cost, and it does not activate the independent
ShadowScheduler's CPU/GPU placement candidate. The per-layer model
cannot infer benefit from repeated slots or overlap. Treat it as
experimental and run independent A/B before promotion. When
`xq_across` is disabled, ranking remains baseline even with the
cost observer enabled. All defaults are OFF.

## Experimental reuse-aware adaptive placement

An idle `PROBE policy reuse on` allocates an optional concurrent
per-expert reuse interval table (maximum 1,048,576 experts, two 64-bit
atomics each). Actual Top-K selected IDs on both verifier paths
update the interval EWMA; the normal adaptive cross-layer ranking is
then weighted 0.5–1.4x according to expected reuse urgency. A very
overdue expert decays instead of remaining pinned forever. This is
a lightweight reuse predictor, **not** an exact SpecMD implementation.

`PROBE policy cost on` and `PROBE policy reuse on` are mutually
exclusive: enabling one detaches the other. `PROBE policy baseline`
detaches both observers and restores unchanged rankings. Defaults stay
OFF. The heuristic does not alter per-layer-only adaptation, slot
ownership, GPU kernels or CUDA copy semantics; real A/B is needed
before treating it as an improvement.

## Comparable-work A/B safeguard

The AEI benchmark gate also requires equal generated token counts and
identical finish reasons for paired decode repetitions. Faster runs
that stopped early, hit a different EOS or generated fewer tokens are
**FAIL**, not accepted throughput improvements. Prefill repetitions
must match prompt depth, input tokens and reused-prefix tokens whenever
those measurements are supplied. Missing or malformed work accounting
is **INDETERMINATE**. Always keep the source JSON artifacts for
independent inspection.

## Continuous trace completeness receipt

For `PROBE trace file PATH`, shutdown now prints exactly one terminal
receipt in the engine log:
`PROBE trace file end written=N dropped_capacity=N dropped_contention=N
io_errors=N complete=0|1`. Completion is true only when both drop
counters and all write/flush/close errors are zero. This receipt is
**separate** from the CSV so malformed tail data cannot forge success.

Validate a captured file and its *matching* shutdown log before feeding
it into `aei-shadow`:

```sh
python tools/aei_trace_file_gate.py events.csv engine.log
```

The validator checks the terminal receipt, exact row count, numeric
bounds, per-stream continuous sequence numbers, stream cardinality and
streaming SHA-256 provenance. A missing, duplicate, or failed receipt,
or a gap in the original sequence, fails closed with exit code 2.
The certificate is a program-generated assessment of the captured
files, not an authenticated signature, GPU performance measurement or
a substitute for the full target engine E2E.

## Cleanup on process termination

The writer's final receipt is also generated by the normal runtime
shutdown path (QUIT or standard input EOF). This reuses the same
`stop_aei_trace` implementation as the idle `PROBE trace off`
command: verifiers are detached, the asynchronous worker drains and
joins, the CSV is closed, and the final outcome is emitted once.
The dispatch error path also attempts cleanup before returning error.
Abrupt power loss, SIGKILL, process crash, or incomplete GPU work
cannot guarantee a complete receipt; the offline validator rejects
any session without one.

## Model-bound learned expert profiles

A learned `--expert-profile-save` file is paired with
`<profile>.model-id`. The identity is derived from the actual packed model's **sorted,
non-profile sections**, their byte sizes and SHA-256 checksums; it
requires a metadata section, an identifiable weight-bearing section,
and populated section digests. An additional SHA-256 compresses the
section manifest into a constant-length versioned model identity. The section names depend on model
format, so optional or nonexistent names do not disable saving.
These digests are already stored by the .basalt packer; no full
weight rehash is needed at each startup. On startup a learned profile is used only if the identity
sidecar matches the currently loaded packed model. Otherwise the
model's original expert profile is used. `PROBE adapt profile=`
also rejects a saved learned profile bound to a different model.

Legacy learned profiles without a sidecar deliberately fall back to
the embedded model profile, rather than guessing based on matching
layer and expert counts. A model with missing section hashes or no identifiable
weight section cannot write a bound learned profile. This is an identity check, not
cryptographic verification of all model file bytes at runtime.

## Live sparse next-expert prefetch (experimental)

`PROBE policy prefetch on` now connects the bounded sparse Top-4
transition table to original verifier Top-K selections. Each request
has its own predecessor, so transitions cannot learn across request
boundaries. The table learns on the inference producer using a
nonblocking try-lock, predicting the next two possible experts after
each selected expert. After at least three observations of an edge,
candidate experts receive a short-lived boost (8,192 observed routing
entries) in the **existing adaptive cache swap ranking**. This
admits likely future GPU residents through the engine's ordinary
asynchronous CUDA Expert copy and swap machinery, respecting its
slot ownership, swap caps, remote ownership and transfer ordering.

This is predictive *admission* during existing adaptation rounds, not
a separate speculative CUDA DMA engine or a forced immediate transfer.
No prefix/routing logits are changed. The policy is explicitly enabled
and mutually exclusive with cost- and reuse-weighted placement; use
`PROBE policy baseline` to detach it and return to the original cache
policy. `PROBE policy stats` prints bounded metadata and skipped
samples. A default launch never allocates or updates the advisor.

## Measured CPU/GPU staged-share scheduling

The device-planned MoE already supports a **GPU-staged share** of
the CPU-tier misses. Its `stage_k` device lookup maps the number of
distinct misses to how many are copied to an existing CUDA staging
buffer. `PROBE policy split on` now connects the actual per-layer
CPU miss time observer to that planner without enabling experimental
cache-rank or next-expert policies. Once enough CPU samples exist,
`PROBE policy split refresh` can explicitly recompute its lookup.

On the first refresh, Basalt measures one pinned-host → **temporary
GPU scratch allocation** copy with CUDA events on the existing staging
stream (at idle). It never overwrites the live Expert staging weights.
The worker synchronizes the copy stream before releasing scratch
buffers, including on partial submission failure. The temporary
measurement is bounded to 16 MiB of host and GPU scratch. It
combines measured DMA nanoseconds per blob with the observed CPU
nanoseconds per miss. The dynamic plan minimizes the estimated maximum
of CPU remaining work and GPU staging work, bounded by the original
staging allocation and host-side minimum threshold. After completion
of subsequent requests, a >=10% change in measured CPU cost triggers
another idle-boundary lookup update. Both verifier chains use their
own existing device lookup, and the previous table is retained in
host metadata for immediate restoration.

`PROBE policy baseline` restores both original CUDA lookup tables,
detaches the observers and restores the standard device allocation.
The policy is **off by default**. No token routing, output logits,
GPU expert arithmetic, or capacity allocation changes. An unsupported
device staging path or insufficient samples leaves the baseline
table in place. The estimated GPU-side overhead still includes a
constant kernel/launch allowance; it is not a full critical-path
GPU kernel timing model.

## Prefill original Top-K capture

During batched Prefill, the GPU's routing results are already copied
to host memory before the CPU's expert-grouping loop. With a PROBE
trace enabled, these original `T*K` IDs are emitted in their token
order *before* the grouping sort, without extra D2H transfers.
Each Prefill run/stage gets a distinct high-bit stream ID and a
monotonic per-stream event sequence. Decode verifier streams use the
other half of the ID space. Layer split stages can interleave safely
because the shadow evaluator tracks per-stream cursor history.

The same nonblocking 4096-event buffer and asynchronous file writer
handle Prefill and Decode. Large prompts can overrun the buffer:
when this happens the final completeness receipt fails closed rather
than treating a truncated stream as evidence. The capture is
experimental and remains OFF by default.

## Predicted host source read-ahead before GPU promotion

The predictive admission policy now also issues up to eight
`ExpertSource::prefetch(layer, expert)` hints for chosen incoming
candidates **before** its ordinary CUDA slot refill. This is the
existing file-backed expert source's mapped-page read-ahead, which can
overlap storage and CPU work with the subsequent cache promotion.
It is used by both the standard adaptive `adapt()` loop and the
accelerated `xq_round()` path. Sources already resident in RAM may
ignore these hints. The predictor never requests extra GPU slots or
changes weight data, and the hint is inactive unless
`PROBE policy prefetch on` was explicitly enabled.

## Automatic request-boundary AEI policy controller

`PROBE policy auto on` enables a deterministic, fixed-memory
explore/exploit state machine. The baseline is measured first, followed
by applicable experimental policies: cross-layer CPU cost placement,
reuse-aware placement, sparse next-expert Prefetch, and measured
CPU/GPU staged split. Modes not applicable to the current cache layout,
GPU staging or model size are skipped; the engine remains on baseline
if there are no available alternatives.

Each arm collects twelve eligible completed decode requests; only
healthy completions with 32 or more generated tokens and a finite,
positive decode time are used. The selector compares mean milliseconds
per generated token against baseline **within matching generated-token
length buckets**. It promotes an arm only with at least four samples
in a comparable bucket and at least 8% lower observed latency per
token. It also samples real Prefill milliseconds per freshly processed
prompt token in independent prompt-length buckets. If any comparable
bucket regresses more than 2% versus baseline, the candidate is
rejected; if one policy has prompt measurements but there is no
comparable bucket, promotion is withheld. Otherwise the original
policy wins. During exploitation it also revisits baseline after each 32 eligible
requests, sampling 8 baseline requests in a dedicated `audit` state.
It restores the winner only when fresh baseline samples show at least
a 5% decode latency-per-token advantage for the candidate in a
comparable output-size bucket without a greater than 2% PP regression.
Otherwise it enters an eight-request baseline cooldown and retries
exploration. After 96 exploitation samples, the controller starts a
fresh full exploration round to accommodate longer-term traffic
changes.
Execution failure or nonfinite latency enters an eight-request
baseline cooldown before another exploration cycle. These boundaries
are initial product heuristics, not guarantees of statistical
significance; highly heterogeneous prompts can bias comparisons.

`PROBE policy auto stats` reports the FSM's current phase, intended
and applied policy, sample count, rejected samples, and transition
count. `PROBE policy auto off` restores the baseline policy and
detaches all auto-managed observers. Explicit manual policy commands
are rejected while auto owns the policy, avoiding competing writers.
Every policy transition occurs **between requests**, before entering
the next Request execution; no switch happens during the verifier or
a CUDA graph. The opt-in feature is OFF by default and preserves the
original scheduler for ordinary launches.

## Automatic startup (no interactive PROBE required)

Start persistent inference with `--serve --aei-auto` to initialize the
AEI FSM automatically after all server GPU caches, Verifier chains and
batch/pipeline resources have been built. This is the same implementation
used by `PROBE policy auto on`, not a separate tuning system.
The default remains OFF, and non-server generation rejects the option.
Request-boundary decisions, experiment isolation and baseline fallback
are otherwise identical to the interactive mode.

## Prefill-to-decode predictor warm start

When `PROBE policy prefetch on` or the automatic FSM selects the
Prefetch arm, both the batched Prefill and Decode Verifier paths now
share the *same* bounded `NextExpertPrefetch` instance. Prefill uses
the exact token-order Expert IDs that the existing CPU grouping step
already received from the GPU, so no new D2H transfer or CUDA kernel
is needed. A per-Prefill-stage predecessor is reset for every run,
preventing false transitions across requests or GPU hand-off stages.
The sparse transition table is guarded by its existing nonblocking
observer lock to tolerate parallel pipeline stages.

This means a long prompt can train the next-expert predictor before
the first decode window. The predicted CPU source read-ahead and
adaptive GPU cache admission subsequently consume those observations.
When the policy changes, Prefill observers are detached before the
shared predictor is destroyed. The original prompt output and GPU
routing logic are untouched.

## Continuity of policy learning across baseline audits

The automatic controller retains the bounded per-policy cost, reuse and
sparse-transition observers while switching temporarily to baseline.
Disabling an observer does not delete its learned state: returning to
Prefetch after an audit immediately reuses its prior causal Expert
transitions. CPU cost models likewise survive non-selected periods.
A newly started automatic session and explicit `PROBE policy auto off`
reset all these optional observers. Only one policy is active at any
given request boundary, and all inactive probes remain detached.

The FSM also keeps audit and cooldown observations separate from the
original baseline exploration samples, preventing a temporary
baseline probe from changing the estimate it is supposed to challenge.
This preservation is opt-in only; normal Basalt startup allocates none
of these policy data structures.

## Two-file profile commit safety

Saving an expert profile now writes a small `.model-id` receipt that
contains both the model's *versioned SHA-256 section-manifest identity*
and the SHA-256 of the exact learned rank file. The receipt itself is
written to a temporary file and atomically replaced only after the rank
file has been written and hashed. A crash or failed sidecar replacement
therefore leaves a *mismatched* old receipt rather than silently
accepting new profile bytes under an old model ID. Both hashes are
checked before loading learned ranks; missing/legacy receipts fall back
to the embedded model profile.

The learned rank file is capped at 64 MiB for this sidecar binding;
large packed weight files are never scanned during startup identity
calculation. These checks detect accidental partial updates and
cross-model profile reuse. They do not authenticate a maliciously
rewritten pair of files or make the pair a single filesystem
transaction.
