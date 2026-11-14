# Proactive Expert read-ahead

In opt-in `PROBE policy prefetch on`, or when `--aei-auto`
selects a Prefetch arm, the runtime now activates a dedicated
host-source prefetch worker. The live Router observer learns
successor edges and sends a bounded, throttled flattened Expert ID
hint to an in-memory queue as soon as the prediction becomes
eligible. The producer uses a try-lock and never calls file I/O.

The worker drains at most 64 hints per polling cycle, drops duplicates
in the same batch and calls the existing
`ExpertSource::prefetch(layer, expert)` on the source adapter.
For mmap-backed experts this is a page-cache read-ahead hint;
resident RAM sources may ignore it. The original adaptive GPU swap
ranking still admits predicted experts when an adaptation round runs.
The two are complementary: page preparation can begin earlier,
before the VRAM promotion is decided.

The queue holds at most 512 IDs. A saturated/contended producer
drops hints rather than blocking inference; hints affect latency only,
not Expert identity, arithmetic, cache ownership or logits. Every
policy switch detaches the observer, joins the worker and discards
outstanding hints. The worker never persists across engine shutdown.

This improves the execution contract for predictive Prefetch but is
not a promise that an OS/filesystem will complete read-ahead by the
next GPU access. A real disk page-cache and target GPU benchmark are
needed to measure the benefit.
