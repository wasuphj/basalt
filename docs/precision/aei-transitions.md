# AEI policy transition contract

Every automatic transition is chosen at a completed request boundary.
The old stage table and previous observer pointers remain valid until
preconditions for the new policy are checked.

If a previously active CPU/GPU staged split must be deactivated, the
runtime first restores the original GPU LUTs. The two Verifier chains
are restored in sequence. If the second restore fails, the runtime
attempts to return the first chain to its prior dynamic lookup and
rejects the policy transition instead of reporting an applied state.
The callback pointers and optional predictive I/O worker are only
detached after that rollback phase succeeds.

Switches between `split` and `prefetch+split` retain the existing
GPU staged-share table and reuse calibration rather than restoring
and rebuilding it unnecessarily. The composite policy activates the
same proactive source Prefetch worker as the standalone Prefetch arm.
An invalid policy enum is rejected before changing any observer or
device state.

A persistent CUDA failure can still require terminating the engine
instead of continuing to request a broken transition. No GPU
partial-reset is claimed to be an atomic CUDA transaction; the
controller records failure and preserves the previous applied arm.
