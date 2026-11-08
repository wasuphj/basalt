# AEI persistent policy evidence

Persistent server mode may opt into a bounded policy checkpoint:

```sh
basalt --serve --aei-auto --aei-state /path/to/aei-policy.state
```

The policy FSM stores sampled per-arm TG and PP statistics and
counters. When a Prefetch observer has learned actual Expert
transitions, the same checkpoint also stores its bounded sparse
transition table in portable binary format. It never stores
in-flight operations, device pointers, Expert residency, CUDA graphs,
the currently applied arm, or permissions to perform actions. On
restart the engine starts in Baseline exploration with those historical
statistics restored. No runtime state is revived from disk.

A checkpoint is accepted only for the same packed model section
manifest and an identical GPU selection and execution configuration
(device identity, model geometry, adaptation settings, peer/split
configuration). These attributes are hashed into the environment
identity, separate from the payload checksum. A missing, damaged or
incompatible checkpoint is ignored, not partially loaded.

Storage uses bounded length-prefixed policy text and optional sparse
transition bytes with SHA-256, written to a sibling
temporary file and atomically renamed. The engine updates it at
64 completed eligible requests and during normal shutdown. It does
not claim atomic durability on sudden power loss or protection from
malicious actors who can rewrite both the file and its checksum.

The switch is opt-in; ordinary Basalt startup uses no persistent
AEI state. To start fresh, omit `--aei-state` or select a new path.
This checkpoint persists policy-selection evidence and the sparse
next-Expert transition table (maximum 4096 hashed sources). It does
not persist large per-Expert reuse intervals, calibrated CPU/GPU cost
observations, in-flight predictions, or actual VRAM cache residency.
