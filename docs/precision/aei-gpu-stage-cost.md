# Measured GPU staged-Expert execution cost

The adaptive CPU/GPU split optimizer can now use three existing measurements:
CPU pool nanoseconds per CPU-tier miss, pinned-host-to-GPU staging DMA
nanoseconds per Expert blob, and GPU PCIe-group execution nanoseconds
per staged Expert.

The GPU cost is derived from the existing `--profile` verifier kernel
stamps for the PCIe Expert-group range. A measured group is divided by
the actual staged count from the current live LUT, not by all CPU misses.
A bounded EWMA becomes eligible after eight observations. It then
informs the cost of placing the next CPU-tier miss in the existing
GPU staged group, with no additional CUDA stream or kernel required.

When GPU profiling is disabled or the measurements are insufficient,
this term stays zero and the previous CPU+DMA planner applies. This
avoids allocating profiling buffers on default runs. Both verifier
chains report the resulting observed values through
`PROBE policy stats` and `PROBE policy split refresh`.

This is a measured PCIe-group term, not a whole GPU critical-path
profile or proof that the GPU and CPU work perfectly overlap.
The `--profile` mode itself has overhead and should not be equated
with uninstrumented throughput. Existing staging slot allocations,
Top-K routing and model numerics are unchanged.
