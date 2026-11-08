# Measured GPU staged-Expert execution cost

The adaptive CPU/GPU split optimizer can now use three existing measurements:
CPU pool nanoseconds per CPU-tier miss, pinned-host-to-GPU staging DMA
nanoseconds per Expert blob, and GPU PCIe-group execution nanoseconds
per staged Expert.

The GPU cost is derived from the existing `--profile` verifier kernel
stamps for the PCIe Expert-group range. CUDA publishes an ordered
host-visible ring with each selected Expert ID and its execution
kind. The observer counts distinct `kind=1` staged Experts in that
ring and divides the measured GPU group time by this count.
`n_cpu` is the remaining host-tier entries *after* staging, so
neither it nor the configured LUT count is a reliable denominator.
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
