# Causal predictive Prefetch reliability

The live sparse Expert predictor now evaluates an edge before inserting
that edge into its frequency table. Its hit and evaluation counters
therefore reflect whether the previous observation would have
predicted the next Expert, not accuracy inflated by training on the
same sample.

Only successors with at least three observed transitions and at least
25% of the source's observed Top-4 edge mass can receive a Prefetch
boost or a proactive source read-ahead hint. Each prediction considers
up to the strongest two successors.

The predictor also maintains a bounded EWMA of causal Top-2 accuracy.
After 128 eligible transitions, observed accuracy below 25%
temporarily suppresses new hints. The model continues learning while
hints are suppressed, so a later shift to a more predictable workload
can enable them again. The previous causal transition history is
still stored in the optional versioned AEI checkpoint, while the
short-lived accuracy and Prefetch hints reset on restart.

`PROBE policy stats` reports evaluated and correct transitions and
the EWMA percentage. This gates I/O and cache-admission hints, not
router decisions or expert arithmetic. The thresholds are defaults
for the experimental Prefetch policy, not guarantees of a throughput
benefit.
