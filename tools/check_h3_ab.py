#!/usr/bin/env python3
"""H3 A/B: the SAME trace through the real scheduler twice — DRR (production default) vs cheapest-expected-first.

Deterministic virtual-time replay (no GPU, no production): identical arrivals, costs, expected costs and service
durations; only the selection policy differs. Reports the gate table: p50/p95/max queue wait, expired (missed
deadlines), makespan and throughput. Nothing here changes production: the service always runs the default policy;
this tool decides whether a policy is worth proposing at all.

  python tools/check_h3_ab.py [--requests 80] [--aging 30] [--rate 2000] [--deadline 120]
"""
import argparse
import heapq
import json
import math
import random
import statistics
import sys

sys.path.insert(0, ".")
from serve.scheduler.core import AgentScheduler, COST, DRR, PREFILL, Request, SchedulerConfig


def make_trace(n: int, seed: int = 7, load: float = 0.85, rate: float = 2000.0) -> list:
    """Agent traffic: tenants with threads (mostly-hot prefix reuse) and cold one-shot jobs, bursty arrivals,
    scaled so the engine sits at exactly `load` of its service capacity: the queue forms but never collapses."""
    rng = random.Random(seed)
    rows, gaps = [], []
    for i in range(n):
        tenant = f"t{i % 3}"
        agent = f"{tenant}-a{i % 5}"
        cost = int(math.exp(rng.uniform(math.log(500), math.log(120000))))
        expected = cost if rng.random() < 0.3 else max(int(cost * rng.uniform(0.001, 0.05)), 1)
        gaps.append(rng.expovariate(1.0 / (0.6 if i % 5 else 2.4)))        # bursts of agent turns, occasional gaps
        rows.append((tenant, agent, cost, expected))
    service = sum(r[3] / rate + 0.4 for r in rows)
    scale = service / load / sum(gaps)                            # exact: sum(arrival gaps) * load = service work
    trace, t = [], 0.0
    for (tenant, agent, cost, expected), gap in zip(rows, gaps):
        t += gap * scale
        trace.append((round(t, 3), tenant, agent, cost, expected))
    return trace


def run(policy: str, trace: list, *, aging_s: float, rate: float, deadline_s: float) -> dict:
    now = [0.0]
    sched = AgentScheduler(SchedulerConfig(enabled=True, max_inflight=1, background=False, policy=policy,
                                           aging_s=aging_s), clock=lambda: now[0])
    events, waits, running = [], [], set()
    for i, ev in enumerate(trace):
        heapq.heappush(events, (ev[0], 0, i))
    grants = {}
    while events:
        t, kind, payload = heapq.heappop(events)
        now[0] = max(now[0], t)
        if kind == 0:                                              # arrival
            _, tenant, agent, cost, expected = trace[payload]
            req = Request(tenant, agent, cost, PREFILL, now[0] + deadline_s, expected)
            try:
                grants[payload] = sched.submit(req)
            except Exception:                                      # queue full: the trace's own limit, equal for both
                grants[payload] = None
        else:
            payload.release()
            running.discard(payload)
        sched.poll()                                               # admit whatever fits at this virtual instant
        for key, g in grants.items():
            if g is not None and key not in running and getattr(g, "t_grant", None) is not None:
                running.add(key)
                waits.append(g.t_grant - g.t_submit)
                heapq.heappush(events, (g.t_grant + g.request.expected() / rate + 0.4, 1, g))
    expired = sum(1 for g in grants.values() if g is not None and g.state == "expired")
    rejected = sum(1 for g in grants.values() if g is None)
    waits.sort()
    pct = lambda q: round(1000 * (waits[min(len(waits) - 1, max(0, math.ceil(q / 100 * len(waits)) - 1))]
                                   if waits else 0), 1)
    computed = sum(x[4] for x in trace)                            # expected tokens actually computed
    return {"policy": policy, "requests": len(trace), "admitted": len(waits), "expired": expired,
            "rejected": rejected, "wait_ms": {"p50": pct(50), "p95": pct(95), "max": pct(100)},
            "sum_wait_s": round(sum(waits), 1), "makespan_s": round(now[0], 1),
            "computed_tokens": computed, "throughput_tok_s": round(computed / max(now[0], 1e-9))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=80)
    ap.add_argument("--aging", type=float, default=30.0)
    ap.add_argument("--rate", type=float, default=2000.0)          # prefill tokens/s for the duration model
    ap.add_argument("--deadline", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--load", type=float, default=0.85)
    args = ap.parse_args()
    trace = make_trace(args.requests, args.seed, args.load, args.rate)
    a = run(DRR, trace, aging_s=args.aging, rate=args.rate, deadline_s=args.deadline)
    b = run(COST, trace, aging_s=args.aging, rate=args.rate, deadline_s=args.deadline)
    verdict = {"drr": a, "cost": b,
               "p95_improved_ms": round(a["wait_ms"]["p95"] - b["wait_ms"]["p95"], 1),
               "adopt_cost": b["wait_ms"]["p95"] < a["wait_ms"]["p95"] and b["expired"] <= a["expired"]
                             and b["wait_ms"]["max"] <= a["wait_ms"]["max"] * 1.10}
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
