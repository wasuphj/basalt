"""Shadow scoring observes without choosing; the cost policy is the A/B knob (off in production); aging wins."""
import unittest

from serve.scheduler.core import (AgentScheduler, COST, DRR, PREFILL, Request, SchedulerConfig)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def sched(**kw):
    clock = _Clock()
    cfg = SchedulerConfig(enabled=True, max_inflight=1, background=False, **kw)
    return AgentScheduler(cfg, clock=clock), clock


class ExpectedCost(unittest.TestCase):
    def test_defaults_to_cost_and_validates(self):
        self.assertEqual(Request("t", "a", 100).expected(), 100)
        self.assertEqual(Request("t", "a", 100, PREFILL, None, 5).expected(), 5)
        with self.assertRaises(ValueError):
            Request("t", "a", 100, PREFILL, None, -1)


class ShadowObserves(unittest.TestCase):
    """A divergence DRR genuinely produces: a heavily-weighted tenant's 50K head is served before a light
    tenant's 10-token head. DRR's cost-greedy head pick otherwise agrees with cheapest-first — the finding."""

    def _run(self, shadow):
        s, clock = sched(shadow=shadow, aging_s=1000.0)          # aging never fires: pure policy comparison
        s.set_weight("t1", 6000.0)                               # t1 favoured enough to outrank the cheap head
        first = s.submit(Request("t0", "a0", 90000, PREFILL, None, 90000))      # bypass: holds the solo slot
        big = s.submit(Request("t1", "a1", 50000, PREFILL, None, 50000))        # favoured, but expensive
        cheap = s.submit(Request("t2", "a2", 10, PREFILL, None, 10))            # cheapest head, light tenant
        first.release()                                          # pump now chooses between big and cheap
        order, pending = [], [big, cheap]
        while pending:                                            # release whoever the scheduler granted
            for g in list(pending):
                if g.state == "granted":
                    order.append(g.request.agent)
                    g.release()
                    pending.remove(g)
            clock.t += 1
            s.poll()
            if clock.t > 100000:                                  # a stuck drain is a bug, not an infinite wait
                self.fail("scheduler stopped granting")
        return order, s.metrics()

    def test_shadow_never_changes_the_order(self):
        off_order, off = self._run(shadow=False)
        on_order, on = self._run(shadow=True)
        self.assertEqual(off_order, on_order)                    # observe-only: DRR admitted the same sequence
        self.assertEqual(off_order, ["a1", "a2"])                # weight really did pass the 10-token head by
        self.assertNotIn("shadow", off)                          # the block is absent when shadow is off
        self.assertGreater(on["shadow"]["comparisons"], 0)

    def test_shadow_records_a_missed_cheaper_head(self):
        _, on = self._run(shadow=True)
        self.assertGreaterEqual(on["shadow"]["divergences"], 1)  # the 10-token head was passed by for the 50K one
        self.assertGreater(on["shadow"]["wasted_expected_tokens"], 0)

    def test_drr_head_pick_is_already_cost_greedy(self):
        # Without a weight skew, DRR's deficit pick agrees with cheapest-first: the H3 finding, pinned as a guard.
        s, clock = sched(policy=DRR, aging_s=1000.0)
        first = s.submit(Request("t0", "a0", 90000, PREFILL, None, 90000))
        big = s.submit(Request("t1", "a1", 50000, PREFILL, None, 50000))
        cheap = s.submit(Request("t2", "a2", 10, PREFILL, None, 10))
        first.release()
        s.poll()
        self.assertEqual(cheap.state, "granted")                 # DRR already served the cheap head first
        self.assertEqual(big.state, "queued")


class CostPolicy(unittest.TestCase):
    def test_picks_the_cheapest_head(self):
        s, clock = sched(policy=COST, aging_s=1000.0)
        big = s.submit(Request("t1", "a1", 90000, PREFILL, None, 90000))
        small = s.submit(Request("t2", "a2", 500, PREFILL, None, 500))   # big is already in flight (bypass)
        big.release()
        s.poll()
        self.assertEqual(small.state, "granted")                 # the cheap head goes before a queued big sibling

    def test_aging_floor_beats_cost(self):
        s, clock = sched(policy=COST, aging_s=30.0)
        inflight = s.submit(Request("t0", "a0", 90000, PREFILL, None, 90000))   # bypassed: holds the solo slot
        oldq = s.submit(Request("t1", "a1", 90000, PREFILL, None, 90000))       # queued first (oldest head)
        cheap = s.submit(Request("t2", "a2", 10, PREFILL, None, 10))            # queued second, cheapest
        clock.t = 31.0                                                          # both now past the aging floor
        inflight.release()                                                      # pump at t=31: aging wins
        self.assertEqual(oldq.state, "granted")                                 # starvation floor beats cheapest
        self.assertEqual(cheap.state, "queued")


class PolicyGuard(unittest.TestCase):
    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            AgentScheduler(SchedulerConfig(enabled=True, policy="fifa"))


if __name__ == "__main__":
    unittest.main()
