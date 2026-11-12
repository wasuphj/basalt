"""Harness hints (H1): the reaper's timing is the only consumer; validation, fencing, TTL and counters."""
import json
import os
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve.api import context_proxy
from serve.api.context_adapter import ContextAdapter
from serve.api.context_lifecycle import ContextLifecycle
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.service import Service
from serve.tests.test_lifecycle import ResidentEngine
from serve.util.text import ByteTokenizer

CID = "11112222-3333-4444-5555-666677778888"


class _Fifo:
    def acquire(self, timeout=None):
        return True

    def release(self):
        pass


class HintTests(unittest.TestCase):
    def setUp(self):
        env = {"BASALT_CONTEXT_COOL_IDLE_S": "600"}     # no reaper thread: the rounds are driven here
        patcher = mock.patch.dict("os.environ", env, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        svc = SimpleNamespace(loaded=lambda: True, fifo=_Fifo(), engine=object())
        self.lc = ContextLifecycle(svc, ContextAdapter("http://registry"))
        self.cooled = []
        self.lc._cool_one = lambda cid: self.cooled.append(cid)

    def hot(self, idle_s=700.0, gen=1):
        self.lc.hot[CID] = {"used": time.monotonic() - idle_s, "gen": gen}

    def test_hint_defers_the_idle_cool(self):
        self.hot()
        code, body = self.lc.note_hint(CID, {"event": "tool_wait_started", "expected_idle_ms": 60000,
                                             "hint_id": "h1"})
        self.assertEqual((code, body["accepted"]), (200, True))
        self.lc._cool_idle()
        self.assertEqual(self.cooled, [])
        self.assertEqual(self.lc._hint_counts["deferral_rounds"], 1)

    def test_no_hint_cools_exactly_as_before(self):
        self.hot()
        self.lc._cool_idle()
        self.assertEqual(self.cooled, [CID])

    def test_expired_hint_falls_back_to_the_old_policy(self):
        self.hot()
        self.lc.note_hint(CID, {"event": "tool_wait_started", "expected_idle_ms": 60000})
        self.lc.hot[CID]["hint"]["expires_at"] = time.monotonic() - 1     # TTL lapsed
        self.lc._cool_idle()
        self.assertEqual(self.cooled, [CID])
        self.assertEqual(self.lc._hint_counts["expired_without_resume"], 1)

    def test_resume_clears_the_hint_and_counts_as_honored(self):
        self.hot()
        self.lc.note_hint(CID, {"event": "tool_wait_started", "expected_idle_ms": 60000})
        self.lc._mark_hot(CID, 2)                     # the resume arrived: fresh entry, no hint, new gen
        self.assertNotIn("hint", self.lc.hot[CID])
        self.assertEqual(self.lc._hint_counts["honored"], 1)
        self.lc._cool_idle()                          # and a long-idle resume is not cooled by a stale hint
        self.assertEqual(self.cooled, [])             # the fresh entry is well within cool_idle_s

    def test_replay_of_the_same_hint_id_is_inert(self):
        self.hot()
        self.lc.note_hint(CID, {"event": "tool_wait_started", "expected_idle_ms": 60000, "hint_id": "h1"})
        self.lc.hot[CID]["hint"]["received_at"] = 0.0                     # prove the replay does not rewrite
        code, body = self.lc.note_hint(CID, {"event": "tool_wait_started", "expected_idle_ms": 9000,
                                             "hint_id": "h1"})
        self.assertEqual((code, body.get("replay")), (200, True))
        self.assertEqual(self.lc.hot[CID]["hint"]["expected_idle_ms"], 60000)
        self.assertEqual(self.lc._hint_counts["accepted"], 1)
        self.assertEqual(self.lc._hint_counts["idempotent_replays"], 1)

    def test_stale_generation_is_rejected(self):
        self.hot(gen=5)
        code, body = self.lc.note_hint(CID, {"event": "tool_wait_started", "expected_idle_ms": 1000,
                                             "generation": 4})
        self.assertEqual(code, 409)
        self.assertNotIn("hint", self.lc.hot[CID])

    def test_validation_and_unknown_context(self):
        self.assertEqual(self.lc.note_hint("no-such", {"event": "e", "expected_idle_ms": 1})[0], 404)
        self.hot()
        for bad in ({}, {"event": 3, "expected_idle_ms": 1}, {"event": "e"},
                    {"event": "e", "expected_idle_ms": -1},
                    {"event": "e", "expected_idle_ms": ContextLifecycle.HINT_MAX_IDLE_MS + 1},
                    {"event": "e", "expected_idle_ms": 1, "resume_probability": 1.5},
                    {"event": "e", "expected_idle_ms": 1, "hint_id": ""}):
            self.assertEqual(self.lc.note_hint(CID, bad)[0], 422, str(bad))
        self.assertNotIn("hint", self.lc.hot[CID])

    def test_probability_is_recorded_but_changes_nothing(self):
        self.hot()
        self.lc.hot["other"] = {"used": time.monotonic() - 700, "gen": 1}
        self.lc.note_hint(CID, {"event": "tool_wait_started", "expected_idle_ms": 60000,
                                "resume_probability": 0.0})
        self.lc.note_hint("other", {"event": "tool_wait_started", "expected_idle_ms": 60000,
                                    "resume_probability": 1.0})
        self.lc._cool_idle()                          # identical timing: the probability does not act in H1
        self.assertEqual(self.cooled, [])
        m = self.lc.hint_metrics()
        self.assertEqual({i["resume_probability"] for i in m["active"].values()}, {0.0, 1.0})
        self.assertEqual(m["counters"]["deferral_rounds"], 2)


class HintRouteTests(unittest.TestCase):
    def test_hint_path_intercepts_and_never_shadows_the_registry(self):
        hint = f"/v1/contexts/{CID}/hint"
        self.assertFalse(context_proxy.is_context_path(hint))        # the registry has no such route
        self.assertTrue(context_proxy.is_context_path(hint[:-len("/hint")]))   # serve intercepts it

    def setUp(self):
        patcher = mock.patch.dict("os.environ", {}, clear=False)
        patcher.start()
        os.environ.pop("BASALT_THREAD_API_KEY", None)
        self.addCleanup(patcher.stop)
        tok = ByteTokenizer()
        self.svc = Service(ResidentEngine(tok), tok,
                           ChatTemplate(Path(__file__).resolve().parents[1] / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.shutdown)
        self.addCleanup(self.httpd.server_close)

    def _post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_route_503_without_a_registry(self):
        self.assertIsNone(self.svc.context)
        code, body = self._post(f"/v1/contexts/{CID}/hint", {"event": "tool_wait_started",
                                                             "expected_idle_ms": 1000})
        self.assertEqual(code, 503)
        self.assertIn("not configured", body["error"]["message"])

    def test_route_passes_the_hint_to_the_lifecycle(self):
        seen = []

        class Stub:
            @staticmethod
            def note_hint(cid, body):
                seen.append((cid, body))
                return 200, {"context_id": cid, "accepted": True, "expires_in_ms": 61000}

        self.svc.context = Stub()
        code, body = self._post(f"/v1/contexts/{CID}/hint", {"event": "tool_wait_started",
                                                             "expected_idle_ms": 60000, "hint_id": "h"})
        self.assertEqual(code, 200)
        self.assertTrue(body["accepted"])
        self.assertEqual(seen[0][0], CID)
        self.assertEqual(seen[0][1]["event"], "tool_wait_started")


if __name__ == "__main__":
    unittest.main()
