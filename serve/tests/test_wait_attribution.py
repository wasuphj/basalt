"""Wait attribution: a queued request records what held the engine ahead of it (ctx/15 bottleneck evidence)."""
import collections
import json
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.engine.mock import MockEngine
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer

TEMPLATE = Path(__file__).resolve().parents[1] / "chat_template.jinja"


class Classes(unittest.TestCase):
    """_wait_class reads the live engine state; _merge_wait folds one request's stretches into the counters."""

    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "x"), tok, ChatTemplate(TEMPLATE))

    def test_the_three_classes_of_one_waiting_stretch(self):
        self.assertEqual(self.svc._wait_class(), "idle_gap")
        self.svc.status.update(busy=True, first_token=None)
        self.assertEqual(self.svc._wait_class(), "behind_prefill")
        self.svc.status["first_token"] = time.time()
        self.assertEqual(self.svc._wait_class(), "behind_decode")
        self.svc.status.update(busy=False)
        self.assertEqual(self.svc._wait_class(), "idle_gap")
        live = {"first_token": None}
        self.svc.live_reqs[1] = (live, collections.deque())
        self.assertEqual(self.svc._wait_class(), "behind_prefill")
        live["first_token"] = time.time()
        self.assertEqual(self.svc._wait_class(), "behind_decode")
        self.svc.live_reqs.clear()

    def test_merge_ignores_a_turn_that_came_around_at_once(self):
        self.svc._merge_wait({"behind_decode": 0.9})
        self.assertEqual(self.svc.wait_attr, {"classes": {}, "total_s": 0.0, "requests": 0})

    def test_merge_accumulates_per_class_and_overall(self):
        self.svc._merge_wait({"behind_decode": 3.0, "idle_gap": 1.5})
        self.svc._merge_wait({"behind_decode": 2.0})
        a = self.svc.wait_attr
        self.assertEqual(a["requests"], 2)
        self.assertAlmostEqual(a["total_s"], 6.5)
        self.assertEqual(a["classes"]["behind_decode"], {"total_s": 5.0, "requests": 2, "max_s": 3.0})
        self.assertEqual(a["classes"]["idle_gap"], {"total_s": 1.5, "requests": 1, "max_s": 1.5})


class Http(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.httpd = None

    def tearDown(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()

    def start(self, script, delay_s):
        self.engine = MockEngine(self.tok, script, max_context=16384, delay_s=delay_s)
        self.svc = Service(self.engine, self.tok, ChatTemplate(TEMPLATE))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def ask(self):
        body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "hi"}]}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status

    def wait_busy(self):
        for _ in range(6000):
            if self.svc.status.get("busy"):
                return
            time.sleep(0.005)
        self.fail("the first request never reached the engine")

    def queued_behind_a_prefill(self):
        # 3 tokens plus the end token at 0.8s: the engine holds first_token=None until 0.8s in, so a request
        # that arrives right after the first one starts is queued behind a prefill for its whole stretch
        self.start("Hi.", 0.8)
        first = threading.Thread(target=self.ask)
        first.start()
        self.wait_busy()
        self.assertEqual(self.ask(), 200)
        first.join(timeout=60)
        return self.svc.wait_attr

    def test_waiting_behind_a_prefill_is_attributed_to_it(self):
        a = self.queued_behind_a_prefill()
        self.assertEqual(set(a["classes"]), {"behind_prefill"})
        self.assertEqual(a["requests"], 1)
        self.assertGreaterEqual(a["classes"]["behind_prefill"]["total_s"], 1.0)

    def test_metrics_exposes_the_counters(self):
        self.queued_behind_a_prefill()
        with urllib.request.urlopen(self.base + "/metrics", timeout=10) as r:
            m = json.loads(r.read())
        a = m["wait_attribution"]
        self.assertEqual(a["requests"], 1)
        self.assertGreaterEqual(a["total_s"], 1.0)
        self.assertEqual(set(a["classes"]), {"behind_prefill"})

    def test_waiting_behind_a_decode_is_attributed_to_it(self):
        self.start("Hello there friend, how are you today?", 0.05)   # first token at 0.05s, ~2s of decoding
        first = threading.Thread(target=self.ask)
        first.start()
        self.wait_busy()
        time.sleep(0.4)                                             # past the first token: the rest decodes
        self.assertEqual(self.ask(), 200)
        first.join(timeout=60)
        a = self.svc.wait_attr
        self.assertEqual(set(a["classes"]), {"behind_decode"})
        self.assertGreaterEqual(a["classes"]["behind_decode"]["total_s"], 1.0)

    def test_a_solo_request_attributes_nothing(self):
        self.start("Hello.", 0.0)
        self.assertEqual(self.ask(), 200)
        self.assertEqual(self.svc.wait_attr, {"classes": {}, "total_s": 0.0, "requests": 0})


if __name__ == "__main__":
    unittest.main()
