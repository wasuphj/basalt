"""previous_response_id binds a thread context only when the registry is configured, and chains turns."""
import types
import unittest

from serve.api import responses as api
from serve.api.context_adapter import ContextAdapter
from serve.mantle.context_client import ResponseThreadMap
from serve.service import Service


class CheckRequestGate(unittest.TestCase):
    def test_stateless_still_rejects(self):
        with self.assertRaises(api.ResponsesError) as e:
            api.check_request({"model": "m", "input": "x", "previous_response_id": "resp_1"})
        self.assertEqual(e.exception.param, "previous_response_id")

    def test_stateful_allows(self):
        api.check_request({"model": "m", "input": "x", "previous_response_id": "resp_1"}, stateful=True)

    def test_conversation_and_background_stay_rejected(self):
        for param in ("conversation", "background"):
            with self.assertRaises(api.ResponsesError):
                api.check_request({"model": "m", "input": "x", param: "z"}, stateful=True)


class BindResponse(unittest.TestCase):
    @staticmethod
    def _svc():
        m = ResponseThreadMap()
        adapter = ContextAdapter("http://registry", resolver=m)
        return types.SimpleNamespace(context=types.SimpleNamespace(ad=adapter), response_threads=m), m

    def test_thread_bound_response_chains(self):
        svc, m = self._svc()
        Service.bind_response(svc, {"thread_id": "t1"}, "resp_1")
        self.assertEqual(m("resp_1"), ("t1", None, None))
        Service.bind_response(svc, {"metadata": {"thread_id": "t2", "branch_id": "b"}}, "resp_2")
        self.assertEqual(m("resp_2"), ("t2", "b", None))

    def test_previous_response_id_chains_forward(self):
        svc, m = self._svc()
        Service.bind_response(svc, {"thread_id": "t1"}, "resp_1")
        Service.bind_response(svc, {"previous_response_id": "resp_1"}, "resp_2")
        self.assertEqual(m("resp_2"), ("t1", None, None))

    def test_unbound_and_unknown_are_not_remembered(self):
        svc, m = self._svc()
        Service.bind_response(svc, {"input": "hi"}, "resp_x")
        Service.bind_response(svc, {"previous_response_id": "resp_never_issued"}, "resp_y")
        self.assertIsNone(m("resp_x"))
        self.assertIsNone(m("resp_y"))

    def test_no_registry_remembers_nothing(self):
        m = ResponseThreadMap()
        svc = types.SimpleNamespace(context=None, response_threads=m)
        Service.bind_response(svc, {"thread_id": "t1"}, "resp_1")
        self.assertIsNone(m("resp_1"))


class Envelope(unittest.TestCase):
    def test_previous_response_id_is_echoed(self):
        asm = api.Assembler({"previous_response_id": "resp_9"}, "model", 0, {}, False)
        self.assertEqual(asm.response["previous_response_id"], "resp_9")


if __name__ == "__main__":
    unittest.main()
