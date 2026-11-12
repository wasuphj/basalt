"""The runner CLI over the orchestrator's mock controller/client: submit, status, recover, configuration errors."""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from serve.mantle import runner
from serve.mantle.orchestrator import Orchestrator

from serve.tests.test_mantle_orchestrator import Mock, handoff


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        other = tempfile.TemporaryDirectory()          # never beside the journal: all() reads every *.json there
        self.addCleanup(other.cleanup)
        self.h = os.path.join(other.name, "handoff.json")
        with open(self.h, "w") as f:
            f.write(handoff().dumps())

    def _patch(self, mock):
        orch = Orchestrator(self.tmp.name, mock, mock)
        orig = runner.build_orchestrator
        runner.build_orchestrator = lambda journal_dir=None: orch
        self.addCleanup(setattr, runner, "build_orchestrator", orig)
        return orch

    def _main(self, *argv) -> tuple:
        out = io.StringIO()
        with redirect_stdout(out):
            code = runner.main(list(argv))
        return code, out.getvalue()

    def test_submit_runs_and_reports(self):
        self._patch(Mock())
        code, out = self._main("submit", self.h, "--key", "k1", "--direction", "demote",
                               "--source", "big", "--target", "little")
        self.assertEqual(code, 0)
        t = json.loads(out)
        self.assertEqual(t["state"], "RUNNING")
        self.assertEqual(t["context_id"], "ctx-1")

    def test_status_roundtrip(self):
        self._patch(Mock())
        self._main("submit", self.h, "--key", "k2", "--direction", "demote",
                   "--source", "big", "--target", "little")
        code, out = self._main("status", "k2")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["key"], "k2")
        code, out = self._main("status", "--all")
        self.assertEqual(code, 0)
        self.assertEqual([t["key"] for t in json.loads(out)], ["k2"])

    def test_recover_finishes_a_half_done_transition(self):
        mock = Mock()
        orch = self._patch(mock)
        orch.begin("k3", "demote", "big", "little", handoff())   # journaled, not run
        code, out = self._main("recover")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)[0]["state"], "RUNNING")

    def test_rolled_back_transition_exits_nonzero(self):
        self._patch(Mock(fail={"release": 99}))
        code, out = self._main("submit", self.h, "--key", "k4", "--direction", "demote",
                               "--source", "big", "--target", "little")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["state"], "ROLLED_BACK")

    def test_a_stray_file_does_not_block_the_journal_read(self):
        self._patch(Mock())
        self._main("submit", self.h, "--key", "k5", "--direction", "demote",
                   "--source", "big", "--target", "little")
        with open(os.path.join(self.tmp.name, "notes.json"), "w") as f:
            f.write('{"schema_version": 1, "not": "a transition"}')      # e.g. a handoff dropped beside the journal
        code, out = self._main("status", "--all")
        self.assertEqual(code, 0)
        self.assertEqual([t["key"] for t in json.loads(out)], ["k5"])    # the stray is skipped, not fatal


class ConfigurationTests(unittest.TestCase):
    def _cleared(self, *names):
        env = {k: v for k, v in os.environ.items() if k not in names}
        patcher = unittest.mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_missing_profiles_exit(self):
        self._cleared("MANTLE_PROFILES")
        with self.assertRaises(SystemExit):
            runner.build_orchestrator()

    def test_missing_registry_exit(self):
        self._cleared("BASALT_THREAD_API_KEY")
        with self.assertRaises(SystemExit):
            runner._adapter()


if __name__ == "__main__":
    unittest.main()
"""The runner CLI over the orchestrator's mock controller/client: submit, status, recover, configuration errors."""
import io
import json
import os
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
