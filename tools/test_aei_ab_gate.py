import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aei_ab_gate import Indeterminate, compare


def report(speed=100.0):
    out = {"model": "fixture-model", "driver": "fixture-driver", "cuda_driver": "fixture-cuda",
           "gpus": [{"name": "fixture gpu"}], "power_limit": "fixture",
           "quality": {"ran": True, "error": "", "kld_p99": 0.04,
                       "top1_pct": 99.9, "ppl": 7.0, "base_ppl": 7.0},
           "results": []}
    for run in range(1, 4):
        out["results"].append({"test": "tg prose", "repetition": run, "tok_s": speed,
                               "tokens": 400, "finish": "length"})
        out["results"].append({"test": "pp16384", "repetition": run, "engine_tok_s": 300.0})
    return out


class AeiGateTest(unittest.TestCase):
    def test_pass(self):
        self.assertEqual(compare(report(), report(120))["verdict"], "pass")

    def test_speed_failure(self):
        self.assertEqual(compare(report(), report(103))["verdict"], "fail")

    def test_quality_failure(self):
        candidate = report(120)
        candidate["quality"]["kld_p99"] = .5
        self.assertEqual(compare(report(), candidate)["verdict"], "fail")

    def test_missing_quality_is_indeterminate(self):
        candidate = report(120)
        del candidate["quality"]
        with self.assertRaises(Indeterminate):
            compare(report(), candidate)

    def test_mismatched_hardware_is_indeterminate(self):
        candidate = report(120)
        candidate["driver"] = "other"
        with self.assertRaises(Indeterminate):
            compare(report(), candidate)

    def test_missing_repetition_is_indeterminate(self):
        candidate = report(120)
        candidate["results"].pop()
        with self.assertRaises(Indeterminate):
            compare(report(), candidate)

    def test_checksum_failure(self):
        candidate = report(120)
        candidate["results"][0]["checksum_ok"] = False
        self.assertEqual(compare(report(), candidate)["verdict"], "fail")


if __name__ == "__main__":
    unittest.main()
