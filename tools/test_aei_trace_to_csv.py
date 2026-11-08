import csv
import io
import unittest
from tools.aei_trace_to_csv import convert


class TraceConversionTest(unittest.TestCase):
    def parse(self, raw, strict=False):
        out = io.StringIO()
        result = convert(io.StringIO(raw), csv.writer(out), strict)
        return result, list(csv.reader(io.StringIO(out.getvalue())))

    def test_valid_interleaved(self):
        trace = """basalt: extra status
AEI 4 0 2 3
AEI 7 0 2 1
PROBE trace end count=2 dropped_capacity=0 dropped_contention=0
AEI 4 2 2 4
PROBE trace end count=1 dropped_capacity=1 dropped_contention=0
"""
        result, rows = self.parse(trace)
        self.assertEqual(result, (3, 1, 2))
        self.assertEqual(rows[1], ["4", "0", "2", "3"])
        self.assertEqual(rows[-1], ["4", "2", "2", "4"])
        with self.assertRaisesRegex(ValueError, "lost 1"):
            self.parse(trace, strict=True)

    def test_reject_duplicate(self):
        with self.assertRaisesRegex(ValueError, "nonmonotonic"):
            self.parse("AEI 1 2 0 3\nAEI 1 2 0 4\nPROBE trace end count=2 dropped_capacity=0 dropped_contention=0\n")

    def test_reject_wrong_count(self):
        with self.assertRaisesRegex(ValueError, "batch count"):
            self.parse("AEI 1 0 0 0\nPROBE trace end count=0 dropped_capacity=0 dropped_contention=0\n")

    def test_reject_bad_event(self):
        with self.assertRaisesRegex(ValueError, "malformed AEI"):
            self.parse("AEI 1 x 0 4\n")

    def test_reject_unterminated_batch(self):
        with self.assertRaisesRegex(ValueError, "unterminated"):
            self.parse("AEI 1 0 0 4\n")

    def test_reject_counter_reset(self):
        trace = "PROBE trace end count=0 dropped_capacity=4 dropped_contention=0\nPROBE trace end count=0 dropped_capacity=2 dropped_contention=0\n"
        with self.assertRaisesRegex(ValueError, "counter reset"):
            self.parse(trace)


if __name__ == "__main__":
    unittest.main()
