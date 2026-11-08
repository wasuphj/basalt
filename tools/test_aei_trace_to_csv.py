import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aei_trace_to_csv import convert, main


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

    def test_atomic_output_preserves_existing_file_on_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "events.csv"
            output.write_text("previous verified result", encoding="utf-8")
            trace = ("AEI 1 0 0 2\n"
                     "PROBE trace end count=1 dropped_capacity=1 dropped_contention=0\n")
            with patch("sys.stdin", io.StringIO(trace)):
                code = main(["--require-lossless", "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"),
                             "previous verified result")
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["events.csv"])

    def test_atomic_output_publishes_validated_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "events.csv"
            trace = ("AEI 5 0 2 3\n"
                     "PROBE trace end count=1 dropped_capacity=0 dropped_contention=0\n")
            with patch("sys.stdin", io.StringIO(trace)):
                code = main(["--require-lossless", "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(output.read_text(encoding="utf-8"),
                             "stream,sequence,layer,expert\n5,0,2,3\n")
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["events.csv"])


if __name__ == "__main__":
    unittest.main()
