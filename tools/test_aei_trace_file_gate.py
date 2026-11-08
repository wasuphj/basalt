import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aei_trace_file_gate import InvalidTrace, verify

class TraceReceiptTest(unittest.TestCase):
    def check_case(self, csv, receipt, expect_success):
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / "events.csv"
            logs = Path(tmp) / "engine.log"
            events.write_text(csv, encoding="utf-8")
            logs.write_text(receipt, encoding="utf-8")
            if expect_success:
                result = verify(events, logs)
                self.assertEqual(result["verdict"], "pass")
                self.assertEqual(result["records"], 3)
                self.assertEqual(len(result["sha256"]), 64)
            else:
                with self.assertRaises(InvalidTrace):
                    verify(events, logs)

    good_csv = ("stream,sequence,layer,expert\n"
                "1,0,0,12\n2,0,0,2\n1,1,0,4\n")
    good_footer = ("status: idle\nPROBE trace file end written=3 "
                   "dropped_capacity=0 dropped_contention=0 io_errors=0 complete=1\n")

    def test_valid_interleaved_streams(self):
        self.check_case(self.good_csv, self.good_footer, True)

    def test_missing_last_event(self):
        self.check_case(self.good_csv[:-8], self.good_footer, False)

    def test_loss_reported(self):
        self.check_case(self.good_csv, self.good_footer.replace("dropped_capacity=0", "dropped_capacity=1"), False)

    def test_io_failure(self):
        self.check_case(self.good_csv, self.good_footer.replace("io_errors=0", "io_errors=1"), False)

    def test_missing_receipt(self):
        self.check_case(self.good_csv, "PROBE trace off\n", False)

    def test_duplicate_receipts(self):
        self.check_case(self.good_csv, self.good_footer * 2, False)

    def test_gap_in_stream(self):
        self.check_case(self.good_csv.replace("1,1,0,4", "1,3,0,4"), self.good_footer, False)

    def test_duplicate_event(self):
        self.check_case(self.good_csv.replace("1,1,0,4", "1,0,0,4"), self.good_footer, False)

    def test_integer_outside_bound(self):
        self.check_case(self.good_csv.replace("1,1,0,4", "1,1,0,4294967296"), self.good_footer, False)

if __name__ == "__main__":
    unittest.main()
