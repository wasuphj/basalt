#!/usr/bin/env python3
"""Verify a saved AEI router CSV against the trace-off completion receipt."""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

FOOTER = re.compile(
    r"PROBE trace file end written=(\d+) dropped_capacity=(\d+) "
    r"dropped_contention=(\d+) io_errors=(\d+) complete=([01])$"
)
MAX_STREAMS = 65536
MAX_UINT64 = (1 << 64) - 1
MAX_UINT32 = (1 << 32) - 1

class InvalidTrace(ValueError):
    pass

def verify(csv_path: Path, log_path: Path) -> dict:
    receipts = []
    with log_path.open(encoding="utf-8") as logs:
        for line in logs:
            match = FOOTER.search(line.rstrip("\r\n"))
            if match:
                receipts.append(tuple(int(v) for v in match.groups()))
    if len(receipts) != 1:
        raise InvalidTrace(f"expected exactly one completion receipt, found {len(receipts)}")
    written, dropped_capacity, dropped_contention, io_errors, complete = receipts[0]
    if max(written, dropped_capacity, dropped_contention, io_errors) > MAX_UINT64:
        raise InvalidTrace("receipt counter outside uint64 range")
    if complete != 1 or dropped_capacity or dropped_contention or io_errors:
        raise InvalidTrace("producer reports missing events or an I/O failure")
    if written <= 0:
        raise InvalidTrace("trace contains no recorded expert events")
    original_stat = csv_path.stat()
    seen = {}
    digest = hashlib.sha256()
    count = 0
    with csv_path.open("rb") as raw:
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        # Replay the data in a bounded streaming second pass. Neither
        # hashing nor sequence validation loads the entire trace into RAM.
        with csv_path.open("r", encoding="utf-8", newline="") as trace:
            reader = csv.reader(trace, strict=True)
            if next(reader, None) != ["stream", "sequence", "layer", "expert"]:
                raise InvalidTrace("invalid CSV header")
            for line, fields in enumerate(reader, 2):
                if len(fields) != 4 or any(not field.isascii() or not field.isdecimal()
                                           for field in fields):
                    raise InvalidTrace(f"invalid CSV row at record {line}")
                stream, sequence, layer, expert = map(int, fields)
                if stream > MAX_UINT64 or sequence > MAX_UINT64 or layer > MAX_UINT32 or expert > MAX_UINT32:
                    raise InvalidTrace(f"router ID outside bounds at record {line}")
                if stream not in seen and len(seen) >= MAX_STREAMS:
                    raise InvalidTrace("stream cursor cardinality bound exceeded")
                expected = seen.get(stream, -1) + 1
                if sequence != expected:
                    raise InvalidTrace(f"missing, duplicated or reordered event at record {line}: "
                                       f"stream={stream}, expected={expected}, got={sequence}")
                seen[stream] = sequence
                count += 1
    except (UnicodeError, csv.Error) as ex:
        raise InvalidTrace(f"malformed CSV: {ex}") from ex
    checked_stat = csv_path.stat()
    if (original_stat.st_size != checked_stat.st_size or
            original_stat.st_mtime_ns != checked_stat.st_mtime_ns):
        raise InvalidTrace("CSV changed during verification")
    if written != count:
        raise InvalidTrace(f"row count mismatch: receipt={written}, file={count}")
    return {"verdict": "pass", "records": count, "streams": len(seen),
            "sha256": digest.hexdigest(), "receipt_complete": True}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("engine_log", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.csv, args.engine_log)
    except (InvalidTrace, OSError) as ex:
        result = {"verdict": "invalid", "reason": str(ex)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "pass" else 2

if __name__ == "__main__":
    sys.exit(main())
