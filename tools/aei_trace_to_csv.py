#!/usr/bin/env python3
"""Convert Basalt's opt-in AEI diagnostic lines to ordered router CSV."""
import argparse
import csv
import re
import sys

EVENT = re.compile(r"AEI ([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+)$")
END = re.compile(r"PROBE trace end count=([0-9]+) dropped_capacity=([0-9]+) dropped_contention=([0-9]+)$")
MAX_U64 = (1 << 64) - 1
MAX_U32 = (1 << 32) - 1
MAX_STREAMS = 65536


def convert(lines, writer, require_lossless=False):
    """Return (events, new_drops, batches); error on corrupted event protocol."""
    writer.writerow(("stream", "sequence", "layer", "expert"))
    previous_sequence = {}
    previous_drops = (0, 0)
    events = 0
    reports = 0
    previous_report_events = 0
    new_drops = 0
    for line_number, raw in enumerate(lines, 1):
        line = raw.rstrip("\r\n")
        match = EVENT.fullmatch(line)
        if match:
            stream, seq, layer, expert = map(int, match.groups())
            if stream > MAX_U64 or seq > MAX_U64 or layer > MAX_U32 or expert > MAX_U32:
                raise ValueError(f"line {line_number}: integer outside router event bounds")
            if stream not in previous_sequence and len(previous_sequence) == MAX_STREAMS:
                raise ValueError(f"line {line_number}: exceeded bounded stream table")
            last = previous_sequence.get(stream)
            if last is not None and seq <= last:
                raise ValueError(f"line {line_number}: duplicate or nonmonotonic stream sequence")
            previous_sequence[stream] = seq
            writer.writerow((stream, seq, layer, expert))
            events += 1
            continue
        if line.startswith("AEI "):
            raise ValueError(f"line {line_number}: malformed AEI event")
        match = END.fullmatch(line)
        if match:
            declared, capacity, contention = map(int, match.groups())
            if declared > 128 or (events - previous_report_events) != declared:
                raise ValueError(f"line {line_number}: inconsistent drain batch count")
            if capacity > MAX_U64 or contention > MAX_U64:
                raise ValueError(f"line {line_number}: drop counter overflow")
            if capacity < previous_drops[0] or contention < previous_drops[1]:
                raise ValueError(f"line {line_number}: drop counter reset inside trace")
            new_drops += capacity - previous_drops[0] + contention - previous_drops[1]
            previous_drops = (capacity, contention)
            previous_report_events = events
            reports += 1
            continue
        if line.startswith("PROBE trace end"):
            raise ValueError(f"line {line_number}: malformed trace summary")
    if events != previous_report_events:
        raise ValueError("unterminated diagnostic batch")
    if require_lossless and new_drops:
        raise ValueError(f"trace lost {new_drops} events (capacity/contention)")
    return events, new_drops, reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-lossless", action="store_true")
    args = parser.parse_args(argv)
    try:
        events, dropped, batches = convert(sys.stdin, csv.writer(sys.stdout), args.require_lossless)
    except ValueError as error:
        print(f"aei-trace-to-csv: {error}", file=sys.stderr)
        return 2
    print(f"aei-trace-to-csv: events={events} batches={batches} new_drops={dropped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
