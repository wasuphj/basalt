#!/usr/bin/env python3
"""Strict side-by-side verdict for two *actual* basalt-bench -o json results."""
import argparse
import json
import math
import statistics
import sys
from collections import defaultdict


class Indeterminate(Exception):
    pass


def finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Indeterminate(f"{name}: missing or nonnumeric")
    if not math.isfinite(value) or value <= 0:
        raise Indeterminate(f"{name}: nonpositive or nonfinite")
    return float(value)


def rows(report):
    grouped = defaultdict(dict)
    for item in report.get("results", []):
        key = item.get("test")
        run = item.get("repetition")
        if not isinstance(key, str) or not isinstance(run, int) or run < 1:
            raise Indeterminate("invalid test/repetition identity")
        if run in grouped[key]:
            raise Indeterminate(f"duplicate result: {key}/{run}")
        grouped[key][run] = item
    return grouped


def quality(report):
    q = report.get("quality")
    if not isinstance(q, dict) or not q.get("ran") or q.get("error"):
        raise Indeterminate("quality check absent, errored or not run")
    return q


def compare(baseline, candidate, min_repetitions=3, min_tg_gain=0.10,
            pp_regression=0.02, max_kld_p99=0.1, min_top1=99.0):
    for key in ("model", "driver", "cuda_driver", "gpus", "power_limit"):
        if baseline.get(key) != candidate.get(key):
            raise Indeterminate(f"unmatched test environment: {key}")
    a, b = rows(baseline), rows(candidate)
    if not a or a.keys() != b.keys() or not any(k.startswith("tg ") for k in a):
        raise Indeterminate("benchmark case lists do not match or lack decode cases")
    qa, qb = quality(baseline), quality(candidate)
    problems, scores = [], {}
    for test in sorted(a):
        if a[test].keys() != b[test].keys() or len(a[test]) < min_repetitions:
            raise Indeterminate(f"unpaired or undersampled case: {test}")
        is_decode = test.startswith("tg ")
        field = "tok_s" if is_decode else "engine_tok_s"
        ar, br = [], []
        for rep in sorted(a[test]):
            left, right = a[test][rep], b[test][rep]
            ar.append(finite_positive(left.get(field), f"baseline {test} {rep}"))
            br.append(finite_positive(right.get(field), f"candidate {test} {rep}"))
            for item in (left, right):
                if item.get("checksum_ok") is False or item.get("finish") in ("error", "failed"):
                    problems.append(f"{test}: checksum or finish failed")
        ratio = statistics.median(br) / statistics.median(ar)
        scores[test] = {"baseline_median": statistics.median(ar),
                        "candidate_median": statistics.median(br),
                        "ratio": ratio, "repetitions": len(ar)}
        if is_decode and ratio < 1.0 + min_tg_gain:
            problems.append(f"{test}: decode gain below required {min_tg_gain:.1%}")
        if not is_decode and ratio < 1.0 - pp_regression:
            problems.append(f"{test}: prefill regression exceeds {pp_regression:.1%}")
    for key in ("kld_p99", "top1_pct", "ppl", "base_ppl"):
        if key not in qa or key not in qb:
            raise Indeterminate(f"quality lacks {key}")
    kld = qb["kld_p99"]
    if isinstance(kld, bool) or not isinstance(kld, (float, int)) or not math.isfinite(kld) or kld < 0:
        raise Indeterminate("candidate kld_p99 is missing or invalid")
    if kld > max_kld_p99:
        problems.append("KLD p99 exceeds threshold")
    if finite_positive(qb["top1_pct"], "candidate top1_pct") < min_top1:
        problems.append("top1 agreement below threshold")
    ppl = finite_positive(qb["ppl"], "candidate ppl")
    base_ppl = finite_positive(qb["base_ppl"], "candidate base_ppl")
    if abs(math.log(ppl / base_ppl)) > 0.02:
        problems.append("candidate PPL diverges more than 2% in log space")
    return {"verdict": "fail" if problems else "pass", "issues": problems,
            "cases": scores, "quality": {"kld_p99": qb["kld_p99"],
                                          "top1_pct": qb["top1_pct"],
                                          "ppl_log_ratio": math.log(ppl / base_ppl)},
            "note": "A pass is a benchmark gate, not proof of full-model correctness or +64MiB RSS."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--min-tg-gain", type=float, default=0.10)
    args = parser.parse_args(argv)
    try:
        if not 0 <= args.min_tg_gain <= 10:
            raise Indeterminate("invalid gain threshold")
        with open(args.baseline, encoding="utf8") as f:
            baseline = json.load(f)
        with open(args.candidate, encoding="utf8") as f:
            candidate = json.load(f)
        result = compare(baseline, candidate, min_tg_gain=args.min_tg_gain)
    except (Indeterminate, ValueError, OSError, json.JSONDecodeError) as err:
        result = {"verdict": "indeterminate", "issues": [str(err)]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return {"pass": 0, "fail": 1, "indeterminate": 2}[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
