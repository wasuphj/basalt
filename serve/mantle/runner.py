"""CLI entry for Mantle BIG/LITTLE transitions: submit a handoff, watch the journal, recover after a crash.

The orchestrator drives the real `vman` controller and the context registry; everything it does is journaled
(fsynced) before and after each effect, so `submit` is safe to re-run with the same key and `recover` finishes
whatever a crash left half-done. Failure is data: a transition that cannot complete rolls back and reports
ROLLBACK_FAILED, it does not crash the runner.

Configuration (environment):
  MANTLE_PROFILES     JSON {model: vman profile}, e.g. {"qwen3.8-flash-next-nvfp4": "strata", "gpt-oss-120b": "llm"}
  MANTLE_JOURNAL_DIR  where transitions are journaled (default /data/runner/strata/state/mantle-journal)
  BASALT_THREAD_API_KEY / _HOST / _PORT  the context registry (same as the serve context adapter)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from .context_client import RegistryContextClient
from .handoff import Handoff, HandoffError
from .orchestrator import RUNNING, Orchestrator
from .vman_controller import default_controller

DEFAULT_JOURNAL = "/data/runner/strata/state/mantle-journal"


def _profiles() -> dict:
    raw = os.environ.get("MANTLE_PROFILES") or ""
    try:
        p = json.loads(raw) if raw.strip() else {}
    except ValueError:
        raise SystemExit("MANTLE_PROFILES must be a JSON object {model: vman profile}")
    if not isinstance(p, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in p.items()):
        raise SystemExit("MANTLE_PROFILES must map model names to vman profiles (strings)")
    return p


def build_orchestrator(journal_dir: Optional[str] = None) -> Orchestrator:
    """The wired orchestrator: real vman, real registry, fsynced journal."""
    profiles = _profiles()
    if not profiles:
        raise SystemExit("MANTLE_PROFILES is not set: the runner needs {model: vman profile} to switch engines")
    directory = journal_dir or os.environ.get("MANTLE_JOURNAL_DIR") or DEFAULT_JOURNAL
    return Orchestrator(directory, default_controller(profiles), RegistryContextClient(_adapter()))


def _adapter():
    from ..api.context_adapter import ContextAdapter
    adapter = ContextAdapter.from_env()
    if adapter is None:
        raise SystemExit("the context registry is not configured (BASALT_THREAD_API_KEY); a handoff must "
                         "re-prefill the transcript under the target fingerprint")
    return adapter


def _report(t) -> dict:
    return {"key": t.key, "state": t.state, "direction": t.direction, "source": t.source_model,
            "target": t.target_model, "context_id": t.context_id, "error": t.error,
            "attempts": t.attempts, "history": t.history}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m serve.mantle.runner",
                                 description="drive journaled Mantle BIG/LITTLE transitions")
    ap.add_argument("--journal-dir", default=None, help="journal directory (default $MANTLE_JOURNAL_DIR)")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("submit", help="journal a handoff and run the transition to a terminal state")
    s.add_argument("handoff", type=argparse.FileType("r"), help="Handoff JSON (plan/evidence/task_dag/transcript_ref)")
    s.add_argument("--key", required=True, help="idempotency key; re-running with the same request is safe")
    s.add_argument("--direction", required=True, choices=("demote", "promote"))
    s.add_argument("--source", required=True, help="model the work leaves (its residency is released)")
    s.add_argument("--target", required=True, help="model the work arrives at (it re-prefills the transcript)")

    g = sub.add_parser("status", help="print a journaled transition (or all of them)")
    g.add_argument("key", nargs="?", default=None)
    g.add_argument("--all", action="store_true")

    sub.add_parser("recover", help="finish every non-terminal journaled transition (after a crash)")

    args = ap.parse_args(argv)
    orch = build_orchestrator(args.journal_dir)

    if args.command == "submit":
        try:
            handoff = Handoff.loads(args.handoff.read())
            args.handoff.close()
        except (HandoffError, ValueError) as e:
            print(f"handoff.json is not a valid handoff: {e}", file=sys.stderr)
            return 2
        try:
            t = orch.submit(args.key, args.direction, args.source, args.target, handoff)
        except Exception as e:                                  # a rolled-back transition is reported, not raised
            print(f"transition {args.key} failed: {type(e).__name__}: {e}", file=sys.stderr)
            t = orch.get(args.key)
            if t is None:
                return 2
        print(json.dumps(_report(t), ensure_ascii=False))
        return 0 if t.state == RUNNING else 1

    if args.command == "status":
        if args.all or args.key is None:
            print(json.dumps([_report(t) for t in orch.journal.all()], ensure_ascii=False, indent=1))
            return 0
        t = orch.get(args.key)
        if t is None:
            print(f"no transition journaled under key {args.key!r}", file=sys.stderr)
            return 1
        print(json.dumps(_report(t), ensure_ascii=False))
        return 0

    done = orch.recover()                                        # submit-style failures inside recovery are journaled
    print(json.dumps([_report(t) for t in done], ensure_ascii=False))
    return 0 if all(t.state == RUNNING for t in done) else 1


if __name__ == "__main__":
    raise SystemExit(main())
