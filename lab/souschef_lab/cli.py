"""`souschef-lab` — one entry point for the lab's commands (docs/LAB.md §1).

    souschef-lab <command> [args…]       (also: python -m souschef_lab.cli …)

Every command is a module of this package with `main(argv) -> int`; the
command's own `--help` comes from that module. Modules are imported only
when their command runs, so `souschef-lab train --dry-run` never imports
torch and a command whose module is still missing fails on its own.
"""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS: dict[str, tuple[str, str]] = {
    # command: (module, one-line help)
    "cases": ("cases", "generate the case set from the manifest (data/cases.jsonl)"),
    "collect": ("collect", "ask the service each case, emulating the browser; resumable log"),
    "verify": ("verify", "re-verify logged rows against their cases"),
    "build": ("dataset", "build train/val/exam JSONL from a collect log"),
    "bestofn": ("bestofn", "best-of-N judging and selection"),
    "pairs": ("pairs", "DPO pairs from a best-of-N log"),
    "eval": ("eval", "run the held-out exam; --compare, --rescore"),
    "passk": ("passk", "pass@k over several eval runs"),
    "exam": ("exam", "the real-phrasing exam: sets, offline retrieval score, end-to-end run, conversation replay"),
    "exam-compare": ("examcmp", "compare exam runs across revisions: pass rate × set × route, Wilson 95 pct CI, cost"),
    "retrain-check": ("retrain_check", "should the model be retrained? exit 0/1/2 + Slack text"),
    "train": ("train", "SFT (Qwen3.5-9B LoRA); --dry-run needs no torch"),
    "dpo": ("dpo", "DPO against the frozen SFT reference; --dry-run"),
    "fields": ("fields", "the service's candidate-fields block (and, --system, its prompt prefix): rerender a built dataset with it, print one, build the CLI"),
    "export": ("export", "GGUF q4_k_m/q8_0 + Ollama Modelfile; --dry-run"),
    "fake-service": ("tests.fake_service", "run the scripted fake service (tests)"),
    "fake-model": ("tests.fake_model", "run the scripted fake /v1/messages model (for the real service binary)"),
}
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="souschef-lab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", metavar="command")
    for name, (_, help_text) in COMMANDS.items():
        sp = sub.add_parser(name, help=help_text, add_help=False)
        sp.add_argument("args", nargs=argparse.REMAINDER)
    return p


def resolve(command: str):
    """The module's `main` for a command; ImportError when the module is absent."""
    module_name = COMMANDS[command][0]
    if module_name.startswith("tests."):
        mod = importlib.import_module(module_name)
    else:
        mod = importlib.import_module(f"{__package__ or 'souschef_lab'}.{module_name}")
    fn = getattr(mod, "main", None)
    if fn is None:
        raise ImportError(f"{mod.__name__} has no main(argv)")
    return fn


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0 if argv else EXIT_USAGE
    ns = parser.parse_args(argv[:1])
    rest = argv[1:]
    try:
        fn = resolve(ns.command)
    except ImportError as e:
        print(f"souschef-lab {ns.command}: not available here ({e})", file=sys.stderr)
        return EXIT_USAGE
    rc = fn(rest)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
