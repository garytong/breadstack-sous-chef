"""Turn a trained adapter into a model Ollama can serve.

Copied from the thermometer's export.py and lab/modelfile.py
(breadstack-thermometer/lab/in-house-model; the pattern, never imported), on
the 9B recipe docs/LAB.md §8: FastModel.from_pretrained(adapter,
load_in_16bit=True) merges the LoRA into the base and save_pretrained_gguf
writes one GGUF per quant — q4_k_m (~5.5 GB for a 9B, small loss) and q8_0
(the exam decides which is served) — then a Modelfile per quant beside it
that reuses the base's chat format: TEMPLATE, RENDERER, PARSER and PARAMETER
lines copied verbatim from `ollama show <base tag> --modelfile`, and num_ctx
65536 so the Sous Chef prompt (rules, the manifest index, the editor context)
and the tool results are never silently cut. The student was trained on the
base's format, so it must be served with it: without RENDERER and PARSER a
qwen3.5 student would get the raw prompt and its tool calls would come back
as plain text.

Fallback (--method merged) when Unsloth's GGUF writer fails: the merged
16-bit model, converted by llama.cpp's own convert_hf_to_gguf.py and
llama-quantize (--llama-cpp <checkout>), or those two commands printed to run
by hand.

  python -m souschef_lab.export --adapter runs/adapters/sous-chef-9b-v1/adapter --name sous-chef-9b-v1 \\
      --base-ollama-tag qwen3.5:9b --dry-run
  python -m souschef_lab.export --adapter … --name sous-chef-9b-v1 --base-ollama-tag qwen3.5:9b --create

--dry-run prints the plan and every command without importing torch
(tests/test_train_dry.py); --base-modelfile <file> stands in for `ollama
show` where no Ollama runs (a dry run, the tests). --create registers each
quant as <name>:<quant>; a Cloud Run GPU image then serves it on /v1/messages
(deferred — docs/LAB.md §10).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import trainkit
from .trainkit import die, say, utc_now

RECIPE = {"quants": "q4_k_m,q8_0", "num_ctx": 65536, "method": "gguf", "base_ollama_tag": "qwen3.5:9b"}
CONFIG_KEYS = ("adapter", "name", "base_ollama_tag", "quants", "num_ctx", "out", "method", "llama_cpp", "ollama",
               "base_modelfile", "create")


# --- The Modelfile (thermometer lab/modelfile.py) -----------------------------------------------------------------------
# Instructions copied from the base: how to format, how to read the reply
# back and when to stop. FROM/ADAPTER point at the base's own files, SYSTEM
# would compete with the service's system prompt, LICENSE and MESSAGE are not
# needed. RENDERER and PARSER need an Ollama client that knows them (0.5.7's
# `ollama create` refuses the Modelfile); check_client looks first.

KEEP = ("TEMPLATE", "RENDERER", "PARSER", "PARAMETER")


@dataclass(frozen=True)
class Instruction:
    keyword: str  # upper-case: FROM, TEMPLATE, PARAMETER, …
    raw: str  # the whole instruction as written, quotes and newlines included

    @property
    def parameter(self) -> str:
        """For PARAMETER lines, the parameter's name ("stop", "num_ctx")."""
        parts = self.raw.split(None, 2)
        return parts[1] if self.keyword == "PARAMETER" and len(parts) > 1 else ""


def parse_modelfile(text: str) -> list[Instruction]:
    """Instructions in order. A value opened with triple quotes runs to the
    closing triple quotes, across lines (how TEMPLATE and LICENSE are
    written); comments and blank lines between instructions are skipped."""
    out: list[Instruction] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        i += 1
        if not stripped or stripped.startswith("#"):
            continue
        keyword, _, rest = stripped.partition(" ")
        raw = [line.rstrip("\n")]
        rest = rest.lstrip()
        if rest.startswith('"""') and rest.count('"""') == 1:
            while i < len(lines):
                raw.append(lines[i])
                i += 1
                if '"""' in raw[-1]:
                    break
        out.append(Instruction(keyword=keyword.upper(), raw="\n".join(raw).strip()))
    return out


def student_modelfile(base_text: str, gguf_name: str, base_tag: str, num_ctx: int | None) -> str:
    """The student's Modelfile: FROM its GGUF, the base's TEMPLATE, RENDERER,
    PARSER and PARAMETERs, and num_ctx set so the long system prompt and the
    tool results are not silently cut (Ollama's default context is small)."""
    keep = [ins for ins in parse_modelfile(base_text) if ins.keyword in KEEP]
    if not any(ins.keyword == "TEMPLATE" for ins in keep):
        raise ValueError(f"`ollama show {base_tag} --modelfile` has no TEMPLATE — is that the right base tag?")
    lines = [
        f"# The Sous Chef lab's student: LoRA-tuned from {base_tag}'s base model.",
        f"# TEMPLATE, RENDERER, PARSER and PARAMETER lines are copied from `ollama show {base_tag} --modelfile`.",
        f"FROM ./{gguf_name}",
    ]
    for ins in keep:
        if num_ctx and ins.parameter == "num_ctx":
            continue
        lines.append(ins.raw)
    if num_ctx:
        lines.append(f"PARAMETER num_ctx {num_ctx}")
    return "\n".join(lines) + "\n"


# --- The Ollama client -------------------------------------------------------------------------------------------------------


def version_key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def client_too_old(version_output: str) -> tuple[str, str] | None:
    """(client, server) versions when `ollama --version` says the client is
    older than the server it reaches; None when they match or the client is
    the newer one."""
    server = re.search(r"ollama version is (\S+)", version_output)
    client = re.search(r"client version is (\S+)", version_output)
    if server and client and version_key(client.group(1)) < version_key(server.group(1)):
        return client.group(1), server.group(1)
    return None


def ollama_modelfile(ollama: str, tag: str) -> str:
    if shutil.which(ollama) is None:
        die(f"the ollama client `{ollama}` was not found — pass --ollama <path>, or --base-modelfile <file> for a dry run")
    r = subprocess.run([ollama, "show", tag, "--modelfile"], capture_output=True, text=True)
    if r.returncode != 0:
        die(f"`ollama show {tag} --modelfile` failed: {r.stderr.strip() or r.stdout.strip()}\n"
            f"Pull the base once so its chat template is available: ollama pull {tag}")
    return r.stdout


def check_client(ollama: str, base_text: str) -> None:
    """Stop before minutes of merging when `ollama create` could not read the
    student's Modelfile: a base that formats with RENDERER/PARSER (qwen3.5)
    needs a client as new as the server."""
    r = subprocess.run([ollama, "--version"], capture_output=True, text=True)
    old = client_too_old(r.stdout + r.stderr)
    if not old:
        return
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434 (the default)")
    msg = f"the ollama client `{ollama}` is {old[0]} but the server at OLLAMA_HOST={host} is {old[1]}"
    if re.search(r"(?m)^(RENDERER|PARSER) ", base_text):
        die(msg + " — this base's Modelfile uses RENDERER/PARSER, which that client refuses; pass "
            "--ollama <a client as new as the server>")
    say(f"note: {msg}")


# --- Files -------------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def writable_copies(out_dir: Path):
    """Files copied into out_dir come out owner-writable while this is open:
    Unsloth's merge copies the base's shards out of the Hugging Face cache
    with shutil.copy2 and rewrites them in place; the hf-xet cache stores its
    blobs read-only (0444) and copy2 carries that mode over."""
    import stat

    root = Path(out_dir).resolve()
    original = shutil.copy2

    def copy2(src, dst, *a, **kw):
        got = original(src, dst, *a, **kw)
        target = Path(got)
        try:
            inside = target.resolve().is_relative_to(root)
        except OSError:
            inside = False
        if inside and target.is_file():
            target.chmod(stat.S_IMODE(target.stat().st_mode) | stat.S_IWUSR)
        return got

    shutil.copy2 = copy2
    try:
        yield
    finally:
        shutil.copy2 = original


def find_gguf(out_dir: Path, quant: str) -> Path | None:
    """Unsloth's writer has used both <dir>/ and <dir>_gguf/ across versions,
    and a multimodal base also gets an mmproj file: find the text model."""
    out_dir = Path(out_dir)
    found = [p for d in (out_dir, out_dir.parent / f"{out_dir.name}_gguf") if d.is_dir()
             for p in d.rglob("*.gguf") if "mmproj" not in p.name.lower()]
    if not found:
        return None
    exact = [p for p in found if quant.lower() in p.name.lower()]
    return max(exact or found, key=lambda p: p.stat().st_size)


def parse_quants(text: str) -> list[str]:
    quants = [q.strip().lower() for q in (text or "").split(",") if q.strip()]
    if not quants:
        raise ValueError("--quants: at least one quant (e.g. q4_k_m,q8_0)")
    return quants


def adapter_base(adapter: Path) -> str | None:
    """The base the adapter was trained on, from its adapter_config.json."""
    cfg = trainkit.read_json(Path(adapter) / "adapter_config.json", {}) or {}
    return cfg.get("base_model_name_or_path")


def adapter_run(adapter: Path) -> dict:
    """The run.json beside the adapter folder ({} when there is none)."""
    return trainkit.read_json(Path(adapter).parent / "run.json", {}) or {}


def llama_cpp_commands(merged: Path, out: Path, name: str, quants: list[str], llama_cpp: Path | None) -> list[list[str]]:
    """The fallback's commands: convert_hf_to_gguf.py to an f16 GGUF, then
    llama-quantize to each quant (run from the llama.cpp checkout)."""
    root = Path(llama_cpp) if llama_cpp else Path("<llama.cpp checkout>")
    f16 = out / f"{name}-f16.gguf"
    cmds = [["python3", str(root / "convert_hf_to_gguf.py"), str(merged), "--outfile", str(f16), "--outtype", "f16"]]
    for q in quants:
        cmds.append([str(root / "build" / "bin" / "llama-quantize"), str(f16), str(out / f"{name}-{q}.gguf"), q.upper()])
    return cmds


# --- The plan ------------------------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="souschef_lab.export", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="settings file (.yaml/.yml/.json) whose keys are these flags; a flag given here wins")
    p.add_argument("--adapter", type=Path, help="the adapter folder train.py or dpo.py wrote (…/adapter)")
    p.add_argument("--name", help="the Ollama model name to create (tags <name>:<quant>)")
    p.add_argument("--base-ollama-tag", default=RECIPE["base_ollama_tag"],
                   help="the base model's Ollama tag, for its chat format (default: %(default)s; confirm with `ollama list`)")
    p.add_argument("--quants", default=RECIPE["quants"], help="GGUF quants, comma-separated (default: %(default)s)")
    p.add_argument("--num-ctx", type=int, default=RECIPE["num_ctx"],
                   help="context window Ollama gives the model (default: %(default)s)")
    p.add_argument("--out", type=Path, help="output folder (default: runs/exports/<name>)")
    p.add_argument("--method", choices=["gguf", "merged"], default=RECIPE["method"],
                   help="gguf: Unsloth writes the GGUFs (default); merged: 16-bit merge, then llama.cpp converts")
    p.add_argument("--llama-cpp", type=Path, help="with --method merged: a llama.cpp checkout to convert with")
    p.add_argument("--ollama", default=os.environ.get("LAB_OLLAMA_BIN", "ollama"),
                   help="the ollama client to run (it reaches OLLAMA_HOST); must be as new as that server "
                        "(default: $LAB_OLLAMA_BIN or `ollama` on PATH)")
    p.add_argument("--base-modelfile", type=Path,
                   help="a saved `ollama show <base> --modelfile` to read instead of running ollama (dry runs, tests)")
    p.add_argument("--create", action="store_true", help="run `ollama create <name>:<quant> -f Modelfile.<quant>` at the end")
    p.add_argument("--dry-run", "--check", dest="dry_run", action="store_true",
                   help="print the plan and every command, then stop (no GPU, no torch, no ollama)")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_parser()
    argv = trainkit.apply_config(p, sys.argv[1:] if argv is None else argv, CONFIG_KEYS)
    args = p.parse_args(argv)
    if args.adapter is None or not args.name:
        p.error("--adapter and --name are required")
    args.adapter = Path(args.adapter)
    args.ollama = os.path.expanduser(str(args.ollama))
    try:
        args.quant_list = parse_quants(args.quants)
    except ValueError as e:
        p.error(str(e))
    if args.num_ctx < 1024:
        p.error("--num-ctx: 1024 or more")
    return args


def plan(args, base_text: str | None = None) -> dict:
    out = Path(args.out) if args.out else trainkit.EXPORTS_DIR / args.name
    run = adapter_run(args.adapter)
    base_hf = adapter_base(args.adapter)
    merged = out / "merged-16bit"
    ggufs = {q: str(out / f"{args.name}-{q}.gguf") for q in args.quant_list}
    d = {
        "kind": "export", "name": args.name, "adapter": str(args.adapter), "adapter_run": run.get("name"),
        "adapter_kind": run.get("kind"), "base": base_hf, "base_ollama_tag": args.base_ollama_tag,
        "output": str(out), "method": args.method, "quants": args.quant_list, "num_ctx": args.num_ctx,
        "config": getattr(args, "config", None), "ggufs": ggufs,
        "modelfiles": {q: str(out / f"Modelfile.{q}") for q in args.quant_list},
        "tags": [f"{args.name}:{q}" for q in args.quant_list], "create": bool(args.create),
        "modelfile_source": str(args.base_modelfile) if args.base_modelfile else f"ollama show {args.base_ollama_tag} --modelfile",
    }
    steps = [f"ollama show {args.base_ollama_tag} --modelfile  (the base's TEMPLATE/RENDERER/PARSER/PARAMETER lines)"
             if not args.base_modelfile else f"read {args.base_modelfile}",
             f"FastModel.from_pretrained({args.adapter}, load_in_16bit=True)  (merges the LoRA into {base_hf or '<base>'})"]
    if args.method == "gguf":
        steps += [f"model.save_pretrained_gguf({out / 'gguf'}, tokenizer, quantization_method={q!r})" for q in args.quant_list]
    else:
        steps.append(f"model.save_pretrained_merged({merged}, tokenizer, save_method='merged_16bit')")
        steps += [" ".join(c) for c in llama_cpp_commands(merged, out, args.name, args.quant_list, args.llama_cpp)]
    steps += [f"write Modelfile.{q} (FROM ./{Path(g).name}, the base's lines, PARAMETER num_ctx {args.num_ctx})"
              for q, g in ggufs.items()]
    if args.create:
        steps += [f"{args.ollama} create {args.name}:{q} -f Modelfile.{q}" for q in args.quant_list]
    d["commands"] = steps
    if base_text is not None:
        try:
            d["modelfile_preview"] = student_modelfile(base_text, Path(ggufs[args.quant_list[0]]).name,
                                                       args.base_ollama_tag, args.num_ctx)
        except ValueError as e:
            d["modelfile_preview"] = f"error: {e}"
    return d


PLAN_ORDER = ("kind", "name", "adapter", "adapter_run", "adapter_kind", "base", "base_ollama_tag", "output", "method",
              "quants", "num_ctx", "config", "ggufs", "modelfiles", "tags", "create", "modelfile_source")


def render_plan(p: dict) -> str:
    text = trainkit.render_plan("Export plan (the thermometer recipe, docs/LAB.md §8)",
                                {k: v for k, v in p.items() if k not in ("commands", "modelfile_preview")}, PLAN_ORDER)
    text += "\nCommands:\n" + "\n".join(f"  $ {c}" for c in p["commands"])
    if p.get("modelfile_preview"):
        text += f"\nModelfile.{p['quants'][0]}:\n" + "\n".join(f"  | {x}" for x in p["modelfile_preview"].splitlines())
    return text


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = args.adapter / "adapter_config.json"
    if not cfg.exists():
        if args.dry_run:
            say(f"note: {args.adapter} has no adapter_config.json yet (train.py writes it); the export would stop here.")
        else:
            die(f"{args.adapter} has no adapter_config.json — pass the adapter folder train.py printed")
    base_text = None
    if args.base_modelfile:
        base_text = Path(args.base_modelfile).read_text(encoding="utf-8")
    elif not args.dry_run:
        base_text = ollama_modelfile(args.ollama, args.base_ollama_tag)  # fail fast, before minutes of merging
        check_client(args.ollama, base_text)
    p = plan(args, base_text)
    say(render_plan(p))
    if args.dry_run:
        if importlib.util.find_spec("unsloth") is None and args.method == "gguf":
            say("The [train] extra is not installed: pip install -e 'lab[train]' before exporting.")
        say("Dry run: nothing written. Drop --dry-run to export.")
        return 0
    if importlib.util.find_spec("unsloth") is None:
        die("the [train] extra is not installed — pip install -e 'lab[train]' first")
    return export(args, p, base_text)


def export(args, p: dict, base_text: str) -> int:
    out = Path(p["output"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    args.adapter = args.adapter.resolve()

    def load_merged_source():
        import unsloth  # noqa: F401  (before transformers, as in train.py)
        from unsloth import FastModel

        say(f"Loading {p['base']} with the adapter from {trainkit.rel(args.adapter)} …")
        model, processor = FastModel.from_pretrained(model_name=str(args.adapter), max_seq_length=args.num_ctx,
                                                     load_in_4bit=False, load_in_16bit=True)
        return model, getattr(processor, "tokenizer", processor)

    ggufs: dict[str, Path] = {}
    if args.method == "gguf":
        model, tok = load_merged_source()
        for q in args.quant_list:
            say(f"Writing a {q} GGUF (merges the adapter first; takes several minutes) …")
            # llama.cpp writes the 16-bit intermediate GGUF to the working
            # directory before Unsloth moves it: run from the export folder.
            with writable_copies(out), contextlib.chdir(out):
                model.save_pretrained_gguf(str(out / "gguf"), tok, quantization_method=q)
            got = find_gguf(out / "gguf", q)
            if got is None:
                die(f"no .gguf file appeared under {out / 'gguf'} for {q} — try --method merged")
            target = out / f"{args.name}-{q}.gguf"
            if got.resolve() != target.resolve():
                shutil.move(str(got), str(target))
            ggufs[q] = target
    else:
        merged = out / "merged-16bit"
        if (merged / "config.json").exists():
            say(f"Reusing the merged model in {trainkit.rel(merged)}")
        else:
            model, tok = load_merged_source()
            say(f"Writing the merged 16-bit model to {trainkit.rel(merged)} …")
            with writable_copies(out), contextlib.chdir(out):
                model.save_pretrained_merged(str(merged), tok, save_method="merged_16bit")
        cmds = llama_cpp_commands(merged.resolve(), out.resolve(), args.name, args.quant_list, args.llama_cpp)
        if not args.llama_cpp:
            say("Now convert it with llama.cpp (from your llama.cpp checkout):\n  " + "\n  ".join(" ".join(c) for c in cmds)
                + "\nthen re-run this command with --llama-cpp <checkout> added (the merge is reused).")
            return 0
        for cmd in cmds:
            target = Path(cmd[-2]) if cmd[0].endswith("llama-quantize") else Path(cmd[cmd.index("--outfile") + 1])
            if target.exists():
                continue
            say("$ " + " ".join(cmd))
            if subprocess.run(cmd, cwd=args.llama_cpp).returncode != 0:
                die(f"{Path(cmd[1] if cmd[0] == 'python3' else cmd[0]).name} failed")
        ggufs = {q: out / f"{args.name}-{q}.gguf" for q in args.quant_list}

    record = {"at": utc_now(), **{k: p[k] for k in ("adapter", "adapter_run", "adapter_kind", "base", "base_ollama_tag",
                                                    "method", "num_ctx")}, "quants": {}}
    for q, g in ggufs.items():
        mf = out / f"Modelfile.{q}"
        try:
            mf.write_text(student_modelfile(base_text, g.name, args.base_ollama_tag, args.num_ctx), encoding="utf-8")
        except ValueError as e:
            die(str(e))
        record["quants"][q] = {"gguf": g.name, "gb": round(g.stat().st_size / 2**30, 1),
                               "sha256": trainkit.file_sha256(g), "modelfile": mf.name, "tag": f"{args.name}:{q}"}
        say(f"GGUF {q}: {trainkit.rel(g)} ({record['quants'][q]['gb']} GB) · Modelfile: {trainkit.rel(mf)}")
    record["minutes"] = round((time.monotonic() - t0) / 60, 1)
    (out / "export.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    if args.create:
        for q in ggufs:
            tag = f"{args.name}:{q}"
            if subprocess.run([args.ollama, "create", tag, "-f", f"Modelfile.{q}"], cwd=out).returncode != 0:
                die(f"`ollama create {tag}` failed (see above)")
            say(f"Created Ollama model {tag}")
        say("Next: point a `messages-compatible` MODEL_ROUTES entry at the Ollama serving it and run the exam "
            "(python -m souschef_lab.eval --route <name>).")
    else:
        for q in ggufs:
            say(f"To register it: (cd {out} && {args.ollama} create {args.name}:{q} -f Modelfile.{q})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
