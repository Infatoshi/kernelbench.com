"""kb — KernelBench operator CLI.

Drives the kernelbench.com monorepo: run sweeps, publish results, deploy the
site, audit/lint runs, and push run artifacts to HuggingFace. Resolves the repo
root from $KB_REPO_ROOT (set by the bin/kb shim) or by walking up from cwd, so
it works from any directory inside the repo.

The GPU-coupled sweep orchestration stays as bench-local shell
(benchmarks/<bench>/scripts/*.sh); this CLI shells out to it. The cross-bench
post-hoc Python (audit, lint, contamination, traces->HF) lives here.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

USAGE = """\
kb — KernelBench operator CLI   (repo: {root})

  kb sweep <harness> <model> [effort]          full deck sweep, parallel containers, unlimited time
  kb run <harness> <model> <problem> [effort]  one problem (problem = e.g. 05_topk_bitonic)
  kb publish [bench]                            rebuild leaderboard + viewers from archives (default: hard; benches: hard|mega|cuda)
  kb deploy [message]                           publish, commit, push -> Vercel deploys
  kb dev                                        preview site locally (localhost:3000)
  kb build                                      next build
  kb audit <run_id>                             print a run's result + annotation verdict
  kb lint <run_id|--all>                        static reward-hack tripwire (scans solution.py)
  kb contamination <hard|mega|cuda|v3|path> [--published <lb.json>]   cross-run contamination audit
  kb traces-to-hf <out_dir> [run_dirs...]       convert run transcripts to HF agent-trace JSONL
  kb push-runs <hard|mega|cuda> [--board h100|b200|rtx3090] [--dataset R] [--dry-run]   convert published runs' traces and push to HF (per-GPU boards upload under <board>/)
  kb brev <up|sync|bootstrap|run|regrade|pull|down> <instance> [...]   Brev GPU worker lifecycle (scripts/brev_worker.sh)
  kb help

Keys live in ~/.env_vars. Bench a new model: add its key, then  kb sweep <harness> <model>.
Harnesses: claude codex cursor gemini grok opencode | zai-claude minimax-claude kimi-claude kinetic-claude deepseek-claude qwen-claude
"""

# harness -> required env key for preflight
_NEED = {
    "kimi-claude": "KIMI_API_KEY",
    "kinetic-claude": "MOONSHOT_API_KEY",  # kinetic-0715; KIMI_API_KEY 401s on it
    "zai-claude": "ZAI_API_KEY",
    "minimax-claude": "MINIMAX_API_KEY",
    "deepseek-claude": "DEEPSEEK_API_KEY",
    "qwen-claude": "DASHSCOPE_API_KEY",
    "longcat-claude": "LONGCAT_API_KEY",
    "tinker": "THINKING_MACHINES_API_KEY",
    "inkling": "THINKING_MACHINES_API_KEY",
    "hy3": "TENCENT_API_KEY",
    "hy3-claude": "TENCENT_API_KEY",  # legacy alias → TokenHub, not Claude Code
    "gemini": "GEMINI_API_KEY",
    "opencode-nemotron": "OPENROUTER_API_KEY",
}


def repo_root() -> Path:
    env = os.environ.get("KB_REPO_ROOT")
    if env and (Path(env) / "benchmarks").is_dir():
        return Path(env).resolve()
    cur = Path.cwd().resolve()
    for cand in (cur, *cur.parents):
        if (cand / "benchmarks" / "hard").is_dir() and (cand / "package.json").is_file():
            return cand
    # last resort: package lives at <root>/kbtool/kb/cli.py
    guess = Path(__file__).resolve().parents[2]
    if (guess / "benchmarks").is_dir():
        return guess
    sys.exit("kb: cannot locate repo root (set KB_REPO_ROOT or run from inside the repo)")


def _load_env_vars() -> dict[str, str]:
    out: dict[str, str] = {}
    envf = Path(os.path.expanduser("~/.env_vars"))
    if not envf.exists():
        return out
    for ln in envf.read_text().splitlines():
        ln = ln.strip()
        if ln.startswith("export "):
            ln = ln[len("export "):]
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def preflight_key(harness: str, model: str = "") -> None:
    need = _NEED.get(harness, "")
    if harness == "opencode":
        if model.startswith(("openrouter-",)) or any(s in model for s in ("/qwen/", "/xiaomi/")) or model.startswith("moonshotai/"):
            need = "OPENROUTER_API_KEY"
        elif model.startswith("deepseek/"):
            need = "DEEPSEEK_API_KEY"
        elif model.startswith("zai/"):
            need = "ZAI_API_KEY"
    if not need:
        return
    env = {**_load_env_vars(), **{k: v for k, v in os.environ.items() if v}}
    if not env.get(need):
        sys.stderr.write(
            f"STOP: {harness}/{model} needs ${need}, which is not set in ~/.env_vars.\n"
            f"  Add it:  echo 'export {need}=sk-...' >> ~/.env_vars   (then rerun)\n"
        )
        sys.exit(3)


def _require_local_gpu() -> None:
    """Eval sessions need a live CUDA GPU. On GPU-less hosts (the Mac control
    plane) fail closed and point at the Brev worker path instead of silently
    running kernels nowhere."""
    if os.environ.get("KB_ALLOW_LOCAL") == "1":
        return
    if shutil.which("nvidia-smi") is None:
        sys.exit(
            "kb: no local GPU (nvidia-smi not found) — this host cannot run eval sessions.\n"
            "  Launch on a Brev worker instead:  kb brev up <name> && kb brev run <name> ...\n"
            "  (see scripts/brev_worker.sh)\n"
            "  Deliberate local override:  KB_ALLOW_LOCAL=1 kb run ...\n"
        )


def _exec(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> int:
    """Replace this process with cmd (preserves TTY/signals)."""
    if cwd is not None:
        os.chdir(cwd)
    if env is not None:
        os.environ.update(env)
    os.execvp(cmd[0], cmd)


def _bench_dir(root: Path, bench: str) -> Path:
    d = root / "benchmarks" / bench
    if not d.is_dir():
        sys.exit(f"kb: no such bench: {bench}")
    return d


def cmd_sweep(root: Path, args: list[str]) -> int:
    _require_local_gpu()
    harness = args[0] if args else ""
    model = args[1] if len(args) > 1 else ""
    preflight_key(harness, model)
    return _exec([str(_bench_dir(root, "hard") / "scripts" / "sweep_deck.sh"), *args])


def cmd_run(root: Path, args: list[str]) -> int:
    if len(args) < 3:
        sys.exit("usage: kb run <harness> <model> <problem> [effort]")
    _require_local_gpu()
    h, m, p = args[0], args[1], args[2]
    effort = args[3:] if len(args) > 3 else []
    preflight_key(h, m)
    hard = _bench_dir(root, "hard")
    problems_root = os.environ.get("KBH_PROBLEMS_ROOT", "problems-rtxpro6000")
    env = {**os.environ, "KBH_AGENT_CONTAINER": "1"}
    os.chdir(hard)
    os.environ.update(env)
    os.execvp("uv", ["uv", "run", "kbh", "run", h, m, f"{problems_root}/{p}", *effort])


def cmd_publish(root: Path, args: list[str]) -> int:
    push = "--push" in args
    args = [a for a in args if a != "--push"]
    bench = args[0] if args else "hard"
    script = {"hard": "publish_v2.sh", "mega": "publish_mega.sh", "cuda": "publish_v2.sh"}.get(bench)
    if not script:
        sys.exit(f"kb publish: no publish script for bench '{bench}' (hard|mega|cuda)")
    pub = _bench_dir(root, bench) / "scripts" / script
    subprocess.run([str(pub)], check=True)
    rc = _rebuild_model_index(root)
    if rc != 0:
        return rc
    if not push:
        return 0
    # --push: publish, then upload the published runs' traces to HF.
    return cmd_push_runs(root, [bench])


def _rebuild_model_index(root: Path) -> int:
    """Regenerate public/data/models.json (the model-centric site reads it).
    Runs after every bench publish; the builder joins all benches."""
    script = root / "scripts" / "build_model_index.py"
    if not script.exists():
        return 0
    print("kb: rebuilding public/data/models.json")
    return subprocess.run(
        ["uv", "run", "--with", "pyyaml", "python", str(script)],
        cwd=root,
    ).returncode


def _leaderboard_run_ids(bench_dir: Path) -> list[str]:
    lb = bench_dir / "results" / "leaderboard.json"
    if not lb.exists():
        return []
    data = json.loads(lb.read_text())
    rids: set[str] = set()
    for m in data.get("models", []):
        for cell in m.get("results", {}).values():
            rid = cell.get("run_id")
            if rid:
                rids.add(rid)
    return sorted(rids)


def _mega_csv_run_ids(root: Path) -> list[str]:
    """Mega has no results/leaderboard.json; its published set is
    public/data/mega/results.csv (built by build_mega_leaderboard.py)."""
    f = root / "public" / "data" / "mega" / "results.csv"
    if not f.exists():
        return []
    with f.open() as fh:
        return sorted({row["run_id"] for row in csv.DictReader(fh) if row.get("run_id")})


def _check_rid_collisions(root: Path, bench: str, rids: list[str], src_dir: str) -> None:
    """The same rid can live in two runs dirs (two different sessions, one per
    GPU board). Artifact paths are namespaced per board (flat = canonical from
    outputs/runs; per-GPU boards upload under <gpu>/), so pushes are
    deterministic — but each rid must exist in THIS push's source dir, and
    cross-dir duplicates are still worth flagging."""
    bench_out = root / "benchmarks" / bench / "outputs"
    dirs = sorted(d for d in bench_out.glob("runs*") if d.is_dir())
    missing = [r for r in rids if not (bench_out / src_dir / r).is_dir()]
    if missing:
        sys.exit(
            f"kb: {len(missing)} published run_ids missing from {src_dir} "
            f"(first: {missing[0]}) — wrong --board, or archive not synced?"
        )
    for rid in rids:
        hits = [str(d.relative_to(root)) for d in dirs if (d / rid).is_dir()]
        if len(hits) > 1:
            sys.stderr.write(
                f"note: {rid} exists in {len(hits)} runs dirs "
                f"({', '.join(hits)}); pushing the {src_dir} copy\n"
            )


def cmd_push_runs(root: Path, args: list[str]) -> int:
    dry = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]
    dataset = None
    if "--dataset" in args:
        i = args.index("--dataset")
        dataset = args[i + 1]
        del args[i:i + 2]
    board = None  # non-canonical GPU board key (h100|b200|rtx3090)
    if "--board" in args:
        i = args.index("--board")
        board = args[i + 1]
        del args[i:i + 2]
    bench = args[0] if args else "hard"
    if bench not in ("hard", "mega", "cuda"):
        sys.exit("kb push-runs: only hard|mega|cuda have trace datasets (v3 is archived)")
    if board and bench != "hard":
        sys.exit("kb push-runs: --board only applies to hard (per-GPU boards)")
    bench_dir = _bench_dir(root, bench)
    dataset = dataset or f"Infatoshi/kernelbench-{bench}-traces"

    if board:
        # Per-GPU board: rids from leaderboard.<board>.json, archives from
        # outputs/runs-<board>, uploaded under <board>/ on HF so they never
        # clobber the canonical flat namespace.
        lb = bench_dir / "results" / f"leaderboard.{board}.json"
        if not lb.exists():
            sys.exit(f"kb push-runs: no such board file {lb}")
        data = json.loads(lb.read_text())
        rids = sorted({c.get("run_id") for m in data.get("models", [])
                       for c in m.get("results", {}).values() if c.get("run_id")})
        src_dir = f"runs-{board}"
    else:
        rids = _leaderboard_run_ids(bench_dir)
        if not rids and bench == "mega":
            rids = _mega_csv_run_ids(root)
        src_dir = "runs"
    if not rids:
        src = "public/data/mega/results.csv" if bench == "mega" else f"{bench}/results/leaderboard.json"
        sys.exit(f"kb push-runs: no run_ids in {src}")
    _check_rid_collisions(root, bench, rids, src_dir)

    staging = root / "runs" / bench / ("traces" if not board else f"traces-{board}")
    staging.mkdir(parents=True, exist_ok=True)
    listfile = root / "runs" / bench / "_rids.txt"
    listfile.write_text("\n".join(rids) + "\n")

    # Other benches reuse hard's transcript parser/converter machinery if the
    # rsynced local copy is missing (mega/cuda normally carry their own).
    conv = bench_dir / "scripts" / "traces_to_hf.py"
    if not conv.exists():
        conv = _bench_dir(root, "hard") / "scripts" / "traces_to_hf.py"
    print(f"converting {len(rids)} published run traces -> {staging} ...")
    bench_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.run(
        ["uv", "run", "python", str(conv), str(staging),
         "--from-list", str(listfile), "--search", f"outputs/{src_dir}"],
        cwd=bench_dir, check=True, env=bench_env,
    )
    produced = sorted(staging.glob("*.jsonl"))
    print(f"  produced {len(produced)} jsonl traces")

    dest = f"{dataset}" + (f" under {board}/" if board else "")
    if dry:
        print(f"[dry-run] would upload {staging} -> {dest} (dataset)")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(dataset, repo_type="dataset", exist_ok=True)
    print(f"uploading {len(produced)} traces -> {dest} ...")
    api.upload_folder(folder_path=str(staging), repo_id=dataset, repo_type="dataset",
                      path_in_repo=(board or ""),
                      commit_message=f"publish {bench} run traces ({len(produced)} runs)"
                                     + (f" [{board}]" if board else ""))
    print(f"done: https://huggingface.co/datasets/{dataset}")
    return 0


def cmd_deploy(root: Path, args: list[str]) -> int:
    msg = args[0] if args else "publish kernelbench results"
    subprocess.run([str(_bench_dir(root, "hard") / "scripts" / "publish_v2.sh")], check=True)
    subprocess.run([str(_bench_dir(root, "cuda") / "scripts" / "publish_v2.sh")], check=True)
    subprocess.run([str(_bench_dir(root, "mega") / "scripts" / "publish_mega.sh")], check=True)
    rc = _rebuild_model_index(root)
    if rc != 0:
        return rc
    os.chdir(root)
    subprocess.run(["git", "add", "-A", "benchmarks/hard/results", "benchmarks/cuda/results",
                    "benchmarks/mega/results", "public/runs", "public/data", "app"], check=True)
    rc = subprocess.run(
        ["git", "-c", "user.email=elliot@arledge.net", "commit", "-m", msg]
    ).returncode
    if rc != 0:
        print("nothing to commit")
        return 0
    subprocess.run(["git", "push", "origin", "master"], check=True)
    print("pushed; Vercel auto-builds.")
    return 0


def _js_runner() -> str:
    """bun where available (repo tracks bun.lock only); npm as fallback."""
    for cand in ("bun", str(Path.home() / ".bun" / "bin" / "bun")):
        if shutil.which(cand):
            return cand
    return "npm"


def cmd_dev(root: Path, args: list[str]) -> int:
    return _exec([_js_runner(), "run", "dev"], cwd=root)


def cmd_build(root: Path, args: list[str]) -> int:
    return _exec([_js_runner(), "run", "build"], cwd=root)


def cmd_audit(root: Path, args: list[str]) -> int:
    if not args:
        sys.exit("usage: kb audit <run_id>")
    rid = args[0]
    hard = _bench_dir(root, "hard")
    d = hard / "outputs" / "runs" / rid
    if not d.is_dir():
        sys.exit(f"no such run: {rid}")
    r = json.loads((d / "result.json").read_text())
    keys = ("harness", "model", "correct", "peak_fraction", "template_mutated", "failure_reason")
    print({k: r.get(k) for k in keys})
    ann = hard / "results" / "annotations" / f"{rid}.yaml"
    if ann.exists():
        print("--- annotation ---")
        print(ann.read_text())
    return 0


def cmd_lint(root: Path, args: list[str]) -> int:
    script = _bench_dir(root, "hard") / "scripts" / "reward_hack_lint.py"
    return _exec(["python3", str(script), *args])


def cmd_contamination(root: Path, args: list[str]) -> int:
    from kb import contamination
    return contamination.run(args, repo_root=root)


def cmd_brev(root: Path, args: list[str]) -> int:
    """Brev worker lifecycle: kb brev <up|sync|bootstrap|run|regrade|pull|down> ..."""
    return _exec([str(root / "scripts" / "brev_worker.sh"), *args])


def cmd_traces_to_hf(root: Path, args: list[str]) -> int:
    script = _bench_dir(root, "hard") / "scripts" / "traces_to_hf.py"
    os.chdir(_bench_dir(root, "hard"))
    os.execvp("uv", ["uv", "run", "python", str(script), *args])


_COMMANDS = {
    "sweep": cmd_sweep,
    "run": cmd_run,
    "publish": cmd_publish,
    "deploy": cmd_deploy,
    "dev": cmd_dev,
    "build": cmd_build,
    "audit": cmd_audit,
    "lint": cmd_lint,
    "contamination": cmd_contamination,
    "brev": cmd_brev,
    "traces-to-hf": cmd_traces_to_hf,
    "push-runs": cmd_push_runs,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "help"
    rest = argv[1:]
    if cmd in ("help", "-h", "--help"):
        print(USAGE.format(root=repo_root()))
        return 0
    fn = _COMMANDS.get(cmd)
    if fn is None:
        sys.stderr.write(f"unknown command: {cmd}\n\n")
        print(USAGE.format(root=repo_root()))
        return 2
    return fn(repo_root(), rest)


if __name__ == "__main__":
    raise SystemExit(main())
