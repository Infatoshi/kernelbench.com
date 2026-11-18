#!/usr/bin/env python3
"""Launch-time sandbox canary — refuse-closed foreign-archive detector.

scripts/lib/sandbox.sh runs this INSIDE the agent bwrap sandbox, before the
model starts. It sweeps the visible tree for anything a live agent must never
see: another run's directory, a stray result.json or solution.py, a pulled
runs-remote-*/runs-lambda-* archive, or the published public/data/*/code tree.
Any hit fails the launch (exit 1). The inverse of the old denylist: instead of
enumerating paths to hide, prove nothing archive-shaped is visible.

Since 2026-11-17 it also sweeps beyond the hide root (--home-sweep: $HOME, the
data mounts, other readable homes) for other checkouts, bench roots, runs*
collections and loose run dirs (find_archives; sandbox.sh calls the same
function on the host with --find-archives to decide what to tmpfs), checks that
every gpu-lock-shimmed binary is the shim (--wrapped), and that /tmp and the home
caches are this run's (--private-dir).

Allowed by name: this run's own directory and the deliberately planted
honeytoken decoy (see sandbox.sh; beacon documented in kbtool/kb/contamination.py).

It also proves session isolation (sandbox.sh passes the flags on every launch):
  * PID namespace: a host sentinel process the harness started for this check
    (--sentinel-pid/--sentinel-token) is neither listed in /proc nor signallable,
    and this process is not in the host's PID namespace (--host-pidns);
  * GPU: NVML enumerates exactly one GPU, the assigned one (--gpu-uuid), and the
    CUDA driver reaches nothing else whether CUDA_VISIBLE_DEVICES is unset, set
    to every index, or set to another card's UUID (--other-gpu-uuid).
The GPU checks use libnvidia-ml/libcuda through ctypes, no torch. On the host,
`--gpu-info INDEX` maps an nvidia-smi (NVML) index to "<uuid> <minor> <count>
<other uuids,comma-separated>" so sandbox.sh can bind exactly /dev/nvidia<minor>.

Stdlib only, no bwrap needed to unit-test: kbtool/tests/test_sandbox_canary.py.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import signal
import subprocess
import sys

RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}_")
REMOTE_ARCHIVE_RE = re.compile(r"^runs-(remote|lambda)-")
# Never descend into these; they are huge and cannot legally hold archives
# (the strict rule that every KernelBench artifact stays in outputs/runs).
PRUNE = {".git", ".venv", "node_modules", "__pycache__", ".cache", ".npm", ".bun"}


# --- foreign archives anywhere the sandbox can reach (2026-11-17) --------------
# The hide root is only this checkout. A stale second checkout (tetra's
# ~/dev/sites/kernelbench.com, a home backup under /mnt) sat outside it with
# hundreds of archives, readable inside the sandbox, and the canary said CLEAN.
# find_archives() is run twice by sandbox.sh: on the host to list what to tmpfs,
# and inside the sandbox (--home-sweep) to prove nothing archive-shaped is left.
# Bounded: no symlinks followed, heavy trees pruned, at most SWEEP_DEPTH levels.
SWEEP_DEPTH = 8
SWEEP_WORKERS = 32      # fork-pool size cap for the sweep
SWEEP_BUDGET = 1500     # directories per work unit before it hands its stack back
SWEEP_PRUNE = PRUNE | {"site-packages", "dist-packages", ".rustup", ".cargo", ".nvm",
                       ".pnpm-store", "proc", "sys", "lost+found"}
SWEEP_EXTRA_ROOTS = ("/data", "/mnt", "/media", "/srv", "/opt", "/scratch", "/workspace")
ARCHIVE_COLLECTION_RE = re.compile(r"^(runs|runs-.+|kb-remote-archives)$")


def sweep_roots(home: str) -> list[str]:
    """$HOME, the usual data mounts, and every other readable home on the box."""
    roots = [home] + [r for r in SWEEP_EXTRA_ROOTS if os.path.isdir(r)]
    homes_parent = os.path.dirname(home.rstrip("/")) or "/"
    try:
        others = sorted(os.listdir(homes_parent))
    except OSError:
        others = []
    for name in others:
        p = os.path.join(homes_parent, name)
        if p != home and os.path.isdir(p) and not os.path.islink(p) and os.access(p, os.R_OK | os.X_OK):
            roots.append(p)
    return roots


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _archive_kind(name: str, subdirs: set[str], entries: set[str]) -> str | None:
    """Why a directory (its name, child dir names, all child names) is foreign
    archive material, or None. Uses only the one scandir the walk already did."""
    if "benchmarks" in subdirs and "kbtool" in subdirs:
        return "KernelBench checkout"
    if "problems" in subdirs and "outputs" in subdirs:
        return "bench root"
    if ARCHIVE_COLLECTION_RE.match(name) and any(RUN_DIR_RE.match(s) for s in subdirs):
        return "run archive collection"
    if RUN_DIR_RE.match(name) and ("result.json" in entries or "solution.py" in entries):
        return "run dir"
    return None


def _walk(job: tuple) -> tuple[list[tuple[str, str]], list[tuple[str, int]]]:
    """Depth-first sweep. job = (starts, skip, max_depth, budget): starts is a
    list of (dir, depth). Visits at most `budget` directories (None: all) and
    returns (hits, leftover) so a pool can rebalance the unvisited rest."""
    starts, skip, max_depth, budget = job
    hits: list[tuple[str, str]] = []
    stack = list(starts)
    visited = 0
    while stack:
        if budget is not None and visited >= budget:
            break
        visited += 1
        path, depth = stack.pop()
        try:
            with os.scandir(path) as it:
                entries = list(it)
        except OSError:
            continue
        # d_type only: no stat per entry, and symlinked dirs are never followed.
        subdirs = [e for e in entries if e.is_dir(follow_symlinks=False)]
        if depth > 0 and not any(_within(s, path) for s in skip):
            kind = _archive_kind(os.path.basename(path), {e.name for e in subdirs},
                                 {e.name for e in entries})
            if kind:
                hits.append((path, kind))
                continue
        if depth >= max_depth:
            continue
        for e in subdirs:
            if e.name in SWEEP_PRUNE or any(_within(e.path, s) for s in skip):
                continue
            stack.append((e.path, depth + 1))
    return hits, stack


def _sweep(starts: list[tuple[str, int]], skip: list[str], max_depth: int) -> list[tuple[str, str]]:
    """All hits under starts. Serially, or (Linux, enough cores) as rounds of
    budgeted work units on a fork pool: a ~400k-directory sweep of tetra's
    $HOME, /mnt and /data took ~14 s on one core, and one wide subtree (a 40k-dir
    cubin cache) defeated a fixed split depth."""
    workers = min(SWEEP_WORKERS, (os.cpu_count() or 1) - 1)
    picklable = getattr(sys.modules.get(_walk.__module__), "_walk", None) is _walk
    if workers < 2 or not picklable or not sys.platform.startswith("linux"):
        return _walk((starts, skip, max_depth, None))[0]
    import multiprocessing
    hits: list[tuple[str, str]] = []
    pending = list(starts)
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        while pending:
            n = min(len(pending), workers * 4)
            jobs = [(pending[i::n], skip, max_depth, SWEEP_BUDGET) for i in range(n)]
            pending = []
            for h, left in pool.imap_unordered(_walk, jobs):
                hits += h
                pending += left
    return hits


def find_archives(roots: list[str], skip: list[str], max_depth: int = SWEEP_DEPTH) -> list[tuple[str, str]]:
    """(path, kind) of every topmost foreign archive tree under roots.

    `skip` holds trees that are handled elsewhere (the hide root, this run).
    A directory that CONTAINS a skipped tree (the live checkout above a smoke
    repo) is never reported itself; the walk descends into it instead, so the
    archives inside it are still reported one by one. Loose run dirs that share
    a parent collapse to that parent when the parent is safe to hide.
    """
    skip = [os.path.abspath(s) for s in skip if s]
    roots_abs = {os.path.abspath(r) for r in roots}
    starts = [(r, 0) for r in dict.fromkeys(os.path.abspath(r) for r in roots)
              if os.path.isdir(r) and not os.path.islink(r) and not any(_within(r, s) for s in skip)]
    hits = sorted(set(_sweep(starts, skip, max_depth)))
    # Collapse sibling loose run dirs to their parent (one mount, not hundreds).
    by_parent: dict[str, list[tuple[str, str]]] = {}
    for path, kind in hits:
        if kind == "run dir":
            by_parent.setdefault(os.path.dirname(path), []).append((path, kind))
    out = [(p, k) for p, k in hits if k != "run dir"]
    for parent, items in by_parent.items():
        unsafe = (parent in roots_abs or parent == "/" or any(_within(s, parent) for s in skip))
        if unsafe:
            out.extend(items)
        else:
            out.append((parent, "run dirs"))
    out.sort()
    # Topmost only.
    top: list[tuple[str, str]] = []
    for p, k in out:
        if not any(_within(p, t) for t, _ in top):
            top.append((p, k))
    return top


def check_private_dir(dst: str, src: str) -> list[str]:
    """dst inside the sandbox must be the very same directory as src (a bind)."""
    try:
        a, b = os.stat(dst), os.stat(src)
    except OSError as e:
        return [f"private dir check failed: {e}"]
    if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
        return [f"{dst} is shared with the host, not this run's {src}"]
    return []


LOCK_SHIM_MARKER = b"kbh-gpu-lock-shim"


def check_wrapped(path: str) -> list[str]:
    """path inside the sandbox must be the gpu-lock shim, not the real binary."""
    try:
        with open(path, "rb") as f:
            head = f.read(256)
    except OSError as e:
        return [f"lock shim missing at {path}: {e}"]
    if not head.startswith(b"#!") or LOCK_SHIM_MARKER not in head:
        return [f"{path} is the real binary inside the sandbox (GPU lock bypassable)"]
    return []


def _is_published_code_dir(dirpath: str, name: str) -> bool:
    parts = dirpath.split(os.sep)
    return name == "code" and "public" in parts and "data" in parts


def scan(
    walk_roots: list[str],
    must_be_hidden: list[str],
    run_id: str,
    allow: list[str],
) -> list[str]:
    """Return a list of human-readable violations; empty means clean."""
    allowed_names = set(allow) | {run_id}
    hits: list[str] = []
    for root in walk_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            keep: list[str] = []
            for name in sorted(dirnames):
                full = os.path.join(dirpath, name)
                if name in allowed_names:
                    continue  # own run dir / honeytoken: skip subtree
                if name in PRUNE:
                    continue
                if REMOTE_ARCHIVE_RE.match(name):
                    hits.append(f"visible pulled-archive dir: {full}")
                    continue
                if RUN_DIR_RE.match(name):
                    hits.append(f"foreign run dir: {full}")
                    continue
                if _is_published_code_dir(dirpath, name):
                    hits.append(f"visible published-code dir: {full}")
                    continue
                keep.append(name)
            dirnames[:] = keep
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                if name == "result.json":
                    hits.append(f"foreign result.json: {full}")
                elif "solution.py" in name:
                    hits.append(f"foreign solution file: {full}")
    for p in must_be_hidden:
        if os.path.isdir(p):
            try:
                entries = os.listdir(p)
            except OSError:
                entries = []
            if entries:
                hits.append(f"must-be-hidden dir visible and non-empty: {p}")
        elif os.path.exists(p):
            hits.append(f"must-be-hidden path visible: {p}")
    return hits


# --- PID namespace -----------------------------------------------------------

def check_pidns(
    sentinel_pid: int,
    token: str,
    host_pidns: str,
    proc: str = "/proc",
    kill=os.kill,
) -> list[str]:
    """Violations if the host's processes are visible or signallable from here."""
    if not token:
        return ["--sentinel-token is required with --sentinel-pid"]
    hits: list[str] = []
    try:
        own_ns = os.readlink(os.path.join(proc, "self", "ns", "pid"))
    except OSError:
        own_ns = ""
    if not own_ns or own_ns == host_pidns:
        hits.append(f"not in a private PID namespace (own {own_ns or '?'}, host {host_pidns})")
    sentinel_seen = False
    for name in os.listdir(proc):
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(proc, name, "cmdline"), "rb") as f:
                cmd = f.read()
        except OSError:
            continue
        # The sentinel runs as argv[0] == token (exec -a). The token also sits in
        # this canary's own argv (and bwrap's), so match argv[0] only.
        if cmd.split(b"\0", 1)[0] == token.encode():
            sentinel_seen = True
            hits.append(f"host sentinel visible as /proc/{name}")
    if not sentinel_seen:
        # A real kill attempt, not kill -0: the sentinel is ours and the host
        # checks it is still alive afterwards.
        try:
            kill(sentinel_pid, signal.SIGKILL)
            hits.append(f"host sentinel pid {sentinel_pid} was signallable from inside")
        except ProcessLookupError:
            pass
        except PermissionError:
            hits.append(f"host pid {sentinel_pid} exists here (EPERM), namespace not private")
    return hits


# --- GPU ---------------------------------------------------------------------

def _uuid_str(raw: bytes) -> str:
    h = raw.hex()
    return f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def nvml_gpus() -> list[tuple[str, int]]:
    """(uuid, device minor) for every GPU NVML can open, in nvidia-smi index order."""
    lib = ctypes.CDLL("libnvidia-ml.so.1")
    if lib.nvmlInit_v2() != 0:
        raise RuntimeError("nvmlInit failed")
    try:
        n = ctypes.c_uint()
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(n)) != 0:
            raise RuntimeError("nvmlDeviceGetCount failed")
        out = []
        for i in range(n.value):
            h = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != 0:
                raise RuntimeError(f"nvmlDeviceGetHandleByIndex({i}) failed")
            buf = ctypes.create_string_buffer(96)
            if lib.nvmlDeviceGetUUID(h, buf, 96) != 0:
                raise RuntimeError(f"nvmlDeviceGetUUID({i}) failed")
            minor = ctypes.c_uint()
            if lib.nvmlDeviceGetMinorNumber(h, ctypes.byref(minor)) != 0:
                raise RuntimeError(f"nvmlDeviceGetMinorNumber({i}) failed")
            out.append((buf.value.decode(), minor.value))
        return out
    finally:
        lib.nvmlShutdown()


def cuda_uuids() -> list[str]:
    """UUIDs the CUDA driver enumerates under this process's environment."""
    lib = ctypes.CDLL("libcuda.so.1")
    rc = lib.cuInit(0)
    if rc == 100:  # CUDA_ERROR_NO_DEVICE
        return []
    if rc != 0:
        raise RuntimeError(f"cuInit failed ({rc})")
    n = ctypes.c_int()
    if lib.cuDeviceGetCount(ctypes.byref(n)) != 0:
        raise RuntimeError("cuDeviceGetCount failed")
    out = []
    for i in range(n.value):
        dev = ctypes.c_int()
        raw = (ctypes.c_ubyte * 16)()
        if lib.cuDeviceGet(ctypes.byref(dev), i) != 0 or lib.cuDeviceGetUuid(raw, dev) != 0:
            raise RuntimeError(f"cuDevice query {i} failed")
        out.append(_uuid_str(bytes(raw)))
    return out


def _cuda_probe_envs(others: list[str], n_host: int) -> dict[str, str | None]:
    """CUDA_VISIBLE_DEVICES values an agent could try (None = unset)."""
    envs: dict[str, str | None] = {"unset": None, "0": "0",
                                   "all-indices": ",".join(str(i) for i in range(max(n_host, 2)))}
    for u in others:
        envs[u] = u
    return envs


def _probe_python() -> str:
    """The interpreter binary itself. Inside the sandbox sys.executable is the
    gpu-lock shim path, and the canary must not queue on the GPU lock."""
    try:
        exe = os.readlink("/proc/self/exe")
        if os.access(exe, os.X_OK):
            return exe
    except OSError:
        pass
    return sys.executable


def probe_cuda_envs(envs: dict[str, str | None]) -> dict[str, list[str] | str]:
    """Run cuda_uuids() in a child per env (CUDA reads the variable at cuInit)."""
    procs = {}
    for label, cvd in envs.items():
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        if cvd is not None:
            env["CUDA_VISIBLE_DEVICES"] = cvd
        procs[label] = subprocess.Popen(
            [_probe_python(), os.path.abspath(__file__), "--cuda-probe"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out: dict[str, list[str] | str] = {}
    for label, p in procs.items():
        text, _ = p.communicate(timeout=120)
        try:
            out[label] = json.loads(text.strip().splitlines()[-1])
        except (ValueError, IndexError):
            out[label] = f"probe failed (exit {p.returncode}): {text.strip()[-300:]}"
    return out


def check_gpu(expected: str, others: list[str], n_host: int,
              nvml=nvml_gpus, cuda=probe_cuda_envs) -> list[str]:
    """Violations unless exactly the assigned GPU is reachable, under any override."""
    hits: list[str] = []
    try:
        seen = [u for u, _ in nvml()]
        if seen != [expected]:
            hits.append(f"NVML sees {seen}, expected only {expected}")
    except (OSError, RuntimeError) as e:
        hits.append(f"NVML failed inside the sandbox: {e}")
    for label, res in cuda(_cuda_probe_envs(others, n_host)).items():
        if isinstance(res, str):
            hits.append(f"CUDA probe CUDA_VISIBLE_DEVICES={label}: {res}")
            continue
        if any(u != expected for u in res):
            hits.append(f"CUDA_VISIBLE_DEVICES={label} reaches another GPU: {res}")
        elif label in ("unset", "0") and res != [expected]:
            hits.append(f"CUDA_VISIBLE_DEVICES={label} does not reach the assigned GPU: {res}")
    return hits


def gpu_info(index: int) -> str:
    gpus = nvml_gpus()
    if not 0 <= index < len(gpus):
        raise RuntimeError(f"GPU index {index} not found (NVML sees {len(gpus)})")
    uuid, minor = gpus[index]
    others = [u for i, (u, _) in enumerate(gpus) if i != index]
    return f"{uuid} {minor} {len(gpus)} {','.join(others) or '-'}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu-info", type=int, help="host mode: print GPU mapping for this index")
    ap.add_argument("--cuda-probe", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--run-id", help="this run's directory basename")
    ap.add_argument("--allow", action="append", default=[],
                    help="additional allowed dir basename (honeytoken decoy)")
    ap.add_argument("--walk", action="append", default=[],
                    help="root to sweep recursively")
    ap.add_argument("--must-be-hidden", action="append", default=[],
                    help="path that must be absent or an empty tmpfs stub")
    ap.add_argument("--sentinel-pid", type=int, help="host PID that must be invisible")
    ap.add_argument("--sentinel-token", help="argv marker of the sentinel")
    ap.add_argument("--host-pidns", default="", help="readlink /proc/self/ns/pid on the host")
    ap.add_argument("--gpu-uuid", help="the only GPU that may be reachable")
    ap.add_argument("--other-gpu-uuid", action="append", default=[],
                    help="another host GPU that must be unreachable")
    ap.add_argument("--host-gpu-count", type=int, default=0)
    ap.add_argument("--find-archives", action="store_true",
                    help="host mode: print foreign archive trees under the sweep roots, one per line")
    ap.add_argument("--home-sweep", metavar="HOME",
                    help="also sweep HOME, the data mounts and other readable homes for foreign archives")
    ap.add_argument("--skip", action="append", default=[],
                    help="tree the home sweep ignores (the hide root, swept by --walk)")
    ap.add_argument("--wrapped", action="append", default=[],
                    help="path that must be the gpu-lock shim inside the sandbox")
    ap.add_argument("--private-dir", action="append", default=[], metavar="DST=SRC",
                    help="DST inside the sandbox must be the same directory as this run's SRC")
    args = ap.parse_args(argv)

    if args.find_archives:
        home = args.home_sweep or os.path.expanduser("~")
        for path, _ in find_archives(sweep_roots(home), args.skip):
            print(path)
        return 0
    if args.cuda_probe:
        print(json.dumps(cuda_uuids()))
        return 0
    if args.gpu_info is not None:
        try:
            print(gpu_info(args.gpu_info))
        except (OSError, RuntimeError) as e:
            print(f"gpu-info: {e}", file=sys.stderr)
            return 1
        return 0
    if not args.run_id:
        ap.error("--run-id is required")

    hits = scan(args.walk, args.must_be_hidden, args.run_id, args.allow)
    if args.home_sweep:
        roots = sweep_roots(args.home_sweep)
        home_hits = [f"foreign {kind} outside the hide root: {path}"
                     for path, kind in find_archives(roots, args.skip)]
        hits += home_hits
        if not home_hits:
            print(f"sandbox canary: no foreign archive under {', '.join(roots)} (depth {SWEEP_DEPTH})")
    lock_hits = [h for p in args.wrapped for h in check_wrapped(p)]
    hits += lock_hits
    if args.wrapped and not lock_hits:
        print(f"sandbox canary: {len(args.wrapped)} interpreter/toolchain paths are gpu-lock shims")
    priv_hits = []
    for spec in args.private_dir:
        dst, _, src = spec.partition("=")
        priv_hits += check_private_dir(dst, src)
    hits += priv_hits
    if args.private_dir and not priv_hits:
        print(f"sandbox canary: per-run {', '.join(s.partition('=')[0] for s in args.private_dir)}")
    if args.sentinel_pid is not None:
        pid_hits = check_pidns(args.sentinel_pid, args.sentinel_token or "", args.host_pidns)
        hits += pid_hits
        if not pid_hits:
            print("sandbox canary: PID namespace private (host sentinel invisible and unsignallable)")
    if args.gpu_uuid:
        gpu_hits = check_gpu(args.gpu_uuid, args.other_gpu_uuid, args.host_gpu_count)
        hits += gpu_hits
        if not gpu_hits:
            print(f"sandbox canary: only {args.gpu_uuid} reachable (NVML + CUDA overrides)")
    if hits:
        print(f"sandbox canary: {len(hits)} violation(s) — REFUSING LAUNCH")
        for h in hits:
            print(f"  {h}")
        return 1
    print("sandbox canary: CLEAN (no foreign archive material visible)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
