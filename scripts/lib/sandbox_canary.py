#!/usr/bin/env python3
"""Launch-time sandbox canary — refuse-closed foreign-archive detector.

scripts/lib/sandbox.sh runs this INSIDE the agent bwrap sandbox, before the
model starts. It sweeps the visible tree for anything a live agent must never
see: another run's directory, a stray result.json or solution.py, a pulled
runs-remote-*/runs-lambda-* archive, or the published public/data/*/code tree.
Any hit fails the launch (exit 1). The inverse of the old denylist: instead of
enumerating paths to hide, prove nothing archive-shaped is visible.

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


def probe_cuda_envs(envs: dict[str, str | None]) -> dict[str, list[str] | str]:
    """Run cuda_uuids() in a child per env (CUDA reads the variable at cuInit)."""
    procs = {}
    for label, cvd in envs.items():
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        if cvd is not None:
            env["CUDA_VISIBLE_DEVICES"] = cvd
        procs[label] = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--cuda-probe"],
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
    args = ap.parse_args(argv)

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
