#!/usr/bin/env bash
# Shared helpers for freezing and replaying agent-authored submissions.
#
# Callers must capture the digest printed by submission_bundle_create and pass
# it back to every verify/extract operation.  Verifying without that captured
# digest only proves that a possibly replaced bundle is internally consistent.

# Resolved before any submission code runs.  Callers deliberately keep these
# paths in shell variables rather than looking tools up through a candidate-
# writable PATH at replay time.
SUBMISSION_ISOLATION_TOOLS_RESOLVED=0
SUBMISSION_ISOLATION_AVAILABLE=0
SUBMISSION_NETWORK_MODE="unavailable"
SUBMISSION_NETWORK_PREFIX=()
SUBMISSION_CLEAN_ENV_PREFIX=()
SUBMISSION_ISOLATED_COMMAND=()
SUBMISSION_ISOLATION_TRUST_ROOT=""
SUBMISSION_ISOLATION_RUN_ROOT=""
SUBMISSION_ISOLATION_EXTRA_READONLY=()
SUBMISSION_BUNDLE_TOOL=""
SUBMISSION_BUNDLE_TOOL_IDENTITY=""
SUBMISSION_TRUSTED_STAGE_TOOL=""
SUBMISSION_TRUSTED_STAGE_TOOL_IDENTITY=""
SUBMISSION_MOUNT_ATTR_PYTHON=""
SUBMISSION_ISOLATION_PYTHON_ROOT=""
IFS= read -r -d '' SUBMISSION_MOUNT_SETATTR_CODE <<'PY' || true
import ctypes
import os
import sys

AT_FDCWD = -100
AT_RECURSIVE = 0x8000
MOUNT_ATTR_RDONLY = 0x00000001
SYS_MOUNT_SETATTR = 442


class MountAttr(ctypes.Structure):
    _fields_ = [
        ("attr_set", ctypes.c_uint64),
        ("attr_clr", ctypes.c_uint64),
        ("propagation", ctypes.c_uint64),
        ("userns_fd", ctypes.c_uint64),
    ]


libc = ctypes.CDLL(None, use_errno=True)
attributes = MountAttr(MOUNT_ATTR_RDONLY, 0, 0, 0)
result = libc.syscall(
    SYS_MOUNT_SETATTR,
    AT_FDCWD,
    os.fsencode(sys.argv[1]),
    AT_RECURSIVE if len(sys.argv) < 3 or sys.argv[2] == "recursive" else 0,
    ctypes.byref(attributes),
    ctypes.sizeof(attributes),
)
if result != 0:
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error), sys.argv[1])
PY

submission_bundle_tool() {
    local helper_dir candidate
    if [ -n "$SUBMISSION_BUNDLE_TOOL" ]; then
        [ -n "$SUBMISSION_BUNDLE_TOOL_IDENTITY" ] \
            && submission_executable_matches \
                "$SUBMISSION_BUNDLE_TOOL" "$SUBMISSION_BUNDLE_TOOL_IDENTITY" \
            || return 1
        printf '%s\n' "$SUBMISSION_BUNDLE_TOOL"
        return 0
    fi
    helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for candidate in \
        "$helper_dir/../submission_bundle.py" \
        "${KB_BENCH_DIR:-}/../../scripts/submission_bundle.py" \
        "${KB_BENCH_DIR:-}/scripts/submission_bundle.py"; do
        if [ -n "$candidate" ] && [ -f "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    echo "submission_bundle.py not found (monorepo or thin-worker layout)" >&2
    return 1
}

# Bind the helper script while the caller is still trusted.  Bundle operations
# happen after host-mode agents return, so resolving this Python source lazily
# would let an agent replace the verifier that is meant to freeze its output.
submission_bind_bundle_tool() {
    local tool identity
    tool="$(submission_bundle_tool)" || return 1
    identity="$(submission_executable_identity "$tool")" || return 1
    SUBMISSION_BUNDLE_TOOL="$tool"
    SUBMISSION_BUNDLE_TOOL_IDENTITY="$identity"
}

submission_trusted_stage_tool() {
    local helper_dir candidate
    if [ -n "$SUBMISSION_TRUSTED_STAGE_TOOL" ]; then
        [ -n "$SUBMISSION_TRUSTED_STAGE_TOOL_IDENTITY" ] \
            && submission_executable_matches \
                "$SUBMISSION_TRUSTED_STAGE_TOOL" \
                "$SUBMISSION_TRUSTED_STAGE_TOOL_IDENTITY" \
            || return 1
        printf '%s\n' "$SUBMISSION_TRUSTED_STAGE_TOOL"
        return 0
    fi
    helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for candidate in \
        "$helper_dir/../trusted_stage.py" \
        "${KB_BENCH_DIR:-}/../../scripts/trusted_stage.py" \
        "${KB_BENCH_DIR:-}/scripts/trusted_stage.py"; do
        if [ -n "$candidate" ] && [ -f "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    echo "trusted_stage.py not found (monorepo or thin-worker layout)" >&2
    return 1
}

submission_bind_trusted_stage_tool() {
    local tool identity
    tool="$(submission_trusted_stage_tool)" || return 1
    identity="$(submission_executable_identity "$tool")" || return 1
    SUBMISSION_TRUSTED_STAGE_TOOL="$tool"
    SUBMISSION_TRUSTED_STAGE_TOOL_IDENTITY="$identity"
    SUBMISSION_ISOLATION_EXTRA_READONLY+=("$tool")
}

submission_bundle_create() {
    local source_dir="$1" bundle_dir="$2" tool
    tool="$(submission_bundle_tool)" || return 1
    "${SUBMISSION_BUNDLE_PYTHON:-python3}" "$tool" create "$source_dir" "$bundle_dir"
}

submission_bundle_verify() {
    local bundle_dir="$1" expected_digest="$2" tool
    tool="$(submission_bundle_tool)" || return 1
    "${SUBMISSION_BUNDLE_PYTHON:-python3}" "$tool" verify "$bundle_dir" \
        --expected-sha256 "$expected_digest"
}

submission_bundle_extract() {
    local bundle_dir="$1" expected_digest="$2" destination="$3" tool
    tool="$(submission_bundle_tool)" || return 1
    "${SUBMISSION_BUNDLE_PYTHON:-python3}" "$tool" extract "$bundle_dir" \
        "$destination" --expected-sha256 "$expected_digest"
}

submission_json_field() {
    local field="$1"
    "${SUBMISSION_BUNDLE_PYTHON:-python3}" -c \
        'import json,sys; value=json.load(sys.stdin)[sys.argv[1]]; print(value)' "$field"
}

# Capture both the path entry and opened target identity of an executable.  The
# shared runner records this before launching a host-mode agent and compares it
# again before provisioning either replay stage, preventing replacement of a
# user-owned uv binary between resolution and use.
submission_executable_identity() {
    local executable="$1"
    "${SUBMISSION_BUNDLE_PYTHON:-python3}" - "$executable" <<'PY'
import hashlib
import json
import os
import stat
import sys

path = os.path.abspath(sys.argv[1])
link_stat = os.lstat(path)
if not (stat.S_ISREG(link_stat.st_mode) or stat.S_ISLNK(link_stat.st_mode)):
    raise SystemExit("executable path is not a file or symlink")
resolved = os.path.realpath(path)
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
fd = os.open(resolved, flags)
try:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit("executable target is not a regular file")
    digest = hashlib.sha256()
    while chunk := os.read(fd, 1024 * 1024):
        digest.update(chunk)
    after = os.fstat(fd)
finally:
    os.close(fd)

fields = lambda value: [
    value.st_dev,
    value.st_ino,
    value.st_mode,
    value.st_uid,
    value.st_gid,
    value.st_size,
    value.st_mtime_ns,
    value.st_ctime_ns,
]
if fields(before) != fields(after):
    raise SystemExit("executable changed while being hashed")
print(json.dumps(
    {
        "path": path,
        "path_stat": fields(link_stat),
        "resolved": resolved,
        "target_stat": fields(before),
        "sha256": digest.hexdigest(),
    },
    separators=(",", ":"),
    sort_keys=True,
))
PY
}

submission_executable_matches() {
    local executable="$1" expected="$2" actual
    actual="$(submission_executable_identity "$executable" 2>/dev/null)" || return 1
    [ "$actual" = "$expected" ]
}

submission_add_isolation_readonly_executable() {
    local executable resolved
    for executable in "$@"; do
        resolved="$("${SUBMISSION_BUNDLE_PYTHON:-python3}" -c \
            'import os,sys; print(os.path.realpath(os.path.abspath(sys.argv[1])))' \
            "$executable")" || return 1
        [ -f "$resolved" ] && [ -x "$resolved" ] || return 1
        SUBMISSION_ISOLATION_EXTRA_READONLY+=("$resolved")
    done
}

# Accept exactly one complete metric line from a successful trusted benchmark.
# Agent code shares stdout with the benchmark, so taking the first substring
# lets a submission spoof the score before the grader prints its own result.
submission_extract_peak_fraction() {
    local log_path="$1"
    "${SUBMISSION_BUNDLE_PYTHON:-python3}" - "$log_path" <<'PY'
import math
import os
import re
import sys

pattern = re.compile(r"^peak_fraction: ([0-9]+(?:\.[0-9]+)?)$")
if os.path.getsize(sys.argv[1]) > 128 * 1024 * 1024:
    raise SystemExit(1)
values = []
with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as stream:
    for line in stream:
        match = pattern.fullmatch(line.rstrip("\r\n"))
        if match is not None:
            values.append(float(match.group(1)))
            if len(values) > 1:
                raise SystemExit(1)
if (
    len(values) != 1
    or not math.isfinite(values[0])
    or not 0 <= values[0] <= 100
):
    raise SystemExit(1)
print(values[0])
PY
}

# Correctness shares stdout with imported submission code.  Accept only the
# grader's complete standalone marker and reject duplicate/spoofed markers.
submission_check_passed() {
    local log_path="$1"
    "${SUBMISSION_BUNDLE_PYTHON:-python3}" - "$log_path" <<'PY'
import os
import sys

if os.path.getsize(sys.argv[1]) > 128 * 1024 * 1024:
    raise SystemExit(1)
markers = 0
with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as stream:
    for line in stream:
        if line.rstrip("\r\n") == "PASS":
            markers += 1
            if markers > 1:
                raise SystemExit(1)
raise SystemExit(0 if markers == 1 else 1)
PY
}

# Hash the trusted replay surface that untrusted solution imports can reach.
# Python bytecode and compiler caches live outside this surface by construction,
# so any change here is a grader/project mutation and invalidates the replay.
submission_trusted_surface_digest() {
    local workspace="$1" problem="$2"
    shift 2
    "${SUBMISSION_BUNDLE_PYTHON:-python3}" - "$workspace" "$problem" "$@" <<'PY'
import hashlib
import os
import stat
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
problem = sys.argv[2]
templates = sys.argv[3:]
paths = [workspace / "pyproject.toml", workspace / "uv.lock"]
python_version = workspace / ".python-version"
if python_version.exists() or python_version.is_symlink():
    paths.append(python_version)
src = workspace / "src"
paths.extend(sorted(src.rglob("*"), key=lambda path: path.relative_to(workspace).as_posix()))
paths.extend(workspace / "problems" / problem / name for name in templates)

digest = hashlib.sha256()
for path in paths:
    relative = path.relative_to(workspace).as_posix()
    metadata = path.lstat()
    if stat.S_ISDIR(metadata.st_mode):
        kind = b"d"
        contents = b""
    elif stat.S_ISREG(metadata.st_mode):
        kind = b"f"
        contents = path.read_bytes()
    else:
        raise SystemExit(f"unsafe trusted replay entry: {relative}")
    digest.update(kind)
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(oct(stat.S_IMODE(metadata.st_mode)).encode("ascii"))
    digest.update(b"\0")
    digest.update(hashlib.sha256(contents).digest())
print(digest.hexdigest())
PY
}

_submission_resolve_tool() {
    local name="$1" candidate directory base
    for candidate in \
        "/usr/bin/$name" "/usr/sbin/$name" "/bin/$name" "/sbin/$name"; do
        [ -f "$candidate" ] && [ -x "$candidate" ] || continue
        directory="$(cd -P -- "${candidate%/*}" && pwd -P)" || return 1
        base="${candidate##*/}"
        printf '%s/%s\n' "$directory" "$base"
        return 0
    done
    return 1
}

_submission_real_directory() {
    [ -d "$1" ] || return 1
    (cd -P -- "$1" && pwd -P)
}

# User-managed Python installations are mounted as a complete read-only tree so
# their standard library and shared objects remain available after pivot_root.
# Reject sockets/devices/FIFOs up front: otherwise an allowlisted runtime tree
# could smuggle a host IPC endpoint into the private root.
submission_validate_isolation_tree() {
    local root="$1"
    "${SUBMISSION_BUNDLE_PYTHON:-python3}" - "$root" <<'PY'
import os
import stat
import sys

root = os.path.realpath(os.path.abspath(sys.argv[1]))
if not os.path.isdir(root) or root == "/":
    raise SystemExit(f"unsafe isolation tree: {root}")
pending = [root]
entries = 0
while pending:
    directory = pending.pop()
    with os.scandir(directory) as children:
        for child in children:
            entries += 1
            if entries > 2_000_000:
                raise SystemExit(f"isolation tree is too large: {root}")
            metadata = child.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(child.path)
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1 and metadata.st_uid != 0:
                    raise SystemExit(
                        f"hard-linked isolation tree entry: {child.path}"
                    )
            elif not stat.S_ISLNK(metadata.st_mode):
                raise SystemExit(
                    f"unsafe isolation tree entry: {child.path}"
                )
PY
}

# Resolve and probe the complete isolation stack before untrusted code runs.
# A network-only namespace is not an acceptable fallback: replay requires one
# user, mount, PID, and network namespace together, plus capability dropping.
submission_resolve_isolation_tools() {
    local probe_root
    if [ "$SUBMISSION_ISOLATION_TOOLS_RESOLVED" = "1" ]; then
        [ "$SUBMISSION_ISOLATION_AVAILABLE" = "1" ]
        return
    fi
    SUBMISSION_ISOLATION_TOOLS_RESOLVED=1
    SUBMISSION_ISOLATION_AVAILABLE=0
    SUBMISSION_NETWORK_MODE="unavailable"
    SUBMISSION_NETWORK_PREFIX=()

    SUBMISSION_UNSHARE_BIN="$(_submission_resolve_tool unshare)" || return 1
    SUBMISSION_MOUNT_BIN="$(_submission_resolve_tool mount)" || return 1
    SUBMISSION_SETPRIV_BIN="$(_submission_resolve_tool setpriv)" || return 1
    SUBMISSION_ENV_BIN="$(_submission_resolve_tool env)" || return 1
    SUBMISSION_BASH_BIN="$(_submission_resolve_tool bash)" || return 1
    SUBMISSION_TRUE_BIN="$(_submission_resolve_tool true)" || return 1
    SUBMISSION_PIVOT_ROOT_BIN="$(_submission_resolve_tool pivot_root)" || return 1
    SUBMISSION_UMOUNT_BIN="$(_submission_resolve_tool umount)" || return 1
    SUBMISSION_MKDIR_BIN="$(_submission_resolve_tool mkdir)" || return 1
    SUBMISSION_LN_BIN="$(_submission_resolve_tool ln)" || return 1
    SUBMISSION_RMDIR_BIN="$(_submission_resolve_tool rmdir)" || return 1
    SUBMISSION_RM_BIN="$(_submission_resolve_tool rm)" || return 1
    SUBMISSION_MKTEMP_BIN="$(_submission_resolve_tool mktemp)" || return 1
    SUBMISSION_HOSTNAME_BIN="$(_submission_resolve_tool hostname)" || return 1
    if [ -x /usr/bin/python3 ]; then
        SUBMISSION_MOUNT_ATTR_PYTHON=/usr/bin/python3
    elif [[ "${SUBMISSION_BUNDLE_PYTHON:-python3}" = /* ]]; then
        SUBMISSION_MOUNT_ATTR_PYTHON="${SUBMISSION_BUNDLE_PYTHON:-python3}"
        [ -f "$SUBMISSION_MOUNT_ATTR_PYTHON" ] \
            && [ -x "$SUBMISSION_MOUNT_ATTR_PYTHON" ] || return 1
    else
        SUBMISSION_MOUNT_ATTR_PYTHON="$(_submission_resolve_tool \
            "${SUBMISSION_BUNDLE_PYTHON:-python3}")" || return 1
    fi

    probe_root="$("$SUBMISSION_MKTEMP_BIN" -d \
        /tmp/.kernelbench-isolation-probe.XXXXXXXXXX)" || return 1
    if ! "$SUBMISSION_UNSHARE_BIN" \
        --user --map-root-user --mount --pid --fork --kill-child=KILL \
        --net --ipc --uts \
        "$SUBMISSION_BASH_BIN" -c '
set -euo pipefail
mount_bin="$1"
mount_attr_python="$2"
mount_attr_code="$3"
pivot_root_bin="$4"
umount_bin="$5"
mkdir_bin="$6"
ln_bin="$7"
rmdir_bin="$8"
hostname_bin="$9"
setpriv_bin="${10}"
env_bin="${11}"
true_bin="${12}"
probe_root="${13}"
"$mount_bin" --make-rprivate /
"$hostname_bin" kernelbench-replay
"$mount_bin" -t tmpfs -o nosuid,nodev,mode=0755 tmpfs "$probe_root"
"$mkdir_bin" -p -- "$probe_root/.oldroot" "$probe_root/usr" "$probe_root/proc"
"$ln_bin" -s usr/bin "$probe_root/bin"
"$ln_bin" -s usr/sbin "$probe_root/sbin"
"$ln_bin" -s usr/lib "$probe_root/lib"
if [ -d /usr/lib64 ]; then
    "$ln_bin" -s usr/lib64 "$probe_root/lib64"
fi
"$mount_bin" --rbind /usr "$probe_root/usr"
"$mount_attr_python" -c "$mount_attr_code" "$probe_root/usr"
"$mount_bin" -t proc -o ro,nosuid,nodev,noexec proc "$probe_root/proc"
cd "$probe_root"
"$pivot_root_bin" . .oldroot
cd /
"$umount_bin" -l /.oldroot
"$rmdir_bin" /.oldroot
"$mount_attr_python" -c "$mount_attr_code" / single
exec "$setpriv_bin" --nnp --bounding-set=-all --inh-caps=-all \
    --ambient-caps=-all "$env_bin" -i PATH=/usr/bin:/bin "$true_bin"
' kernelbench-isolation-probe "$SUBMISSION_MOUNT_BIN" \
        "$SUBMISSION_MOUNT_ATTR_PYTHON" "$SUBMISSION_MOUNT_SETATTR_CODE" \
        "$SUBMISSION_PIVOT_ROOT_BIN" "$SUBMISSION_UMOUNT_BIN" \
        "$SUBMISSION_MKDIR_BIN" "$SUBMISSION_LN_BIN" "$SUBMISSION_RMDIR_BIN" \
        "$SUBMISSION_HOSTNAME_BIN" "$SUBMISSION_SETPRIV_BIN" \
        "$SUBMISSION_ENV_BIN" "$SUBMISSION_TRUE_BIN" "$probe_root" \
        >/dev/null 2>&1; then
        "$SUBMISSION_RM_BIN" -rf -- "$probe_root"
        return 1
    fi
    "$SUBMISSION_RM_BIN" -rf -- "$probe_root" || return 1

    SUBMISSION_ISOLATION_AVAILABLE=1
    SUBMISSION_NETWORK_MODE="unshare-user-mount-pid-net-private-root-v1"
    return 0
}

# Record the immutable root and run root while they are still trusted.  In the
# monorepo layout the helper is <root>/scripts/lib; on thin workers it is
# <bench>/scripts/lib, so the same default protects all shipped grader code.
submission_configure_isolation() {
    local run_root="$1" trust_root="${2:-}" helper_dir python_path
    submission_resolve_isolation_tools || return 1
    if [ -z "$trust_root" ]; then
        helper_dir="$(cd -P -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" \
            || return 1
        trust_root="$(cd -P -- "$helper_dir/../.." && pwd -P)" || return 1
    fi
    SUBMISSION_ISOLATION_TRUST_ROOT="$(_submission_real_directory "$trust_root")" \
        || return 1
    SUBMISSION_ISOLATION_RUN_ROOT="$(_submission_real_directory "$run_root")" \
        || return 1
    [ "$SUBMISSION_ISOLATION_TRUST_ROOT" != "/" ] || return 1
    [ "$SUBMISSION_ISOLATION_RUN_ROOT" != "/" ] || return 1
    case "$SUBMISSION_ISOLATION_TRUST_ROOT/" in
        /usr/*) return 1 ;;
    esac
    case "$SUBMISSION_ISOLATION_RUN_ROOT/" in
        /usr/*) return 1 ;;
    esac

    SUBMISSION_ISOLATION_PYTHON_ROOT=""
    python_path="${REAL_PYTHON:-}"
    if [ -n "$python_path" ]; then
        python_path="$("${SUBMISSION_BUNDLE_PYTHON:-python3}" -c \
            'import os,sys; print(os.path.realpath(os.path.abspath(sys.argv[1])))' \
            "$python_path")" || return 1
        case "$python_path" in
            /usr/*|/bin/*|/sbin/*|/lib/*|/lib64/*) ;;
            /*/bin/*)
                SUBMISSION_ISOLATION_PYTHON_ROOT="${python_path%/bin/*}"
                SUBMISSION_ISOLATION_PYTHON_ROOT="$(_submission_real_directory \
                    "$SUBMISSION_ISOLATION_PYTHON_ROOT")" || return 1
                [ "$SUBMISSION_ISOLATION_PYTHON_ROOT" != "/" ] || return 1
                ;;
            *) return 1 ;;
        esac
    fi
}

# Compatibility selector used by result metadata.  It intentionally performs
# no PATH lookup or late probe; callers must have resolved the full stack before
# launching the agent/submission.
submission_select_network_isolation() {
    SUBMISSION_NETWORK_PREFIX=()
    if [ "$SUBMISSION_ISOLATION_TOOLS_RESOLVED" != "1" ] \
        || [ "$SUBMISSION_ISOLATION_AVAILABLE" != "1" ]; then
        SUBMISSION_NETWORK_MODE="unavailable"
        return 1
    fi
    SUBMISSION_NETWORK_MODE="unshare-user-mount-pid-net-private-root-v1"
    return 0
}

submission_reset_caches() {
    local cache_root="$1"
    if [ -z "$cache_root" ] || [ "$cache_root" = "/" ]; then
        echo "refusing unsafe replay cache root: $cache_root" >&2
        return 1
    fi
    rm -rf -- "$cache_root" || return 1
    mkdir -p "$cache_root/torch_extensions" "$cache_root/triton" \
        "$cache_root/cuda" "$cache_root/torchinductor" \
        "$cache_root/xdg" "$cache_root/pycache" "$cache_root/tmp" \
        "$cache_root/uv" || return 1
    export TORCH_EXTENSIONS_DIR="$cache_root/torch_extensions"
    export TRITON_CACHE_DIR="$cache_root/triton"
    export CUDA_CACHE_PATH="$cache_root/cuda"
    export TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
    export XDG_CACHE_HOME="$cache_root/xdg"
    export PYTHONPYCACHEPREFIX="$cache_root/pycache"
    export TMPDIR="$cache_root/tmp"
    export TEMP="$cache_root/tmp"
    export TMP="$cache_root/tmp"
    export UV_CACHE_DIR="$cache_root/uv"
    export UV_LINK_MODE="copy"
}

# Build an absolute env(1) -i prefix for untrusted imports.  Only grading and
# CUDA controls are copied; credentials, proxy routes, Python injection knobs,
# SSH agents, and arbitrary operator state are absent by construction.
submission_select_clean_environment() {
    local replay_home="$1" name trusted_path uv_cache
    SUBMISSION_CLEAN_ENV_PREFIX=()
    submission_resolve_isolation_tools || return 1
    rm -rf -- "$replay_home" || return 1
    mkdir -p "$replay_home/.config" "$replay_home/.local/share" \
        "$replay_home/.local/state" "$replay_home/.cache/uv" || return 1
    uv_cache="${UV_CACHE_DIR:-$replay_home/.cache/uv}"
    mkdir -p "$uv_cache" || return 1
    trusted_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    if [[ "${CUDA_HOME:-}" = /* ]] && [ -d "$CUDA_HOME/bin" ]; then
        trusted_path="$CUDA_HOME/bin:$trusted_path"
    fi
    SUBMISSION_CLEAN_ENV_PREFIX=(
        "$SUBMISSION_ENV_BIN" -i
        "HOME=$replay_home"
        "USER=kernelbench"
        "LOGNAME=kernelbench"
        "SHELL=$SUBMISSION_BASH_BIN"
        "PATH=$trusted_path"
        "LANG=C.UTF-8"
        "LC_ALL=C.UTF-8"
        "TZ=UTC"
        "XDG_CONFIG_HOME=$replay_home/.config"
        "XDG_DATA_HOME=$replay_home/.local/share"
        "XDG_STATE_HOME=$replay_home/.local/state"
        "UV_OFFLINE=1"
        "UV_CACHE_DIR=$uv_cache"
        "UV_LINK_MODE=copy"
        "PYTHONNOUSERSITE=1"
        "PIP_NO_INDEX=1"
    )
    for name in \
        CUDA_HOME CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER NVIDIA_VISIBLE_DEVICES \
        LD_LIBRARY_PATH TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR CUDA_CACHE_PATH \
        TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME PYTHONPYCACHEPREFIX \
        TMPDIR TEMP TMP KBH_HARDWARE KBH_NUMERIC_STRESS; do
        if [[ -v "$name" ]]; then
            SUBMISSION_CLEAN_ENV_PREFIX+=("$name=${!name}")
        fi
    done
    while IFS= read -r name; do
        case "$name" in
            KBH_BENCHMARK_BASELINES|KBH_*_BENCHMARK_BASELINES)
                SUBMISSION_CLEAN_ENV_PREFIX+=("$name=${!name}")
                ;;
        esac
    done < <(compgen -e)
}

# Construct (but do not run) the isolated command.  The caller retains control
# of timeout/GPU locking and opens stdout/stderr before executing this array.
# Log paths live outside the writable mount, but submission code inherits the
# open descriptors and can seek or truncate them; captured text is diagnostic,
# not an authoritative completion channel.
#
# Usage:
#   submission_build_isolated_command STAGE PROJECT PROBLEM [template ...] -- CMD...
submission_build_isolated_command() {
    local stage_root="$1" project_root="$2" problem_dir="$3" template path
    local tree_real private_root
    local new_root python_root logical_python_root cuda_root cuda_real_root
    local library_root library_real_root
    local trusted_count tree_count etc_count device_count env_count
    local -a templates=() trusted_paths=() readonly_trees=()
    local -a etc_paths=() device_paths=() command=()
    shift 3
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
        templates+=("$1")
        shift
    done
    [ "$#" -gt 0 ] || { echo "isolated replay command delimiter is missing" >&2; return 1; }
    shift
    [ "$#" -gt 0 ] || { echo "isolated replay command is empty" >&2; return 1; }
    command=("$@")

    [ "$SUBMISSION_ISOLATION_AVAILABLE" = "1" ] || {
        echo "full replay namespace isolation is unavailable" >&2
        return 1
    }
    [ -n "$SUBMISSION_ISOLATION_TRUST_ROOT" ] \
        && [ -n "$SUBMISSION_ISOLATION_RUN_ROOT" ] || {
        echo "replay isolation roots were not configured" >&2
        return 1
    }
    case "${command[0]}" in
        /*) ;;
        *) echo "isolated replay executable must be an absolute path" >&2; return 1 ;;
    esac
    [ -f "${command[0]}" ] && [ -x "${command[0]}" ] || {
        echo "isolated replay executable is not a regular executable: ${command[0]}" >&2
        return 1
    }

    stage_root="$(_submission_real_directory "$stage_root")" || return 1
    project_root="$(_submission_real_directory "$project_root")" || return 1
    problem_dir="$(_submission_real_directory "$problem_dir")" || return 1
    case "$stage_root/" in
        "$SUBMISSION_ISOLATION_RUN_ROOT"/*) ;;
        *) echo "replay stage is outside the configured run root" >&2; return 1 ;;
    esac
    case "$stage_root/" in
        /tmp/*)
            echo "replay stage cannot live under the host /tmp mount" >&2
            return 1
            ;;
    esac
    case "$project_root/" in
        "$SUBMISSION_ISOLATION_RUN_ROOT"/*) ;;
        *) echo "replay project is outside the configured run root" >&2; return 1 ;;
    esac
    case "$problem_dir/" in
        "$project_root"/*) ;;
        *) echo "replay problem is outside its project" >&2; return 1 ;;
    esac

    new_root="$stage_root/.isolation-root"
    "$SUBMISSION_RM_BIN" -rf -- "$new_root" || return 1
    "$SUBMISSION_MKDIR_BIN" -p "$new_root" || return 1
    [ -d "$stage_root/cache/tmp" ] || {
        echo "private replay tmp directory is missing" >&2
        return 1
    }

    for template in "${templates[@]}"; do
        case "$template" in
            ""|.|..|*/*|*\\*)
                echo "unsafe trusted template name: $template" >&2
                return 1
                ;;
        esac
    done
    for template in \
        "$project_root/src" "$project_root/pyproject.toml" \
        "$project_root/uv.lock" "$project_root/.python-version" \
        "$project_root/.venv"; do
        if [ -e "$template" ] || [ -L "$template" ]; then
            trusted_paths+=("$template")
        fi
    done
    for template in "${templates[@]}"; do
        if [ -e "$problem_dir/$template" ] || [ -L "$problem_dir/$template" ]; then
            trusted_paths+=("$problem_dir/$template")
        fi
    done
    for template in "${trusted_paths[@]}"; do
        case "$template/" in
            "$SUBMISSION_ISOLATION_RUN_ROOT"/*) ;;
            *) echo "trusted replay path is outside the run root: $template" >&2; return 1 ;;
        esac
    done
    trusted_paths+=("${SUBMISSION_ISOLATION_EXTRA_READONLY[@]}")

    python_root="$SUBMISSION_ISOLATION_PYTHON_ROOT"
    if [ -n "$python_root" ]; then
        readonly_trees+=("$python_root")
        logical_python_root="$("$SUBMISSION_MOUNT_ATTR_PYTHON" - \
            "${command[0]}" <<'PY'
import os
import sys

current = os.path.abspath(sys.argv[1])
seen = set()
while os.path.islink(current):
    if current in seen:
        raise SystemExit("Python executable symlink loop")
    seen.add(current)
    target = os.readlink(current)
    current = os.path.normpath(
        target if os.path.isabs(target) else os.path.join(os.path.dirname(current), target)
    )
marker = f"{os.sep}bin{os.sep}"
if marker in current:
    print(current.split(marker, 1)[0])
PY
        )" || return 1
        if [ -n "$logical_python_root" ] \
            && [ "$logical_python_root" != "$python_root" ]; then
            [ -d "$logical_python_root" ] || return 1
            [ "$(_submission_real_directory "$logical_python_root")" \
                = "$python_root" ] || return 1
            readonly_trees+=("$logical_python_root")
        fi
    fi
    cuda_root="${CUDA_HOME:-}"
    if [ -n "$cuda_root" ]; then
        case "$cuda_root" in
            /*) ;;
            *) echo "CUDA_HOME must be absolute for isolated replay" >&2; return 1 ;;
        esac
        [ -d "$cuda_root" ] || return 1
        cuda_real_root="$(_submission_real_directory "$cuda_root")" || return 1
        [ "$cuda_real_root" != "/" ] || return 1
        readonly_trees+=("$cuda_real_root")
        if [ "$cuda_root" != "$cuda_real_root" ] \
            && [[ "$cuda_root" != /usr/* ]]; then
            readonly_trees+=("$cuda_root")
        fi
    fi
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        IFS=: read -r -a library_roots <<< "$LD_LIBRARY_PATH"
        for library_root in "${library_roots[@]}"; do
            case "$library_root" in
                "")
                    echo "unsafe LD_LIBRARY_PATH entry for isolated replay" >&2
                    return 1
                    ;;
                /*) ;;
                *)
                    echo "LD_LIBRARY_PATH entries must be absolute for isolated replay" >&2
                    return 1
                    ;;
            esac
            [ -d "$library_root" ] || continue
            case "$library_root/" in
                /usr/*|/lib/*|/lib64/*) continue ;;
            esac
            library_real_root="$(_submission_real_directory "$library_root")" \
                || return 1
            [ "$library_real_root" != "/" ] || return 1
            readonly_trees+=("$library_real_root")
            if [ "$library_root" != "$library_real_root" ]; then
                readonly_trees+=("$library_root")
            fi
        done
    fi
    for path in "${readonly_trees[@]}"; do
        submission_validate_isolation_tree "$path" || return 1
        tree_real="$(_submission_real_directory "$path")" || return 1
        for private_root in \
            "$SUBMISSION_ISOLATION_TRUST_ROOT" \
            "$SUBMISSION_ISOLATION_RUN_ROOT" "${HOME:-}"; do
            [ -n "$private_root" ] || continue
            private_root="$(_submission_real_directory "$private_root")" \
                || return 1
            case "$private_root/" in
                "$tree_real"/*)
                    echo "replay runtime tree is too broad: $path" >&2
                    return 1
                    ;;
            esac
        done
    done
    for path in "${readonly_trees[@]}" "${trusted_paths[@]}"; do
        case "$path" in
            /*) ;;
            *) echo "replay allowlist path is not absolute: $path" >&2; return 1 ;;
        esac
        [ -e "$path" ] || {
            echo "replay allowlist path is missing: $path" >&2
            return 1
        }
    done

    for path in \
        /etc/ld.so.cache /etc/ld.so.conf /etc/ld.so.conf.d /etc/alternatives \
        /etc/nsswitch.conf; do
        if [ -e "$path" ] || [ -L "$path" ]; then
            etc_paths+=("$path")
        fi
    done
    for path in \
        /dev/null /dev/zero /dev/full /dev/random /dev/urandom \
        /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm* \
        /dev/nvidia-caps/* /dev/dri/renderD*; do
        if [ -e "$path" ]; then
            device_paths+=("$path")
        fi
    done

    trusted_count="${#trusted_paths[@]}"
    tree_count="${#readonly_trees[@]}"
    etc_count="${#etc_paths[@]}"
    device_count="${#device_paths[@]}"
    env_count="${#SUBMISSION_CLEAN_ENV_PREFIX[@]}"
    SUBMISSION_ISOLATED_COMMAND=(
        "$SUBMISSION_UNSHARE_BIN"
        --user --map-root-user --mount --pid --fork --kill-child=KILL
        --net --ipc --uts
        "$SUBMISSION_BASH_BIN" -c '
set -euo pipefail
mount_bin="$1"
setpriv_bin="$2"
mount_attr_python="$3"
mount_attr_code="$4"
pivot_root_bin="$5"
umount_bin="$6"
mkdir_bin="$7"
ln_bin="$8"
rmdir_bin="$9"
hostname_bin="${10}"
new_root="${11}"
stage_root="${12}"
workdir="${13}"
trusted_count="${14}"
shift 14
trusted_paths=()
for ((i = 0; i < trusted_count; i++)); do
    trusted_paths+=("$1")
    shift
done
tree_count="$1"
shift
readonly_trees=()
for ((i = 0; i < tree_count; i++)); do
    readonly_trees+=("$1")
    shift
done
etc_count="$1"
shift
etc_paths=()
for ((i = 0; i < etc_count; i++)); do
    etc_paths+=("$1")
    shift
done
device_count="$1"
shift
device_paths=()
for ((i = 0; i < device_count; i++)); do
    device_paths+=("$1")
    shift
done
env_count="$1"
shift
env_args=()
for ((i = 0; i < env_count; i++)); do
    env_args+=("$1")
    shift
done
[ "$1" = "--" ]
shift
[ "$#" -gt 0 ]

make_parent() {
    target="$new_root$1"
    "$mkdir_bin" -p -- "${target%/*}"
}
readonly_bind() {
    source="$1"
    target="$new_root$source"
    make_parent "$source"
    if [ -d "$source" ]; then
        "$mkdir_bin" -p -- "$target"
        "$mount_bin" --rbind "$source" "$target"
        "$mount_attr_python" -c "$mount_attr_code" "$target"
    else
        if [ ! -e "$target" ] && [ ! -L "$target" ]; then
            : > "$target"
        fi
        "$mount_bin" --bind "$source" "$target"
        "$mount_bin" -o remount,bind,ro "$target"
    fi
}

"$mount_bin" --make-rprivate /
"$hostname_bin" kernelbench-replay

# Start from an empty filesystem.  Nothing from the host remains reachable by
# pathname unless it is explicitly mounted below; this also defeats alternate
# hard links to host AF_UNIX sockets, which finite path masking cannot do.
"$mount_bin" -t tmpfs -o nosuid,nodev,mode=0755,size=16m,nr_inodes=4096 \
    tmpfs "$new_root"
"$mkdir_bin" -p -- \
    "$new_root/.oldroot" "$new_root/usr" "$new_root/etc" \
    "$new_root/proc" "$new_root/sys" "$new_root/run" \
    "$new_root/tmp" "$new_root/dev" "$new_root/dev/shm"
printf "kernelbench:x:0:0:kernelbench:$stage_root/home:/bin/bash\n" \
    > "$new_root/etc/passwd"
printf "kernelbench:x:0:\n" > "$new_root/etc/group"
printf "127.0.0.1 localhost kernelbench-replay\n::1 localhost kernelbench-replay\n" \
    > "$new_root/etc/hosts"
"$ln_bin" -s usr/bin "$new_root/bin"
"$ln_bin" -s usr/sbin "$new_root/sbin"
"$ln_bin" -s usr/lib "$new_root/lib"
if [ -d /usr/lib64 ]; then
    "$ln_bin" -s usr/lib64 "$new_root/lib64"
fi

"$mount_bin" --rbind /usr "$new_root/usr"
"$mount_attr_python" -c "$mount_attr_code" "$new_root/usr"
for path in "${etc_paths[@]}"; do
    readonly_bind "$path"
done
"$mount_bin" --rbind /sys "$new_root/sys"
"$mount_attr_python" -c "$mount_attr_code" "$new_root/sys"
"$mount_bin" -t proc -o ro,nosuid,nodev,noexec proc "$new_root/proc"
"$mount_bin" -t tmpfs -o ro,nosuid,nodev,noexec,mode=0755,size=16m,nr_inodes=4096 \
    tmpfs "$new_root/run"
"$mount_bin" -t tmpfs -o nosuid,nodev,mode=1777,size=1g,nr_inodes=131072 \
    tmpfs "$new_root/tmp"
"$mount_bin" -t tmpfs -o nosuid,mode=0755,size=16m,nr_inodes=4096 \
    tmpfs "$new_root/dev"
"$mkdir_bin" -p -- "$new_root/dev/shm"
"$ln_bin" -s /proc/self/fd "$new_root/dev/fd"
"$ln_bin" -s /proc/self/fd/0 "$new_root/dev/stdin"
"$ln_bin" -s /proc/self/fd/1 "$new_root/dev/stdout"
"$ln_bin" -s /proc/self/fd/2 "$new_root/dev/stderr"
for path in "${device_paths[@]}"; do
    target="$new_root$path"
    make_parent "$path"
    : > "$target"
    "$mount_bin" --bind "$path" "$target"
    "$mount_bin" -o remount,bind,ro "$target"
done
"$mount_attr_python" -c "$mount_attr_code" "$new_root/dev"
"$mount_bin" -t tmpfs -o nosuid,nodev,mode=1777,size=4g,nr_inodes=262144 \
    tmpfs "$new_root/dev/shm"

# Preserve only the exact clean stage plus immutable runtime/tool roots.  The
# stage bind is deliberately non-recursive so the temporary new-root mount,
# which lives below the stage on the host, is not reflected back into itself.
"$mkdir_bin" -p -- "$new_root$stage_root"
"$mount_bin" --bind "$stage_root" "$new_root$stage_root"
"$mount_bin" -o remount,bind,rw,nosuid,nodev "$new_root$stage_root"
for path in "${readonly_trees[@]}"; do
    readonly_bind "$path"
done
for path in "${trusted_paths[@]}"; do
    readonly_bind "$path"
done

cd -- "$new_root"
"$pivot_root_bin" . .oldroot
cd /
"$umount_bin" -l /.oldroot
"$rmdir_bin" /.oldroot
"$mount_attr_python" -c "$mount_attr_code" / single

[ ! -e /.oldroot ]
cd -- "$workdir"
# Bound every regular file descriptor inherited by or opened from the replay,
# including the parent-opened stdout/stderr logs, to 128 MiB.
ulimit -f 131072
ulimit -c 0
# Do not expose unrelated parent descriptors (agent sockets, directory handles,
# or lock files) to submission code.  stdout/stderr are intentional log FDs.
for descriptor_path in /proc/self/fd/*; do
    descriptor="${descriptor_path##*/}"
    case "$descriptor" in
        0|1|2|*[!0-9]*) continue ;;
    esac
    eval "exec ${descriptor}>&-" 2>/dev/null || true
done
exec "$setpriv_bin" --nnp --bounding-set=-all --inh-caps=-all \
    --ambient-caps=-all "${env_args[@]}" "$@"
' kernelbench-isolated-replay
        "$SUBMISSION_MOUNT_BIN" "$SUBMISSION_SETPRIV_BIN"
        "$SUBMISSION_MOUNT_ATTR_PYTHON" "$SUBMISSION_MOUNT_SETATTR_CODE"
        "$SUBMISSION_PIVOT_ROOT_BIN" "$SUBMISSION_UMOUNT_BIN"
        "$SUBMISSION_MKDIR_BIN" "$SUBMISSION_LN_BIN" "$SUBMISSION_RMDIR_BIN"
        "$SUBMISSION_HOSTNAME_BIN" "$new_root" "$stage_root" "$problem_dir"
        "$trusted_count"
        "${trusted_paths[@]}" "$tree_count" "${readonly_trees[@]}"
        "$etc_count" "${etc_paths[@]}" "$device_count" "${device_paths[@]}"
        "$env_count" "${SUBMISSION_CLEAN_ENV_PREFIX[@]}"
        -- "${command[@]}"
    )
}

# Prepare a new repo-shaped workspace from an immutable bundle.  The bundle is
# extracted before trusted benchmark templates are copied in; project metadata
# and src always come from the canonical bench, never the agent workspace.
submission_prepare_replay() {
    local bundle_dir="$1" expected_digest="$2" bench_root="$3"
    local template_root="$4" replay_root="$5" problem="$6"
    shift 6
    local workspace="$replay_root/repo"
    local problem_dir="$workspace/problems/$problem"
    local template

    if [ -z "$replay_root" ] || [ "$replay_root" = "/" ]; then
        echo "refusing unsafe replay root: $replay_root" >&2
        return 1
    fi
    rm -rf -- "$replay_root"
    mkdir -p "$workspace/problems"
    submission_bundle_extract "$bundle_dir" "$expected_digest" "$problem_dir" \
        >/dev/null || return 1

    cp -a "$bench_root/src" "$workspace/src"
    cp -p "$bench_root/pyproject.toml" "$workspace/pyproject.toml"
    cp -p "$bench_root/uv.lock" "$workspace/uv.lock"
    if [ -f "$bench_root/.python-version" ]; then
        cp -p "$bench_root/.python-version" "$workspace/.python-version"
    fi
    for template in "$@"; do
        if [ -e "$template_root/$template" ]; then
            cp -p "$template_root/$template" "$problem_dir/$template"
        fi
    done
}

# Publish a verified regular file without following a destination symlink.  The
# run directory existed while host-mode agent code was active, so even this
# compatibility view must treat its old entries as attacker-controlled.  Both
# the temporary file and the final replace are relative to one opened parent
# directory, which also closes the usual path-swap race around os.replace.
submission_atomic_copy_regular() {
    local source="$1" destination="$2"
    "${SUBMISSION_BUNDLE_PYTHON:-python3}" - "$source" "$destination" <<'PY'
import os
import secrets
import stat
import sys

source = os.path.abspath(sys.argv[1])
destination = os.path.abspath(sys.argv[2])
parent = os.path.dirname(destination)
name = os.path.basename(destination)
if not name or name in {".", ".."}:
    raise SystemExit("invalid publication destination")

source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
source_flags |= getattr(os, "O_NOFOLLOW", 0)
source_fd = os.open(source, source_flags)
parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
parent_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
parent_fd = os.open(parent, parent_flags)
temporary = None
temporary_fd = None
try:
    before = os.fstat(source_fd)
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit("publication source is not a regular file")
    for _ in range(128):
        candidate = f".{name}.tmp-{secrets.token_hex(12)}"
        try:
            temporary_fd = os.open(
                candidate,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            continue
        temporary = candidate
        break
    if temporary_fd is None or temporary is None:
        raise SystemExit("could not allocate publication temporary file")

    while chunk := os.read(source_fd, 1024 * 1024):
        view = memoryview(chunk)
        while view:
            written = os.write(temporary_fd, view)
            view = view[written:]
    after = os.fstat(source_fd)
    stable_fields = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if stable_fields(before) != stable_fields(after):
        raise SystemExit("publication source changed while being copied")
    os.fchmod(temporary_fd, stat.S_IMODE(before.st_mode) & 0o777)
    os.fsync(temporary_fd)
    os.close(temporary_fd)
    temporary_fd = None
    os.replace(
        temporary,
        name,
        src_dir_fd=parent_fd,
        dst_dir_fd=parent_fd,
    )
    temporary = None
    os.fsync(parent_fd)
finally:
    if temporary_fd is not None:
        os.close(temporary_fd)
    if temporary is not None:
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
    os.close(parent_fd)
    os.close(source_fd)
PY
}

# Replace an archive-controlled output entry with a fresh regular file without
# following symlinks or treating a planted directory as a redirection target.
# Callers use this after agent descendants are gone (or while holding the
# cross-run trust lock), so the later shell redirection cannot be raced.
submission_prepare_output_file() {
    local destination="$1"
    "${SUBMISSION_BUNDLE_PYTHON:-python3}" - "$destination" <<'PY'
import os
import shutil
import stat
import sys

destination = os.path.abspath(sys.argv[1])
parent = os.path.dirname(destination)
name = os.path.basename(destination)
if not name or name in {".", ".."}:
    raise SystemExit("invalid output destination")
flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
parent_fd = os.open(parent, flags)
try:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISDIR(metadata.st_mode):
            shutil.rmtree(name, dir_fd=parent_fd)
        else:
            os.unlink(name, dir_fd=parent_fd)
    output_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    output_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    output_fd = os.open(name, output_flags, 0o600, dir_fd=parent_fd)
    os.fsync(output_fd)
    os.close(output_fd)
    os.fsync(parent_fd)
finally:
    os.close(parent_fd)
PY
}

# Remove and recreate one direct child directory of a pinned archive root.  A
# symlink such as regrade_replays -> /victim is unlinked rather than traversed.
submission_reset_run_subdirectory() {
    local run_dir="$1" name="$2"
    "${SUBMISSION_BUNDLE_PYTHON:-python3}" - "$run_dir" "$name" <<'PY'
import os
import shutil
import stat
import sys

root = os.path.abspath(sys.argv[1])
name = sys.argv[2]
if not name or name in {".", ".."} or "/" in name or "\\" in name:
    raise SystemExit("invalid archive subdirectory name")
flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
root_fd = os.open(root, flags)
try:
    try:
        metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISDIR(metadata.st_mode):
            shutil.rmtree(name, dir_fd=root_fd)
        else:
            os.unlink(name, dir_fd=root_fd)
    os.mkdir(name, 0o700, dir_fd=root_fd)
    os.fsync(root_fd)
finally:
    os.close(root_fd)
PY
}

# Maintain the historical run-root solution.py + scratch/ view, but derive it
# from a verified extraction rather than from a workspace after executing the
# submission.  Existing audit and publishing tools continue to work unchanged.
submission_project_legacy_archive() {
    local extracted_problem="$1" run_dir="$2" entry base
    [ -f "$extracted_problem/solution.py" ] || return 1
    submission_atomic_copy_regular \
        "$extracted_problem/solution.py" "$run_dir/solution.py" || return 1
    rm -rf -- "$run_dir/scratch"
    shopt -s nullglob dotglob
    for entry in "$extracted_problem"/*; do
        base="$(basename "$entry")"
        [ "$base" = "." ] && continue
        [ "$base" = ".." ] && continue
        [ "$base" = "solution.py" ] && continue
        mkdir -p "$run_dir/scratch"
        cp -a "$entry" "$run_dir/scratch/"
    done
    shopt -u nullglob dotglob
}

# Read JSON on stdin and durably publish it with a same-directory os.replace.
# result.json callers invoke this only after every other run artifact is final.
submission_atomic_write_json() {
    local destination="$1"
    "${SUBMISSION_BUNDLE_PYTHON:-python3}" -c '
import json, os, pathlib, sys, tempfile
destination = pathlib.Path(sys.argv[1])
value = json.load(sys.stdin)
fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=4, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fchmod(stream.fileno(), 0o644)
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
except BaseException:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
    raise
' "$destination"
}
