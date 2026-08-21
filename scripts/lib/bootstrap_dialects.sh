#!/usr/bin/env bash
# Provision the pinned CUDA dialect toolchains used by the shared runner.

set -euo pipefail

CUDA_OXIDE_REPOSITORY="https://github.com/NVlabs/cuda-oxide.git"
CUDA_OXIDE_REVISION="6c5458fe991bbde32c5bee74d87822aef1b5a691"
CUDA_OXIDE_TOOLCHAIN="nightly-2026-04-03"
CUTILE_RUST_REPOSITORY="https://github.com/NVlabs/cutile-rs.git"
CUTILE_RUST_REVISION="a3ed99d225befcb19f75ec8d81708eb35818fee2"
CUTILE_RUST_TOOLCHAIN="1.89.0"
CUTILE_RUST_CUDA_TILE_REVISION="0859212ad19f71133a9b940c05323286cbf28a05"
CUDA_TILE_PYTHON_VERSION="1.5.0"
CUTLASS_DSL_VERSION="4.7.0"
CUDA_TOOLKIT_VERSION="13.3.1"

export CARGO_HOME="${KBH_CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${KBH_RUSTUP_HOME:-$HOME/.rustup}"
export PATH="$CARGO_HOME/bin:$HOME/.local/bin:$PATH"
SOURCE_ROOT="$CARGO_HOME/kbh-dialects"
CUDA_OXIDE_ROOT="$SOURCE_ROOT/cuda-oxide-$CUDA_OXIDE_REVISION"
CUTILE_RUST_ROOT="$SOURCE_ROOT/cutile-rs-$CUTILE_RUST_REVISION"
TARGET_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/kernelbench-dialects"
CUDA_TOOLKIT_INSTALL="$CARGO_HOME/kbh-cuda-toolkit-$CUDA_TOOLKIT_VERSION"

for command in curl git uv; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "missing bootstrap command: $command" >&2
        exit 3
    }
done

find_libclang() {
    find /usr/lib -path '*/llvm-*/lib/libclang.so' -print -quit 2>/dev/null
}
libclang="$(find_libclang)"
if [ -z "$libclang" ]; then
    command -v sudo >/dev/null 2>&1 || {
        echo "libclang is missing and sudo is unavailable" >&2
        exit 3
    }
    sudo apt-get update -qq
    sudo apt-get install -y -qq libclang-dev
    libclang="$(find_libclang)"
fi
if [ -z "$libclang" ]; then
    echo "libclang is still unavailable after provisioning" >&2
    exit 3
fi
export LIBCLANG_PATH="$(dirname "$libclang")"

if [ ! -x "$CARGO_HOME/bin/rustup" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain none
fi
RUSTUP="$CARGO_HOME/bin/rustup"
"$RUSTUP" toolchain install "$CUTILE_RUST_TOOLCHAIN" --profile minimal
"$RUSTUP" toolchain install "$CUDA_OXIDE_TOOLCHAIN" --profile minimal \
    --component rust-src --component rustc-dev --component rust-analyzer \
    --component clippy --component rustfmt --component llvm-tools
"$RUSTUP" default "$CUTILE_RUST_TOOLCHAIN"

install_repository() {
    local name="$1"
    local repository="$2"
    local revision="$3"
    local destination="$4"
    local with_submodules="$5"

    if [ ! -e "$destination" ]; then
        mkdir -p "$destination"
        git -C "$destination" init --quiet
        git -C "$destination" remote add origin "$repository"
        git -C "$destination" fetch --quiet --depth 1 origin "$revision"
        git -C "$destination" checkout --quiet --detach FETCH_HEAD
    fi
    if [ ! -d "$destination/.git" ]; then
        echo "$name bootstrap path is not a git checkout: $destination" >&2
        exit 3
    fi
    local actual_revision
    actual_revision="$(git -C "$destination" rev-parse HEAD)"
    if [ "$actual_revision" != "$revision" ]; then
        echo "$name revision mismatch: expected $revision, found $actual_revision" >&2
        exit 3
    fi
    if ! git -C "$destination" diff --quiet || \
        ! git -C "$destination" diff --cached --quiet; then
        echo "$name checkout is dirty: $destination" >&2
        exit 3
    fi
    if [ "$with_submodules" = "1" ]; then
        git -C "$destination" submodule sync --quiet --recursive
        git -C "$destination" submodule update --init --recursive --depth 1
    fi
}

install_repository cuda-oxide "$CUDA_OXIDE_REPOSITORY" \
    "$CUDA_OXIDE_REVISION" "$CUDA_OXIDE_ROOT" 0
install_repository cuTile-Rust "$CUTILE_RUST_REPOSITORY" \
    "$CUTILE_RUST_REVISION" "$CUTILE_RUST_ROOT" 1

actual_submodule="$(git -C "$CUTILE_RUST_ROOT" rev-parse HEAD:cuda-tile-rs/cuda-tile)"
if [ "$actual_submodule" != "$CUTILE_RUST_CUDA_TILE_REVISION" ]; then
    echo "cuTile Rust CUDA Tile submodule mismatch: expected $CUTILE_RUST_CUDA_TILE_REVISION, found $actual_submodule" >&2
    exit 3
fi

link_revision() {
    local target="$1"
    local link="$2"
    if [ -e "$link" ] && [ ! -L "$link" ]; then
        echo "refusing to replace non-symlink dialect path: $link" >&2
        exit 3
    fi
    ln -sfn "$target" "$link"
}
link_revision "$CUDA_OXIDE_ROOT" "$CARGO_HOME/cuda-oxide"
link_revision "$CUTILE_RUST_ROOT" "$CARGO_HOME/cutile-rs"

python_bin="${KBH_DIALECT_PYTHON:-.venv/bin/python}"
if [ ! -x "$python_bin" ]; then
    echo "canonical Python environment is missing: $python_bin (run uv sync first)" >&2
    exit 3
fi
uv pip install --target "$CUDA_TOOLKIT_INSTALL" --upgrade \
    "cuda-toolkit[all]==$CUDA_TOOLKIT_VERSION"
cuda_toolkit_root="$CUDA_TOOLKIT_INSTALL/nvidia/cu13"
if [ ! -x "$cuda_toolkit_root/bin/nvcc" ]; then
    echo "pinned CUDA compiler is missing: $cuda_toolkit_root/bin/nvcc" >&2
    exit 3
fi
ln -sfn "$cuda_toolkit_root/bin/nvcc" "$(dirname "$python_bin")/nvcc"
export CUDA_HOME="$cuda_toolkit_root"
export CUDA_TOOLKIT_PATH="$cuda_toolkit_root"

mkdir -p "$TARGET_ROOT/cuda-oxide" "$TARGET_ROOT/cutile-rust"
(
    cd "$CUDA_OXIDE_ROOT"
    cargo "+$CUDA_OXIDE_TOOLCHAIN" fetch --locked
    CARGO_TARGET_DIR="$TARGET_ROOT/cuda-oxide" \
        cargo "+$CUDA_OXIDE_TOOLCHAIN" check --locked
)
(
    cd "$CUTILE_RUST_ROOT"
    cargo "+$CUTILE_RUST_TOOLCHAIN" fetch --locked
    CARGO_TARGET_DIR="$TARGET_ROOT/cutile-rust" \
        cargo "+$CUTILE_RUST_TOOLCHAIN" check --locked
)

"$python_bin" -I - <<PY
from importlib.metadata import version

expected = {
    "cuda-tile": "$CUDA_TILE_PYTHON_VERSION",
    "nvidia-cutlass-dsl": "$CUTLASS_DSL_VERSION",
}
actual = {name: version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"Python CUDA dialect version mismatch: expected {expected}, found {actual}")
import cuda.tile  # noqa: F401
import cutlass.cute  # noqa: F401
import triton  # noqa: F401
PY

printf 'CUDA dialect environment ready:\n'
printf '  cuda-oxide %s (%s)\n' "$CUDA_OXIDE_REVISION" "$CUDA_OXIDE_TOOLCHAIN"
printf '  cutile-rs  %s (%s; cuda-tile %s)\n' \
    "$CUTILE_RUST_REVISION" "$CUTILE_RUST_TOOLCHAIN" "$CUTILE_RUST_CUDA_TILE_REVISION"
printf '  cuda-tile Python %s; nvidia-cutlass-dsl %s\n' \
    "$CUDA_TILE_PYTHON_VERSION" "$CUTLASS_DSL_VERSION"
printf '  CUDA toolkit %s at %s\n' "$CUDA_TOOLKIT_VERSION" "$cuda_toolkit_root"
