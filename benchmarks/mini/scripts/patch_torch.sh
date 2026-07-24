#!/bin/bash
# torch 2.11.0 + inductor CSE typing hotfix.
#
# torch._inductor.codegen.cutedsl.cutedsl_kernel.CuteDSLSubgraphInfo has:
#     cse: Optional[CSE[Any]] = None
# but CSE is Generic[CSEVariableType, AugmentedKeyT] (two params). On any
# torch.compile call that touches the cutedsl codegen path (which fires on
# Blackwell SM120), this raises TypeError at class definition time.
#
# Upstream fix landed in main branch; not yet in 2.11.0 release. Patch locally.
# Re-run this after every `uv sync` or torch upgrade.
#
# See https://github.com/pytorch/pytorch/issues/174094

set -euo pipefail

patch_file() {
    local f="$1"
    if [ ! -f "$f" ]; then
        return 0
    fi
    if grep -q "cse: Optional\[CSE\[Any, Any\]\] = None" "$f"; then
        echo "  [skip] already patched: $f"
        return 0
    fi
    if grep -q "cse: Optional\[CSE\[Any\]\] = None" "$f"; then
        sed -i.bak 's|cse: Optional\[CSE\[Any\]\] = None|cse: Optional[CSE[Any, Any]] = None|' "$f"
        # Remove stale pyc
        rm -f "$(dirname "$f")/__pycache__/cutedsl_kernel.cpython-"*.pyc
        echo "  [fix]  patched: $f"
    else
        echo "  [skip] line not found (already a newer torch?): $f"
    fi
}

echo "Applying torch inductor CSE typing hotfix..."

# Venv torch (uv creates .venv by default)
if [ -d .venv ]; then
    for f in .venv/lib/python*/site-packages/torch/_inductor/codegen/cutedsl/cutedsl_kernel.py; do
        patch_file "$f"
    done
fi

# System-wide torch (user install, python3.12)
for f in "$HOME/.local/lib/python"*/site-packages/torch/_inductor/codegen/cutedsl/cutedsl_kernel.py; do
    patch_file "$f"
done

echo "Done."
