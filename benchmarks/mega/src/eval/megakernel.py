"""Megakernel authenticity evidence extractor.

This is the DETERMINISTIC half of the mega "is this really a fused megakernel?"
gate. It does NOT decide authenticity by itself -- deciding is the LLM judge's
job (see docs/megakernel_authenticity_judge.md). This module only gathers
objective, auditable signals and renders the judge prompt:

  * recursive static source  -- solution.py + every LOCAL module it imports
  * kernel count             -- @triton.jit / load_inline / __global__ void
  * forbidden-import hits     -- AST import matching (NOT raw substring)
  * graph / compile tripwire  -- CUDA graphs, torch.compile, on CODE only
  * codegen tripwire          -- exec/eval/compile/importlib + writing .py/.cu
  * obfuscation tripwire      -- getattr(x, "a"+"b"), string-concat banned tokens

Why deterministic-evidence + LLM-judge instead of a hard mechanical gate: the
red-team battery (tests/test_megakernel_evidence.py) proved a raw substring scan
is BOTH leaky (misses getattr/importlib obfuscation, A5/A6) AND brittle
(false-positives on an honest disclaimer comment, A7). So the tripwires here are
ADVISORY: they are strong hints the judge must resolve, never an auto-reject.
The one bright line that stays a hard fail lives in check.py: importing a banned
model/kernel library (AST-matched), because that is unambiguous.
"""
from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from pathlib import Path

HARNESS = {"reference", "baseline", "shapes", "check", "benchmark", "problem"}

# textual tripwire patterns (run on CODE with string/comment contents removed)
_GRAPH_RE = re.compile(
    r"CUDAGraph|cuda\.graph\b|cuda\.graphs\b|make_graphed_callables|"
    r"graph\.replay|capture_begin|graph_pool_handle"
)
_COMPILE_RE = re.compile(r"torch\.compile|torch\._dynamo|torchdynamo")
_CODE_FILE_RE = re.compile(r"\.(py|cu|cpp|cc|cxx|so|ptx)\b")
_BANNED_TOKEN_RE = re.compile(r"CUDAGraph|cuda\.graph|torch\.compile|make_graphed_callables")


# --------------------------------------------------------------------------- #
# import graph
# --------------------------------------------------------------------------- #
def dotted_imports(src: str) -> set[str]:
    """Every fully-dotted imported module name in `src` (best-effort)."""
    out: set[str] = set()
    try:
        tree = ast.parse(src)
    except Exception:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                out.add(node.module)
    return out


def _top_local(src: str) -> set[str]:
    return {m.split(".")[0] for m in dotted_imports(src)}


def _local_module_index(root: Path) -> dict[str, Path]:
    """Map local module name -> file, over root + root/scratch (like the builder).

    Some harnesses (cursor, and archived claude runs) keep the actual kernel in a
    sidecar module that solution.py imports (e.g. `import kernels`) but store it
    under scratch/. Resolve those too; never follow site-packages/.venv. Top-level
    files win over scratch/ on a name clash.
    """
    root = Path(root)
    index: dict[str, Path] = {}
    scratch = root / "scratch"
    scratch_files = sorted(scratch.rglob("*.py")) if scratch.is_dir() else []
    top_files = sorted(root.glob("*.py"))
    for p in scratch_files + top_files:  # top-level last so it wins
        if any(part in (".venv", "site-packages", "__pycache__") for part in p.parts):
            continue
        if p.stem in HARNESS or p.name == "solution.py":
            continue
        index[p.stem] = p
    return index


def gather_source(root: Path, entry: str = "solution.py") -> tuple[str, set[str], list[str]]:
    """Concatenate `entry` + every LOCAL module it imports, recursively.

    Returns (concatenated_source, all_dotted_imports, modules_read). Harness
    files (reference/shapes/check/...) are never followed. Local modules are
    resolved over root + root/scratch (see _local_module_index).
    """
    root = Path(root)
    entry_path = root / entry
    src = entry_path.read_text(errors="ignore") if entry_path.exists() else ""
    all_imports = set(dotted_imports(src))
    index = _local_module_index(root)
    seen = {Path(entry).stem}
    modules_read: list[str] = []
    queue = [m for m in _top_local(src) if m not in HARNESS and m in index]
    while queue:
        m = queue.pop()
        if m in seen:
            continue
        seen.add(m)
        try:
            s = index[m].read_text(errors="ignore")
        except Exception:
            continue
        modules_read.append(m)
        src += "\n" + s
        all_imports |= dotted_imports(s)
        for mm in _top_local(s):
            if mm not in seen and mm not in HARNESS and mm in index:
                queue.append(mm)
    return src, all_imports, modules_read


# --------------------------------------------------------------------------- #
# forbidden imports (the one bright-line; mirrors check.py)
# --------------------------------------------------------------------------- #
def _ban_module(entry: str) -> str:
    """Reduce a forbidden-list entry to the module name it bans.

    'transformers' -> transformers; 'fla.ops' -> fla.ops;
    'import reference' -> reference; 'from baseline' -> baseline.
    """
    e = entry.strip()
    for pre in ("import ", "from "):
        if e.startswith(pre):
            e = e[len(pre):].strip()
    return e.split()[0] if e else ""


def forbidden_import_hits(all_imports: set[str], forbidden: list[str]) -> list[tuple[str, str]]:
    """Which forbidden entries are actually imported (AST-matched, not substring).

    Only entries that denote a module are considered; textual technique bans
    like 'torch.compile' / 'CUDAGraph' are handled as ADVISORY tripwires, not
    here. Matching is exact or dotted-prefix so 'marlin' does not hit 'marlinx'.
    """
    hits: list[tuple[str, str]] = []
    for entry in forbidden:
        mod = _ban_module(entry)
        if not mod:
            continue
        # skip technique bans that are not module imports
        if any(tok in mod for tok in (".compile", "CUDAGraph", "graph", "graphed")):
            continue
        for imp in all_imports:
            if imp == mod or imp.startswith(mod + ".") or mod.startswith(imp + "."):
                hits.append((entry, imp))
                break
    return hits


# --------------------------------------------------------------------------- #
# code-only view (strip comments + string CONTENTS, keep structure)
# --------------------------------------------------------------------------- #
def strip_code(src: str) -> str:
    """Return `src` with comments removed and string literals blanked to "".

    Keeps identifiers/operators intact so `torch.compile(` still reads as code,
    but a docstring that merely mentions 'torch.compile' disappears, and a
    "CUDAGra"+"ph" concat becomes ""+"" so the literal token never forms.
    """
    out: list[str] = []
    try:
        toks = tokenize.generate_tokens(io.StringIO(src).readline)
        for tok in toks:
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type in (tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)):
                out.append('""')
                continue
            out.append(tok.string)
    except Exception:
        # tokenizer choked (e.g. syntax error); fall back to raw
        return src
    return " ".join(out)


# --------------------------------------------------------------------------- #
# kernel count
# --------------------------------------------------------------------------- #
def count_kernels(src: str) -> dict:
    triton = len(re.findall(r"@triton\.jit", src))
    cuda_inline = len(re.findall(r"load_inline|cpp_extension\.load\b", src))
    cuda_global = len(re.findall(r"__global__\s+void", src))
    return {
        "triton_jit": triton,
        "cuda_inline": cuda_inline,
        "cuda_global": cuda_global,
        "total": triton + cuda_inline + cuda_global,
    }


# --------------------------------------------------------------------------- #
# tripwires (advisory)
# --------------------------------------------------------------------------- #
def _ast_obfuscation(src: str) -> dict:
    """AST-level obfuscation signals that survive string stripping."""
    getattr_concat = False
    concat_banned = False
    exec_eval = False
    importlib_use = False
    writes_code = False
    has_code_file_str = False
    try:
        tree = ast.parse(src)
    except Exception:
        return {
            "getattr_concat": False, "concat_banned_token": False,
            "exec_eval_compile": False, "importlib": False, "writes_code_file": False,
        }

    def _str_of(node) -> str | None:
        # fold a (possibly nested) Add of str constants into one string
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = _str_of(node.left)
            right = _str_of(node.right)
            if left is not None and right is not None:
                return left + right
        return None

    for node in ast.walk(tree):
        # any Add of string constants -> does it fold into a banned token?
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            folded = _str_of(node)
            if folded and _BANNED_TOKEN_RE.search(folded):
                concat_banned = True
        if isinstance(node, ast.Call):
            fn = node.func
            fname = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
            if fname == "getattr" and len(node.args) >= 2:
                a = node.args[1]
                if isinstance(a, ast.BinOp) and isinstance(a.op, ast.Add):
                    getattr_concat = True
            if fname in ("exec", "eval", "compile"):
                exec_eval = True
            if fname == "import_module":
                importlib_use = True
            if fname in ("write_text", "write") or (
                fname == "open" and any(
                    isinstance(x, ast.Constant) and x.value in ("w", "wb", "a", "w+", "x")
                    for x in list(node.args) + [kw.value for kw in node.keywords]
                )
            ):
                # writing SOMETHING; flag code-write only if a code-file string is around
                writes_code = True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _CODE_FILE_RE.search(node.value):
                has_code_file_str = True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or ""
            names = [n.name for n in getattr(node, "names", [])]
            if mod == "importlib" or "importlib" in names or any(n.startswith("importlib") for n in names):
                importlib_use = True

    return {
        "getattr_concat": getattr_concat,
        "concat_banned_token": concat_banned,
        "exec_eval_compile": exec_eval,
        "importlib": importlib_use,
        "writes_code_file": bool(writes_code and has_code_file_str),
    }


def tripwires(src: str) -> dict:
    code = strip_code(src)
    obf = _ast_obfuscation(src)
    graph = bool(_GRAPH_RE.search(code))
    compile_ = bool(_COMPILE_RE.search(code))
    codegen = bool(
        obf["exec_eval_compile"] or obf["importlib"] or obf["writes_code_file"]
    )
    obfuscation = bool(obf["getattr_concat"] or obf["concat_banned_token"])
    return {
        "graph": graph,
        "compile": compile_,
        "codegen": codegen,
        "obfuscation": obfuscation,
        "detail": obf,
    }


# --------------------------------------------------------------------------- #
# top-level evidence bundle
# --------------------------------------------------------------------------- #
def _load_forbidden(root: Path) -> list[str]:
    y = root / "problem.yaml"
    if not y.exists():
        return []
    try:
        import yaml
        return yaml.safe_load(y.read_text()).get("forbidden", []) or []
    except Exception:
        return []


def extract_evidence(root: Path, forbidden: list[str] | None = None) -> dict:
    """Full deterministic evidence bundle for one solution directory."""
    root = Path(root)
    src, all_imports, modules = gather_source(root)
    if forbidden is None:
        forbidden = _load_forbidden(root)
    tw = tripwires(src)
    return {
        "root": str(root),
        "modules_read": modules,
        "imports": sorted(all_imports),
        "kernel_count": count_kernels(src),
        "forbidden_import_hits": forbidden_import_hits(all_imports, forbidden),
        "tripwires": tw,
        "source_chars": len(src),
        "_source": src,  # not for JSON dumping by default; used by judge prompt
    }


def render_judge_prompt(evidence: dict, problem: str = "", constraint: str = "") -> str:
    kc = evidence["kernel_count"]
    tw = evidence["tripwires"]
    src = evidence.get("_source", "")
    return f"""You are auditing a KernelBench-Mega submission for MEGAKERNEL AUTHENTICITY.

PROBLEM: {problem}
MANDATE: {constraint or "The scored step must be a genuine fused megakernel -- one custom kernel launch per step (see problem PROMPT.txt/problem.yaml constraints). CUDA graphs, torch.compile, and per-op kernel loops are NOT megakernels: they cut launch overhead without fusing on-chip dataflow."}

DETERMINISTIC EVIDENCE (advisory -- you decide, do not rubber-stamp):
  kernels found: {kc}
  tripwires:
    graph/CUDA-graph present in code : {tw['graph']}
    torch.compile / dynamo present   : {tw['compile']}
    runtime codegen (exec/importlib) : {tw['codegen']}
    string/getattr obfuscation       : {tw['obfuscation']}
    detail: {json.dumps(tw['detail'])}
  forbidden library imports (hard-fail elsewhere): {evidence['forbidden_import_hits']}
  local modules folded in: {evidence['modules_read']}

RECURSIVE STATIC SOURCE (solution.py + imported local modules):
<<<SOURCE
{src}
SOURCE

Decide, reasoning from the CODE not from comments/docstrings (which may lie):
  1. Is the SCORED/timed path a genuine fused custom kernel (Triton @jit or
     load_inline/__global__) that does the real work, invoked per the mandate?
  2. Or does it hide launches via CUDA graph / torch.compile / a per-op kernel
     loop, or is it plain eager PyTorch dressed up as a megakernel?
  3. Note any obfuscation (getattr string-concat, importlib codegen) as evidence
     of intent to evade the bright-line import/technique bans.

Output exactly:
  megakernel_authentic: true|false
  reason: <2-4 sentences grounded in specific lines>
"""


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Megakernel authenticity evidence extractor")
    ap.add_argument("path", help="run dir, solution dir, or path to solution.py")
    ap.add_argument("--prompt", action="store_true", help="also print the rendered judge prompt")
    ap.add_argument("--problem", default="", help="problem name for the prompt")
    args = ap.parse_args(argv)

    p = Path(args.path)
    root = p.parent if p.name == "solution.py" else p
    ev = extract_evidence(root)
    printable = {k: v for k, v in ev.items() if k != "_source"}
    print(json.dumps(printable, indent=2))
    if args.prompt:
        print("\n" + "=" * 70 + "\nJUDGE PROMPT\n" + "=" * 70)
        print(render_judge_prompt(ev, problem=args.problem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
