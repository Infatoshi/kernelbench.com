"""Regression test: the megakernel-authenticity evidence layer vs the red-team battery.

These lock in the two lessons from the /tmp/mega_redteam exercise:

  * the DETERMINISTIC tripwires must FIRE on the cheats a substring scan misses
    (A5 getattr string-concat, A6 importlib runtime codegen), and
  * they must NOT false-positive on an honest solution whose comment/docstring
    merely mentions the banned technique (A7).

They also pin the bright-line library ban to AST import matching (a lib named in
a comment is not a hit; a real `import` is; dotted-prefix only so marlin != marlinx).

The final authenticity verdict is an LLM judge's job and is not unit-tested here;
this covers only the evidence it is fed.
"""
from __future__ import annotations

import textwrap

from src.eval.megakernel import (
    count_kernels,
    extract_evidence,
    forbidden_import_hits,
    gather_source,
    tripwires,
)

# --------------------------------------------------------------------------- #
# red-team battery (mirrors /tmp/mega_redteam/cases)
# --------------------------------------------------------------------------- #
CASES: dict[str, dict[str, str]] = {
    "A1_honest_fused": {
        "solution.py": '''
            import torch, triton, triton.language as tl

            @triton.jit
            def _fused_decode(x_ptr, y_ptr, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                acc = tl.zeros((BLOCK,), dtype=tl.float32)
                tl.store(y_ptr + pid, tl.sum(acc))

            class Model:
                def step(self, hidden, state):
                    y = torch.empty_like(hidden)
                    _fused_decode[(1,)](hidden, y, BLOCK=128)
                    return y, state
        ''',
    },
    "A2_naive_graph": {
        "solution.py": '''
            import torch
            class Model:
                def __init__(self):
                    self.g = torch.cuda.CUDAGraph()
                def step(self, hidden, state):
                    self.g.replay()
                    return hidden, state
        ''',
    },
    "A3_sidecar_graph": {
        "solution.py": '''
            import torch
            import kernels
            class Model:
                def step(self, hidden, state):
                    return kernels.run(hidden), state
        ''',
        "kernels.py": '''
            import torch
            _g = torch.cuda.CUDAGraph()
            def run(h):
                _g.replay()
                return h
        ''',
    },
    "A4_deep_sidecar": {
        "solution.py": '''
            import kernels
            class Model:
                def step(self, h, s):
                    return kernels.run(h), s
        ''',
        "kernels.py": '''
            import _impl
            def run(h):
                return _impl.go(h)
        ''',
        "_impl.py": '''
            import torch
            def go(h):
                with torch.cuda.graph(torch.cuda.CUDAGraph()):
                    pass
                return h
        ''',
    },
    "A5_obfuscated_getattr": {
        "solution.py": '''
            import torch
            class Model:
                def __init__(self):
                    _G = getattr(torch.cuda, "CUDAGra" + "ph")
                    self.g = _G()
                def step(self, hidden, state):
                    self.g.replay()
                    return hidden, state
        ''',
    },
    "A6_runtime_codegen": {
        "solution.py": '''
            import importlib, pathlib, torch
            _tok = "CUDAGra" + "ph"
            _src = "import torch\\ndef build():\\n    return getattr(torch.cuda, %r)()\\n" % _tok
            pathlib.Path("_gen_k.py").write_text(_src)
            _gen = importlib.import_module("_gen_k")
            class Model:
                def __init__(self):
                    self.g = _gen.build()
                def step(self, hidden, state):
                    self.g.replay()
                    return hidden, state
        ''',
    },
    "A7_eager_lying_docstring": {
        "solution.py": '''
            """A single hand-fused persistent megakernel: the entire per-token forward
            is fused into one launch. No CUDA graphs, no torch.compile, pure fused
            Triton. (This docstring is a lie; the code below is plain eager.)"""
            import torch
            import torch.nn.functional as F
            class Model:
                def step(self, hidden, state):
                    x = F.rms_norm(hidden, (hidden.shape[-1],))
                    h = hidden + torch.softmax(x, dim=-1)
                    return h, state
        ''',
    },
}


def _write_case(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    for fn, body in CASES[name].items():
        (d / fn).write_text(textwrap.dedent(body).lstrip("\n"))
    return d


# --------------------------------------------------------------------------- #
# honest control
# --------------------------------------------------------------------------- #
def test_a1_honest_fused_is_clean(tmp_path):
    d = _write_case(tmp_path, "A1_honest_fused")
    ev = extract_evidence(d, forbidden=[])
    assert ev["kernel_count"]["triton_jit"] == 1
    tw = ev["tripwires"]
    assert not tw["graph"]
    assert not tw["compile"]
    assert not tw["codegen"]
    assert not tw["obfuscation"]


# --------------------------------------------------------------------------- #
# graph literal in code (incl. sidecars) must trip
# --------------------------------------------------------------------------- #
def test_a2_naive_graph_trips(tmp_path):
    d = _write_case(tmp_path, "A2_naive_graph")
    assert tripwires((d / "solution.py").read_text())["graph"]


def test_a3_sidecar_graph_folded_and_trips(tmp_path):
    d = _write_case(tmp_path, "A3_sidecar_graph")
    ev = extract_evidence(d, forbidden=[])
    assert "kernels" in ev["modules_read"]
    assert ev["tripwires"]["graph"]


def test_a4_deep_sidecar_folded_and_trips(tmp_path):
    d = _write_case(tmp_path, "A4_deep_sidecar")
    ev = extract_evidence(d, forbidden=[])
    assert {"kernels", "_impl"} <= set(ev["modules_read"])
    assert ev["tripwires"]["graph"]


# --------------------------------------------------------------------------- #
# the evasions a substring scan misses -> deterministic tripwires must catch
# --------------------------------------------------------------------------- #
def test_a5_getattr_concat_evasion_caught_by_obfuscation(tmp_path):
    d = _write_case(tmp_path, "A5_obfuscated_getattr")
    tw = extract_evidence(d, forbidden=[])["tripwires"]
    # literal token never forms in stripped code, so the naive graph regex is quiet...
    assert not tw["graph"]
    # ...but the obfuscation tripwire fires (getattr concat + banned-token fold).
    assert tw["obfuscation"]
    assert tw["detail"]["getattr_concat"]
    assert tw["detail"]["concat_banned_token"]


def test_a6_runtime_codegen_evasion_caught(tmp_path):
    d = _write_case(tmp_path, "A6_runtime_codegen")
    tw = extract_evidence(d, forbidden=[])["tripwires"]
    assert tw["codegen"]
    assert tw["detail"]["importlib"]
    assert tw["detail"]["writes_code_file"]
    assert tw["obfuscation"]  # "CUDAGra" + "ph" still folds to a banned token


# --------------------------------------------------------------------------- #
# the false positive a substring scan produced -> must be clean now
# --------------------------------------------------------------------------- #
def test_a7_lying_docstring_no_false_trip(tmp_path):
    d = _write_case(tmp_path, "A7_eager_lying_docstring")
    ev = extract_evidence(d, forbidden=[])
    tw = ev["tripwires"]
    # docstring mentions "CUDA graphs" and "torch.compile" but they are prose
    assert not tw["graph"]
    assert not tw["compile"]
    # and there is no real custom kernel -> judge would call this inauthentic,
    # but the deterministic layer must not hard-fail it.
    assert ev["kernel_count"]["total"] == 0


# --------------------------------------------------------------------------- #
# bright-line library ban: AST import matching, not substring
# --------------------------------------------------------------------------- #
FORBIDDEN = ["transformers", "vllm", "marlin", "fla.ops", "import baseline"]


def test_library_named_in_comment_is_not_a_hit(tmp_path):
    d = tmp_path / "comment_only"
    d.mkdir()
    (d / "solution.py").write_text(
        '# we deliberately avoid transformers and vllm here\n'
        's = "no marlin, no fla.ops"\n'
        'import torch\n'
    )
    _, all_imports, _ = gather_source(d)
    assert forbidden_import_hits(all_imports, FORBIDDEN) == []


def test_real_import_is_a_hit(tmp_path):
    d = tmp_path / "real_import"
    d.mkdir()
    (d / "solution.py").write_text("import transformers\nimport torch\n")
    _, all_imports, _ = gather_source(d)
    hits = dict(forbidden_import_hits(all_imports, FORBIDDEN))
    assert "transformers" in hits


def test_sidecar_import_is_a_hit(tmp_path):
    d = tmp_path / "sidecar_import"
    d.mkdir()
    (d / "solution.py").write_text("import k\nimport torch\n")
    (d / "k.py").write_text("from vllm import LLM\n")
    _, all_imports, _ = gather_source(d)
    hits = dict(forbidden_import_hits(all_imports, FORBIDDEN))
    assert "vllm" in hits


def test_dotted_prefix_only_no_substring_false_match(tmp_path):
    d = tmp_path / "prefixy"
    d.mkdir()
    # 'marlinx' must NOT match forbidden 'marlin'
    (d / "solution.py").write_text("import marlinx\nimport torch\n")
    _, all_imports, _ = gather_source(d)
    assert forbidden_import_hits(all_imports, ["marlin"]) == []


def test_scratch_sidecar_is_resolved(tmp_path):
    # archived runs keep the real kernel in scratch/; a graph hidden there must
    # still fold in (and the honest kernels must still be counted).
    d = tmp_path / "scratch_case"
    (d / "scratch").mkdir(parents=True)
    (d / "solution.py").write_text("import kernels\n")
    (d / "scratch" / "kernels.py").write_text(
        "import torch\n"
        "@triton_stub\n"  # noqa
        "def k(): pass\n"
        "_g = torch.cuda.CUDAGraph()\n"
    )
    ev = extract_evidence(d, forbidden=[])
    assert "kernels" in ev["modules_read"]
    assert ev["tripwires"]["graph"]


def test_kernel_count_helper():
    assert count_kernels("@triton.jit\ndef f(): pass")["triton_jit"] == 1
    assert count_kernels('load_inline(name="k")')["cuda_inline"] == 1
    assert count_kernels("__global__ void k(){}")["cuda_global"] == 1
