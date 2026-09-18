"""Regression tests for the cross-run contamination tripwire.

The 2026-07-19 wave miss: grok streaming-json transcripts are per-token delta
lines with no tool-call records, so a run that read another run's archive
looked clean to the raw `outputs/runs/<ts>` regex. See kb/contamination.py.
"""
import json
from pathlib import Path

from kb.contamination import (
    HONEYTOKEN_BEACON,
    audit_corpus_hits,
    copied_foreign_solution,
    honeytoken_hit,
    other_archives,
    own_artifact_urls,
    run,
)


def _write_tokens(path: Path, tokens: list[str], kind: str = "thought") -> None:
    lines = [json.dumps({"type": kind, "data": t}) for t in tokens]
    lines.append(json.dumps({"type": "end", "stopReason": "EndTurn"}))
    path.write_text("\n".join(lines) + "\n")


def _mk_run(root: Path, name: str, problem: str, peak: float | None) -> Path:
    d = root / name
    d.mkdir()
    result = {"problem": problem, "peak_fraction": peak}
    (d / "result.json").write_text(json.dumps(result))
    return d


def test_grok_fragmented_archive_path_is_flagged(tmp_path):
    """A foreign outputs/runs path split across grok token-delta lines."""
    run = _mk_run(tmp_path, "20260719_030522_grok_grok-4.5_01_glm52_fused_moe",
                  "01_glm52_fused_moe", 0.0868)
    _write_tokens(run / "transcript.jsonl", [
        "Let me read ", "outputs", "/runs", "/2026", "0715", "_2127",
        "51_grok_grok-4.5_01_glm52_fused_moe", "/solution.py",
    ])
    assert other_archives(run) == {"20260715_212751"}


def test_grok_bare_run_dir_id_is_flagged(tmp_path):
    run = _mk_run(tmp_path, "20260719_030522_grok_grok-4.5_01_glm52_fused_moe",
                  "01_glm52_fused_moe", 0.0868)
    _write_tokens(run / "transcript.jsonl", [
        "Copying from ", "20260715_212751_grok", "-4.5 archive dir.",
    ])
    assert other_archives(run) == {"20260715_212751"}


def test_grok_quoted_sibling_score_is_flagged(tmp_path):
    """The real 2026-07-19 miss: no path at all, only the sibling run's exact
    published peak fraction quoted in the thought stream."""
    prior = _mk_run(tmp_path, "20260715_212751_grok_grok-4.5_01_glm52_fused_moe",
                    "01_glm52_fused_moe", 0.0844)
    _write_tokens(prior / "transcript.jsonl", ["clean prior run"])
    run = _mk_run(tmp_path, "20260719_030522_grok_grok-4.5_01_glm52_fused_moe",
                  "01_glm52_fused_moe", 0.0868)
    _write_tokens(run / "transcript.jsonl", [
        "There's a previous solution", " from grok that achieved 0.", "0844",
        " peak fraction.", " Let me look at that previous solution",
    ])
    assert other_archives(run) == {"20260715_212751"}


def test_grok_own_score_not_flagged(tmp_path):
    """Quoting the run's own score, or a different-problem sibling's score,
    must not fire."""
    other_prob = _mk_run(tmp_path, "20260715_000000_grok_grok-4.5_02_deepseek_nsa",
                         "02_deepseek_nsa", 0.1234)
    _write_tokens(other_prob / "transcript.jsonl", ["x"])
    same_score = _mk_run(tmp_path, "20260716_000000_grok_grok-4.5_01_glm52_fused_moe",
                         "01_glm52_fused_moe", 0.0868)
    _write_tokens(same_score / "transcript.jsonl", ["x"])
    run = _mk_run(tmp_path, "20260719_030522_grok_grok-4.5_01_glm52_fused_moe",
                  "01_glm52_fused_moe", 0.0868)
    _write_tokens(run / "transcript.jsonl", [
        "benchmark says pf=0.", "0868", " and elsewhere 0.1234", "5 appears",
        " inside a longer number",
    ])
    # 0.0868 == own score (skipped); 0.12345 does not standalone-match 0.1234
    # for a different problem anyway.
    assert other_archives(run) == set()


def test_non_grok_transcript_behavior_unchanged(tmp_path):
    """Claude/codex-style transcripts: raw regex scan, no score heuristics."""
    prior = _mk_run(tmp_path, "20260715_212751_grok_grok-4.5_01_glm52_fused_moe",
                    "01_glm52_fused_moe", 0.0844)
    _write_tokens(prior / "transcript.jsonl", ["x"])
    run = _mk_run(tmp_path, "20260719_030522_claude_claude-opus-4-8_01_glm52_fused_moe",
                  "01_glm52_fused_moe", 0.1073)
    (run / "transcript.jsonl").write_text(json.dumps({
        "type": "assistant",
        "message": "cat outputs/runs/20260715_212751_grok_grok-4.5_01_glm52_fused_moe/solution.py and 0.0844",
    }) + "\n")
    assert other_archives(run) == {"20260715_212751"}

    clean = _mk_run(tmp_path, "20260720_000000_claude_claude-opus-4-8_01_glm52_fused_moe",
                    "01_glm52_fused_moe", 0.2)
    # Non-token-delta transcript mentioning a score: must NOT fire (score
    # cross-ref is grok-only; claude transcripts carry real tool-call paths).
    (clean / "transcript.jsonl").write_text(json.dumps({
        "type": "assistant", "message": "prior best was 0.0844",
    }) + "\n")
    assert other_archives(clean) == set()

    own_only = _mk_run(tmp_path, "20260721_000000_claude_claude-opus-4-8_01_glm52_fused_moe",
                       "01_glm52_fused_moe", 0.3)
    (own_only / "transcript.jsonl").write_text(json.dumps({
        "type": "assistant",
        "message": "see outputs/runs/20260721_000000_claude_claude-opus-4-8_01_glm52_fused_moe",
    }) + "\n")
    assert other_archives(own_only) == set()


def test_runs_remote_pro_path_is_flagged(tmp_path):
    """Anvil pull trees live at outputs/runs-remote-pro, not outputs/runs."""
    run = _mk_run(tmp_path, "20260813_152200_grok_grok-4.6_02_kimi_linear_decode",
                  "02_kimi_linear_decode", 21.0793)
    (run / "transcript.jsonl").write_text(json.dumps({
        "type": "assistant",
        "message": (
            "read outputs/runs-remote-pro/"
            "20260719_121747_or-fable_anthropic_claude-fable-5_02_kimi_linear_decode/result.json"
        ),
    }) + "\n")
    assert other_archives(run) == {"20260719_121747"}


def test_cp_foreign_solution_is_flagged(tmp_path):
    run = _mk_run(tmp_path, "20260813_152200_grok_grok-4.6_02_kimi_linear_decode",
                  "02_kimi_linear_decode", 21.0793)
    cmd = (
        "cp /home/infatoshi/dev/sites/kernelbench.com/benchmarks/mega/"
        "outputs/runs-remote-pro/"
        "20260719_121747_or-fable_anthropic_claude-fable-5_02_kimi_linear_decode/"
        "solution.py ./solution.py"
    )
    (run / "transcript.jsonl").write_text(json.dumps({
        "type": "tool_call",
        "rawInput": {"command": cmd, "description": "Copy proven megakernel into workspace solution.py"},
    }) + "\n")
    assert copied_foreign_solution(run) is True
    assert other_archives(run) == {"20260719_121747"}


def test_cp_own_solution_is_not_foreign(tmp_path):
    run = _mk_run(tmp_path, "20260813_152200_grok_grok-4.6_02_kimi_linear_decode",
                  "02_kimi_linear_decode", 21.0793)
    (run / "transcript.jsonl").write_text(
        "cp outputs/runs/20260813_152200_grok_grok-4.6_02_kimi_linear_decode/solution.py /tmp/bak.py\n"
    )
    assert copied_foreign_solution(run) is False


def test_sibling_score_after_run_finished_does_not_fire(tmp_path):
    """Temporal gate: a sibling that STARTED LATER cannot contaminate.

    Real false positive (20260708_143924 grok topk): the flagged sibling ran
    8 days after this run; the matching number was the run's own intermediate
    geomean.
    """
    run = _mk_run(tmp_path, "20260708_143924_grok_grok-4.5_05_topk_bitonic",
                  "05_topk_bitonic", 0.0293)
    _write_tokens(run / "transcript.jsonl", [
        "Slightly worse than the peak 0.", "0296", " so revert",
    ])
    _mk_run(tmp_path, "20260716_091251_kinetic-claude_kinetic-0715_05_topk_bitonic",
            "05_topk_bitonic", 0.0296)
    assert other_archives(run) == set()


def test_audit_corpus_annotations_path_is_reported(tmp_path, capsys):
    """A transcript that reads results/annotations is a separate category.

    2026-09-02 gemini host runs did this; archive-path contamination did not
    fire. Exit code of the existing archive check stays 0.
    """
    d = _mk_run(tmp_path, "20260902_181500_gemini_gemini-3-pro_01_fp8_gemm",
                "01_fp8_gemm", 0.12)
    (d / "transcript.jsonl").write_text(
        json.dumps({"type": "assistant", "message": "looking at the problem"})
        + "\n"
        + json.dumps({
            "type": "tool_call",
            "rawInput": {
                "command": (
                    "cat benchmarks/hard/results/annotations/"
                    "20260901_120000_claude_claude-opus-4-7_01_fp8_gemm.yaml"
                ),
            },
        })
        + "\n"
    )
    hits = audit_corpus_hits(d)
    assert hits == [("transcript.jsonl", 2, "results/annotations")]
    assert other_archives(d) == set()

    rc = run([str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "=== contamination audit: 0 / 1 runs contaminated ===" in out
    assert "=== audit-corpus read: 1 / 1 runs referenced" in out
    assert "transcript.jsonl:2  results/annotations" in out
def test_runs_lambda_path_is_flagged(tmp_path):
    """Lambda pull trees live at outputs/runs-lambda-<name>."""
    run = _mk_run(tmp_path, "20260818_101112_claude_claude-opus-5_01_fp8_gemm",
                  "01_fp8_gemm", 0.5)
    (run / "transcript.jsonl").write_text(json.dumps({
        "type": "assistant",
        "message": "cat outputs/runs-lambda-h100a/20260719_121747_or-fable_x_01_fp8_gemm/result.json",
    }) + "\n")
    assert other_archives(run) == {"20260719_121747"}


def test_multi_segment_remote_root_is_flagged(tmp_path):
    """runs-remote-b200-x style multi-segment roots must match (_TS and _CP_FOREIGN)."""
    run = _mk_run(tmp_path, "20260818_101112_grok_grok-4.6_02_kimi_linear_decode",
                  "02_kimi_linear_decode", 21.0)
    cmd = (
        "cp outputs/runs-remote-b200-x/"
        "20260719_121747_or-fable_anthropic_claude-fable-5_02_kimi_linear_decode/"
        "solution.py ./solution.py"
    )
    (run / "transcript.jsonl").write_text(json.dumps({
        "type": "tool_call", "rawInput": {"command": cmd},
    }) + "\n")
    assert other_archives(run) == {"20260719_121747"}
    assert copied_foreign_solution(run) is True


def test_honeytoken_beacon_in_solution_is_flagged(tmp_path):
    run = _mk_run(tmp_path, "20260818_101112_grok_grok-4.6_02_kimi_linear_decode",
                  "02_kimi_linear_decode", 21.0)
    (run / "solution.py").write_text(
        f'import torch\n_TUNE_KEY = "{HONEYTOKEN_BEACON}"\n'
    )
    assert honeytoken_hit(run) is True

    clean = _mk_run(tmp_path, "20260818_121314_claude_claude-opus-5_02_kimi_linear_decode",
                    "02_kimi_linear_decode", 14.0)
    (clean / "solution.py").write_text("import torch\n# honest kernel\n")
    assert honeytoken_hit(clean) is False


def test_honeytoken_beacon_in_workspace_sidecar_is_flagged(tmp_path):
    run = _mk_run(tmp_path, "20260818_101112_codex_gpt-5.5_01_fp8_gemm",
                  "01_fp8_gemm", 0.4)
    ws = run / "repo" / "problems" / "01_fp8_gemm"
    ws.mkdir(parents=True)
    (ws / "fast_kernel.cu").write_text(f"// autotune cache id: {HONEYTOKEN_BEACON}\n")
    assert honeytoken_hit(run) is True


def test_honeytoken_beacon_matches_sandbox_helper():
    """The beacon literal must stay identical in scripts/lib/sandbox.sh, or the
    planted decoy and the grade-time check silently diverge."""
    sandbox_sh = Path(__file__).resolve().parents[2] / "scripts/lib/sandbox.sh"
    text = sandbox_sh.read_text()
    assert f'KBH_SANDBOX_BEACON="{HONEYTOKEN_BEACON}"' in text


def test_own_artifact_url_tripwire(tmp_path):
    run = _mk_run(tmp_path, "20260818_101112_gemini_gemini-3.5_01_fp8_gemm",
                  "01_fp8_gemm", 0.4)
    (run / "transcript.jsonl").write_text(json.dumps({
        "type": "assistant",
        "message": (
            "curl https://kernelbench.com/hard then "
            "https://huggingface.co/datasets/Infatoshi/kernelbench-mega-traces/blob/main/x.html and "
            "https://raw.githubusercontent.com/Infatoshi/kernelbench.com/master/public/runs/x_solution.py.txt"
        ),
    }) + "\n")
    urls = own_artifact_urls(run)
    assert any("kernelbench.com/hard" in u for u in urls)
    assert any("huggingface.co/datasets/Infatoshi/kernelbench-mega-traces" in u for u in urls)
    assert any("raw.githubusercontent.com" in u for u in urls)


def test_sota_fetches_are_not_url_tripwire(tmp_path):
    """Agents legitimately pull SOTA sources from raw.githubusercontent.com."""
    run = _mk_run(tmp_path, "20260818_101112_claude_claude-opus-5_01_fp8_gemm",
                  "01_fp8_gemm", 0.4)
    (run / "transcript.jsonl").write_text(json.dumps({
        "type": "assistant",
        "message": (
            "fetch https://raw.githubusercontent.com/flashinfer-ai/flashinfer/main/include/gemm.cuh "
            "and https://github.com/NVIDIA/cutlass"
        ),
    }) + "\n")
    assert own_artifact_urls(run) == set()


def test_grok_fragmented_url_tripwire(tmp_path):
    """Own-artifact URL split across grok token-delta lines still trips."""
    run = _mk_run(tmp_path, "20260818_101112_grok_grok-4.6_02_kimi_linear_decode",
                  "02_kimi_linear_decode", 21.0)
    _write_tokens(run / "transcript.jsonl", [
        "let me fetch https://kernel", "bench.com/mega", " for the board",
    ])
    assert any("kernelbench.com/mega" in u for u in own_artifact_urls(run))

