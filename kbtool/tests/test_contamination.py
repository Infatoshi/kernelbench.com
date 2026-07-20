"""Regression tests for the cross-run contamination tripwire.

The 2026-07-19 wave miss: grok streaming-json transcripts are per-token delta
lines with no tool-call records, so a run that read another run's archive
looked clean to the raw `outputs/runs/<ts>` regex. See kb/contamination.py.
"""
import json
from pathlib import Path

from kb.contamination import other_archives


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
