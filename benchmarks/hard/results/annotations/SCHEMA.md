# Annotation schema

Annotations attach human commentary to specific benchmark runs. They live in
`results/annotations/<run_id>.yaml`, where `<run_id>` matches the directory
name of a run under `outputs/runs/` (e.g.
`20260428_040539_claude_claude-opus-4-7_01_fp8_gemm`).

The website rendering layer reads these files alongside `leaderboard.json` and
the per-run `result.json` to add side-margin notes, callouts, and pull quotes
on the leaderboard page.

## Format

```yaml
run_id: <matches outputs/runs/ directory name>
model: <model name only — display label>
harness: <claude | codex | kimi | opencode>
effort: <max | xhigh | "" if default>
problem: <problem dir name, e.g. 01_fp8_gemm>
peak_fraction: <number, or null if not correct>

verdict: <one of: clean | rubric_leak | reward_hack | interesting | bug>

summary: >
  One-paragraph plain-language description of what's happening in this run,
  written for a leaderboard reader, not a model auditor. Surfaces the headline,
  not the procedural detail.

quotes:
  - file: solution.py | transcript.jsonl | check.log | stderr.log
    lines: [<start>, <end>]   # inclusive; or [N] for a single line
    label: <short caption shown above the pull quote>
    text: |
      <verbatim content from the file at those lines, for the website to render
      directly without re-fetching solution.py>

implication: >
  What this means for how the leaderboard cell should be read. The "so what"
  paragraph. For rubric_leak verdicts, this should explain how the cell's
  number diverges from the skill the problem name implies.
```

## Verdict semantics

- **clean** — solution does the work the problem name promises; peak_fraction
  reflects real kernel-engineering skill on the targeted algorithmic challenge.
  Use this to highlight exemplary implementations, not just to confirm
  absence-of-cheating.
- **rubric_leak** — solution passes the correctness check but takes a
  shortcut the rubric was meant to forbid (the rubric just didn't enforce it
  tightly enough). The peak_fraction is real, but it measures the wrong skill.
- **reward_hack** — solution actively games the eval (precision downcast to
  inflate "speedup", PyTorch wrapper masquerading as a custom kernel,
  hardcoded test-input handling, baseline gaming). Cell number should be
  treated as invalid.
- **interesting** — neither leak nor hack, but worth surfacing: novel
  algorithm choice, surprising failure mode, unique approach, etc.
- **bug** — harness/infra issue distorting the result (timeout, sandbox
  failure, library missing on test rig). Cell number is unreliable.

## Conventions

- Quotes should be VERBATIM from the source file at the specified lines,
  including indentation and surrounding context as it appears. Don't
  paraphrase — the consumer expects to render this directly with monospace
  formatting.
- Use 1-indexed line numbers (matches editor / GitHub line-anchor convention).
- Keep `summary` to one paragraph, ideally under 60 words.
- `implication` is optional for `clean` verdicts; required for everything else.
- One YAML document per run. To annotate multiple aspects of the same run,
  add multiple entries under `quotes`.

## Adding annotations

By hand: write the YAML file. The format is small enough that a tool isn't
strictly necessary. If a tool gets built, it should generate this schema, not
replace it.
