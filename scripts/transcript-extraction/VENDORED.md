# Vendored: transcript / reasoning extraction

Vendored from `ai-data-extraction` (GitHub `0xSero/ai-data-extraction`) so
KernelBench is self-contained — the canonical complete extractor for agent
transcripts across every harness format (codex / claude-code / cursor / gemini /
opencode / continue / trae / windsurf). This is the reference for "fully exposed
reasoning" in the agent-timeline viewers (`benchmarks/*/src/viewer`), which
otherwise under-extract (e.g. an opus transcript with 216 thinking blocks can
render as 1).

Each `extract_<agent>.py` pulls full conversations — messages, tool use, diffs,
reasoning — into a clean `{messages:[{role,content,...}], session_id, ...}`
JSONL. `extract_all.sh` runs them all.

Note: codex/gpt-5.5 encrypts its chain-of-thought, so only sparse reasoning
*summaries* exist in any codex transcript; claude-routed harnesses (opus, glm,
kimi, deepseek, minimax) carry full thinking blocks.
