"""Convert archived run transcripts into HuggingFace agent-trace JSONL.

Emits one <run_id>.jsonl per run in the native Claude-Code session schema that
the HF agent-trace-viewer auto-detects (user/assistant lines, content blocks,
uuid/parentUuid chain). Works for any harness because it goes through the
shared src.viewer.parsers normalizer. Secrets are redacted (raw transcripts
contain real API keys).

Usage:
    uv run python scripts/traces_to_hf.py <out_dir> <run_dir> [<run_dir> ...]
    uv run python scripts/traces_to_hf.py <out_dir> --from-list <rids.txt> --search <dir> [--search <dir> ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid as uuidlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MONOREPO_ROOT))

from scripts.redaction import redact_text  # noqa: E402

from src.viewer.parsers import parse  # noqa: E402


def _redact(s):
    return redact_text(s) if isinstance(s, str) else s


def _new_uuid() -> str:
    return str(uuidlib.uuid4())


def _ts(base_epoch: int, i: int) -> str:
    import datetime
    t = datetime.datetime.utcfromtimestamp(base_epoch + i)
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _pick_transcript(run_dir: Path) -> Path | None:
    """Prefer richest source per harness.

    Grok: agent_home/.grok/sessions/**/chat_history.jsonl has the real tool
    timeline; harness transcript.jsonl is only token-delta streaming-json.
    """
    run_dir = Path(run_dir)
    # Explicit high-fidelity session files first.
    for name in ("codex_session.jsonl",):
        p = run_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    grok_histories = sorted(
        (run_dir / "agent_home" / ".grok" / "sessions").rglob("chat_history.jsonl")
        if (run_dir / "agent_home" / ".grok" / "sessions").is_dir()
        else [],
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if grok_histories:
        return grok_histories[0]
    for name in ("transcript.jsonl", "transcript.txt"):
        p = run_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def convert(run_dir: Path, out_dir: Path) -> Path | None:
    run_dir = Path(run_dir)
    rid = run_dir.name
    tpath = _pick_transcript(run_dir)
    if tpath is None:
        return None
    try:
        session = parse(tpath)
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP {rid}: parse error {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
        return None

    # base epoch from run_id timestamp prefix (YYYYMMDD_HHMMSS) if present
    base = 0
    m = re.match(r"(\d{8})_(\d{6})", rid)
    if m:
        import datetime
        base = int(datetime.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp())

    sid = session.session_id or rid
    cwd = _redact(session.cwd or "/workspace")
    model = session.model or "unknown"
    # Recover model from run_id when parsers leave it as unknown/None
    if model in (None, "unknown") and "_grok_" in rid:
        # <ts>_grok_<model>_<problem>
        after = rid.split("_grok_", 1)[1]
        # strip trailing _NN_problem
        mm = re.match(r"(.+?)_\d{2}_", after)
        model = mm.group(1) if mm else after
    if model in (None, "unknown") and "_codex_" in rid:
        after = rid.split("_codex_", 1)[1]
        mm = re.match(r"(.+?)_\d{2}_", after)
        model = mm.group(1) if mm else after
    # Human-readable title from the run_id: <model> · <problem>
    _parts = rid.split("_")
    _title = f"{model} · {rid}"
    lines: list[dict] = [
        {"type": "custom-title", "customTitle": _title, "sessionId": sid},
        {"type": "ai-title", "aiTitle": _title, "sessionId": sid},
        {"type": "mode", "mode": "normal", "sessionId": sid},
    ]
    parent = None

    def emit(typ: str, message: dict) -> None:
        nonlocal parent
        u = _new_uuid()
        rec = {
            "parentUuid": parent,
            "isSidechain": False,
            "userType": "external",
            "cwd": cwd,
            "sessionId": sid,
            "version": "2.1.181",
            "gitBranch": "",
            "entrypoint": "cli",
            "type": typ,
            "message": message,
            "uuid": u,
            "timestamp": _ts(base, len(lines)),
        }
        if typ == "assistant":
            rec["requestId"] = f"req_{u[:24]}"
        else:
            rec["promptId"] = _new_uuid()
        lines.append(rec)
        parent = u

    # Skip giant system prompts / skill dumps that bloat the viewer; keep the
    # actual user_query / PROMPT and all assistant/tool turns.
    def _keep_user_text(text: str) -> bool:
        t = text.lstrip()
        if t.startswith("<user_query>") or t.startswith("I need you to"):
            return True
        if t.startswith("<user_info>"):
            return False
        if t.startswith("<system-reminder>") or t.startswith("[system]"):
            return False
        if t.startswith("You are Grok") or t.startswith("You are Codex"):
            return False
        # keep ordinary short notes; drop multi-KB skill catalogs
        if len(text) > 12000 and ("SKILL.md" in text or "skills are available" in text):
            return False
        return True

    for e in session.events:
        if e.role == "assistant":
            blocks: list[dict] = []
            if e.reasoning:
                blocks.append({"type": "thinking", "thinking": _redact(e.reasoning), "signature": ""})
            if e.text:
                blocks.append({"type": "text", "text": _redact(e.text)})
            for tc in e.tool_calls:
                args = json.loads(_redact(json.dumps(tc.args, default=str))) if tc.args else {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.call_id or _new_uuid(),
                    "name": tc.name,
                    "input": args,
                })
            if not blocks:
                continue
            msg = {"role": "assistant", "content": blocks, "id": _new_uuid(), "model": model,
                   "stop_reason": None, "type": "message"}
            if e.usage:
                msg["usage"] = {
                    "input_tokens": e.usage.input_tokens, "output_tokens": e.usage.output_tokens,
                    "cache_read_input_tokens": e.usage.cache_read_tokens,
                    "cache_creation_input_tokens": e.usage.cache_write_tokens,
                }
            emit("assistant", msg)
        elif e.role in ("user", "tool"):
            if e.tool_result is not None:
                # Cap huge tool dumps so the HF viewer stays responsive
                body = e.tool_result.content or ""
                if len(body) > 80000:
                    body = body[:80000] + "\n…[truncated]"
                content = [{
                    "type": "tool_result",
                    "tool_use_id": e.tool_result.call_id or _new_uuid(),
                    "content": _redact(body),
                    "is_error": bool(e.tool_result.is_error),
                }]
                emit("user", {"role": "user", "content": content})
            elif e.text and _keep_user_text(e.text):
                emit("user", {"role": "user", "content": _redact(e.text)})
        elif e.role in ("system", "compaction", "error") and e.text:
            # Drop the full system prompt; keep short end/status notes only.
            if e.subtype == "system" or len(e.text) > 2000:
                continue
            label = (e.subtype or e.role)
            emit("user", {"role": "user", "content": _redact(f"[{label}] {e.text}")})

    # Need at least one real conversation record beyond the title/mode headers.
    if len(lines) <= 3:
        print(f"  SKIP {rid}: no conversation events after convert", file=sys.stderr)
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{rid}.jsonl"
    out.write_text("\n".join(json.dumps(ln) for ln in lines) + "\n")
    return out



def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--from-list", type=Path, help="file of run_ids, one per line")
    ap.add_argument("--search", action="append", type=Path, default=[], help="dir(s) to find run_ids in")
    args = ap.parse_args(argv)

    targets: list[Path] = list(args.run_dirs)
    if args.from_list:
        rids = [r.strip() for r in args.from_list.read_text().split() if r.strip()]
        for rid in rids:
            found = next((s / rid for s in args.search if (s / rid).is_dir()), None)
            if found:
                targets.append(found)
            else:
                print(f"  MISSING dir for {rid}", file=sys.stderr)

    ok = 0
    for d in targets:
        if convert(d, args.out_dir):
            ok += 1
    print(f"converted {ok}/{len(targets)} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
