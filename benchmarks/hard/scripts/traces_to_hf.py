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
import os
import re
import sys
import uuid as uuidlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.viewer.parsers import parse  # noqa: E402

# ---- redaction ---------------------------------------------------------------
_ENV_VALS: list[str] = []
_envf = Path(os.path.expanduser("~/.env_vars"))
if _envf.exists():
    for _ln in _envf.read_text().splitlines():
        if "=" in _ln and "export" in _ln:
            _v = _ln.split("=", 1)[1].strip().strip('"').strip("'")
            if len(_v) >= 12:
                _ENV_VALS.append(_v)
_ENV_VALS = sorted(set(_ENV_VALS), key=len, reverse=True)
_PATS = [re.compile(p) for p in [
    r"sk-ant-oat01-[A-Za-z0-9_\-]+", r"sk-ant-api[A-Za-z0-9_\-]+",
    r"sk-proj-[A-Za-z0-9_\-]+", r"AIzaSy[A-Za-z0-9_\-]{20,}",
    r"sk-[a-z]{2,}-[A-Za-z0-9_\-]{16,}", r"sk-[A-Za-z0-9]{24,}",
    r"ghp_[A-Za-z0-9]{20,}", r"gho_[A-Za-z0-9]{20,}", r"hf_[A-Za-z0-9]{20,}",
    r"Bearer\s+[A-Za-z0-9._\-]{20,}",
]]


def _redact(s):
    if not isinstance(s, str):
        return s
    for v in _ENV_VALS:
        if v:
            s = s.replace(v, "REDACTED")
    for p in _PATS:
        s = p.sub("REDACTED", s)
    return s


def _new_uuid() -> str:
    return str(uuidlib.uuid4())


def _ts(base_epoch: int, i: int) -> str:
    import datetime
    t = datetime.datetime.utcfromtimestamp(base_epoch + i)
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def convert(run_dir: Path, out_dir: Path) -> Path | None:
    run_dir = Path(run_dir)
    rid = run_dir.name
    # pick the richest transcript
    tpath = None
    for name in ("codex_session.jsonl", "transcript.jsonl", "transcript.txt"):
        p = run_dir / name
        if p.exists() and p.stat().st_size > 0:
            tpath = p
            break
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
                content = [{
                    "type": "tool_result",
                    "tool_use_id": e.tool_result.call_id or _new_uuid(),
                    "content": _redact(e.tool_result.content or ""),
                    "is_error": bool(e.tool_result.is_error),
                }]
                emit("user", {"role": "user", "content": content})
            elif e.text:
                emit("user", {"role": "user", "content": _redact(e.text)})
        elif e.role in ("system", "compaction", "error") and e.text:
            # carry as a user-role note so it stays visible without a special schema
            label = (e.subtype or e.role)
            emit("user", {"role": "user", "content": _redact(f"[{label}] {e.text}")})

    if not lines:
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
