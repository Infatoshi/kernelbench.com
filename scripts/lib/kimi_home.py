"""Build the per-run Kimi Code home (KIMI_CODE_HOME) for the native `kimi` harness.

Usage: kimi_home.py <home_dir> <model_alias> [effort]

The Kimi Code CLI reads everything (config.toml, AGENTS.md, skills, sessions, logs,
credentials) from KIMI_CODE_HOME, default ~/.kimi-code. A benchmark session gets its
own home inside the run archive so it never sees the operator's AGENTS.md, host hooks,
skills or old sessions, and its own session store lands in the archive:

- config.toml: the host config with `[[hooks]]` and `[secondary_model]` dropped (the
  subagent pool then stays on the requested model), `default_model` set to the model,
  thinking pinned on at the requested effort (default max) on both `[thinking]` and the
  model entry, skill merging off. The host model table, providers and OAuth refs are kept.
  `kimi-code/k3.1-pro` is added with its published entry when the host table lacks it.
- credentials/ and oauth/: symlinks to the host dirs, not copies. The OAuth access
  token lives about 15 minutes; a refresh rotates the refresh token, so it must be
  written back where the host CLI and the next run read it, under the same cross-process
  refresh lock (oauth/<slot>.lock). The credential slot name is a hash of the OAuth host
  and base URL, not of the home path, so the copied config resolves the same slot.
  The runner deletes both links once the session ends.
- device_id and region are copied (small identity/region markers, no secrets).

Prints one summary line; exits 2 when the model or effort cannot be honoured.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tomllib
from pathlib import Path

DEFAULT_EFFORT = "max"
# Published model-table entries (Kimi Code managed provider) for models a host
# config may predate. Values copied from a logged-in 2026-11 host config.
KNOWN_MODELS = {
    "kimi-code/k3.1-pro": {
        "provider": "managed:kimi-code",
        "model": "k3.1-pro",
        "max_context_size": 1048576,
        "capabilities": ["thinking", "always_thinking", "image_in", "video_in",
                         "tool_use", "dynamically_loaded_tools"],
        "display_name": "K3.1 Pro",
        "support_efforts": ["low", "high", "max"],
        "default_effort": "high",
    },
}

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(k: str) -> str:
    return k if _BARE_KEY.match(k) else json.dumps(k, ensure_ascii=False)


def _value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[ " + ", ".join(_value(x) for x in v) + " ]" if v else "[]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(f"unsupported TOML value {type(v).__name__}")


def _is_table_array(v) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(x, dict) for x in v)


def dump_toml(data: dict) -> str:
    out: list[str] = []

    def emit(table: dict, path: list[str]) -> None:
        for k, v in table.items():
            if not isinstance(v, dict) and not _is_table_array(v):
                out.append(f"{_key(k)} = {_value(v)}")
        for k, v in table.items():
            if isinstance(v, dict):
                out.append("")
                out.append("[" + ".".join(_key(p) for p in [*path, k]) + "]")
                emit(v, [*path, k])
        for k, v in table.items():
            if _is_table_array(v):
                for item in v:
                    out.append("")
                    out.append("[[" + ".".join(_key(p) for p in [*path, k]) + "]]")
                    emit(item, [*path, k])

    emit(data, [])
    return "\n".join(out).lstrip("\n") + "\n"


def build_config(host: dict, model: str, effort: str) -> tuple[dict, int]:
    cfg = dict(host)
    hooks = cfg.pop("hooks", None)
    cfg.pop("secondary_model", None)
    models = dict(cfg.get("models") or {})
    if model not in models:
        if model not in KNOWN_MODELS:
            raise SystemExit(f"STOP: model {model!r} is not in the host Kimi config "
                             f"(~/.kimi-code/config.toml [models]); known: {sorted(models)}")
        models[model] = dict(KNOWN_MODELS[model])
    entry = dict(models[model])
    efforts = entry.get("support_efforts")
    if efforts and effort not in efforts:
        raise SystemExit(f"STOP: effort {effort!r} not supported by {model} (support_efforts={efforts})")
    entry["default_effort"] = effort
    models[model] = entry
    cfg["models"] = models
    cfg["default_model"] = model
    cfg["default_permission_mode"] = "auto"
    cfg["merge_all_available_skills"] = False
    cfg["thinking"] = {**(cfg.get("thinking") or {}), "enabled": True, "effort": effort}
    return cfg, len(hooks) if isinstance(hooks, list) else 0


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(__doc__.splitlines()[2], file=sys.stderr)
        return 2
    home = Path(argv[1])
    model = argv[2]
    effort = (argv[3] if len(argv) == 4 else "") or DEFAULT_EFFORT
    host_home = Path.home() / ".kimi-code"
    host_cfg = host_home / "config.toml"
    if not host_cfg.is_file():
        print(f"STOP: {host_cfg} not found; run `kimi login` on this host first", file=sys.stderr)
        return 2
    with host_cfg.open("rb") as f:
        host = tomllib.load(f)
    try:
        cfg, dropped = build_config(host, model, effort)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2
    home.mkdir(parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    (home / "config.toml").write_text(dump_toml(cfg))
    os.chmod(home / "config.toml", 0o600)
    for name in ("device_id", "region"):
        if (host_home / name).is_file():
            shutil.copyfile(host_home / name, home / name)
            os.chmod(home / name, 0o600)
    for name in ("credentials", "oauth"):
        (host_home / name).mkdir(mode=0o700, exist_ok=True)
        link = home / name
        if link.is_symlink() or link.exists():
            if not link.is_symlink():
                print(f"STOP: {link} exists and is not a symlink", file=sys.stderr)
                return 2
            link.unlink()
        link.symlink_to(host_home / name, target_is_directory=True)
    print(f"kimi home: {home} model={model} thinking_effort={effort} host_hooks_dropped={dropped}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
