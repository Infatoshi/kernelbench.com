"""Map a live nvidia-smi GPU name onto a hardware key.

Stdlib only — the harness calls this before uv/torch exist on a fresh worker.
A 3090, Quadro RTX 6000, or RTX 6000 Ada must never grade as RTX_PRO_6000.
H100 SXM and H100 PCIe are different keys. Unknown names refuse closed.
"""
from __future__ import annotations

import argparse
import json
import sys

# More specific fragments first. A match returns the key or None (hard refuse).
_RULES: list[tuple[str, str | None]] = [
    ("QUADRO RTX 6000", None),
    ("RTX 6000 ADA", None),
    ("RTX PRO 6000", "RTX_PRO_6000"),
    ("H100 SXM", "H100_SXM"),
    ("H100 80GB HBM3", "H100_SXM"),
    ("H100", "H100"),
    ("B200", "B200"),
    ("GB200", "B200"),
    ("GEFORCE RTX 3090", None),
    ("RTX 3090", None),
]


def key_from_smi_name(name: str) -> str | None:
    """Return the hardware key, or None when the SKU is unknown or forbidden."""
    n = (name or "").upper()
    if not n.strip():
        return None
    for fragment, key in _RULES:
        if fragment in n:
            return key
    return None


def matches_claimed(name: str, claimed: str) -> bool:
    live = key_from_smi_name(name)
    return live is not None and live == claimed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smi-name", required=True)
    p.add_argument("--claimed", default="")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    key = key_from_smi_name(args.smi_name)
    payload = {
        "smi_name": args.smi_name,
        "hardware_key": key,
        "claimed": args.claimed or None,
        "ok": bool(args.claimed) and key == args.claimed,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(key or "UNKNOWN")
    if args.claimed:
        return 0 if payload["ok"] else 2
    return 0 if key else 1


if __name__ == "__main__":
    sys.exit(main())
