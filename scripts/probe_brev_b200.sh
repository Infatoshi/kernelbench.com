#!/usr/bin/env bash
# Probe Brev for B200 availability every 5 minutes.
# Logs: ~/.local/log/brev-b200-probe.log
# Sticky alert file when available: ~/.local/log/B200_AVAILABLE.txt
set -euo pipefail

LOG_DIR="${HOME}/.local/log"
LOG="${LOG_DIR}/brev-b200-probe.log"
STATE="${LOG_DIR}/brev-b200-probe.state"
ALERT="${LOG_DIR}/B200_AVAILABLE.txt"
mkdir -p "$LOG_DIR"

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

# Full search dump, then filter for B200-class SKUs (not B300).
raw="$(brev search 2>&1 || true)"
hits="$(printf '%s\n' "$raw" | rg -i 'B200|GB200|verda_B200|hyperstack_B200|b200' | rg -vi 'B300' || true)"

hit=0
if [[ -n "${hits// }" ]]; then
  hit=1
fi

{
  echo "---- ${ts} hit=${hit} ----"
  if [[ "$hit" -eq 1 ]]; then
    printf '%s\n' "$hits"
  else
    echo "(no B200/GB200 SKUs)"
    # keep a breadcrumb of what blackwell *is* listed
    printf '%s\n' "$raw" | rg -i 'B300|Blackwell|H200' | head -5 || true
  fi
} >>"$LOG"

prev="$(cat "$STATE" 2>/dev/null || echo 0)"
echo "$hit" >"$STATE"

if [[ "$hit" -eq 1 ]]; then
  printf '%s\n%s\n' "$ts" "$hits" >"$ALERT"
  if [[ "$prev" != "1" ]]; then
    echo "NOTIFY ${ts}: Brev B200 AVAILABLE" >>"$LOG"
    logger -t brev-b200-probe "B200 AVAILABLE at ${ts}" 2>/dev/null || true
    if command -v notify-send >/dev/null 2>&1; then
      notify-send -u critical "Brev B200 available" "$hits" 2>/dev/null || true
    fi
  fi
else
  rm -f "$ALERT"
fi

exit 0
