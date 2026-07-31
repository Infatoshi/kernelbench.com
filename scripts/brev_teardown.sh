#!/usr/bin/env bash
# Reliable brev instance teardown, portable across Linux and macOS.
#
# `brev delete` has a hidden interactive "are you sure?" confirmation that
# SILENTLY HANGS with no TTY (prints nothing, never deletes; `brev stop` and
# `yes | brev delete` also no-op). The Linux fix is util-linux `script -qec`;
# macOS `script` takes different flags and the -qec form does nothing there,
# so on Darwin we drive the TTY with /usr/bin/expect (ships with macOS).
#
# Usage: brev_teardown.sh <instance-name>
# Exits 0 only after `brev ls` no longer lists the instance.
set -uo pipefail
NAME="${1:?usage: brev_teardown.sh <instance-name>}"
BREV="${BREV:-brev}"

# listing() must DISTINGUISH "brev ls worked and the instance is absent" from
# "brev ls itself failed" (expired auth, network, missing CLI). Treating a
# failed listing as absence reports a successful teardown of a still-billing
# instance — the exact failure this script exists to prevent.
listing() { "$BREV" ls 2>/dev/null; }

if ! OUT="$(listing)"; then
  echo "brev_teardown: ERROR: 'brev ls' failed — cannot confirm state of '$NAME' (billing may continue!)" >&2
  exit 1
fi
if ! grep -qx "$NAME" <<<"$(awk '{print $1}' <<<"$OUT")"; then
  echo "brev_teardown: no instance named '$NAME' in brev ls — nothing to do"
  exit 0
fi

case "$(uname -s)" in
  Darwin)
    /usr/bin/expect <<EOF
set timeout 180
spawn $BREV delete $NAME
sleep 2
send "y\r"
expect eof
EOF
    ;;
  *)
    script -qec "$BREV delete $NAME" /dev/null <<< "y"
    ;;
esac

# Deletion is async; poll until a SUCCESSFUL listing no longer shows the
# instance. A failed `brev ls` mid-poll is retried, never counted as gone.
for _ in $(seq 1 24); do
  if OUT="$(listing)"; then
    if ! grep -qx "$NAME" <<<"$(awk '{print $1}' <<<"$OUT")"; then
      echo "TEARDOWN OK: '$NAME' no longer in brev ls"
      exit 0
    fi
  else
    echo "brev_teardown: WARN: 'brev ls' failed, retrying" >&2
  fi
  sleep 10
done
echo "TEARDOWN FAILED: '$NAME' still in brev ls — delete manually (billing continues!)" >&2
"$BREV" ls >&2 || true
exit 1
