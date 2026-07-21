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

listed() { "$BREV" ls 2>/dev/null | awk '{print $1}' | grep -qx "$NAME"; }

if ! listed; then
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

# Deletion is async; poll until the instance drops out of brev ls.
for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
  if ! listed; then
    echo "TEARDOWN OK: '$NAME' no longer in brev ls"
    exit 0
  fi
  sleep 10
done
echo "TEARDOWN FAILED: '$NAME' still in brev ls — delete manually (billing continues!)" >&2
"$BREV" ls >&2 || true
exit 1
