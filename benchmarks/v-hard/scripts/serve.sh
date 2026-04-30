#!/bin/bash
# Serve a viewer output directory over HTTP for remote browsing via SSH tunnel.
#
# Usage:
#   ./scripts/serve.sh                      # serve outputs/runs/
#   ./scripts/serve.sh /tmp/viewer_demo     # serve a specific dir
#   ./scripts/serve.sh /tmp/viewer_demo 9000  # custom port
#
# On your Mac, open a separate terminal and run:
#   ssh -N -L <port>:localhost:<port> <ssh-host>
# Then point your browser at http://localhost:<port>

set -euo pipefail

DIR="${1:-outputs/runs}"
PORT="${2:-8000}"

if [ ! -d "$DIR" ]; then
    echo "directory not found: $DIR" >&2
    exit 1
fi

cd "$DIR"
echo "Serving $(pwd) on http://localhost:${PORT}"
echo "On your Mac:  ssh -N -L ${PORT}:localhost:${PORT} <ssh-host>"
echo "Then browse:  http://localhost:${PORT}"
echo
exec python3 -m http.server "$PORT"
