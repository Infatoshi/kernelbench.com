"""Anthropic-compat pass-through proxy that pins an OpenRouter provider.

Claude Code cannot add OpenRouter's `provider` routing field to its request
bodies, but OpenRouter's /api/v1/messages honors it (verified 2026-08-01:
provider.order=["novita"] + allow_fallbacks=false served Novita, is_byok
false). This proxy sits between Claude Code and OpenRouter and injects that
field into every POST JSON body, streaming the response back untouched.

Usage (the or-fable harness branch starts this when KBH_OR_PROVIDER is set):
    uv run python scripts/or_provider_proxy.py <port> <provider-slug>
Env: OR_PROXY_UPSTREAM overrides the upstream base (default
https://openrouter.ai/api). The client's own auth headers pass through.
"""
import http.client
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import os

UPSTREAM = os.environ.get("OR_PROXY_UPSTREAM", "https://openrouter.ai/api")
_U = urlsplit(UPSTREAM)

# Hop-by-hop headers that must not be forwarded either direction.
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    provider_slug = ""

    def log_message(self, fmt, *args):  # quiet; errors still raise
        sys.stderr.write("or-proxy: " + fmt % args + "\n")

    def _forward(self, body: bytes | None) -> None:
        conn_cls = http.client.HTTPSConnection if _U.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(_U.netloc, timeout=3600)
        path = (_U.path.rstrip("/") + self.path) if _U.path else self.path
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        conn.request(self.command, path, body=body, headers=headers)
        resp = conn.getresponse()
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in _HOP:
                self.send_header(k, v)
        chunked = (resp.getheader("Transfer-Encoding") or "").lower() == "chunked"
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        while True:
            chunk = resp.read(16384)
            if not chunk:
                break
            if chunked:
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            else:
                self.wfile.write(chunk)
            self.wfile.flush()
        if chunked:
            self.wfile.write(b"0\r\n\r\n")
        conn.close()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                data["provider"] = {"order": [self.provider_slug], "allow_fallbacks": False}
                body = json.dumps(data).encode()
        except Exception:
            pass  # non-JSON bodies forward untouched
        self._forward(body)

    def do_GET(self):
        self._forward(None)

    def do_HEAD(self):
        # Claude Code >= 2.1.x preflights the base URL with HEAD /api/hello and
        # aborts the whole session on a non-2xx; answer locally rather than
        # forwarding (upstream path mapping 404s it).
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


def main() -> int:
    port = int(sys.argv[1])  # 0 = OS-assigned; the bound URL is printed below
    Handler.provider_slug = sys.argv[2]
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    sys.stderr.write(
        f"or-proxy: listening http://127.0.0.1:{srv.server_address[1]} -> {UPSTREAM} "
        f"(provider={sys.argv[2]})\n"
    )
    sys.stderr.flush()
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
