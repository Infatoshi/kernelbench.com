#!/usr/bin/env python3
"""Tiny OpenAI-compatible adapter for NVIDIA NVCF Nemotron chat endpoints.

The shared Nemotron 3 Ultra NVCF endpoint is invoked as:

    POST /v2/nvcf/pexec/functions/<function_id>

OpenCode expects an OpenAI-compatible `/v1/chat/completions` server. This proxy
translates the request shape and, when the caller asks for streaming, emits a
single Server-Sent Events completion chunk followed by `[DONE]`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DEFAULT_FUNCTION_ID = "948fe171-ce7a-4332-8bc0-5e14e90259f9"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


def _env_key() -> str | None:
    return (
        os.environ.get("NGC_API_KEY")
        or os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("NVCF_API_KEY")
    )


def _nvcf_url() -> str:
    base = os.environ.get("NVCF_BASE_URL", "https://api.nvcf.nvidia.com").rstrip("/")
    function_id = os.environ.get("NVCF_FUNCTION_ID", DEFAULT_FUNCTION_ID)
    return f"{base}/v2/nvcf/pexec/functions/{function_id}"


def _upstream_model(model: str | None) -> str:
    if model in {None, "", "nemotron-3-ultra"}:
        return os.environ.get("NVCF_MODEL", DEFAULT_MODEL)
    return model


def _completion_id() -> str:
    return f"chatcmpl-nvcf-{int(time.time() * 1000)}"


def _json_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    payload: dict[str, Any],
) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _error_payload(message: str, *, code: str = "nvcf_proxy_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": "api_error", "code": code}}


def _coerce_openai_completion(
    upstream: dict[str, Any],
    *,
    model: str,
    completion_id: str,
) -> dict[str, Any]:
    if "choices" in upstream and isinstance(upstream.get("choices"), list):
        upstream.setdefault("id", completion_id)
        upstream.setdefault("object", "chat.completion")
        upstream.setdefault("created", int(time.time()))
        upstream.setdefault("model", model)
        return upstream

    text = upstream.get("text") or upstream.get("content") or json.dumps(upstream)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": upstream.get("usage", {}),
    }


def _stream_completion(
    handler: BaseHTTPRequestHandler,
    completion: dict[str, Any],
) -> None:
    handler.send_response(200)
    handler.send_header("content-type", "text/event-stream")
    handler.send_header("cache-control", "no-cache")
    handler.send_header("connection", "keep-alive")
    handler.end_headers()

    choice = (completion.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    chunk: dict[str, Any] = {
        "id": completion.get("id", _completion_id()),
        "object": "chat.completion.chunk",
        "created": completion.get("created", int(time.time())),
        "model": completion.get("model", DEFAULT_MODEL),
        "choices": [
            {
                "index": choice.get("index", 0),
                "delta": {"role": message.get("role", "assistant")},
                "finish_reason": None,
            }
        ],
    }
    content = message.get("content")
    if content is not None:
        chunk["choices"][0]["delta"]["content"] = content
    if message.get("tool_calls") is not None:
        chunk["choices"][0]["delta"]["tool_calls"] = message["tool_calls"]
    handler.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())

    done = {
        "id": chunk["id"],
        "object": "chat.completion.chunk",
        "created": chunk["created"],
        "model": chunk["model"],
        "choices": [
            {
                "index": choice.get("index", 0),
                "delta": {},
                "finish_reason": choice.get("finish_reason", "stop"),
            }
        ],
    }
    handler.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
    handler.wfile.write(b"data: [DONE]\n\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def do_GET(self) -> None:
        if self.path == "/health":
            _json_response(self, 200, {"ok": True})
            return
        _json_response(self, 404, _error_payload(f"unknown path: {self.path}", code="not_found"))

    def do_POST(self) -> None:
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            _json_response(self, 404, _error_payload(f"unknown path: {self.path}", code="not_found"))
            return
        key = _env_key()
        if not key:
            _json_response(self, 401, _error_payload("NGC_API_KEY is required", code="missing_api_key"))
            return

        try:
            length = int(self.headers.get("content-length", "0"))
            request_body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            _json_response(self, 400, _error_payload(f"invalid JSON request: {exc}", code="bad_request"))
            return

        stream = bool(request_body.pop("stream", False))
        requested_model = request_body.get("model")
        model = _upstream_model(requested_model)
        response_model = requested_model or model
        request_body["model"] = model
        request_body.setdefault("max_tokens", int(os.environ.get("NVCF_MAX_TOKENS", "4096")))

        data = json.dumps(request_body).encode("utf-8")
        req = urllib.request.Request(
            _nvcf_url(),
            data=data,
            headers={
                "authorization": f"Bearer {key}",
                "content-type": "application/json",
                "accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                req,
                timeout=float(os.environ.get("NVCF_TIMEOUT_SECONDS", "900")),
            ) as resp:
                upstream = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            _json_response(
                self,
                exc.code,
                _error_payload(f"NVCF HTTP {exc.code}: {body}", code="nvcf_http_error"),
            )
            return
        except (OSError, json.JSONDecodeError) as exc:
            _json_response(self, 502, _error_payload(f"NVCF request failed: {exc}"))
            return

        completion = _coerce_openai_completion(
            upstream,
            model=response_model,
            completion_id=_completion_id(),
        )
        if stream:
            _stream_completion(self, completion)
        else:
            _json_response(self, 200, completion)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    host, port = server.server_address
    print(f"NVCF proxy listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
